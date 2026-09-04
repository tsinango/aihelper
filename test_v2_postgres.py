"""Optional PostgreSQL integration tests for the V2 learning loop.

Set ``V2_TEST_DATABASE_URL`` to an initialized database containing migration
013.  Every test rolls its writes back; only PostgreSQL identity sequences may
advance, which is harmless.
"""

from __future__ import annotations

import json
import os
import unittest
import uuid

import psycopg
from psycopg.rows import dict_row

from v2.learning import learn_turn


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


class SequenceLLM:
    def __init__(self, facts: list[str]):
        self._facts = iter(facts)

    def extract(self, _messages, max_tokens=800):
        content = next(self._facts)
        return json.dumps(
            {
                "facts": [
                    {
                        "title": "PostgreSQL integration fact",
                        "content": content,
                        "entity_name": content.split()[0],
                    }
                ]
            },
            ensure_ascii=False,
        )


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2PostgresIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        # Keep an outer transaction open. learn_turn() then uses savepoints,
        # and tearDown can roll back every row written by the test.
        self.conn.execute("SELECT 1")

    def tearDown(self):
        self.conn.rollback()
        self.conn.close()

    def _fact(self, suffix: str) -> str:
        return f"V2-TEST-{uuid.uuid4().hex[:12].upper()} {suffix}"

    def test_explicit_confirmation_promotes_knowledge_and_keeps_sources(self):
        fact = self._fact("机箱高度是 1U")
        first = learn_turn(self.conn, fact, llm_service=SequenceLLM([fact]))
        self.assertEqual(first["status"], "awaiting_confirmation")
        session_type = self.conn.execute(
            "SELECT session_type FROM v2_learning_sessions WHERE thread_id=%s AND status='active'",
            (first["thread_id"],),
        ).fetchone()["session_type"]
        self.assertEqual(session_type, "active_inbox")

        confirmed = learn_turn(self.conn, "对", thread_id=first["thread_id"])
        self.assertEqual(confirmed["status"], "confirmed")
        knowledge = self.conn.execute(
            "SELECT id, trust, active FROM v2_knowledge WHERE content=%s",
            (fact,),
        ).fetchone()
        self.assertEqual(knowledge["trust"], "user_confirmed")
        self.assertTrue(knowledge["active"])
        source_kinds = {
            row["source_kind"]
            for row in self.conn.execute(
                "SELECT source_kind FROM v2_knowledge_sources WHERE knowledge_id=%s",
                (knowledge["id"],),
            ).fetchall()
        }
        self.assertEqual(source_kinds, {"user_input", "user_confirmation"})

    def test_skip_uses_the_persisted_skipped_status(self):
        fact = self._fact("支持测试功能")
        first = learn_turn(self.conn, fact, llm_service=SequenceLLM([fact]))
        result = learn_turn(self.conn, "跳过", thread_id=first["thread_id"])
        self.assertEqual(result["status"], "skipped")
        status = self.conn.execute(
            "SELECT status FROM v2_learning_proposals WHERE thread_id=%s",
            (first["thread_id"],),
        ).fetchone()["status"]
        self.assertEqual(status, "skipped")

    def test_correction_retires_rejected_interpretation_without_deleting_it(self):
        original = self._fact("支持测试功能")
        corrected = original.replace("支持", "不支持")
        llm = SequenceLLM([original, corrected])
        first = learn_turn(self.conn, original, llm_service=llm)
        second = learn_turn(
            self.conn,
            corrected,
            thread_id=first["thread_id"],
            llm_service=llm,
        )
        self.assertEqual(second["status"], "awaiting_confirmation")

        rows = self.conn.execute(
            "SELECT content, active FROM v2_knowledge WHERE content IN (%s, %s) ORDER BY content",
            (original, corrected),
        ).fetchall()
        self.assertEqual(len(rows), 2)
        states = {row["content"]: row["active"] for row in rows}
        self.assertFalse(states[original])
        self.assertTrue(states[corrected])
        proposal_statuses = {
            row["fact_text"]: row["status"]
            for row in self.conn.execute(
                "SELECT fact_text, status FROM v2_learning_proposals WHERE thread_id=%s",
                (first["thread_id"],),
            ).fetchall()
        }
        self.assertEqual(proposal_statuses[original], "corrected")
        self.assertEqual(proposal_statuses[corrected], "pending_confirmation")

    def test_correction_keeps_other_independent_provisional_source_active(self):
        original = self._fact("支持测试功能")
        corrected = original.replace("支持", "不支持")
        first = learn_turn(self.conn, original, llm_service=SequenceLLM([original]))
        second = learn_turn(self.conn, original, llm_service=SequenceLLM([original]))

        learn_turn(
            self.conn,
            corrected,
            thread_id=first["thread_id"],
            llm_service=SequenceLLM([corrected]),
        )

        knowledge = self.conn.execute(
            "SELECT id, active FROM v2_knowledge WHERE content=%s",
            (original,),
        ).fetchone()
        self.assertTrue(knowledge["active"])
        active_sources = self.conn.execute(
            "SELECT count(*) AS count FROM v2_knowledge_sources WHERE knowledge_id=%s AND active=TRUE",
            (knowledge["id"],),
        ).fetchone()["count"]
        self.assertEqual(active_sources, 1)
        self.assertNotEqual(first["thread_id"], second["thread_id"])


if __name__ == "__main__":
    unittest.main()
