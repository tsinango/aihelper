"""Optional PostgreSQL integration tests for the V2 learning loop.

Set ``V2_TEST_DATABASE_URL`` to an initialized database containing migration
013.  Every test rolls its writes back; only PostgreSQL identity sequences may
advance, which is harmless.
"""

from __future__ import annotations

import json
import os
import re
import unittest
import uuid

import psycopg
from psycopg.rows import dict_row

from v2.learning import learn_turn


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


class SequenceLLM:
    def __init__(self, facts: list[str], decisions: list[dict | str] | None = None):
        self._facts = iter(facts)
        self._decisions = iter(decisions or ["NEW"] * len(facts))

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

    def judge(self, messages, max_tokens=1000):
        decision = next(self._decisions)
        if isinstance(decision, dict):
            return json.dumps(decision, ensure_ascii=False)
        if decision in {"AUTO_CONFIRM", "AUTO_ENRICH", "AUTO_CONFLICT"}:
            match = re.search(r'"id":(\d+)', messages[-1]["content"])
            if not match:
                raise AssertionError(f"{decision} expected a retrieved candidate")
            action = decision.removeprefix("AUTO_")
            return json.dumps({
                "decision": action,
                "knowledge_id": int(match.group(1)),
                "question": (
                    "这个补充具体适用于哪个硬件版本？"
                    if action == "ENRICH" else
                    "这两条产品说法中，哪一条适用于当前版本？"
                    if action == "CONFLICT" else None
                ),
                "reason": "同一事实",
            }, ensure_ascii=False)
        return json.dumps({
            "decision": decision,
            "knowledge_id": None,
            "question": None,
            "reason": "测试判定",
        }, ensure_ascii=False)


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        vector = [1.0] + [0.0] * 2047
        return [vector for _ in texts]


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
        first = learn_turn(
            self.conn,
            fact,
            llm_service=SequenceLLM([fact]),
            embedding_client=FakeEmbedder(),
        )
        self.assertEqual(first["status"], "awaiting_confirmation")
        session_type = self.conn.execute(
            "SELECT session_type FROM v2_learning_sessions WHERE thread_id=%s AND status='active'",
            (first["thread_id"],),
        ).fetchone()["session_type"]
        self.assertEqual(session_type, "active_inbox")

        confirmed = learn_turn(
            self.conn,
            "对",
            thread_id=first["thread_id"],
            embedding_client=FakeEmbedder(),
        )
        self.assertEqual(confirmed["status"], "confirmed")
        knowledge = self.conn.execute(
            "SELECT id, trust, active FROM v2_knowledge WHERE content=%s",
            (fact,),
        ).fetchone()
        self.assertEqual(knowledge["trust"], "user_confirmed")
        self.assertTrue(knowledge["active"])
        embedded = self.conn.execute(
            "SELECT embedding_model IS NOT NULL AS embedded FROM v2_knowledge WHERE id=%s",
            (knowledge["id"],),
        ).fetchone()["embedded"]
        self.assertTrue(embedded)
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
        llm = SequenceLLM([original, corrected], ["NEW", "NEW"])
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
        second = learn_turn(
            self.conn,
            original,
            llm_service=SequenceLLM([original], ["AUTO_CONFIRM"]),
        )

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

    def test_unclear_asks_for_product_detail_before_confirmation(self):
        ambiguous = self._fact("新版和以前不一样")
        precise = ambiguous.replace("新版和以前不一样", "硬件 revision B 使用新接口")
        unclear = {
            "decision": "UNCLEAR",
            "knowledge_id": None,
            "question": "这里的新版具体指哪个硬件 revision？",
            "reason": "新版含义不明确",
        }
        llm = SequenceLLM([ambiguous, precise], [unclear, "NEW"])
        first = learn_turn(self.conn, ambiguous, llm_service=llm)
        self.assertEqual(first["status"], "awaiting_clarification")
        self.assertEqual(first["message"]["message_type"], "clarification")
        self.assertIn("硬件 revision", first["message"]["content"])
        self.assertIn("固件版本", first["message"]["content"])
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) AS count FROM v2_knowledge WHERE content=%s",
                (ambiguous,),
            ).fetchone()["count"],
            0,
        )

        second = learn_turn(
            self.conn,
            "指硬件 revision B",
            thread_id=first["thread_id"],
            llm_service=llm,
        )
        self.assertEqual(second["message"]["message_type"], "question")
        self.assertIn(precise, second["message"]["content"])
        statuses = [
            row["status"]
            for row in self.conn.execute(
                "SELECT status FROM v2_learning_proposals WHERE thread_id=%s ORDER BY id",
                (first["thread_id"],),
            ).fetchall()
        ]
        self.assertEqual(statuses, ["superseded", "pending_confirmation"])

    def test_semantic_confirm_reuses_existing_knowledge(self):
        original = self._fact("机箱高度是 1U")
        paraphrase = original.replace("机箱高度是", "机身高度为")
        first = learn_turn(
            self.conn,
            original,
            llm_service=SequenceLLM([original]),
            embedding_client=FakeEmbedder(),
        )
        learn_turn(
            self.conn,
            "对",
            thread_id=first["thread_id"],
            embedding_client=FakeEmbedder(),
        )

        second = learn_turn(
            self.conn,
            paraphrase,
            llm_service=SequenceLLM([paraphrase], ["AUTO_CONFIRM"]),
            embedding_client=FakeEmbedder(),
        )
        self.assertEqual(second["proposal"]["comparison_result"], "CONFIRM")
        learn_turn(
            self.conn,
            "对",
            thread_id=second["thread_id"],
            embedding_client=FakeEmbedder(),
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT count(*) AS count FROM v2_knowledge WHERE entity_name=%s AND active=TRUE",
                (original.split()[0],),
            ).fetchone()["count"],
            1,
        )

    def test_enrich_and_conflict_ask_before_any_confirmation(self):
        base = self._fact("支持声音检测")
        first = learn_turn(
            self.conn,
            base,
            llm_service=SequenceLLM([base]),
            embedding_client=FakeEmbedder(),
        )
        learn_turn(
            self.conn,
            "对",
            thread_id=first["thread_id"],
            embedding_client=FakeEmbedder(),
        )

        for action, text in (
            ("AUTO_ENRICH", base + "，旧版需要支架"),
            ("AUTO_CONFLICT", base.replace("支持", "不支持")),
        ):
            with self.subTest(action=action):
                turn = learn_turn(
                    self.conn,
                    text,
                    llm_service=SequenceLLM([text], [action]),
                    embedding_client=FakeEmbedder(),
                )
                self.assertEqual(turn["status"], "awaiting_clarification")
                self.assertEqual(turn["message"]["message_type"], "clarification")
                self.assertIn(
                    turn["proposal"]["comparison_result"],
                    {"ENRICH", "CONFLICT"},
                )
                self.assertEqual(
                    self.conn.execute(
                        "SELECT count(*) AS count FROM v2_knowledge WHERE trust='user_confirmed' AND content=%s",
                        (text,),
                    ).fetchone()["count"],
                    0,
                )
                skipped = learn_turn(
                    self.conn,
                    "跳过",
                    thread_id=turn["thread_id"],
                )
                self.assertEqual(skipped["status"], "skipped")

    def test_conflict_clarification_keeps_inactive_audit_and_closes_source(self):
        base = self._fact("支持声音检测")
        first = learn_turn(
            self.conn,
            base,
            llm_service=SequenceLLM([base]),
            embedding_client=FakeEmbedder(),
        )
        learn_turn(
            self.conn,
            "对",
            thread_id=first["thread_id"],
            embedding_client=FakeEmbedder(),
        )

        conflict = base.replace("支持", "不支持")
        turn = learn_turn(
            self.conn,
            conflict,
            llm_service=SequenceLLM([conflict], ["AUTO_CONFLICT"]),
            embedding_client=FakeEmbedder(),
        )
        conflict_knowledge = self.conn.execute(
            "SELECT id, active, trust FROM v2_knowledge WHERE content=%s",
            (conflict,),
        ).fetchone()
        self.assertEqual(conflict_knowledge["trust"], "conflicted")
        self.assertFalse(conflict_knowledge["active"])

        clarified = base.replace("支持声音检测", "硬件 revision B 不支持声音检测")
        learn_turn(
            self.conn,
            "只有硬件 revision B 不支持",
            thread_id=turn["thread_id"],
            llm_service=SequenceLLM([clarified], ["NEW"]),
        )
        source = self.conn.execute(
            """
            SELECT active, resolution
            FROM v2_knowledge_sources
            WHERE knowledge_id=%s
            """,
            (conflict_knowledge["id"],),
        ).fetchone()
        self.assertFalse(source["active"])
        self.assertEqual(source["resolution"], "superseded")
        self.assertIsNotNone(self.conn.execute(
            "SELECT id FROM v2_knowledge WHERE id=%s AND active=FALSE",
            (conflict_knowledge["id"],),
        ).fetchone())


if __name__ == "__main__":
    unittest.main()
