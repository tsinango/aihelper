"""PostgreSQL integration tests for the Phase 3.2 correction loop.

Set ``V2_TEST_DATABASE_URL`` to an initialized database containing
migrations 001-022.  Every test commits through the real service functions
and then removes its letter-tagged rows explicitly (FK children first), so
no test data survives.  Letters-only tags keep the model-identifier regex
out of conflict checks.
"""

from __future__ import annotations

import json
import os
import random
import string
import unittest
import uuid

import psycopg
from psycopg.rows import dict_row

from v2.answering import answer_question
from v2.feedback import (
    FeedbackConflict,
    StaleRevision,
    close_feedback,
    confirm_feedback,
    count_unresolved_feedback,
    create_feedback,
    get_feedback,
    list_unresolved_feedback,
    retest_feedback,
    set_answer_verdict,
)
from v2.retrieval import retrieve_for_answer
from v2.service import edit_knowledge


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


class FakeLLM:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def judge(self, messages, max_tokens=600):
        self.calls.append((messages, max_tokens))
        if self.error:
            raise self.error
        return self.responses.pop(0)


def answered_json(answer="Ответ на русском языке."):
    return json.dumps({
        "status": "answered", "answer": answer,
        "clarifying_question": "", "source_indexes": [0],
        "confidence": 0.9,
    }, ensure_ascii=False)


def _tag():
    return "ZZ" + "".join(random.choice(string.ascii_uppercase) for _ in range(8))


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2FeedbackPostgresTest(unittest.TestCase):
    def setUp(self):
        self.tag = _tag()
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def tearDown(self):
        try:
            self.conn.rollback()
            with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM v2_knowledge_history WHERE knowledge_id IN ("
                        "SELECT id FROM v2_knowledge WHERE title LIKE %s)",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge_sources WHERE excerpt LIKE %s",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_answer_feedback WHERE idempotency_key LIKE %s",
                        (f"{self.tag}-%",),
                    )
                    cur.execute(
                        "DELETE FROM v2_learning_proposals WHERE fact_text LIKE %s",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge WHERE title LIKE %s",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_raw_evidence WHERE source_label LIKE %s",
                        ("Answer feedback",),
                    )
                    cur.execute(
                        "DELETE FROM v2_answer_runs WHERE idempotency_key LIKE %s",
                        (f"{self.tag}-%",),
                    )
        finally:
            self.conn.rollback()
            self.conn.close()

    def _factory(self):
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def _run(self, question="Подскажите, где в настройках камеры включается ONVIF?"):
        with self._factory() as conn:
            result = answer_question(
                f"{self.tag} {question}",
                context={},
                idempotency_key=f"{self.tag}-{uuid.uuid4().hex[:8]}",
                db_factory=self._factory,
                llm_service=FakeLLM([answered_json("старый ответ")]),
            )
        self.conn.commit()
        return result

    def _submit_experience(self, run_id, key="exp", text=None, **kwargs):
        params = {
            "answer_run_id": int(run_id),
            "idempotency_key": f"{self.tag}-{key}",
            "feedback_kind": "save_experience",
            "correction_text": text or f"{self.tag} ONVIF включается в разделе настроек Network.",
            "applicability": {"models": ["DS-2CD2387G2P-LSU/SL"]},
        }
        params.update(kwargs)
        item, duplicate = create_feedback(self.conn, **params)
        self.conn.commit()
        return item, duplicate

    # -- reply_only ------------------------------------------------------
    def test_reply_only_creates_no_knowledge_or_proposal(self):
        run = self._run()
        before = self.conn.execute("SELECT count(*) AS n FROM v2_knowledge").fetchone()["n"]
        item, duplicate = create_feedback(
            self.conn, answer_run_id=int(run["run_id"]),
            idempotency_key=f"{self.tag}-reply",
            feedback_kind="reply_only", correction_text=f"{self.tag} reply text",
        )
        self.conn.commit()
        self.assertFalse(duplicate)
        self.assertEqual(item["status"], "closed")
        after = self.conn.execute("SELECT count(*) AS n FROM v2_knowledge").fetchone()["n"]
        self.assertEqual(before, after)
        proposals = self.conn.execute(
            "SELECT count(*) AS n FROM v2_learning_proposals WHERE fact_text LIKE %s",
            (f"{self.tag} %",),
        ).fetchone()["n"]
        self.assertEqual(proposals, 0)

    # -- save_experience submit/confirm ----------------------------------
    def test_provisional_experience_is_not_answer_eligible(self):
        run = self._run()
        item, _ = self._submit_experience(run["run_id"])
        with self._factory() as conn:
            retrieved = retrieve_for_answer(conn, f"{self.tag} ONVIF", embedder=None)
        self.assertNotIn(int(item["knowledge_id"]),
                         retrieved["diagnostics"]["eligible_ids"])

    def test_confirm_makes_experience_answer_eligible(self):
        run = self._run()
        item, _ = self._submit_experience(run["run_id"])
        knowledge, duplicate = confirm_feedback(
            self.conn, int(item["id"]), reviewer_label="pg-test",
        )
        self.conn.commit()
        self.assertFalse(duplicate)
        self.assertEqual(knowledge["trust"], "user_confirmed")
        self.assertEqual(knowledge["unit_kind"], "experience")
        with self._factory() as conn:
            retrieved = retrieve_for_answer(conn, f"{self.tag} ONVIF", embedder=None)
        self.assertIn(int(knowledge["id"]), retrieved["diagnostics"]["eligible_ids"])
        again, duplicate = confirm_feedback(self.conn, int(item["id"]))
        self.conn.commit()
        self.assertTrue(duplicate)
        self.assertEqual(int(again["id"]), int(knowledge["id"]))

    def test_duplicate_submit_returns_stored_feedback(self):
        run = self._run()
        first, _ = self._submit_experience(run["run_id"], key="dup")
        second, duplicate = self._submit_experience(run["run_id"], key="dup")
        self.assertTrue(duplicate)
        self.assertEqual(int(first["id"]), int(second["id"]))
        count = self.conn.execute(
            "SELECT count(*) AS n FROM v2_answer_feedback WHERE idempotency_key=%s",
            (f"{self.tag}-dup",),
        ).fetchone()["n"]
        self.assertEqual(count, 1)

    def test_update_target_requires_matching_revision(self):
        run = self._run()
        item, _ = self._submit_experience(run["run_id"], key="base")
        knowledge, _ = confirm_feedback(self.conn, int(item["id"]))
        self.conn.commit()
        revision = int(knowledge["revision"])
        fresh, _ = create_feedback(
            self.conn, answer_run_id=int(run["run_id"]),
            idempotency_key=f"{self.tag}-stale2",
            feedback_kind="save_experience",
            correction_text=f"{self.tag} обновлённый текст подтверждения",
            target_knowledge_id=int(knowledge["id"]),
            expected_revision=revision,
        )
        self.conn.commit()
        with self._factory() as conn2:
            edit_knowledge(conn2, int(knowledge["id"]), "изменено другим инженером", None)
            conn2.commit()
        with self.assertRaises(StaleRevision):
            confirm_feedback(self.conn, int(fresh["id"]), confirmed_text=f"{self.tag} опоздавшее подтверждение")
        self.conn.rollback()
        stored = get_feedback(self.conn, int(fresh["id"]))
        self.assertEqual(stored["status"], "open")

    # -- retest + verdict -------------------------------------------------
    def test_retest_creates_a_new_linked_run(self):
        run = self._run()
        item, _ = self._submit_experience(run["run_id"])
        confirm_feedback(self.conn, int(item["id"]))
        self.conn.commit()
        # Lexical retrieval finds the confirmed text; the fake judge cites it.
        new = retest_feedback(
            int(item["id"]), db_factory=self._factory,
            llm_service=FakeLLM([answered_json("новый ответ")]),
            idempotency_key=f"{self.tag}-retest",
        )
        self.assertNotEqual(int(new["run_id"]), int(run["run_id"]))
        self.assertEqual(int(new["retest_of"]), int(run["run_id"]))
        self.assertEqual(int(new["feedback_id"]), int(item["id"]))
        self.assertEqual(new["answer_status"], "answered")
        with self._factory() as conn:
            from v2.answering import get_answer_run
            old = get_answer_run(conn, int(run["run_id"]))
        # The old run is immutable: still the original unsupported triage.
        self.assertEqual(old["answer_status"], "unsupported")
        self.assertEqual(old["answer_text"], "")

    def test_verdict_and_gap_queue(self):
        run = self._run()
        set_answer_verdict(self.conn, int(run["run_id"]), verdict="fail",
                           reason="pg-test", reviewer_label="pg-test")
        self.conn.commit()
        self.assertEqual(count_unresolved_feedback(self.conn), 0)
        item, _ = create_feedback(
            self.conn, answer_run_id=int(run["run_id"]),
            idempotency_key=f"{self.tag}-gap",
            feedback_kind="retrieval_failure",
            correction_text=f"{self.tag} knowledge exists but was missed",
        )
        self.conn.commit()
        self.assertEqual(count_unresolved_feedback(self.conn), 1)
        queued = list_unresolved_feedback(self.conn)
        self.assertEqual(int(queued[0]["id"]), int(item["id"]))
        close_feedback(self.conn, int(item["id"]))
        self.conn.commit()
        self.assertEqual(count_unresolved_feedback(self.conn), 0)
        with self.assertRaises(FeedbackConflict):
            set_answer_verdict(self.conn, int(run["run_id"]), verdict="maybe")

    def test_edit_knowledge_bumps_revision_and_clears_embedding(self):
        run = self._run()
        item, _ = self._submit_experience(run["run_id"])
        knowledge, _ = confirm_feedback(self.conn, int(item["id"]))
        self.conn.commit()
        before_revision = int(knowledge["revision"])
        updated = edit_knowledge(
            self.conn, int(knowledge["id"]), f"{self.tag} отредактированное содержание", None,
            applicability={"models": ["DS-2CD2387G2P-LSU/SL"], "firmware": "5.7"},
        )
        self.conn.commit()
        self.assertEqual(int(updated["revision"]), before_revision + 1)
        self.assertEqual(updated["applicability"]["firmware"], "5.7")
        row = self.conn.execute(
            "SELECT embedding, revision FROM v2_knowledge WHERE id=%s",
            (int(knowledge["id"]),),
        ).fetchone()
        self.assertIsNone(row["embedding"])
        history = self.conn.execute(
            "SELECT action, after_json FROM v2_knowledge_history "
            "WHERE knowledge_id=%s ORDER BY id DESC LIMIT 2",
            (int(knowledge["id"]),),
        ).fetchall()
        actions = [entry["action"] for entry in history]
        self.assertIn("confirm", actions)
        self.assertIn("edit", actions)
        stored = get_feedback(self.conn, int(item["id"]))
        self.assertEqual(stored["status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
