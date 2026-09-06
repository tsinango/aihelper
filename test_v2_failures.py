"""Unit tests for the Phase 5.3 failure-driven improvement loop.

Failure classification is pure and runs without a database; the failure
list, the 10-case retrieval gate, and the evaluate flag run against
PostgreSQL when ``V2_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import json
import os
import unittest

import psycopg
from psycopg.rows import dict_row

from v2.feedback import (
    FAILURE_ACTIONS,
    FAILURE_CATEGORIES,
    RETRIEVAL_GATE_NEEDED,
    classify_run_failure,
    list_failures,
    retrieval_gate_progress,
)


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


def _run(**kwargs):
    base = {
        "execution_status": "completed", "answer_status": "unsupported",
        "reason_code": "no_eligible_evidence",
        "retrieval_trace": {"eligible_knowledge_ids": [], "candidate_knowledge_ids": []},
    }
    base.update(kwargs)
    return base


class ClassifyTest(unittest.TestCase):
    def test_all_categories_have_actions(self):
        self.assertEqual(len(FAILURE_CATEGORIES), 6)
        for category in FAILURE_CATEGORIES:
            self.assertTrue(FAILURE_ACTIONS[category])

    def test_feedback_labels_outrank_heuristics(self):
        cases = {
            "retrieval_failure": "retrieval_failure",
            "generation_failure": "generation_failure",
            "missing_information": "knowledge_missing",
            "field_result_failure": "applicability_version_failure",
        }
        for kind, expected in cases.items():
            category, _ = classify_run_failure(
                _run(answer_status="answered", reason_code="grounded_answer"),
                {"feedback_kind": kind},
            )
            self.assertEqual(category, expected, kind)

    def test_service_and_version_failures(self):
        category, _ = classify_run_failure(_run(execution_status="failed"))
        self.assertEqual(category, "service_failure")
        category, _ = classify_run_failure(_run(answer_status="service_error"))
        self.assertEqual(category, "service_failure")
        category, _ = classify_run_failure(_run(
            answer_status="needs_clarification", reason_code="missing_model"))
        self.assertEqual(category, "applicability_version_failure")
        category, _ = classify_run_failure(_run(reason_code="model_not_covered"))
        self.assertEqual(category, "applicability_version_failure")

    def test_evidence_presence_splits_the_rest(self):
        trace = {"eligible_knowledge_ids": [1], "candidate_knowledge_ids": [1]}
        category, _ = classify_run_failure(_run(retrieval_trace=trace))
        self.assertEqual(category, "generation_failure")
        trace = {"eligible_knowledge_ids": [1], "candidate_knowledge_ids": []}
        category, _ = classify_run_failure(_run(retrieval_trace=trace))
        self.assertEqual(category, "knowledge_missing")
        trace = {"eligible_knowledge_ids": [],
                 "document": {"candidate_block_ids": [9]}, "candidate_knowledge_ids": []}
        category, _ = classify_run_failure(_run(retrieval_trace=trace))
        self.assertEqual(category, "knowledge_missing")
        category, _ = classify_run_failure(_run())
        self.assertEqual(category, "missing_source")


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2FailuresPostgresTest(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def tearDown(self):
        try:
            self.conn.rollback()
            with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM v2_answer_feedback WHERE idempotency_key LIKE 'ZZDOC-%'"
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge_history WHERE knowledge_id IN ("
                        "SELECT id FROM v2_knowledge WHERE title LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge_sources WHERE knowledge_id IN ("
                        "SELECT id FROM v2_knowledge WHERE title LIKE 'ZZDOC %')"
                    )
                    cur.execute("DELETE FROM v2_knowledge WHERE title LIKE 'ZZDOC %'")
                    cur.execute("DELETE FROM v2_raw_evidence WHERE source_label LIKE 'ZZDOC %'")
                    cur.execute("DELETE FROM v2_answer_runs WHERE idempotency_key LIKE 'ZZDOC-%'")
        finally:
            self.conn.rollback()
            self.conn.close()

    def _factory(self):
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def _eligible_knowledge(self):
        with self._factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO v2_knowledge(title, content, trust, active)
                    VALUES('ZZDOC recallable fact', 'ZZDOC recallable content', 'provisional', TRUE)
                    RETURNING id
                    """,
                )
                knowledge_id = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    INSERT INTO v2_raw_evidence(
                        evidence_type, author_role, content, source_label
                    ) VALUES('user_input', 'product_expert', 'ZZDOC evidence', 'ZZDOC label')
                    RETURNING id
                    """,
                )
                evidence_id = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    INSERT INTO v2_knowledge_sources(
                        knowledge_id, raw_evidence_id, source_kind, relation,
                        source_role, excerpt, active, resolution
                    ) VALUES(%s, %s, 'user_input', 'supports', 'supporting',
                             'ZZDOC excerpt', TRUE, 'accepted')
                    """,
                    (knowledge_id, evidence_id),
                )
                cur.execute(
                    "UPDATE v2_knowledge SET trust='user_confirmed' WHERE id=%s",
                    (knowledge_id,),
                )
            conn.commit()
            return knowledge_id

    def _run_row(self, key, status="unsupported", reason="no_eligible_evidence",
                 trace=None, verdict=None):
        from psycopg.types.json import Jsonb

        with self._factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO v2_answer_runs(
                        idempotency_key, question, execution_status,
                        answer_status, reason_code, retrieval_trace,
                        reviewer_verdict
                    ) VALUES(%s, 'ZZDOC question', 'completed', %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (key, status, reason, Jsonb(trace or {}), verdict),
                )
                run_id = int(cur.fetchone()["id"])
            conn.commit()
            return run_id

    def test_list_and_gate(self):
        from v2.feedback import create_feedback

        knowledge_id = self._eligible_knowledge()
        run_failed = self._run_row("ZZDOC-f-1")
        self._run_row("ZZDOC-f-2", status="service_error", reason="llm_timeout")
        judged = self._run_row("ZZDOC-f-3", reason="llm_unsupported",
                               trace={"eligible_knowledge_ids": [knowledge_id],
                                      "candidate_knowledge_ids": [knowledge_id]})
        self._run_row("ZZDOC-ok-1", status="answered", reason="grounded_answer")
        with self._factory() as conn:
            item, _ = create_feedback(
                conn, answer_run_id=run_failed,
                idempotency_key="ZZDOC-fb-gate",
                feedback_kind="retrieval_failure",
                correction_text="ZZDOC should have matched",
                expected_knowledge_ids=[knowledge_id],
            )
            conn.commit()
        with self._factory() as conn:
            failures = list_failures(conn, days=1, limit=50)
            kinds = {item["run_id"]: item["category"] for item in failures}
            self.assertEqual(kinds[run_failed], "retrieval_failure")
            judged_item = next(item for item in failures if item["run_id"] == judged)
            self.assertEqual(judged_item["category"], "generation_failure")
            self.assertNotIn("ZZDOC-ok-1", [
                item["question"] for item in failures
                if item["run_id"] not in (run_failed, judged)
            ])
            gate = retrieval_gate_progress(conn)
            self.assertEqual(gate["qualifying_cases"], 1)
            self.assertEqual(gate["needed"], RETRIEVAL_GATE_NEEDED)
            self.assertFalse(gate["gate_open"])

    def test_expected_ids_validation(self):
        from v2.feedback import FeedbackConflict, create_feedback

        run_id = self._run_row("ZZDOC-f-9")
        with self._factory() as conn:
            with self.assertRaises(FeedbackConflict):
                create_feedback(
                    conn, answer_run_id=run_id,
                    idempotency_key="ZZDOC-fb-bad",
                    feedback_kind="reply_only", correction_text="ZZDOC x",
                    expected_knowledge_ids=[1],
                )
            conn.rollback()
            with self.assertRaises(FeedbackConflict):
                create_feedback(
                    conn, answer_run_id=run_id,
                    idempotency_key="ZZDOC-fb-bad2",
                    feedback_kind="retrieval_failure",
                    correction_text="ZZDOC x",
                    expected_knowledge_ids=["nope"],
                )
            conn.rollback()

    def test_ineligible_expected_ids_do_not_count(self):
        from v2.feedback import create_feedback

        run_id = self._run_row("ZZDOC-f-8")
        with self._factory() as conn:
            create_feedback(
                conn, answer_run_id=run_id,
                idempotency_key="ZZDOC-fb-inelig",
                feedback_kind="retrieval_failure",
                correction_text="ZZDOC points at nothing eligible",
                expected_knowledge_ids=[424242],
            )
            conn.commit()
            gate = retrieval_gate_progress(conn)
            self.assertEqual(gate["qualifying_cases"], 0)
            self.assertFalse(gate["gate_open"])


class FailuresApiTest(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch

        import app as app_module

        self.app_module = app_module
        self._previous_api_key = app_module.settings["api_key"]
        app_module.settings["api_key"] = "test-key"
        self.db_patch = patch.object(app_module, "db", return_value=_DummyConn())
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(app_module.settings.__setitem__, "api_key", self._previous_api_key)

    def test_failures_shape(self):
        from unittest.mock import patch

        from app import v2_failures

        item = {"run_id": 3, "question": "q", "answer_status": "unsupported",
                "reason_code": "no_eligible_evidence", "reviewer_verdict": None,
                "category": "missing_source", "default_action": "补资料",
                "feedback_id": None, "feedback_kind": "",
                "eligible_count": 0, "candidate_count": 0, "created_at": None}
        gate = {"qualifying_cases": 0, "needed": 10, "gate_open": False, "cases": []}
        with patch.object(self.app_module, "list_failures", lambda conn, **__: [item]), patch.object(
            self.app_module, "retrieval_gate_progress", lambda conn: gate
        ):
            response = v2_failures(x_api_key="test-key")
        self.assertEqual(response["total"], 1)
        self.assertEqual(response["items"][0]["category"], "missing_source")
        self.assertFalse(response["retrieval_gate"]["gate_open"])


class _DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):  # pragma: no cover - module functions are patched above
        raise AssertionError("database must not be touched")


if __name__ == "__main__":
    unittest.main()
