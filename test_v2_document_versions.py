"""Tests for Phase 4.3 whole-document coverage.

Worker interleave order is covered without a database by patching the
queue readers.  Coverage accounting, evidence_only shelving, crash
recovery/resume, idempotent relearning, edit consistency, and a
fixture-based question set run against PostgreSQL when
``V2_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import psycopg
from psycopg.rows import dict_row

import worker as worker_module
from worker import pump_once


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


class DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def commit(self):
        pass

    def rollback(self):
        pass

    def cursor(self):  # pragma: no cover - queue readers are patched below
        raise AssertionError("database must not be touched")


class PumpOnceTest(unittest.TestCase):
    def _pump(self, inbox_ids, doc_job, llm_configured=True):
        calls = []
        stages_seen = {}

        def fake_list(conn, limit=1):
            return list(inbox_ids)

        def fake_claim(conn, stages):
            stages_seen["stages"] = tuple(stages)
            return dict(doc_job) if doc_job else None

        with patch.object(worker_module, "list_queued_job_ids", fake_list), patch.object(
            worker_module, "claim_document_job", fake_claim
        ):
            result = pump_once(
                db_factory=DummyConn,
                run_inbox_job=lambda job_id: calls.append(("inbox", job_id)),
                run_document_job=lambda job_id: calls.append(("document", job_id)),
                llm_configured=llm_configured,
            )
        return result, calls, stages_seen.get("stages")

    def test_inbox_beats_document_jobs(self):
        result, calls, _ = self._pump([101], {"id": 7})
        self.assertEqual(result, "inbox")
        self.assertEqual(calls, [("inbox", 101)])

    def test_document_runs_only_when_inbox_is_empty(self):
        result, calls, stages = self._pump([], {"id": 7})
        self.assertEqual(result, "document")
        self.assertEqual(calls, [("document", 7)])
        self.assertEqual(stages, ("parse", "learn"))

    def test_idle_when_both_queues_are_empty(self):
        result, calls, _ = self._pump([], None)
        self.assertEqual(result, "idle")
        self.assertEqual(calls, [])

    def test_no_learn_claim_without_a_model(self):
        _, _, stages = self._pump([], {"id": 7}, llm_configured=False)
        self.assertEqual(stages, ("parse",))


class DocumentCoverageApiTest(unittest.TestCase):
    """Route-level tests: HTTP mapping, not service logic."""

    def setUp(self):
        from unittest.mock import patch

        import app as app_module

        self.app_module = app_module
        self._previous_api_key = app_module.settings["api_key"]
        app_module.settings["api_key"] = "test-key"
        self.db_patch = patch.object(app_module, "db", return_value=DummyConn())
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(app_module.settings.__setitem__, "api_key", self._previous_api_key)

    def _patch(self, name, value):
        from unittest.mock import patch

        patcher = patch.object(self.app_module, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_coverage_and_learn_reset_shapes(self):
        from fastapi import HTTPException

        from app import v2_document_coverage, v2_learn_document_version

        self._patch("get_version", lambda conn, _: {"id": 9})
        coverage = {
            "version_id": 9, "total_blocks": 4,
            "by_state": {"knowledge": 1, "evidence_only": 2, "needs_review": 1},
            "with_destination": 4, "unfinished_blocks": [], "jobs": [],
            "complete": True,
        }
        self._patch("version_coverage", lambda conn, _: coverage)
        response = v2_document_coverage(9, x_api_key="test-key")
        self.assertTrue(response["complete"])
        self.assertEqual(response["total_blocks"], 4)

        seen = {}

        def fake_queue(conn, version_id, reset_evidence_only=False):
            seen["reset"] = reset_evidence_only
            return []

        self._patch("queue_learn_jobs", fake_queue)
        v2_learn_document_version(9, {"reset_evidence_only": True}, x_api_key="test-key")
        self.assertTrue(seen["reset"])

        self._patch("get_version", lambda conn, _: None)
        with self.assertRaises(HTTPException) as caught:
            v2_document_coverage(4242, x_api_key="test-key")
        self.assertEqual(caught.exception.status_code, 404)


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2DocumentCoveragePostgresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        self._inbox_threads: list[int] = []

    def tearDown(self):
        try:
            self.conn.rollback()
            with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    if self._inbox_threads:
                        cur.execute(
                            "DELETE FROM v2_inbox_processing_jobs WHERE thread_id = ANY(%s)",
                            (self._inbox_threads,),
                        )
                        cur.execute(
                            "DELETE FROM v2_inbox_messages WHERE thread_id = ANY(%s)",
                            (self._inbox_threads,),
                        )
                        cur.execute(
                            "DELETE FROM v2_inbox_threads WHERE id = ANY(%s)",
                            (self._inbox_threads,),
                        )
                        cur.execute(
                            "DELETE FROM v2_raw_evidence WHERE raw_payload->>'thread_id' = ANY(%s)",
                            ([str(thread_id) for thread_id in self._inbox_threads],),
                        )
                    cur.execute(
                        "DELETE FROM v2_document_jobs WHERE version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_document_blocks WHERE version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_inbox_processing_jobs WHERE idempotency_key LIKE 'ZZDOC-%'"
                    )
                    cur.execute(
                        "DELETE FROM v2_inbox_messages WHERE thread_id IN ("
                        "SELECT id FROM v2_inbox_threads WHERE external_thread_id LIKE 'ZZDOC-%')"
                    )
                    cur.execute(
                        "DELETE FROM v2_inbox_threads WHERE external_thread_id LIKE 'ZZDOC-%'"
                    )
                    cur.execute(
                        "DELETE FROM v2_learning_proposals WHERE origin_document_version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
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
                    cur.execute(
                        "DELETE FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %'"
                    )
                    cur.execute("DELETE FROM v2_answer_runs WHERE idempotency_key LIKE 'ZZDOC-%'")
        finally:
            self.conn.rollback()
            self.conn.close()
            self.tmp.cleanup()

    def _factory(self):
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def _parsed_version(self, key="ZZDOC cov", data=None, filename="m.pdf"):
        import sys

        sys.path.insert(0, ".")
        from test_v2_documents import _pdf_bytes
        from v2.document_processing import process_document_job
        from v2.documents import claim_document_job, create_version

        version, _ = create_version(
            self.conn, base_dir=self.tmp.name, document_key=key,
            version_label="v1", filename=filename, content=data or _pdf_bytes(),
        )
        self.conn.commit()
        with self._factory() as conn:
            job = claim_document_job(conn, ("parse",))
            conn.commit()
        process_document_job(int(job["id"]), db_factory=self._factory, base_dir=self.tmp.name)
        return version

    def _extractor_citing(self, wanted: int):
        class FakeExtractor:
            def extract_structured(self, messages, schema, max_tokens):
                blocks = json.loads(messages[1]["content"])["blocks"][:wanted]
                return json.dumps({"units": [{
                    "title": "ZZDOC covered procedure",
                    "unit_kind": "procedure",
                    "content": "ZZDOC " + " ".join(block["text"][:80] for block in blocks),
                    "applicability": {},
                    "ordered_steps": ["ZZDOC step one", "ZZDOC step two"],
                    "expected_result": "ZZDOC done",
                    "sources": [
                        {"block_id": block["block_id"], "excerpt": block["text"][:100]}
                        for block in blocks
                    ],
                }]})
        return FakeExtractor()

    def _learn_all(self, version_id, extractor):
        from v2.document_learning import queue_learn_jobs
        from v2.document_processing import process_document_job
        from v2.documents import claim_document_job

        jobs = queue_learn_jobs(self.conn, int(version_id))
        self.conn.commit()
        for _ in jobs:
            with self._factory() as conn:
                job = claim_document_job(conn, ("learn",))
                conn.commit()
            process_document_job(
                int(job["id"]), db_factory=self._factory, base_dir=self.tmp.name,
                llm_service=extractor,
            )

    def test_pump_prefers_inbox_over_documents_on_real_queues(self):
        from v2.processing import enqueue_inbox_job

        version = self._parsed_version()
        with self._factory() as conn:
            inbox_job = enqueue_inbox_job(
                conn, "ZZDOC some learning input", thread_id=None,
                channel="web", idempotency_key="ZZDOC-pump-1",
            )
            conn.commit()
        self._inbox_threads.append(int(inbox_job["thread_id"]))
        from v2.documents import claim_document_job

        with self._factory() as conn:
            doc_job = claim_document_job(conn, ("parse",))
            # No parse job remains (already processed); queue a learn marker.
            self.assertIsNone(doc_job)
        from v2.document_learning import queue_learn_jobs

        jobs = queue_learn_jobs(self.conn, int(version["id"]))
        self.conn.commit()
        self.assertTrue(jobs)
        calls = []
        first = pump_once(
            db_factory=self._factory,
            run_inbox_job=lambda job_id: calls.append(("inbox", job_id)),
            run_document_job=lambda job_id: calls.append(("document", job_id)),
            llm_configured=True,
        )
        self.assertEqual(first, "inbox")
        self.assertEqual(calls, [("inbox", int(inbox_job["id"]))])

    def test_crash_recovery_resumes_from_per_context_checkpoint(self):
        from v2.document_learning import queue_learn_jobs
        from v2.document_processing import recover_document_jobs
        from v2.documents import claim_document_job, get_document_job

        version = self._parsed_version()
        jobs = queue_learn_jobs(self.conn, int(version["id"]))
        self.conn.commit()
        with self._factory() as conn:
            job = claim_document_job(conn, ("learn",))
            conn.commit()
        # Simulate a crash after claim: row stays processing.
        with self._factory() as conn:
            stuck = get_document_job(conn, int(job["id"]))
            self.assertEqual(stuck["status"], "processing")
        calls = []
        recover_document_jobs(
            db_factory=self._factory, process_job=lambda job_id: calls.append(job_id))
        self.assertEqual(calls, [int(job["id"])])
        with self._factory() as conn:
            resumed = get_document_job(conn, int(job["id"]))
            self.assertEqual(resumed["status"], "queued")

    def test_unused_blocks_become_evidence_only_and_coverage_closes(self):
        from v2.document_learning import queue_learn_jobs
        from v2.documents import get_blocks, version_coverage

        version = self._parsed_version()
        with self._factory() as conn:
            blocks = get_blocks(conn, int(version["id"]))
        teachable = [block for block in blocks if str(block.get("evidence_text") or "").strip()]
        self._learn_all(version["id"], self._extractor_citing(1))
        with self._factory() as conn:
            after = get_blocks(conn, int(version["id"]))
            states = {block["block_key"]: block["processing_state"] for block in after}
            # The single cited block is a proposal; other teachable blocks
            # are shelved as evidence, never left pending.
            self.assertIn("proposal", set(states.values()))
            self.assertIn("evidence_only", set(states.values()))
            self.assertNotIn("pending", set(states.values()))
            coverage = version_coverage(conn, int(version["id"]))
        self.assertEqual(coverage["total_blocks"], len(blocks))
        self.assertEqual(coverage["with_destination"], len(blocks))
        self.assertEqual(coverage["unfinished_blocks"], [])
        self.assertTrue(coverage["complete"])
        with self._factory() as conn:
            stored = conn.execute(
                "SELECT status FROM v2_document_versions WHERE id=%s",
                (int(version["id"]),),
            ).fetchone()
        self.assertEqual(stored["status"], "complete")

    def test_relearn_is_idempotent_and_reset_reopens_evidence(self):
        from v2.document_learning import list_document_proposals, queue_learn_jobs

        version = self._parsed_version(key="ZZDOC relearn")
        self._learn_all(version["id"], self._extractor_citing(2))
        with self._factory() as conn:
            first = list_document_proposals(conn, int(version["id"]))
        # Re-queue without reset: no pending blocks, no new jobs.
        jobs = queue_learn_jobs(self.conn, int(version["id"]))
        self.conn.commit()
        self.assertEqual(jobs, [])
        with self._factory() as conn:
            second = list_document_proposals(conn, int(version["id"]))
        self.assertEqual(len(first), len(second))
        # Reset reopens shelved blocks for one more pass; same content
        # fingerprints to the same proposals, never duplicates.
        jobs = queue_learn_jobs(self.conn, int(version["id"]), reset_evidence_only=True)
        self.conn.commit()
        self.assertTrue(jobs)
        self._learn_all(version["id"], self._extractor_citing(2))
        with self._factory() as conn:
            third = list_document_proposals(conn, int(version["id"]))
        self.assertEqual(len(third), len(first))

    def test_edit_details_keeps_history_and_invalidates_embedding(self):
        from v2.document_learning import confirm_document_proposal, list_document_proposals
        from v2.service import edit_knowledge

        version = self._parsed_version(key="ZZDOC edit")
        self._learn_all(version["id"], self._extractor_citing(2))
        with self._factory() as conn:
            proposals = list_document_proposals(conn, int(version["id"]))
            knowledge, _ = confirm_document_proposal(conn, int(proposals[0]["id"]))
            conn.commit()
            before_revision = int(knowledge["revision"])
            updated = edit_knowledge(
                conn, int(knowledge["id"]), knowledge["content"], None,
                details_json={"title": "ZZDOC covered procedure",
                              "ordered_steps": ["ZZDOC step one", "ZZDOC step two",
                                                "ZZDOC step three"]},
            )
            conn.commit()
        self.assertEqual(int(updated["revision"]), before_revision + 1)
        with self._factory() as conn:
            row = conn.execute(
                "SELECT embedding, revision FROM v2_knowledge WHERE id=%s",
                (int(knowledge["id"]),),
            ).fetchone()
            self.assertIsNone(row["embedding"])
            history = conn.execute(
                "SELECT action, after_json FROM v2_knowledge_history "
                "WHERE knowledge_id=%s ORDER BY id DESC LIMIT 1",
                (int(knowledge["id"]),),
            ).fetchone()
        self.assertEqual(history["action"], "edit")
        self.assertIn("ZZDOC step three", json.dumps(history["after_json"], ensure_ascii=False))

    def test_fixture_question_set_over_confirmed_units(self):
        import sys

        sys.path.insert(0, ".")
        from test_v2_documents import _pptx_bytes
        from v2.answering import answer_question
        from v2.document_learning import confirm_document_proposal, list_document_proposals

        pdf_version = self._parsed_version(key="ZZDOC qa-pdf")
        pptx_version = self._parsed_version(
            key="ZZDOC qa-pptx", data=_pptx_bytes(), filename="d.pptx")
        self._learn_all(pdf_version["id"], self._extractor_citing(2))
        self._learn_all(pptx_version["id"], self._extractor_citing(2))

        class FakeJudge:
            def __init__(self, answer):
                self.answer = answer

            def judge(self, messages, max_tokens=600):
                return json.dumps({
                    "status": "answered", "answer": self.answer,
                    "clarifying_question": "", "source_indexes": [0],
                    "confidence": 0.9,
                })

        with self._factory() as conn:
            for version_id in (pdf_version["id"], pptx_version["id"]):
                for proposal in list_document_proposals(conn, int(version_id)):
                    confirm_document_proposal(conn, int(proposal["id"]))
            conn.commit()
        cases = [
            ("ZZDOC-qa-1", "ZZDOC how do I add a user?",
             "ZZDOC Open Users and tap Add.", True),
            ("ZZDOC-qa-2", "ZZDOC which imaging mode needs light?",
             "ZZDOC Color needs illumination.", True),
            ("ZZDOC-qa-3", "ZZDOC what did the speaker notes remind?",
             "ZZDOC Light first.", True),
            # No lexical overlap with the English fixture: honestly uncovered.
            ("ZZDOC-qa-4", "Как сбросить пароль администратора?", "", False),
        ]
        for key, question, answer, answerable in cases:
            with self._factory() as conn:
                run = answer_question(
                    question, context={},
                    idempotency_key=key,
                    db_factory=self._factory,
                    llm_service=FakeJudge(answer) if answerable else None,
                )
            if answerable:
                self.assertEqual(run["answer_status"], "answered", question)
                self.assertTrue(run["evidence_snapshot"], question)
            else:
                self.assertEqual(run["answer_status"], "unsupported", question)


if __name__ == "__main__":
    unittest.main()
