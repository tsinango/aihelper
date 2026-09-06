"""Tests for Phase 4.3 whole-document coverage and Phase 5.2 revalidation.

Worker interleave order is covered without a database by patching the
queue readers.  Coverage accounting, evidence_only shelving, crash
recovery/resume, idempotent relearning, edit consistency, version
comparison/impact/lineage, and a fixture-based question set run against
PostgreSQL when ``V2_TEST_DATABASE_URL`` is set.
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

    def test_impact_and_revalidate_shapes(self):
        from fastapi import HTTPException

        from app import v2_document_impact, v2_revalidate_document_version

        comparison = {
            "added": [{"section": "Warnings", "blocks": []}],
            "changed": [{"section": "Quick Start Guide", "blocks": []}],
            "removed": [], "unmatched": [], "global_scope_changed": False,
        }
        affected = [{"id": 8, "title": "t", "trust": "user_confirmed",
                     "validation_status": "validated", "revision": 1}]
        self._patch("get_version", lambda conn, _: {"id": 9})
        self._patch("compare_versions", lambda conn, *_, **__: comparison)
        self._patch("affected_knowledge", lambda conn, *_, **__: affected)
        impact = v2_document_impact(9, previous_version_id=7, x_api_key="test-key")
        self.assertEqual(impact["previous_version_id"], 7)
        self.assertEqual(
            [item["knowledge_id"] for item in impact["affected_knowledge"]], [8])

        self._patch("record_revalidation",
                    lambda conn, *_,
                    **__: {"id": 9, "previous_version_id": 7,
                           "change_summary": {"changed": ["Quick Start Guide"]}})
        stored = v2_revalidate_document_version(
            9, {"previous_version_id": 7}, x_api_key="test-key")
        self.assertEqual(stored["previous_version_id"], 7)

        from v2.documents import DocumentNotFound

        def missing(conn, *_, **__):
            raise DocumentNotFound("gone")

        self._patch("compare_versions", missing)
        with self.assertRaises(HTTPException) as caught:
            v2_document_impact(9, previous_version_id=7, x_api_key="test-key")
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(HTTPException) as caught:
            v2_revalidate_document_version(9, {"previous_version_id": "x"},
                                           x_api_key="test-key")
        self.assertEqual(caught.exception.status_code, 400)


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


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2DocumentRevalidationPostgresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def tearDown(self):
        try:
            self.conn.rollback()
            with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM v2_document_jobs WHERE version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_document_blocks WHERE version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
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

    def _upload_parsed(self, key, label, lines):
        from v2.document_processing import process_document_job
        from v2.documents import claim_document_job, create_version

        version, _ = create_version(
            self.conn, base_dir=self.tmp.name, document_key=key,
            version_label=label, filename="m.pdf",
            content=_pdf_lines_bytes(lines),
        )
        self.conn.commit()
        with self._factory() as conn:
            job = claim_document_job(conn, ("parse",))
            conn.commit()
        process_document_job(int(job["id"]), db_factory=self._factory, base_dir=self.tmp.name)
        return version

    def _two_versions(self, key="ZZDOC reval"):
        old = self._upload_parsed(key, "v1", _manual_v1_lines())
        new = self._upload_parsed(key, "v2", _manual_v2_lines())
        return old, new

    def _unit_for_section(self, version_id, fragment, title):
        """Trusted validated unit citing the first block containing fragment."""

        with self._factory() as conn:
            block = conn.execute(
                """
                SELECT b.id, b.raw_evidence_id, r.content
                FROM v2_document_blocks b
                JOIN v2_raw_evidence r ON r.id=b.raw_evidence_id
                WHERE b.version_id=%s AND r.content LIKE %s
                ORDER BY b.ord LIMIT 1
                """,
                (int(version_id), f"%{fragment}%"),
            ).fetchone()
            assert block, fragment
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO v2_knowledge(
                        title, content, trust, active, unit_kind,
                        origin_document_version_id, validation_status
                    ) VALUES(%s, %s, 'user_confirmed', TRUE, 'fact', %s, 'validated')
                    RETURNING id
                    """,
                    (title, f"ZZDOC {block['content'][:200]}", int(version_id)),
                )
                knowledge_id = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    INSERT INTO v2_knowledge_sources(
                        knowledge_id, raw_evidence_id, source_kind, relation,
                        source_role, excerpt, active, resolution
                    ) VALUES(%s, %s, 'other', 'supports', 'supporting',
                             'ZZDOC excerpt', TRUE, 'accepted')
                    """,
                    (knowledge_id, int(block["raw_evidence_id"])),
                )
            conn.commit()
            return knowledge_id

    def test_compare_finds_changed_added_removed(self):
        from v2.documents import compare_versions

        old, new = self._two_versions()
        with self._factory() as conn:
            comparison = compare_versions(conn, int(old["id"]), int(new["id"]))
        sections = {group: sorted(item["section"] for item in comparison[group])
                    for group in ("added", "changed", "removed", "unmatched")}
        self.assertEqual(sections["added"], ["Quick Start Guide / Warnings"])
        # The Guide text changed even though it flows onto page 2: page
        # numbers never count as content changes, section paths do.
        self.assertEqual(sections["changed"], ["Quick Start Guide"])
        self.assertEqual(sections["removed"], ["Quick Start Guide / Fingerprints"])
        self.assertEqual(sections["unmatched"], [])
        self.assertFalse(comparison["global_scope_changed"])

    def test_affected_lists_only_changed_area_units(self):
        from v2.documents import affected_knowledge, compare_versions

        old, new = self._two_versions(key="ZZDOC affected")
        guide = self._unit_for_section(old["id"], "tap Add", "ZZDOC Guide unit")
        appendix = self._unit_for_section(old["id"], "dry and safe", "ZZDOC Appendix unit")
        with self._factory() as conn:
            comparison = compare_versions(conn, int(old["id"]), int(new["id"]))
            affected = affected_knowledge(conn, int(old["id"]), comparison)
        ids = [int(item["id"]) for item in affected]
        self.assertIn(guide, ids)
        self.assertNotIn(appendix, ids)

    def test_global_change_conservatively_affects_everything(self):
        from v2.documents import affected_knowledge, compare_versions

        old = self._upload_parsed("ZZDOC global", "v1", [[
            (b"F2", 16, 72, 720, b"Warnings"),
            (b"F1", 12, 72, 700, b"Read everything first."),
            (b"F2", 16, 72, 660, b"Setup"),
            (b"F1", 12, 72, 640, b"Follow the steps."),
        ]])
        new = self._upload_parsed("ZZDOC global", "v2", [[
            (b"F2", 16, 72, 720, b"Warnings"),
            (b"F1", 12, 72, 700, b"Read everything twice."),
            (b"F2", 16, 72, 660, b"Setup"),
            (b"F1", 12, 72, 640, b"Follow the steps."),
        ]])
        setup = self._unit_for_section(old["id"], "Follow the steps", "ZZDOC Setup unit")
        with self._factory() as conn:
            comparison = compare_versions(conn, int(old["id"]), int(new["id"]))
            self.assertTrue(comparison["global_scope_changed"])
            affected = affected_knowledge(conn, int(old["id"]), comparison)
        self.assertIn(setup, [int(item["id"]) for item in affected])

    def test_revalidate_records_without_touching_old_units(self):
        from v2.documents import (
            DocumentError,
            affected_knowledge,
            compare_versions,
            lineage_version_ids,
            record_revalidation,
        )

        old, new = self._two_versions(key="ZZDOC record")
        guide = self._unit_for_section(old["id"], "tap Add", "ZZDOC Guide unit")
        with self._factory() as conn:
            comparison = compare_versions(conn, int(old["id"]), int(new["id"]))
            affected = affected_knowledge(conn, int(old["id"]), comparison)
            stored = record_revalidation(
                conn, int(new["id"]), int(old["id"]), comparison, affected)
            conn.commit()
            self.assertEqual(int(stored["previous_version_id"]), int(old["id"]))
            summary = stored["change_summary"]
            self.assertIn("Quick Start Guide", summary["changed"])
            self.assertIn(guide, summary["affected_knowledge_ids"])
            self.assertEqual(lineage_version_ids(conn, int(new["id"])),
                             [int(new["id"]), int(old["id"])])
            # Old units keep serving: validation untouched.
            row = conn.execute(
                "SELECT validation_status FROM v2_knowledge WHERE id=%s", (guide,),
            ).fetchone()
            self.assertEqual(row["validation_status"], "validated")
            with self.assertRaises(DocumentError):
                record_revalidation(conn, int(new["id"]), int(new["id"]), comparison, [])
            conn.rollback()

    def test_lineage_gate_keeps_old_units_from_new_questions(self):
        import json

        from v2.answering import answer_question
        from v2.documents import compare_versions, record_revalidation
        from v2.retrieval import retrieve_for_answer

        old, new = self._two_versions(key="ZZDOC gate")
        guide = self._unit_for_section(old["id"], "tap Add", "ZZDOC Guide unit")
        with self._factory() as conn:
            comparison = compare_versions(conn, int(old["id"]), int(new["id"]))
            from v2.documents import affected_knowledge

            record_revalidation(conn, int(new["id"]), int(old["id"]),
                                comparison, affected_knowledge(conn, int(old["id"]), comparison))
            conn.commit()

        class FakeJudge:
            def judge(self, messages, max_tokens=600):
                return json.dumps({
                    "status": "answered", "answer": "ZZDOC open Users and tap Add.",
                    "clarifying_question": "", "source_indexes": [0],
                    "confidence": 0.9,
                })

        # No request version: old unit answers as before.
        with self._factory() as conn:
            plain = retrieve_for_answer(conn, "ZZDOC tap Add", embedder=None)
            self.assertIn(guide, plain["diagnostics"]["eligible_ids"])
        # Explicit old version: still eligible.
        with self._factory() as conn:
            scoped_old = retrieve_for_answer(
                conn, "ZZDOC tap Add", embedder=None, request_version_id=int(old["id"]))
            self.assertIn(guide, scoped_old["diagnostics"]["eligible_ids"])
        # Explicit new version: older-lineage unit stays out.
        with self._factory() as conn:
            scoped_new = retrieve_for_answer(
                conn, "ZZDOC tap Add", embedder=None, request_version_id=int(new["id"]))
            self.assertNotIn(guide, scoped_new["diagnostics"]["eligible_ids"])
            run = answer_question(
                "ZZDOC tap Add", context={"document_version_id": int(new["id"])},
                idempotency_key="ZZDOC-gate-1", db_factory=self._factory,
                llm_service=FakeJudge(),
            )
        self.assertEqual(run["answer_status"], "unsupported")
        self.assertEqual(run["retrieval_trace"]["request_version_id"], int(new["id"]))

    def test_validation_toggle_changes_eligibility_with_history(self):
        from v2.retrieval import retrieve_for_answer
        from v2.service import edit_knowledge

        old, _ = self._two_versions(key="ZZDOC toggle")
        guide = self._unit_for_section(old["id"], "tap Add", "ZZDOC Guide unit")
        with self._factory() as conn:
            current = conn.execute(
                "SELECT content FROM v2_knowledge WHERE id=%s", (guide,)).fetchone()
            updated = edit_knowledge(
                conn, guide, current["content"], None, validation_status="needs_revalidation",
            )
            conn.commit()
            self.assertEqual(updated["validation_status"], "needs_revalidation")
            gated = retrieve_for_answer(conn, "ZZDOC tap Add", embedder=None)
            self.assertNotIn(guide, gated["diagnostics"]["eligible_ids"])
            history = conn.execute(
                "SELECT action FROM v2_knowledge_history WHERE knowledge_id=%s ORDER BY id DESC LIMIT 1",
                (guide,),
            ).fetchone()
            self.assertEqual(history["action"], "revalidate")
            edit_knowledge(conn, guide, current["content"], None,
                           validation_status="validated")
            conn.commit()
            admitted = retrieve_for_answer(conn, "ZZDOC tap Add", embedder=None)
            self.assertIn(guide, admitted["diagnostics"]["eligible_ids"])


def _pdf_lines_bytes(pages) -> bytes:
    """Minimal PDF from per-page line lists of (font, size, x, y, text)."""

    count = len(pages)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(b"%d 0 R" % (3 + index) for index in range(count))
        + b"] /Count %d >>" % count,
    ]
    for index in range(count):
        # Page objects are 3..2+count; fonts follow; streams last.
        stream_no = 3 + count + 2 + index
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> "
            b"/Contents %d 0 R >>" % (3 + count, 4 + count, stream_no)
        )
    objects.extend([
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ])
    for page_lines in pages:
        content = b"".join(
            b"BT /%s %d Tf %d %d Td (%s) Tj ET\n" % (font, size, x, y, text)
            for font, size, x, y, text in page_lines
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, xref_at)
    )
    return bytes(out)


def _manual_v1_lines():
    return [[
        (b"F2", 24, 72, 720, b"Quick Start Guide"),
        (b"F1", 12, 72, 690, b"Add a user before anything else."),
        (b"F1", 12, 72, 672, b"Open Users and tap Add to create one."),
        (b"F2", 16, 72, 640, b"Fingerprints"),
        (b"F1", 12, 72, 622, b"Enroll two fingerprints per user."),
        (b"F2", 16, 72, 590, b"Appendix"),
        (b"F1", 12, 72, 572, b"Keep the manual dry and safe."),
    ]]


def _manual_v2_lines():
    # Changed Guide step, removed Fingerprints section, added Warnings;
    # the whole Appendix section moves to page 2 with identical text, so
    # pagination alone must not count as a content change.
    return [
        [
            (b"F2", 24, 72, 720, b"Quick Start Guide"),
            (b"F1", 12, 72, 690, b"Add a user before anything else."),
            (b"F1", 12, 72, 672, b"Open Users and tap Add twice to create one."),
            (b"F2", 16, 72, 640, b"Warnings"),
            (b"F1", 12, 72, 622, b"Do not share admin accounts."),
        ],
        [
            (b"F2", 16, 72, 720, b"Appendix"),
            (b"F1", 12, 72, 702, b"Keep the manual dry and safe."),
        ],
    ]


class ComparePureTest(unittest.TestCase):
    def test_global_section_detection(self):
        from v2.documents import _is_global_section

        self.assertTrue(_is_global_section("Warnings and Scope"))
        self.assertTrue(_is_global_section("适用范围"))
        self.assertFalse(_is_global_section("Quick Start Guide"))

    def test_signature_ignores_whitespace_case(self):
        from v2.documents import _section_text_signature

        self.assertEqual(
            _section_text_signature("  Open  Users\n"),
            _section_text_signature("open users"),
        )


if __name__ == "__main__":
    unittest.main()
