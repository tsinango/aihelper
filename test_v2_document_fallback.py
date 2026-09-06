"""Unit tests for the Phase 5.1 bounded source fallback.

A fake store serves Knowledge rows, document versions/blocks, and answer
runs so trigger selection, budget bounds, whole-block fitting, gates, and
the no-downgrade rule are exercised without PostgreSQL.  Real persistence
runs in the PostgreSQL integration class below.
"""

from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from v2.answering import (
    answer_question,
    build_document_snapshot,
    normalize_document_decision,
)
from v2.retrieval import (
    explicit_source_request,
    high_risk_operation,
    retrieve_document_evidence,
)


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


def _unwrap(value):
    return getattr(value, "obj", value)


class FakeCursor:
    def __init__(self, state):
        self.state = state
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def execute(self, query, params=()):
        text = " ".join(str(query).split())
        self._result = []
        state = self.state
        if text.startswith("INSERT INTO v2_answer_runs"):
            key = params[0]
            run_id = state["next_id"]
            state["next_id"] += 1
            now = datetime.now(timezone.utc)
            state["runs"][key] = {
                "id": run_id, "idempotency_key": key, "question": params[1],
                "context_json": _unwrap(params[2]), "request_hash": params[3],
                "execution_status": "started", "answer_status": "service_error",
                "answer_text": "", "clarifying_question": "", "reason_code": "",
                "evidence_snapshot": [], "retrieval_trace": {},
                "model": params[4], "prompt_version": params[5],
                "retest_of": params[6], "feedback_id": params[7],
                "llm_requests": 0, "latency_ms": 0,
                "reviewer_verdict": None, "reviewer_reason": "",
                "reviewer_label": "", "reviewed_at": None,
                "created_at": now, "updated_at": now,
            }
            self._result = [{"id": run_id}]
        elif "FROM v2_answer_runs" in text and "WHERE id=" in text:
            rows = [r for r in state["runs"].values() if int(r["id"]) == int(params[0])]
            self._result = [dict(r) for r in rows]
        elif "FROM v2_answer_runs" in text:
            row = state["runs"].get(params[0])
            self._result = [dict(row)] if row else []
        elif text.startswith("UPDATE v2_answer_runs") and "execution_status='started'" in text:
            for row in state["runs"].values():
                if int(row["id"]) == int(params[0]) and row["execution_status"] == "started":
                    row["updated_at"] = datetime.now(timezone.utc)
        elif text.startswith("UPDATE v2_answer_runs"):
            (execution_status, answer_status, answer_text, clarifying,
             reason, snapshot, trace, llm_requests, latency_ms, run_id) = params
            for row in state["runs"].values():
                if int(row["id"]) == int(run_id):
                    row.update({
                        "execution_status": execution_status, "answer_status": answer_status,
                        "answer_text": answer_text, "clarifying_question": clarifying,
                        "reason_code": reason, "evidence_snapshot": _unwrap(snapshot),
                        "retrieval_trace": _unwrap(trace),
                        "llm_requests": llm_requests, "latency_ms": latency_ms,
                        "updated_at": datetime.now(timezone.utc),
                    })
        elif "FROM v2_knowledge k" in text:
            self._result = [dict(row) for row in state["knowledge"]]
        elif "FROM v2_knowledge_sources s" in text:
            wanted = {int(item) for item in params[0]}
            self._result = [dict(item) for item in state["sources"]
                            if int(item.get("knowledge_id")) in wanted]
        elif "FROM v2_document_blocks b" in text:
            # Emulate the SQL qualification gate: authenticity, version
            # status, block state, and non-empty text.
            allowed_auth, allowed_status, allowed_state = params[0], params[1], params[2]
            self._result = [dict(row) for row in state["doc_rows"]
                            if row.get("source_authenticity") in allowed_auth
                            and row.get("status") in allowed_status
                            and row.get("processing_state") in allowed_state
                            and str(row.get("block_text") or "") != ""]
        else:  # pragma: no cover
            raise AssertionError(f"unhandled query: {text[:110]}")


class FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return FakeCursor(self.state)


def _knowledge_row(identifier=1, content="ONVIF включается в настройках камеры, это подтверждённый факт."):
    return {
        "id": identifier, "title": "t", "content": content, "entity_id": None,
        "entity_name": "", "legacy_entity_name": "",
        "trust": "user_confirmed", "active": True,
        "embedding": None, "embedding_model": None,
        "unit_kind": "fact", "applicability": {}, "revision": 1,
        "origin_document_version_id": None, "validation_status": None,
        "created_at": None, "updated_at": None,
    }


def _knowledge_source(identifier=1):
    return {
        "source_id": identifier, "knowledge_id": identifier,
        "source_kind": "user_input", "source_role": "supporting",
        "excerpt": "摘录", "relation": "supports", "resolution": "accepted",
        "raw_evidence_id": identifier, "evidence_type": "user_input",
        "source_label": "label", "source_locator": "locator",
        "evidence_status": "active",
    }


def _doc_row(block_id=11, text="ONVIF включается в разделе Network.", **kwargs):
    row = {
        "id": 5, "document_key": "MANUAL", "version_label": "v1",
        "title": "Manual", "source_authenticity": "official_vendor",
        "status": "parsed", "applicability": {},
        "block_id": block_id, "block_key": f"p1-{block_id}",
        "page_no": 1, "slide_no": None, "block_type": "paragraph",
        "section_path": ["Network"], "processing_state": "pending",
        "block_text": text,
    }
    row.update(kwargs)
    return row


def _state(knowledge=None, sources=None, doc_rows=None):
    return {"runs": {}, "next_id": 1,
            "knowledge": knowledge or [], "sources": sources or [],
            "doc_rows": doc_rows or []}


class FakeJudge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def judge(self, messages, max_tokens=600):
        self.calls.append((messages, max_tokens))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _answered(answer, index=0):
    return json.dumps({
        "status": "answered", "answer": answer,
        "clarifying_question": "", "source_indexes": [index],
        "confidence": 0.9,
    }, ensure_ascii=False)


def _unsupported(conflict=False):
    return json.dumps({
        "status": "unsupported", "answer": "",
        "clarifying_question": "", "source_indexes": [],
        "confidence": 0.0, "conflict": conflict,
    }, ensure_ascii=False)


class TriggerTest(unittest.TestCase):
    def test_explicit_request_keywords(self):
        self.assertTrue(explicit_source_request("请核对原文表格"))
        self.assertTrue(explicit_source_request("Check the manual page"))
        self.assertTrue(explicit_source_request("проверь оригинал"))
        self.assertFalse(explicit_source_request("Камера не включается, что делать?"))

    def test_high_risk_keywords(self):
        self.assertTrue(high_risk_operation("Как сбросить пароль?"))
        self.assertTrue(high_risk_operation("恢复出厂步骤是什么？"))
        self.assertFalse(high_risk_operation("Где включается ONVIF?"))


class DecisionTest(unittest.TestCase):
    def _evidence(self):
        return [{"version_id": 5, "block_id": 11, "locator": "page 1",
                 "section_path": ["Network"], "block_type": "paragraph",
                 "source_authenticity": "official_vendor",
                 "text": "ONVIF включается в разделе Network."}]

    def test_answered_and_conflict_shapes(self):
        decision = normalize_document_decision(
            _answered("ONVIF включается в разделе Network."), self._evidence(), "ru")
        self.assertEqual(decision["status"], "answered")
        self.assertFalse(decision["conflict"])
        decision = normalize_document_decision(_unsupported(conflict=True), self._evidence(), "ru")
        self.assertEqual(decision["status"], "unsupported")
        self.assertTrue(decision["conflict"])
        self.assertEqual(decision["reason_code"], "llm_unsupported")

    def test_snapshot_copies_block_provenance(self):
        snapshot = build_document_snapshot(self._evidence(), [0])
        (entry,) = snapshot
        self.assertEqual(entry["evidence_type"], "document_block")
        self.assertIsNone(entry["knowledge_id"])
        self.assertIn("Network", entry["title"])
        self.assertEqual(entry["sources"][0]["source_locator"], "v2-doc:5:page 1")


class RescueTest(unittest.TestCase):
    QUESTION = "Подскажите, где в настройках камеры включается ONVIF?"

    def _run(self, state, judge, **kwargs):
        factory = lambda: FakeConn(state)
        return answer_question(
            self.QUESTION, context={}, idempotency_key=f"k-{len(state['runs'])}",
            db_factory=factory, llm_service=judge, **kwargs,
        )

    def test_no_knowledge_is_rescued_from_original_text(self):
        state = _state(doc_rows=[_doc_row()])
        run = self._run(state, FakeJudge([_answered("ONVIF включается в разделе Network.")]))
        self.assertEqual(run["answer_status"], "answered")
        self.assertEqual(run["reason_code"], "grounded_document_fallback")
        self.assertEqual(run["llm_requests"], 1)
        (entry,) = run["evidence_snapshot"]
        self.assertEqual(entry["evidence_type"], "document_block")
        self.assertEqual(entry["block_id"], 11)
        trace = run["retrieval_trace"]["document"]
        self.assertEqual(trace["trigger"], "no_knowledge")

    def test_empty_fallback_keeps_honest_refusal(self):
        state = _state(doc_rows=[])
        run = self._run(state, FakeJudge([]))
        self.assertEqual(run["answer_status"], "unsupported")
        # The trigger is recorded, but no blocks were read and no model called.
        self.assertEqual(run["retrieval_trace"]["document"]["candidate_block_ids"], [])
        self.assertEqual(run["llm_requests"], 0)

    def test_insufficient_excerpts_stay_unsupported(self):
        state = _state(doc_rows=[_doc_row()])
        run = self._run(state, FakeJudge([_unsupported()]))
        self.assertEqual(run["answer_status"], "unsupported")
        self.assertEqual(run["llm_requests"], 1)

    def test_oversize_section_asks_to_narrow(self):
        state = _state(doc_rows=[_doc_row(
            text="ONVIF включается в разделе Network. " + "длинный текст. " * 3000,
        )])
        judge = FakeJudge([])
        run = self._run(state, judge)
        self.assertEqual(run["answer_status"], "needs_clarification")
        self.assertEqual(run["reason_code"], "document_section_too_large")
        self.assertEqual(judge.calls, [])

    def test_qualification_gates(self):
        # Unverified versions never auto-quote.
        state = _state(doc_rows=[_doc_row(source_authenticity="unverified")])
        run = self._run(state, FakeJudge([]))
        self.assertEqual(run["answer_status"], "unsupported")
        self.assertEqual(run["retrieval_trace"]["document"]["candidate_block_ids"], [])
        # Blocks under human review never auto-quote.
        state = _state(doc_rows=[_doc_row(processing_state="needs_review")])
        run = self._run(state, FakeJudge([]))
        self.assertEqual(run["retrieval_trace"]["document"]["candidate_block_ids"], [])
        # Contradicting version scope never auto-quotes.
        state = _state(doc_rows=[_doc_row(
            applicability={"versions": ["5.6.11"]},
            text="ONVIF включается в разделе Network версии 5.6.11.",
        )])
        run = answer_question(
            "Подскажите, где включается ONVIF в версии 5.7.0?",
            context={}, idempotency_key="k scope",
            db_factory=lambda: FakeConn(state), llm_service=FakeJudge([]),
        )
        self.assertEqual(run["retrieval_trace"]["document"]["candidate_block_ids"], [])

    def test_version_scoping(self):
        rows = [_doc_row(block_id=11, id=5), _doc_row(block_id=21, id=6)]
        state = _state(doc_rows=rows)
        run = answer_question(
            self.QUESTION, context={"document_version_id": 6},
            idempotency_key="k scoped", db_factory=lambda: FakeConn(state),
            llm_service=FakeJudge([_answered("ONVIF включается в разделе Network.")]),
        )
        (entry,) = run["evidence_snapshot"]
        self.assertEqual(entry["document_version_id"], 6)


class VerifyTest(unittest.TestCase):
    QUESTION = "Подскажите, где в настройках камеры включается ONVIF?"

    def test_explicit_check_verifies_a_grounded_draft(self):
        state = _state(knowledge=[_knowledge_row()],
                       sources=[_knowledge_source()],
                       doc_rows=[_doc_row()])
        judge = FakeJudge([
            _answered("Подтверждённый факт про камеру."),
            _answered("ONVIF включается в разделе Network."),
        ])
        run = answer_question(
            self.QUESTION, context={}, idempotency_key="k verify",
            db_factory=lambda: FakeConn(state), llm_service=judge,
            check_sources=True,
        )
        self.assertEqual(run["answer_status"], "answered")
        self.assertEqual(run["reason_code"], "document_verified_answer")
        self.assertEqual(run["llm_requests"], 2)
        kinds = [entry.get("evidence_type", "knowledge") for entry in run["evidence_snapshot"]]
        self.assertIn("document_block", kinds)
        self.assertEqual(run["retrieval_trace"]["document"]["trigger"], "explicit_check")

    def test_empty_verify_never_downgrades_a_grounded_draft(self):
        state = _state(knowledge=[_knowledge_row()], sources=[_knowledge_source()])
        run = answer_question(
            self.QUESTION, context={}, idempotency_key="k keep",
            db_factory=lambda: FakeConn(state),
            llm_service=FakeJudge([_answered("Подтверждённый факт про камеру.")]),
            check_sources=True,
        )
        self.assertEqual(run["answer_status"], "answered")
        self.assertEqual(run["reason_code"], "grounded_answer")
        self.assertEqual(run["llm_requests"], 1)

    def test_contradiction_surfaces_instead_of_guessing(self):
        state = _state(knowledge=[_knowledge_row()],
                       sources=[_knowledge_source()],
                       doc_rows=[_doc_row()])
        judge = FakeJudge([
            _answered("Подтверждённый факт про камеру."),
            _unsupported(conflict=True),
        ])
        run = answer_question(
            self.QUESTION, context={}, idempotency_key="k conflict",
            db_factory=lambda: FakeConn(state), llm_service=judge,
            check_sources=True,
        )
        self.assertEqual(run["answer_status"], "unsupported")
        self.assertEqual(run["reason_code"], "knowledge_document_conflict")
        self.assertIn("block_ids", run["retrieval_trace"]["document"]["conflict"])

    def test_clarification_never_triggers_fallback(self):
        state = _state(
            knowledge=[_knowledge_row(1, "Камера DS-100 умеет снимать ночью."),
                       _knowledge_row(2, "Камера DS-200 умеет снимать днём.")],
            sources=[_knowledge_source(1), _knowledge_source(2)],
            doc_rows=[_doc_row()],
        )
        judge = FakeJudge([])
        run = answer_question(
            "Что умеет устройство?", context={}, idempotency_key="k clarify",
            db_factory=lambda: FakeConn(state), llm_service=judge,
        )
        self.assertEqual(run["answer_status"], "needs_clarification")
        self.assertEqual(judge.calls, [])
        self.assertNotIn("document", run["retrieval_trace"])


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2DocumentFallbackPostgresTest(unittest.TestCase):
    def setUp(self):
        import tempfile

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
                        "DELETE FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %'"
                    )
                    cur.execute("DELETE FROM v2_raw_evidence WHERE source_label LIKE 'ZZDOC %'")
                    cur.execute("DELETE FROM v2_answer_runs WHERE idempotency_key LIKE 'ZZDOC-%'")
        finally:
            self.conn.rollback()
            self.conn.close()
            self.tmp.cleanup()

    def _factory(self):
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def _official_version(self, authenticity="official_vendor"):
        import sys

        sys.path.insert(0, ".")
        from test_v2_documents import _pdf_bytes
        from v2.document_processing import process_document_job
        from v2.documents import claim_document_job, create_version

        version, _ = create_version(
            self.conn, base_dir=self.tmp.name, document_key="ZZDOC fallback",
            version_label="v1", filename="m.pdf", content=_pdf_bytes(),
            source_authenticity="unverified",
        )
        self.conn.commit()
        with self._factory() as conn:
            job = claim_document_job(conn, ("parse",))
            conn.commit()
        process_document_job(int(job["id"]), db_factory=self._factory, base_dir=self.tmp.name)
        with self._factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE v2_document_versions SET source_authenticity=%s WHERE id=%s",
                    (authenticity, int(version["id"])),
                )
            conn.commit()
        return version

    def _judge(self, answer):
        class FakeJudge:
            def judge(self, messages, max_tokens=600):
                return json.dumps({
                    "status": "answered", "answer": answer,
                    "clarifying_question": "", "source_indexes": [0],
                    "confidence": 0.9,
                })
        return FakeJudge()

    def test_rescue_answers_from_qualified_original_text(self):
        from v2.answering import answer_question

        version = self._official_version()
        run = answer_question(
            "ZZDOC how do I add a user?", context={},
            idempotency_key="ZZDOC-fb-1", db_factory=self._factory,
            llm_service=self._judge("ZZDOC open Users and tap Add."),
        )
        self.assertEqual(run["answer_status"], "answered")
        self.assertEqual(run["reason_code"], "grounded_document_fallback")
        (entry,) = run["evidence_snapshot"]
        self.assertEqual(entry["evidence_type"], "document_block")
        self.assertEqual(int(entry["document_version_id"]), int(version["id"]))
        self.assertTrue(entry["locator"].startswith("page"))

    def test_unverified_originals_stay_human_only(self):
        from v2.answering import answer_question

        self._official_version(authenticity="unverified")
        run = answer_question(
            "ZZDOC how do I add a user?", context={},
            idempotency_key="ZZDOC-fb-2", db_factory=self._factory,
            llm_service=self._judge("ZZDOC open Users and tap Add."),
        )
        self.assertEqual(run["answer_status"], "unsupported")
        self.assertEqual(run["retrieval_trace"]["document"]["candidate_block_ids"], [])


if __name__ == "__main__":
    unittest.main()
