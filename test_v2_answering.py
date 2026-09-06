"""Unit tests for the Phase 3.1 read-only answer service.

A fake in-memory ``v2_answer_runs`` table plus fake knowledge rows exercise
triage, citation validation, failure mapping, snapshots, and idempotency
without PostgreSQL.  Real SQL persistence is covered by the PostgreSQL
integration tests.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from psycopg.errors import UniqueViolation

from v2.answering import (
    AnswerConflict,
    AnswerInProgress,
    answer_question,
    normalize_answer_decision,
    normalize_context,
    request_hash,
    triage_without_candidates,
)


def _unwrap(value):
    return getattr(value, "obj", value)


class FakeRunsCursor:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        text = " ".join(str(query).split())
        self._result = []
        if text.startswith("INSERT INTO v2_answer_runs"):
            key = params[0]
            if self.state.get("fail_insert_once"):
                self.state["fail_insert_once"] = False
                raise UniqueViolation("duplicate key value violates unique constraint")
            if key in self.state["runs_by_key"]:
                raise UniqueViolation("duplicate key value violates unique constraint")
            run_id = self.state["next_id"]
            self.state["next_id"] += 1
            now = datetime.now(timezone.utc)
            self.state["runs_by_key"][key] = {
                "id": run_id, "idempotency_key": key, "question": params[1],
                "context_json": _unwrap(params[2]), "request_hash": params[3],
                "execution_status": "started", "answer_status": "service_error",
                "answer_text": "", "clarifying_question": "", "reason_code": "",
                "evidence_snapshot": [], "retrieval_trace": {},
                "model": params[4], "prompt_version": params[5],
                "llm_requests": 0, "latency_ms": 0,
                "created_at": now, "updated_at": now,
            }
            self._result = [{"id": run_id}]
        elif text.startswith("SELECT") and "FROM v2_answer_runs" in text:
            if "WHERE id=" in text:
                row = self.state["runs_by_id"].get(int(params[0]))
                # runs_by_id mirrors runs_by_key
                if row is None:
                    for candidate in self.state["runs_by_key"].values():
                        if int(candidate["id"]) == int(params[0]):
                            row = candidate
                self._result = [dict(row)] if row else []
            else:
                row = self.state["runs_by_key"].get(params[0])
                self._result = [dict(row)] if row else []
        elif "UPDATE v2_answer_runs" in text and "execution_status='started'" in text:
            for row in self.state["runs_by_key"].values():
                if int(row["id"]) == int(params[0]) and row["execution_status"] == "started":
                    row["updated_at"] = datetime.now(timezone.utc)
        elif text.startswith("UPDATE v2_answer_runs"):
            (execution_status, answer_status, answer_text, clarifying,
             reason, snapshot, trace, llm_requests, latency_ms, run_id) = params
            for row in self.state["runs_by_key"].values():
                if int(row["id"]) == int(run_id):
                    row.update({
                        "execution_status": execution_status,
                        "answer_status": answer_status,
                        "answer_text": answer_text,
                        "clarifying_question": clarifying,
                        "reason_code": reason,
                        "evidence_snapshot": _unwrap(snapshot),
                        "retrieval_trace": _unwrap(trace),
                        "llm_requests": int(llm_requests),
                        "latency_ms": int(latency_ms),
                        "updated_at": datetime.now(timezone.utc),
                    })
        elif "AS source_id" in text:
            self._result = list(self.state["sources"])
        else:
            self._result = list(self.state["knowledge"])

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class FakeConnection:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return FakeRunsCursor(self.state)

    def rollback(self):
        pass


def knowledge_row(identifier, title, content, entity="", trust="user_confirmed"):
    return {
        "id": identifier, "title": title, "content": content,
        "entity_name": entity, "legacy_entity_name": entity,
        "trust": trust, "active": True,
        "embedding": None, "embedding_model": None,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 2, 2, tzinfo=timezone.utc),
    }


def knowledge_source(knowledge_id, identifier=1):
    return {
        "source_id": identifier, "knowledge_id": knowledge_id,
        "source_kind": "user_confirmation", "source_role": "primary",
        "excerpt": "摘录 Встроенная выдержка",
        "relation": "supports", "resolution": "accepted",
        "raw_evidence_id": identifier, "evidence_type": "user_input",
        "source_label": "label", "source_locator": "locator-1",
        "evidence_status": "active",
    }


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


class FakeEmbedder:
    def __init__(self, error=None):
        self.error = error

    def encode(self, texts, **kwargs):
        if self.error:
            raise self.error
        raise AssertionError("no vector expected in these tests")


def answered_json(answer="Ответ на русском языке.", indexes=(0,)):
    return json.dumps({
        "status": "answered", "answer": answer,
        "clarifying_question": "", "source_indexes": list(indexes),
        "confidence": 0.9,
    }, ensure_ascii=False)


class V2AnsweringTest(unittest.TestCase):
    def setUp(self):
        self.state = {
            "runs_by_key": {}, "runs_by_id": {}, "next_id": 1,
            "knowledge": [], "sources": [],
            "fail_insert_once": False,
        }
        self.db_factory = lambda: FakeConnection(self.state)

    def ask(self, question, **kwargs):
        kwargs.setdefault("db_factory", self.db_factory)
        return answer_question(question, **kwargs)

    def seed(self, rows, sources=()):
        self.state["knowledge"] = list(rows)
        self.state["sources"] = list(sources)

    def test_answered_path_snapshots_cited_evidence(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持标准 19 英寸机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json("F-NR-208E/2 устанавливается в стандартную 19-дюймовую стойку.")])
        result = self.ask("F-NR-208E/2 如何安装在机架上？", llm_service=llm)
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["reason_code"], "grounded_answer")
        self.assertEqual(result["llm_requests"], 1)
        self.assertFalse(result["duplicate"])
        snapshot = result["evidence_snapshot"]
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["knowledge_id"], 1)
        self.assertIn("19 英寸", snapshot[0]["content"])
        self.assertEqual(snapshot[0]["sources"][0]["source_locator"], "locator-1")
        self.assertEqual(snapshot[0]["trust"], "user_confirmed")

    def test_missing_model_is_clarification_without_llm_call(self):
        self.seed([
            knowledge_row(1, "F-X1 安装", "F-X1 支持机架安装", "F-X1"),
            knowledge_row(2, "F-Y2 安装", "F-Y2 支持机架安装", "F-Y2"),
        ])
        llm = FakeLLM([answered_json()])
        result = self.ask("设备 安装 机架 方法", llm_service=llm)
        self.assertEqual(result["answer_status"], "needs_clarification")
        self.assertEqual(result["reason_code"], "missing_model")
        self.assertIn("F-X1", result["clarifying_question"])
        self.assertEqual(result["llm_requests"], 0)
        self.assertEqual(llm.calls, [])

    def test_no_evidence_is_unsupported(self):
        self.seed([])
        llm = FakeLLM([answered_json()])
        result = self.ask("Неизвестное устройство мигает?", llm_service=llm)
        self.assertEqual(result["answer_status"], "unsupported")
        self.assertEqual(result["reason_code"], "no_eligible_evidence")
        self.assertEqual(llm.calls, [])

    def test_covered_other_model_is_unsupported_not_clarification(self):
        self.seed([knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")])
        llm = FakeLLM([answered_json()])
        result = self.ask("F-NR-232X/2 如何安装在机架上？", llm_service=llm)
        self.assertEqual(result["answer_status"], "unsupported")
        self.assertEqual(result["reason_code"], "model_not_covered")
        self.assertEqual(llm.calls, [])

    def test_versioned_candidate_answers_versionless_model_question(self):
        self.seed([knowledge_row(
            1, "升级到 5.7", "将 IDS-TCM203-A 从 5.6.11 升级到 5.7 版本", "IDS-TCM203-A",
        )])
        llm = FakeLLM([answered_json("Обновите IDS-TCM203-A до версии 5.7.")])
        result = self.ask("IDS-TCM203-A 升级失败怎么办？", llm_service=llm)
        # The question names the model but no version, so version exclusion
        # cannot fire; the single-scope candidate goes to the grounded LLM.
        self.assertEqual(result["answer_status"], "answered")

    def test_version_conflict_without_model_match_is_unsupported(self):
        self.seed([knowledge_row(
            1, "升级 5.6.11", "IDS-TCM203-A 版本 5.6.11 支持升级", "IDS-TCM203-A",
        )])
        llm = FakeLLM([answered_json()])
        result = self.ask("IDS-TCM203-A 4.5.7 升级失败怎么办？", llm_service=llm)
        self.assertEqual(result["answer_status"], "needs_clarification")
        self.assertEqual(result["reason_code"], "missing_version")

    def test_llm_timeout_is_service_error(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM(error=TimeoutError("timed out"))
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        self.assertEqual(result["answer_status"], "service_error")
        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(result["reason_code"], "llm_timeout")
        self.assertEqual(result["clarifying_question"], "")
        self.assertIn("TimeoutError", result["retrieval_trace"].get("llm_error", ""))

    def test_rate_limit_is_service_error(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        error = RuntimeError("rate limited")
        error.status_code = 429
        llm = FakeLLM(error=error)
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        self.assertEqual(result["answer_status"], "service_error")
        self.assertEqual(result["reason_code"], "llm_rate_limited")

    def test_bad_citation_fails_closed_to_unsupported(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json(indexes=(7,))])
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        self.assertEqual(result["answer_status"], "unsupported")
        self.assertEqual(result["reason_code"], "citation_invalid")

    def test_non_russian_answer_fails_closed(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json(answer="Install it into the rack.")])
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        self.assertEqual(result["answer_status"], "unsupported")
        self.assertEqual(result["reason_code"], "citation_invalid")

    def test_unparsable_llm_output_is_service_error(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM(["конечно, вот ответ без всякого JSON"])
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        self.assertEqual(result["answer_status"], "service_error")
        self.assertEqual(result["reason_code"], "llm_bad_response")

    def test_llm_requested_clarification_is_kept(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        body = json.dumps({
            "status": "needs_clarification",
            "clarifying_question": "Какая высота юнита?",
            "source_indexes": [], "confidence": 0.2,
        }, ensure_ascii=False)
        llm = FakeLLM([body])
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        self.assertEqual(result["answer_status"], "needs_clarification")
        self.assertEqual(result["reason_code"], "llm_requested_clarification")

    def test_embedding_failure_still_answers_lexically(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json("F-NR-208E/2 устанавливается в стойку.")])
        result = self.ask(
            "F-NR-208E/2 如何安装？", llm_service=llm,
            embedding_client=FakeEmbedder(error=RuntimeError("embedding down")),
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertTrue(result["retrieval_trace"]["lexical_only"])

    def test_duplicate_key_returns_same_run_without_new_llm_call(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json("F-NR-208E/2 устанавливается в стойку.")])
        first = self.ask("F-NR-208E/2 如何安装？", llm_service=llm, idempotency_key="key-1")
        second = self.ask("F-NR-208E/2 如何安装？", llm_service=llm, idempotency_key="key-1")
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(llm.calls), 1)

    def test_same_key_different_payload_is_conflict(self):
        self.seed([])
        llm = FakeLLM()
        self.ask("Первый вопрос?", llm_service=llm, idempotency_key="key-9")
        with self.assertRaises(AnswerConflict):
            self.ask("Совсем другой вопрос?", llm_service=llm, idempotency_key="key-9")

    def test_unique_violation_race_returns_existing_run(self):
        self.seed([])
        llm = FakeLLM()
        first = self.ask("Первый вопрос?", llm_service=llm, idempotency_key="key-race")
        self.state["fail_insert_once"] = True
        # Same payload racing the insert resolves to the stored run.
        second = self.ask("Первый вопрос?", llm_service=llm, idempotency_key="key-race")
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["run_id"], second["run_id"])

    def test_snapshot_is_immutable_after_knowledge_edit(self):
        self.seed(
            [knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持标准 19 英寸机架安装", "F-NR-208E/2")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json("F-NR-208E/2 устанавливается в стойку.")])
        result = self.ask("F-NR-208E/2 如何安装？", llm_service=llm)
        # Later Knowledge edits must not rewrite the stored snapshot.
        self.state["knowledge"][0]["content"] = "F-NR-208E/2 需要特殊导轨安装"
        stored = self.state["runs_by_key"][result["idempotency_key"]]
        self.assertIn("19 英寸", stored["evidence_snapshot"][0]["content"])
        self.assertNotIn("特殊导轨", stored["evidence_snapshot"][0]["content"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ask("   ", llm_service=FakeLLM())

    def test_qualifier_names_cited_scope_not_codec(self):
        self.seed(
            [knowledge_row(1, "T1000 编码", "T1000 支持 H.265 编码 设置方法", "T1000")],
            [knowledge_source(1)],
        )
        llm = FakeLLM([answered_json("Включите H.265 в настройках T1000.")])
        result = self.ask("H.265 编码 设置 方法", llm_service=llm)
        self.assertEqual(result["answer_status"], "answered")
        self.assertIn("T1000", result["answer_text"])
        self.assertNotIn("H.265", result["answer_text"].split("порядок такой:")[0])

    def test_qualifier_uses_cited_candidate_only(self):
        self.seed(
            [
                knowledge_row(1, "T1000 安装", "T1000 支持机架安装", "T1000"),
                knowledge_row(2, "通用安装说明", "设备安装需要断电操作", ""),
            ],
            [knowledge_source(1, identifier=1), knowledge_source(2, identifier=2)],
        )
        # The model cites only the generic first-ranked candidate: no T1000
        # qualifier from the uncited scoped candidate may leak in.
        body = json.dumps({
            "status": "answered", "answer": "Перед установкой отключите питание.",
            "clarifying_question": "", "source_indexes": [0], "confidence": 0.8,
        }, ensure_ascii=False)
        llm = FakeLLM([body])
        result = self.ask("设备 安装 断电", llm_service=llm)
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["evidence_snapshot"][0]["knowledge_id"], 2)
        self.assertNotIn("T1000", result["answer_text"])

    def test_request_hash_is_stable(self):
        left = request_hash("  F-NR-208E/2 安装？ ", normalize_context({"b": 1, "a": 2}))
        right = request_hash("f-nr-208e/2 安装？", normalize_context({"a": 2, "b": 1}))
        self.assertEqual(left, right)


class V2AnswerDecisionTest(unittest.TestCase):
    def test_out_of_range_citations_fail_closed(self):
        decision = normalize_answer_decision(
            answered_json(answer="Ответ на русском.", indexes=(3,)), [{"index": 0}],
        )
        self.assertEqual(decision["status"], "unsupported")
        self.assertEqual(decision["reason_code"], "citation_invalid")

    def test_triage_prefers_version_question_over_silence(self):
        diagnostics = {
            "query_models": ["IDS-TCM203-A"],
            "query_versions": ["457"],
            "topical_excluded": [{
                "knowledge_id": 1, "reason": "version_conflict",
                "scope_models": ["IDS-TCM203-A"], "scope_versions": ["5611"],
            }],
            "topical_scopes": [["IDS-TCM203-A"]],
        }
        triage = triage_without_candidates("q", diagnostics)
        self.assertEqual(triage["answer_status"], "needs_clarification")
        self.assertEqual(triage["reason_code"], "missing_version")


if __name__ == "__main__":
    unittest.main()
