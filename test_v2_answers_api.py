"""Route-level tests for POST/GET /api/v2/answers.

Route functions are called directly (the established pattern in this repo)
with a fake in-memory runs table and fake knowledge rows; no PostgreSQL or
network is required.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import HTTPException
from psycopg.errors import UniqueViolation

import app as app_module
from app import V2AnswerIn, _v2_answer_response, v2_create_answer, v2_get_answer


def _unwrap(value):
    return getattr(value, "obj", value)


class FakeCursor:
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
            if key in self.state["runs"]:
                raise UniqueViolation("duplicate key")
            run_id = self.state["next_id"]
            self.state["next_id"] += 1
            now = datetime.now(timezone.utc)
            self.state["runs"][key] = {
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
        elif "FROM v2_answer_runs" in text:
            if "WHERE id=" in text:
                rows = [r for r in self.state["runs"].values() if int(r["id"]) == int(params[0])]
            else:
                row = self.state["runs"].get(params[0])
                rows = [row] if row else []
            self._result = [dict(r) for r in rows]
        elif text.startswith("UPDATE v2_answer_runs") and "execution_status='started'" in text:
            for row in self.state["runs"].values():
                if int(row["id"]) == int(params[0]) and row["execution_status"] == "started":
                    row["updated_at"] = datetime.now(timezone.utc)
        elif text.startswith("UPDATE v2_answer_runs"):
            (execution_status, answer_status, answer_text, clarifying,
             reason, snapshot, trace, llm_requests, latency_ms, run_id) = params
            for row in self.state["runs"].values():
                if int(row["id"]) == int(run_id):
                    row.update({
                        "execution_status": execution_status, "answer_status": answer_status,
                        "answer_text": answer_text, "clarifying_question": clarifying,
                        "reason_code": reason, "evidence_snapshot": _unwrap(snapshot),
                        "retrieval_trace": _unwrap(trace),
                        "llm_requests": int(llm_requests), "latency_ms": int(latency_ms),
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
        return FakeCursor(self.state)

    def rollback(self):
        pass


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


def knowledge_row(identifier, title, content, entity=""):
    return {
        "id": identifier, "title": title, "content": content,
        "entity_name": entity, "legacy_entity_name": entity,
        "trust": "user_confirmed", "active": True,
        "embedding": None, "embedding_model": None,
        "created_at": None, "updated_at": None,
    }


def knowledge_source(knowledge_id):
    return {
        "source_id": knowledge_id, "knowledge_id": knowledge_id,
        "source_kind": "user_confirmation", "source_role": "primary",
        "excerpt": "摘录", "relation": "supports", "resolution": "accepted",
        "raw_evidence_id": knowledge_id, "evidence_type": "user_input",
        "source_label": "label", "source_locator": "locator",
        "evidence_status": "active",
    }


def answered_json():
    return json.dumps({
        "status": "answered", "answer": "F-NR-208E/2 安装在机架上。",
        "clarifying_question": "", "source_indexes": [0], "confidence": 0.9,
    }, ensure_ascii=False)


class V2AnswersApiTest(unittest.TestCase):
    def setUp(self):
        self.state = {"runs": {}, "next_id": 1, "knowledge": [], "sources": []}
        self.llm = FakeLLM([answered_json()])
        self._previous_api_key = app_module.settings["api_key"]
        app_module.settings["api_key"] = "test-key"
        self.db_patch = patch.object(app_module, "db", return_value=FakeConnection(self.state))
        self.llm_patch = patch.object(app_module, "llm", self.llm)
        self.embedder_patch = patch.object(app_module, "embedder", None)
        self.db_patch.start()
        self.llm_patch.start()
        self.embedder_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.llm_patch.stop()
        self.embedder_patch.stop()
        app_module.settings["api_key"] = self._previous_api_key

    def seed(self):
        self.state["knowledge"] = [
            knowledge_row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2"),
        ]
        self.state["sources"] = [knowledge_source(1)]

    def post(self, question, key="test-key", idempotency_key=None):
        return v2_create_answer(
            V2AnswerIn(question=question), x_api_key=key, idempotency_key=idempotency_key,
        )

    def test_post_answers_and_returns_citations(self):
        self.seed()
        response = self.post("F-NR-208E/2 如何安装？", idempotency_key="route-1")
        self.assertEqual(response["answer_status"], "answered")
        self.assertEqual(response["idempotency_key"], "route-1")
        self.assertFalse(response["duplicate"])
        self.assertEqual(len(response["citations"]), 1)
        self.assertEqual(response["citations"][0]["knowledge_id"], 1)
        self.assertEqual(response["model"], app_module.OPENROUTER_DEFAULT_MODEL)

    def test_post_duplicate_returns_same_run(self):
        self.seed()
        first = self.post("F-NR-208E/2 如何安装？", idempotency_key="route-dup")
        second = self.post("F-NR-208E/2 如何安装？", idempotency_key="route-dup")
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(self.llm.calls), 1)

    def test_post_same_key_different_payload_is_409(self):
        self.seed()
        self.post("Первый вопрос?", idempotency_key="route-conflict")
        with self.assertRaises(HTTPException) as raised:
            self.post("Другой вопрос?", idempotency_key="route-conflict")
        self.assertEqual(raised.exception.status_code, 409)

    def test_post_rejects_bad_api_key(self):
        with self.assertRaises(HTTPException) as raised:
            self.post("F-NR-208E/2 如何安装？", key="wrong")
        self.assertEqual(raised.exception.status_code, 401)

    def test_get_returns_stored_run(self):
        self.seed()
        created = self.post("F-NR-208E/2 如何安装？", idempotency_key="route-get")
        fetched = v2_get_answer(int(created["run_id"]), x_api_key="test-key")
        self.assertEqual(fetched["run_id"], created["run_id"])
        self.assertEqual(fetched["answer_status"], "answered")
        self.assertEqual(fetched["evidence_snapshot"][0]["knowledge_id"], 1)

    def test_get_missing_run_is_404(self):
        with self.assertRaises(HTTPException) as raised:
            v2_get_answer(424242, x_api_key="test-key")
        self.assertEqual(raised.exception.status_code, 404)

    def test_response_shape_condenses_citations(self):
        response = _v2_answer_response({
            "run_id": 7, "idempotency_key": "k", "question": "q",
            "execution_status": "completed", "answer_status": "answered",
            "answer_text": "a", "clarifying_question": "", "reason_code": "grounded_answer",
            "evidence_snapshot": [{
                "knowledge_id": 3, "title": "T", "entity_name": "E", "trust": "user_confirmed",
                "scope_models": ["T1000"], "scope_versions": [],
                "sources": [{"source_kind": "user_confirmation", "excerpt": "x"}],
            }],
            "retrieval_trace": {}, "model": "m", "prompt_version": "v",
            "llm_requests": 1, "latency_ms": 5, "duplicate": False,
            "created_at": None,
        })
        self.assertEqual(response["citations"][0]["scope_models"], ["T1000"])
        self.assertEqual(response["evidence_snapshot"][0]["knowledge_id"], 3)


if __name__ == "__main__":
    unittest.main()
