"""Regression tests for the human-review knowledge contract.

These tests intentionally keep the database behind a tiny recording fake.  The
review API is PostgreSQL-backed in production, but its state-machine guarantees
are useful to verify without requiring a running database or embedding model.
"""

import unittest
from unittest.mock import patch

from fastapi import HTTPException

import app
from app import (
    _review_normalize_payload,
    _review_candidate_payload,
    _review_scope,
    _upsert_verified_draft,
    _verified_searchable_text,
    publish_verified_knowledge,
    retrieve_case_memory,
    retrieve_verified_knowledge,
)


class RecordingCursor:
    """Minimal context-manager cursor with deterministic fetch results."""

    def __init__(self, fetchone_results=(), fetchall_results=()):
        self.fetchone_results = list(fetchone_results)
        self.fetchall_results = list(fetchall_results)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self.fetchone_results.pop(0) if self.fetchone_results else None

    def fetchall(self):
        return self.fetchall_results.pop(0) if self.fetchall_results else []


class RecordingConnection:
    def __init__(self, cursor):
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cursor_value


def verified_row(scope=None):
    return {
        "verified_knowledge_id": 7,
        "knowledge_key": "door_station.admin_mode",
        "title": "Open administrator mode",
        "scope": scope or {},
        "claims": [{"claim": "Hold the home screen to open administrator authentication."}],
        "procedure_steps": ["Hold the home screen", "Enter the administrator password"],
        "conditions": [],
        "exceptions": [],
        "warnings": [],
        "question_patterns": ["как зайти в режим админ"],
        "evidence": [],
        "aliases": [],
        "version": 1,
        "publication_status": "draft",
        "production_answer_allowed": False,
        "source_candidate_id": "C-1",
        "verified_by": "reviewer",
    }


class ReviewPayloadTest(unittest.TestCase):
    def test_partial_effective_payload_is_completed_before_rendering(self):
        candidate = {
            "candidate_id": "CASE-1",
            "effective_payload": {
                "title": "Stored title",
                "answer_text": "Stored answer",
            },
            "title": "Stored title",
            "answer_text": "Stored answer",
            "scope": {"models": ["DS-K1T320"]},
            "claims": [{"claim": "The operation is supported."}],
            "procedure_steps": ["Open settings"],
            "conditions": ["Only for administrator mode"],
            "exceptions": ["Not supported on old firmware"],
            "warnings": ["Do not reboot during the operation"],
            "question_patterns": ["How do I do it?"],
            "scope_level": "model",
        }

        payload = _review_candidate_payload(candidate)

        self.assertEqual(payload["claims"][0]["claim"], "The operation is supported.")
        self.assertEqual(payload["procedure_steps"], ["Open settings"])
        self.assertEqual(payload["conditions"], ["Only for administrator mode"])
        self.assertEqual(payload["scope"]["models"], ["DS-K1T320"])

    def test_scope_is_explicit_and_model_specific_fields_are_preserved(self):
        scope = _review_scope({
            "models": [" DS-KV9503 ", "", 123],
            "product_families": "door station",
            "unexpected": ["must not be persisted"],
        })

        self.assertEqual(scope["models"], ["DS-KV9503", "123"])
        self.assertEqual(scope["product_families"], ["door station"])
        self.assertNotIn("unexpected", scope)

    def test_human_payload_cannot_reenable_production_on_save(self):
        payload = _review_normalize_payload(
            {
                "scope": {"models": ["DS-KV9503"]},
                "claims": ["The password is the activation password."],
                "production_answer_allowed": True,
            },
            {"knowledge_key": "door_station.admin_mode", "production_answer_allowed": True},
        )

        self.assertEqual(payload["scope"]["models"], ["DS-KV9503"])
        self.assertEqual(payload["claims"][0]["claim"], "The password is the activation password.")
        self.assertFalse(payload["production_answer_allowed"])

    def test_searchable_text_contains_patterns_claims_and_model_scope(self):
        text = _verified_searchable_text({
            "knowledge_key": "door_station.admin_mode",
            "title": "Administrator mode",
            "question_patterns": ["как зайти в админ"],
            "scope": {"models": ["DS-KV9503"], "product_families": ["Villa Door Station"]},
            "claims": [{"claim": "Hold the home screen."}],
            "procedure_steps": ["Enter the password"],
        })

        for value in ("door_station.admin_mode", "как зайти в админ", "DS-KV9503", "Hold the home screen"):
            self.assertIn(value, text)

    def test_related_knowledge_score_is_only_a_suggestion_signal(self):
        current = {"knowledge_key": "door_station.admin_mode", "title": "Admin mode", "scope": {"models": ["DS-KV9503"]}, "question_patterns": ["как зайти в админ"]}
        same = {"knowledge_key": "door_station.admin_mode", "title": "Administrator mode", "scope": {"models": ["DS-KV9503"]}, "question_patterns": ["режим админ"]}
        different_model = {"knowledge_key": "door_station.admin_mode", "title": "Administrator mode", "scope": {"models": ["DS-KV9504"]}, "question_patterns": ["режим админ"]}
        self.assertGreater(app._review_knowledge_match_score(current, same), app._review_knowledge_match_score(current, different_model))


class RetrievalScopeTest(unittest.TestCase):
    def test_published_knowledge_retrieval_marks_exact_model_scope(self):
        row = verified_row({"models": ["DS-KV9503"]})
        cursor = RecordingCursor(fetchall_results=([dict(row, vector_score=0.9)], [], []))
        result, trace = retrieve_verified_knowledge(
            RecordingConnection(cursor), "Как зайти в режим админ DS-KV9503", [0.1, 0.2], limit=3
        )

        self.assertEqual(result[0]["scope_match"], "exact")
        self.assertEqual(trace["verified_knowledge_scope_matches"][0]["scope_match"], "exact")
        self.assertEqual(result[0]["source_type"], "verified_knowledge")

    def test_generic_and_model_scoped_knowledge_are_distinguishable(self):
        generic = verified_row({})
        model_specific = verified_row({"models": ["DS-KV9503"]})
        cursor = RecordingCursor(fetchall_results=([dict(generic, vector_score=0.9)], [], []))
        generic_result, _ = retrieve_verified_knowledge(
            RecordingConnection(cursor), "Как зайти в режим админ", [0.1], limit=3
        )
        cursor = RecordingCursor(fetchall_results=([dict(model_specific, vector_score=0.9)], [], []))
        specific_result, _ = retrieve_verified_knowledge(
            RecordingConnection(cursor), "Как зайти в режим админ DS-KV9503", [0.1], limit=3
        )

        self.assertEqual(generic_result[0]["scope_match"], "generic")
        self.assertEqual(specific_result[0]["scope_match"], "exact")


class ReviewPublicationTest(unittest.TestCase):
    def test_verified_draft_insert_binds_source_and_reviewer(self):
        cursor = RecordingCursor(
            fetchone_results=[None, {"verified_knowledge_id": 42}],
            fetchall_results=[[], []],
        )
        payload = {
            "knowledge_key": "nvr.temperature_range",
            "title": "NVR temperature range",
            "knowledge_type": "product_fact",
            "answer_text": "-10°C to +55°C",
            "scope_level": "generic",
            "scope": {},
            "claims": [],
            "procedure_steps": [],
            "conditions": [],
            "exceptions": [],
            "warnings": [],
            "question_patterns": [],
        }

        result = _upsert_verified_draft(cursor, "C-1", payload, "reviewer")

        insert_sql, params = next(call for call in cursor.calls if "INSERT INTO verified_knowledge" in call[0])
        self.assertEqual(result, 42)
        self.assertEqual(insert_sql.count("%s"), len(params))
        self.assertEqual(params[-3:-1], ("C-1", "reviewer"))

    def test_only_approved_draft_can_be_published_and_enables_case_memory(self):
        draft = verified_row()
        published = dict(draft, publication_status="published", production_answer_allowed=True)
        cursor = RecordingCursor(
            fetchone_results=[draft, {"review_status": "approved"}, published]
        )
        conn = RecordingConnection(cursor)
        with patch.dict(app.settings, {"api_key": "test-key"}), patch("app.db", return_value=conn), patch(
            "app._verified_embedding", side_effect=AssertionError("approval must not call embedding")
        ) as embedding:
            result = publish_verified_knowledge(7, {"reviewer": "human"}, "test-key")

        self.assertTrue(result["production_answer_allowed"])
        self.assertEqual(result["publication_status"], "published")
        sql = "\n".join(call[0] for call in cursor.calls)
        self.assertIn("UPDATE case_knowledge_memory", sql)
        self.assertIn("source_status='verified'", sql)
        self.assertIn("answer_allowed=TRUE", sql)
        self.assertIn("publication_status='published'", sql)
        embedding.assert_not_called()

    def test_pending_candidate_cannot_be_published(self):
        cursor = RecordingCursor(fetchone_results=[verified_row(), {"review_status": "pending"}])
        with patch.dict(app.settings, {"api_key": "test-key"}), patch("app.db", return_value=RecordingConnection(cursor)):
            with self.assertRaises(HTTPException) as raised:
                publish_verified_knowledge(7, {}, "test-key")

        self.assertEqual(raised.exception.status_code, 409)

    def test_publication_does_not_require_an_embedding_client(self):
        draft = verified_row()
        cursor = RecordingCursor(fetchone_results=[draft, {"review_status": "approved"}, dict(draft, publication_status="published", production_answer_allowed=True)])
        with patch.dict(app.settings, {"api_key": "test-key"}), patch("app.db", return_value=RecordingConnection(cursor)), patch.object(app, "embedder", None):
            result = publish_verified_knowledge(7, {"reviewer": "human"}, "test-key")
        self.assertTrue(result["production_answer_allowed"])
        sql = "\n".join(call[0] for call in cursor.calls)
        self.assertIn("embedding_status='pending'", sql)
        self.assertIn("embedding=NULL", sql)


if __name__ == "__main__":
    unittest.main()
