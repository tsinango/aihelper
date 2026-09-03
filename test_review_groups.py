"""Regression tests for semantic review grouping and lossless aggregation."""

import unittest
from unittest.mock import patch

import app


def candidate(candidate_id, question, *, knowledge_key="device.reset", embedding=None,
              claims=None, steps=None, conditions=None, warnings=None, case_id=None):
    payload = {
        "candidate_id": candidate_id,
        "knowledge_key": knowledge_key,
        "title": question,
        "knowledge_type": "troubleshooting",
        "scope": {"models": [], "brands": [], "product_families": [], "series": [],
                  "hardware_revisions": [], "firmware_versions": [], "software_versions": [],
                  "operating_modes": []},
        "question_patterns": [question],
        "claims": claims or [],
        "procedure_steps": steps or [],
        "conditions": conditions or [],
        "exceptions": [],
        "warnings": warnings or [],
        "answer_text": "\n".join(steps or []),
        "scope_level": "generic",
        "confidence": "medium",
        "freshness_sensitive": False,
    }
    return {
        "id": int(candidate_id.split("-")[-1]) if candidate_id.split("-")[-1].isdigit() else 1,
        "candidate_id": candidate_id,
        "knowledge_key": knowledge_key,
        "title": question,
        "frequency": 1,
        "effective_payload": payload,
        "_embedding": embedding,
        "_case_ids": [case_id] if case_id is not None else [],
    }


class ReviewGroupAggregationTest(unittest.TestCase):
    def test_deterministic_grouping_needs_no_v1_1_or_embedding(self):
        left = candidate("C-1", "reset administrator password on DS-K1T320")
        right = candidate("C-2", "reset forgotten admin password DS-K1T320")

        self.assertTrue(app._review_group_should_join(left, right, 0.76))
        self.assertEqual(app._review_group_build.__defaults__[-1], "deterministic")

    def test_semantically_similar_candidates_have_high_similarity(self):
        left = candidate("C-1", "reset device password", embedding=[1.0, 0.0])
        right = candidate("C-2", "change forgotten administrator password", embedding=[0.98, 0.1])
        self.assertGreaterEqual(app._review_group_similarity(left, right), 0.76)

    def test_aggregate_retains_all_features_and_source_provenance(self):
        members = [
            candidate("C-1", "Password reset fails", case_id=101,
                      claims=[{"claim": "The red LED indicates reset mode."}],
                      steps=["Open settings"]),
            candidate("C-2", "Administrator password reset", case_id=202,
                      conditions=["Hold the home screen"],
                      warnings=["Do not reboot during reset"]),
        ]
        aggregate, facts, conflicts = app._review_group_aggregate(members)

        self.assertFalse(conflicts)
        self.assertIn("Open settings", aggregate["procedure_steps"])
        self.assertIn("Hold the home screen", aggregate["conditions"])
        self.assertIn("Do not reboot during reset", aggregate["warnings"])
        self.assertTrue(any(fact["source_candidate_id"] == "C-1" for fact in facts))
        self.assertTrue(any(fact["support_case_id"] == 202 for fact in facts))

    def test_conflicting_claims_are_reported(self):
        members = [
            candidate("C-1", "Does reset work", claims=[{"claim": "Reset is supported."}]),
            candidate("C-2", "Does reset work", claims=[{"claim": "Reset is not supported."}]),
        ]
        _, _, conflicts = app._review_group_aggregate(members)
        self.assertTrue(conflicts)
        self.assertEqual(conflicts[0]["type"], "claim")

    def test_openrouter_failure_uses_lossless_mechanical_draft(self):
        payload = {
            "answer_text": "Initial answer",
            "procedure_steps": ["Open settings"],
            "conditions": ["Only for model X"],
            "exceptions": [],
            "warnings": ["Do not reboot"],
            "claims": [{"claim": "The operation is supported."}],
        }

        class FailingOpenRouter:
            def complete(self, *_args, **_kwargs):
                raise RuntimeError("OpenRouter unavailable")

        with patch.object(app, "llm", FailingOpenRouter()):
            draft = app._review_group_polish_with_llm(payload)

        self.assertIn("Open settings", draft)
        self.assertIn("Do not reboot", draft)
        self.assertIn("The operation is supported", draft)


class ReviewGroupContractTest(unittest.TestCase):
    def test_review_group_endpoints_are_exposed(self):
        expected = {
            ("POST", "/api/review/groups/build"),
            ("GET", "/api/review/groups"),
            ("GET", "/api/review/groups/{group_id}"),
            ("PATCH", "/api/review/groups/{group_id}"),
            ("POST", "/api/review/groups/{group_id}/approve"),
            ("POST", "/api/review/groups/{group_id}/recompute"),
        }
        actual = {
            (method.upper(), route.path)
            for route in app.app.routes
            for method in getattr(route, "methods", ())
        }
        self.assertTrue(expected.issubset(actual))

    def test_group_statuses_include_published_and_member_merge_state(self):
        self.assertIn("published", app.REVIEW_GROUP_STATUSES)
        self.assertIn("merged", app.REVIEW_STATUSES)
        self.assertIn("merged", app.REVIEW_ANSWER_STATUSES)


if __name__ == "__main__":
    unittest.main()
