import json
import unittest
from pathlib import Path

from evaluate_local_qwen import (
    LocalQwenClient,
    MODEL_SPECS,
    _actual_answer_status,
    build_llama_command,
    prepare_golden_samples,
    summarize,
)


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": '{"supported":true,"confidence":0.9,"source_indexes":[0],"answer":"Готово"}'}}],
            "timings": {
                "prompt_n": 40,
                "predicted_n": 8,
                "predicted_ms": 1100,
                "predicted_per_second": 7.27,
            },
        }


class FakeClient:
    def __init__(self):
        self.payload = None

    def post(self, url, json):
        self.payload = (url, json)
        return FakeResponse()

    def close(self):
        return None


class LocalQwenEvaluationTest(unittest.TestCase):
    def test_command_is_loopback_sequential_and_reasoning_off(self):
        command = build_llama_command(
            Path("/usr/local/bin/llama"),
            Path("/models/Qwen3.5-2B-Q4_K_M.gguf"),
            MODEL_SPECS["2b"]["alias"],
            18902,
        )
        self.assertIn("--host", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertIn("--offline", command)
        self.assertEqual(command[command.index("--threads") + 1], "2")

    def test_client_uses_openai_compatible_structured_request_and_llama_timings(self):
        client = LocalQwenClient("http://127.0.0.1:18902", "qwen3.5-2b-q4-k-m")
        fake = FakeClient()
        client.client = fake
        response = client.complete([{"role": "user", "content": "test"}])
        self.assertEqual(response["content"], '{"supported":true,"confidence":0.9,"source_indexes":[0],"answer":"Готово"}')
        self.assertEqual(response["prompt_tokens"], 40)
        self.assertEqual(response["completion_tokens"], 8)
        self.assertEqual(response["generation_ms"], 1100)
        self.assertAlmostEqual(response["tokens_per_second"], 7.27)
        self.assertEqual(fake.payload[1]["temperature"], 0)
        self.assertEqual(fake.payload[1]["stream"], False)
        client.close()

    def test_answer_status_matches_production_fail_closed_branches(self):
        self.assertEqual(_actual_answer_status({"supported": True}, "общий вопрос", {"scope_match": "generic"}), "answered")
        self.assertEqual(_actual_answer_status({"supported": False}, "общий вопрос", {"scope_match": "unspecified"}), "needs_clarification")
        self.assertEqual(_actual_answer_status({"supported": False}, "DS-2CD1234", {"scope_match": "generic"}), "unsupported")
        self.assertEqual(_actual_answer_status({"supported": False}, "DS-2CD1234", {"scope_match": "unspecified", "retrieved_document_models": []}), "unsupported")

    def test_summary_does_not_turn_two_sample_smoke_test_into_accuracy_claim(self):
        rows = [
            {"model_name": "2b", "structure_pass": True, "applicability_pass": True, "tokens_per_second": 7.3, "generation_ms": 1000},
            {"model_name": "4b", "structure_pass": True, "applicability_pass": False, "tokens_per_second": 3.6, "generation_ms": 2000},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["2b"]["count"], 1)
        self.assertEqual(summary["4b"]["applicability_pass"], 0)
        self.assertEqual(summary["2b"]["median_tokens_per_second"], 7.3)

    def test_real_golden_set_has_required_volume_and_edge_case_coverage(self):
        import app

        samples, metadata = prepare_golden_samples(
            None,
            Path("data/golden_set.json"),
            Path("data/telegram_knowledge_review.json"),
            150,
            app,
        )
        self.assertEqual(metadata["version"], "golden-v2")
        self.assertGreaterEqual(len(samples), 50)
        self.assertLessEqual(len(samples), 150)
        tags = {tag for sample in samples for tag in sample["golden_tags"]}
        self.assertTrue({"model_confusion", "condition_restriction", "insufficient_evidence", "multiple_knowledge_hit"} <= tags)
        self.assertEqual(len({sample["sample_key"] for sample in samples}), len(samples))
        self.assertTrue(any(len(sample["evidence"]) == 2 for sample in samples if "multiple_knowledge_hit" in sample["golden_tags"]))

    def test_golden_samples_use_real_source_questions_and_references(self):
        import app

        samples, _metadata = prepare_golden_samples(
            None,
            Path("data/golden_set.json"),
            Path("data/telegram_knowledge_review.json"),
            135,
            app,
        )
        confusion = next(sample for sample in samples if "model_confusion" in sample["golden_tags"])
        insufficient = next(sample for sample in samples if "insufficient_evidence" in sample["golden_tags"])
        self.assertEqual(confusion["retrieval_mode"], "golden_reference_snapshot")
        self.assertEqual(confusion["evidence"][0]["source_type"], "golden_reference")
        self.assertTrue(confusion["must_refuse"])
        self.assertEqual(insufficient["evidence"], [])
        self.assertTrue(insufficient["must_clarify"] or insufficient["must_refuse"])


class EvaluateV2ReadinessTest(unittest.TestCase):
    """Phase 3.0: the evaluation runner must validate golden and sidecar
    integrity without touching the network or a database."""

    def _golden(self, samples):
        return {
            "version": "golden-v2",
            "labeling": {
                "required_fields": [
                    "expected_answer_status",
                    "expected_scope",
                    "expected_knowledge_keys",
                    "evidence_case_ids",
                    "must_clarify",
                    "must_refuse",
                ],
            },
            "samples": samples,
        }

    def _sample(self, key, status="answered", tags=("direct_answer",)):
        return {
            "sample_key": key,
            "question": f"问题 {key}？",
            "expected_answer_status": status,
            "expected_scope": {},
            "expected_knowledge_keys": [],
            "evidence_case_ids": [1],
            "must_clarify": status == "needs_clarification",
            "must_refuse": status == "unsupported",
            "tags": list(tags),
        }

    def _sidecar(self, cases):
        return {"version": "v2-eval-cases-1", "cases": cases}

    def _case(self, key, category, status="answered"):
        return {
            "sample_key": key,
            "category": category,
            "expected_answer_status": status,
            "v2_knowledge_ids": [],
            "mapping_status": "pending_expert_mapping",
            "paraphrases": [],
            "forbidden_assertions": [],
        }

    def test_valid_pair_passes_with_quota(self):
        from evaluate_v2 import check_golden_set, check_sidecar

        statuses = ["answered"] * 15 + ["needs_clarification"] * 5 + ["unsupported"] * 5
        samples = [self._sample(f"k-{i:02d}", status=s) for i, s in enumerate(statuses)]
        for i in range(5):
            samples.append(self._sample(f"b-{i}", tags=("condition_restriction",)))
        errors, warnings = [], []
        by_key = check_golden_set(self._golden(samples), errors, warnings)
        cases = [self._case(f"k-{i:02d}", "answerable") for i in range(15)]
        cases += [self._case(f"k-{15 + i}", "clarify", "needs_clarification") for i in range(5)]
        cases += [self._case(f"k-{20 + i}", "unsupported", "unsupported") for i in range(5)]
        cases += [self._case(f"b-{i}", "boundary") for i in range(5)]
        check_sidecar(self._sidecar(cases), by_key, errors, warnings)
        self.assertEqual(errors, [])

    def test_missing_quota_and_status_mismatch_are_errors(self):
        from evaluate_v2 import check_golden_set, check_sidecar

        samples = [self._sample("only-one")]
        errors, warnings = [], []
        by_key = check_golden_set(self._golden(samples), errors, warnings)
        cases = [self._case("only-one", "clarify", "answered")]
        check_sidecar(self._sidecar(cases), by_key, errors, warnings)
        self.assertTrue(any("quota not met" in error for error in errors))
        self.assertTrue(any("clarify but labelled" in error for error in errors))

    def test_unknown_sample_key_is_an_error(self):
        from evaluate_v2 import check_golden_set, check_sidecar

        errors, warnings = [], []
        by_key = check_golden_set(self._golden([self._sample("real")]), errors, warnings)
        check_sidecar(self._sidecar([self._case("ghost", "answerable")]), by_key, errors, warnings)
        self.assertTrue(any("unknown golden sample_key" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
