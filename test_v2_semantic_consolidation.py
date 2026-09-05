import json
import unittest
from unittest.mock import patch

from v2.bulk import extraction_coverage_is_complete, segment_bulk_text
from v2.learning import UNDERSTANDING_SYSTEM_PROMPT, _model_facts, learn_turn


TANDEMVU_SOURCE = (
    "Cameras with TandemVu technology provide the ability to monitor an entire area "
    "while zooming in to inspect specific security incidents without creating blind "
    "spots related to zooming, tilting, or panning. This means that users can see "
    "the big picture of the scene while also capturing every detail virtually all "
    "the time, thus enhancing security."
)
TANDEMVU_CLAIM_1 = TANDEMVU_SOURCE.split(" This means", 1)[0]
TANDEMVU_CLAIM_2 = "This means" + TANDEMVU_SOURCE.split(" This means", 1)[1]


class SemanticExtractor:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def extract(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        return json.dumps(self.response, ensure_ascii=False)


def semantic_response(source, claims, units):
    return {
        "claims": [{"id": claim_id, "text": text} for claim_id, text, _, _ in claims],
        "knowledge_units": units,
        "coverage": {
            "complete": True,
            "claims": [
                {
                    "id": claim_id,
                    "text": text,
                    "knowledge_unit_indexes": indexes,
                    "disposition": disposition,
                }
                for claim_id, text, indexes, disposition in claims
            ],
            "uncovered_claims": [],
        },
    }


class SemanticConsolidationTest(unittest.TestCase):
    def test_prompt_distinguishes_claims_from_semantic_knowledge_units(self):
        self.assertIn("多个 claims 可以支持同一个 knowledge_unit", UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("一个句子可能有多个 units，多句话也可能只有一个 unit", UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("每个被映射到 knowledge_unit 的 claim 的技术意义都必须体现在 canonical_fact 中", UNDERSTANDING_SYSTEM_PROMPT)
        self.assertNotIn("每个 facts 项只能包含一个事实", UNDERSTANDING_SYSTEM_PROMPT)

    def test_tandemvu_is_one_semantic_unit_with_multiple_claims(self):
        canonical = (
            "TandemVu cameras can maintain visibility of the overall monitored scene "
            "while inspecting local details with zoom or PTZ, reducing blind spots "
            "caused by changing the viewing direction or zoom level."
        )
        response = semantic_response(
            TANDEMVU_SOURCE,
            [
                ("c1", TANDEMVU_CLAIM_1, [0], "knowledge"),
                ("c2", TANDEMVU_CLAIM_2, [0], "knowledge"),
            ],
            [{
                "title": "TandemVu 全景与细节监控",
                "canonical_fact": canonical,
                "entity_name": "TandemVu",
                "supporting_claim_ids": ["c1", "c2"],
                "source_excerpt": TANDEMVU_SOURCE,
                "derived": False,
            }],
        )
        extractor = SemanticExtractor(response)
        facts, fallback = _model_facts(TANDEMVU_SOURCE, extractor)

        self.assertFalse(fallback)
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["content"], canonical)
        self.assertEqual(facts[0]["supporting_claim_ids"], ["c1", "c2"])
        self.assertEqual(facts[0]["source_excerpt"], TANDEMVU_SOURCE)

    def test_cohesive_prose_is_not_split_into_sentence_segments(self):
        segments = segment_bulk_text(TANDEMVU_SOURCE)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], TANDEMVU_SOURCE)

    def test_many_claims_can_map_to_one_unit_in_coverage(self):
        facts = [{"content": "保守归并后的 TandemVu 能力", "source_excerpt": TANDEMVU_SOURCE}]
        coverage = semantic_response(
            TANDEMVU_SOURCE,
            [
                ("c1", TANDEMVU_CLAIM_1, [0], "knowledge"),
                ("c2", TANDEMVU_CLAIM_2, [0], "knowledge"),
            ],
            [],
        )["coverage"]
        self.assertTrue(extraction_coverage_is_complete(TANDEMVU_SOURCE, facts, coverage))

    def test_one_sentence_with_two_independent_parameters_stays_two_units(self):
        source = "The camera has 8 MP resolution and supports PoE."
        response = semantic_response(
            source,
            [
                ("c1", "The camera has 8 MP resolution", [0], "knowledge"),
                ("c2", "supports PoE", [1], "knowledge"),
            ],
            [
                {
                    "title": "Resolution",
                    "canonical_fact": "The camera has 8 MP resolution.",
                    "entity_name": "camera",
                    "supporting_claim_ids": ["c1"],
                    "source_excerpt": source,
                },
                {
                    "title": "Power over Ethernet",
                    "canonical_fact": "The camera supports PoE.",
                    "entity_name": "camera",
                    "supporting_claim_ids": ["c2"],
                    "source_excerpt": source,
                },
            ],
        )
        facts, fallback = _model_facts(source, SemanticExtractor(response))

        self.assertFalse(fallback)
        self.assertEqual(len(facts), 2)
        self.assertEqual([fact["supporting_claim_ids"] for fact in facts], [["c1"], ["c2"]])

    def test_different_models_are_not_consolidated(self):
        source = "Model A supports Wi-Fi. Model B supports PoE."
        response = semantic_response(
            source,
            [
                ("c1", "Model A supports Wi-Fi", [0], "knowledge"),
                ("c2", "Model B supports PoE", [1], "knowledge"),
            ],
            [
                {
                    "title": "Model A Wi-Fi",
                    "canonical_fact": "Model A supports Wi-Fi.",
                    "entity_name": "Model A",
                    "supporting_claim_ids": ["c1"],
                    "source_excerpt": "Model A supports Wi-Fi.",
                },
                {
                    "title": "Model B PoE",
                    "canonical_fact": "Model B supports PoE.",
                    "entity_name": "Model B",
                    "supporting_claim_ids": ["c2"],
                    "source_excerpt": "Model B supports PoE.",
                },
            ],
        )
        facts, fallback = _model_facts(source, SemanticExtractor(response))

        self.assertFalse(fallback)
        self.assertEqual([fact["entity_name"] for fact in facts], ["Model A", "Model B"])

    def test_marketing_claim_can_be_covered_without_creating_knowledge(self):
        source = "This dramatically improves security and provides an excellent user experience."
        response = semantic_response(
            source,
            [("c1", source, [], "non_knowledge")],
            [],
        )
        facts, fallback = _model_facts(source, SemanticExtractor(response))

        self.assertFalse(fallback)
        self.assertEqual(facts, [])

    def test_conditions_remain_in_separate_units(self):
        source = "The camera supports 4K at 25 fps. It supports 1080p at 60 fps."
        response = semantic_response(
            source,
            [
                ("c1", "The camera supports 4K at 25 fps", [0], "knowledge"),
                ("c2", "It supports 1080p at 60 fps", [1], "knowledge"),
            ],
            [
                {
                    "title": "4K frame rate",
                    "canonical_fact": "The camera supports 4K at 25 fps.",
                    "entity_name": "camera",
                    "supporting_claim_ids": ["c1"],
                    "source_excerpt": "The camera supports 4K at 25 fps.",
                },
                {
                    "title": "1080p frame rate",
                    "canonical_fact": "The camera supports 1080p at 60 fps.",
                    "entity_name": "camera",
                    "supporting_claim_ids": ["c2"],
                    "source_excerpt": "It supports 1080p at 60 fps.",
                },
            ],
        )
        facts, fallback = _model_facts(source, SemanticExtractor(response))

        self.assertFalse(fallback)
        self.assertEqual([fact["content"] for fact in facts], [
            "The camera supports 4K at 25 fps.",
            "The camera supports 1080p at 60 fps.",
        ])

    def test_semantic_unit_is_confirmed_once(self):
        canonical = "TandemVu cameras keep the overall scene visible while inspecting local details."
        response = semantic_response(
            TANDEMVU_SOURCE,
            [
                ("c1", TANDEMVU_CLAIM_1, [0], "knowledge"),
                ("c2", TANDEMVU_CLAIM_2, [0], "knowledge"),
            ],
            [{
                "title": "TandemVu monitoring",
                "canonical_fact": canonical,
                "entity_name": "TandemVu",
                "supporting_claim_ids": ["c1", "c2"],
                "source_excerpt": TANDEMVU_SOURCE,
            }],
        )
        plan_result = {"id": 30, "status": "pending_confirmation", "comparison_result": "NEW"}
        with patch("v2.learning.create_thread", return_value={"id": 7}), patch(
            "v2.learning._lock_thread"
        ), patch("v2.learning._ensure_session", return_value={"id": 9, "session_type": "active_inbox", "question_budget": 5}), patch(
            "v2.learning._pending_proposal", return_value=None
        ), patch("v2.learning._pending_batch", return_value=None), patch(
            "v2.learning._insert_evidence", return_value={"id": 10}
        ), patch("v2.learning._insert_message", return_value={"id": 11, "thread_id": 7}), patch(
            "v2.learning._plan_fact", return_value=plan_result
        ) as plan, patch("v2.learning._resume_paused_proposals"), patch(
            "v2.learning._next_question",
            return_value=(
                {"id": 12, "content": f"我理解为：{canonical}。对吗？"},
                f"我理解为：{canonical}。对吗？",
                plan_result,
            ),
        ), patch("v2.learning.thread_response", return_value={"thread": {"id": 7}, "messages": []}):
            result = learn_turn(object(), TANDEMVU_SOURCE, llm_service=SemanticExtractor(response))

        self.assertEqual(result["status"], "awaiting_confirmation")
        plan.assert_called_once()
        self.assertEqual(plan.call_args.kwargs["fact"]["content"], canonical)

    def test_invalid_claim_mapping_fails_closed(self):
        source = "Model A supports Wi-Fi. Model B supports PoE."
        response = semantic_response(
            source,
            [
                ("c1", "Model A supports Wi-Fi", [0], "knowledge"),
                ("c2", "Model B supports PoE", [1], "knowledge"),
            ],
            [{
                "title": "Model A",
                "canonical_fact": "Model A supports Wi-Fi.",
                "entity_name": "Model A",
                "supporting_claim_ids": ["c1", "c2"],
                "source_excerpt": source,
            }],
        )
        facts, fallback = _model_facts(source, SemanticExtractor(response))

        self.assertTrue(fallback)
        self.assertEqual(facts[0]["content"], source)


if __name__ == "__main__":
    unittest.main()
