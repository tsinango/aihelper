from __future__ import annotations

import json
import unittest

from v2.learning import (
    _compose_fact_content,
    _consolidate_related_units,
    _meaningful_tokens,
    _model_facts,
    _postprocess_semantic_units,
)


class _LegacyLLM:
    def __init__(self, payload):
        self.payload = payload

    def extract(self, _messages, max_tokens=800):
        return json.dumps(self.payload, ensure_ascii=False)


class _StructuredLLM:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def extract_structured(self, messages, schema, max_tokens=800):
        self.calls.append((messages, schema, max_tokens))
        return json.dumps(next(self.payloads), ensure_ascii=False)


def _structured_payload(source, *, scope="", conditions=None):
    return {
        "claims": [{"id": "c1", "text": source}],
        "knowledge_units": [{
            "title": "Поддержка PoE+",
            "canonical_fact": "Камера поддерживает PoE+.",
            "entity_name": "F-X",
            "supporting_claim_ids": ["c1"],
            "source_excerpt": source,
            "derived": False,
            "scope": scope,
            "conditions": conditions or [],
        }],
        "coverage": {
            "complete": True,
            "claims": [{
                "id": "c1",
                "text": source,
                "knowledge_unit_indexes": [0],
                "disposition": "knowledge",
            }],
            "uncovered_claims": [],
        },
    }


class LearningConsistencyTest(unittest.TestCase):
    def test_legacy_facts_keep_scope_and_conditions(self):
        source = "F-X поддерживает режим при низкой освещенности."
        facts, fallback = _model_facts(
            source,
            _LegacyLLM({
                "facts": [{
                    "title": "Режим",
                    "content": source,
                    "entity_name": "F-X",
                    "source_excerpt": source,
                    "scope": " ночная съёмка ",
                    "conditions": "при низкой освещенности",
                }],
            }),
        )
        self.assertFalse(fallback)
        self.assertEqual(facts[0]["scope"], "ночная съёмка")
        self.assertEqual(facts[0]["conditions"], ["при низкой освещенности"])

    def test_scope_and_conditions_are_materialized_once_in_content(self):
        fact = {
            "content": "F-X supports the feature [scope: indoor]",
            "scope": " indoor ",
            "conditions": ["when powered", "WHEN   POWERED", "when configured"],
        }
        result = _compose_fact_content(fact)
        self.assertEqual(
            result["content"],
            "F-X supports the feature [scope: indoor] — when powered; when configured",
        )

    def test_structured_and_legacy_paths_materialize_metadata_before_proposal(self):
        structured = _postprocess_semantic_units([{
            "content": "Камера поддерживает режим.",
            "entity_name": "F-X",
            "scope": "ночная съёмка",
            "conditions": ["при низкой освещенности"],
        }])
        legacy, fallback = _model_facts(
            "Камера поддерживает режим.",
            _LegacyLLM({"facts": [{
                "title": "Режим",
                "content": "Камера поддерживает режим.",
                "entity_name": "F-X",
                "scope": "ночная съёмка",
                "conditions": ["при низкой освещенности"],
            }]}),
        )
        self.assertFalse(fallback)
        self.assertIn("ночная съёмка", structured[0]["content"])
        self.assertIn("при низкой освещенности", structured[0]["content"])
        self.assertEqual(legacy[0]["content"], structured[0]["content"])

    def test_scope_and_conditions_merge_deterministically(self):
        facts = _consolidate_related_units([
            {
                "content": "Камера поддерживает режим TandemVu.",
                "entity_name": "F-X",
                "scope": "Ночная съёмка",
                "conditions": ["при низкой освещенности"],
            },
            {
                "content": "Камера использует режим TandemVu для уменьшения слепых зон.",
                "entity_name": "F-X",
                "scope": " ночная   съёмка ",
                "conditions": ["При низкой освещенности", "при записи"],
            },
        ])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["scope"], "Ночная съёмка")
        self.assertEqual(facts[0]["conditions"], [
            "при низкой освещенности", "при записи",
        ])

    def test_scope_or_disjoint_conditions_keep_units_separate(self):
        base = {
            "content": "Камера поддерживает режим TandemVu.",
            "entity_name": "F-X",
        }
        scoped = dict(base, scope="дневная съёмка")
        other_scope = dict(base, scope="ночная съёмка")
        self.assertEqual(len(_consolidate_related_units([scoped, other_scope])), 2)

        daytime = dict(base, conditions=["при дневном свете"])
        night = dict(base, conditions=["при низкой освещенности"])
        self.assertEqual(len(_consolidate_related_units([daytime, night])), 2)

    def test_tokenization_covers_english_chinese_and_russian(self):
        self.assertIn("h.265", _meaningful_tokens("Камера поддерживает H.265"))
        tokens = _meaningful_tokens(
            "Camera supports resolution; камера поддерживает разрешение; 中文功能",
            "Camera камера",
        )
        self.assertIn("resolution", tokens)
        self.assertIn("разрешение", tokens)
        self.assertIn("中", tokens)
        self.assertIn("文", tokens)
        self.assertIn("功", tokens)
        self.assertIn("能", tokens)

    def test_russian_conjoined_predicates_split_in_legacy_fallback(self):
        source = "Камера имеет 8 МП и поддерживает PoE."
        facts, fallback = _model_facts(
            source,
            _LegacyLLM({"facts": [{
                "title": "Характеристики",
                "content": source,
                "entity_name": "F-X",
                "source_excerpt": source,
                "scope": "для помещений",
                "conditions": ["при подключении питания"],
            }]}),
            normalize_to_russian=True,
        )
        self.assertFalse(fallback)
        self.assertEqual([item["content"].split(" — ", 1)[0] for item in facts], [
            "Камера имеет 8 МП", "Камера поддерживает PoE",
        ])
        self.assertTrue(all("для помещений" in item["content"] for item in facts))
        self.assertTrue(all("при подключении питания" in item["content"] for item in facts))

    def test_russian_paraphrases_with_same_technical_marker_consolidate(self):
        facts = _postprocess_semantic_units([
            {
                "content": "Камера поддерживает H.265.",
                "entity_name": "F-X",
                "source_excerpt": "Камера поддерживает H.265.",
            },
            {
                "content": "Кодек H.265 поддерживается камерой.",
                "entity_name": "F-X",
                "source_excerpt": "Кодек H.265 поддерживается камерой.",
            },
        ])
        self.assertEqual(len(facts), 1)
        self.assertIn("H.265", facts[0]["content"])

    def test_russian_independent_attributes_stay_separate(self):
        facts = _postprocess_semantic_units([{
            "content": "Камера имеет разрешение 8 Мп и поддерживает PoE+.",
            "entity_name": "F-X",
            "source_excerpt": "Камера имеет разрешение 8 Мп и поддерживает PoE+.",
        }])
        self.assertEqual([fact["content"] for fact in facts], [
            "Камера имеет разрешение 8 Мп.",
            "Камера поддерживает PoE+.",
        ])

    def test_conditions_are_preserved_for_english_chinese_and_russian_sources(self):
        sources = (
            "The camera supports PoE+ when used with a compatible power sourcing device.",
            "在使用兼容供电设备时，该摄像机支持 PoE+。",
            "Камера поддерживает PoE+ при использовании совместимого источника питания.",
        )
        for source in sources:
            with self.subTest(source=source):
                facts, fallback = _model_facts(
                    source,
                    _LegacyLLM({"facts": [{
                        "title": "Поддержка PoE+",
                        "content": "Камера поддерживает PoE+.",
                        "entity_name": "F-X",
                        "source_excerpt": source,
                        "conditions": ["при использовании совместимого источника питания"],
                    }]}),
                    normalize_to_russian=True,
                )
                self.assertFalse(fallback)
                self.assertIn("PoE+", facts[0]["content"])
                self.assertIn("при использовании совместимого источника питания", facts[0]["content"])

    def test_structured_output_conditions_are_materialized_before_proposal(self):
        source = "The camera supports PoE+ when used with a compatible power sourcing device."
        payload = _structured_payload(
            source,
            conditions=["при использовании совместимого источника питания"],
        )
        facts, fallback = _model_facts(source, _StructuredLLM([payload]), normalize_to_russian=True)
        self.assertFalse(fallback)
        self.assertEqual(facts[0]["content"].count("при использовании совместимого источника питания"), 1)

    def test_english_condition_fails_then_repair_uses_russian(self):
        source = "The camera supports PoE+ when used with a compatible PSE."
        llm = _StructuredLLM([
            _structured_payload(source, conditions=["when used with a compatible PSE"]),
            _structured_payload(source, conditions=["при использовании совместимого PSE"]),
        ])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("при использовании совместимого PSE", facts[0]["content"])
        self.assertNotIn("when used with", facts[0]["content"])

    def test_english_scope_fails_then_repair_uses_russian(self):
        source = "The camera supports PoE+ for outdoor installation."
        llm = _StructuredLLM([
            _structured_payload(source, scope="for outdoor installation"),
            _structured_payload(source, scope="для наружной установки"),
        ])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("для наружной установки", facts[0]["content"])
        self.assertNotIn("for outdoor", facts[0]["content"])

    def test_russian_scope_and_condition_are_accepted(self):
        source = "Камера поддерживает PoE+ для наружной установки."
        llm = _StructuredLLM([_structured_payload(
            source,
            scope="для наружной установки",
            conditions=["при использовании совместимого источника питания"],
        )])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(llm.calls), 1)

    def test_russian_prose_with_technical_markers_is_accepted(self):
        source = "Камера поддерживает PoE+ и ONVIF Profile T."
        llm = _StructuredLLM([_structured_payload(
            source,
            conditions=["при использовании PoE+ и ONVIF Profile T"],
        )])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("PoE+ и ONVIF Profile T", facts[0]["content"])

    def test_non_russian_condition_fails_closed_after_one_repair(self):
        source = "The camera supports PoE+ when used with a compatible PSE."
        invalid = _structured_payload(source, conditions=["when used with a compatible PSE"])
        llm = _StructuredLLM([invalid, invalid])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertTrue(fallback)
        self.assertEqual(facts, [])
        self.assertEqual(len(llm.calls), 2)

    def test_non_russian_condition_is_rejected_on_default_learning_path(self):
        source = "The camera supports PoE+ when used with a compatible PSE."
        invalid = _structured_payload(source, conditions=["when used with a compatible PSE"])
        llm = _StructuredLLM([invalid, invalid])
        facts, fallback = _model_facts(source, llm)

        self.assertTrue(fallback)
        self.assertEqual(facts, [])
        self.assertEqual(len(llm.calls), 2)

    def test_chinese_condition_fails_closed_after_one_repair(self):
        source = "在使用兼容供电设备时，该摄像机支持 PoE+。"
        invalid = _structured_payload(source, conditions=["在使用兼容供电设备时"])
        llm = _StructuredLLM([invalid, invalid])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertTrue(fallback)
        self.assertEqual(facts, [])
        self.assertEqual(len(llm.calls), 2)

    def test_russian_fallback_never_returns_non_russian_fact(self):
        facts, fallback = _model_facts(
            "Model A supports Wi-Fi.", None, normalize_to_russian=True,
        )
        self.assertTrue(fallback)
        self.assertEqual(facts, [])


if __name__ == "__main__":
    unittest.main()
