import json
import unittest

from v2.learning import (
    RUSSIAN_NORMALIZATION_SYSTEM_PROMPT,
    UNDERSTANDING_SYSTEM_PROMPT,
    _has_russian_prose,
    _model_facts,
)


class RussianPipelineLLM:
    """Test double for the two prompts used by the production OpenRouter path."""

    def __init__(self, extraction, normalization):
        self.extraction = extraction
        self.normalization = normalization
        self.calls = []

    def extract(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        if messages[0]["content"] == RUSSIAN_NORMALIZATION_SYSTEM_PROMPT:
            return json.dumps(self.normalization, ensure_ascii=False)
        return json.dumps(self.extraction, ensure_ascii=False)


class RetryRussianPipelineLLM(RussianPipelineLLM):
    def __init__(self, extraction_responses):
        super().__init__(None, {"knowledge_units": []})
        self.extraction_responses = iter(extraction_responses)

    def extract(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        if messages[0]["content"] == RUSSIAN_NORMALIZATION_SYSTEM_PROMPT:
            return json.dumps(self.normalization, ensure_ascii=False)
        return json.dumps(next(self.extraction_responses), ensure_ascii=False)


def legacy_extraction(source, content, *, title="Fact", entity_name=""):
    return {
        "facts": [{
            "title": title,
            "content": content,
            "entity_name": entity_name,
            "source_excerpt": source,
        }]
    }


class RussianKnowledgeLearningTest(unittest.TestCase):
    def test_extraction_prompt_requires_russian_knowledge(self):
        self.assertIn("title 和 canonical_fact 必须用俄语", UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("无论原文是中文、英文还是其他语言", UNDERSTANDING_SYSTEM_PROMPT)

    def test_english_fact_is_normalized_to_russian_and_keeps_markers(self):
        source = "DS-2CD7A26G0/P-IZHS supports 4K at 25 fps and PoE."
        llm = RussianPipelineLLM(
            legacy_extraction(source, source, title="Camera capability", entity_name="DS-2CD7A26G0/P-IZHS"),
            {
                "knowledge_units": [{
                    "index": 0,
                    "title": "Поддержка 4K и PoE",
                    "canonical_fact": "DS-2CD7A26G0/P-IZHS поддерживает 4K при 25 fps и PoE.",
                }],
            },
        )
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(facts), 1)
        self.assertTrue(_has_russian_prose(facts[0]["content"]))
        self.assertEqual(facts[0]["entity_name"], "DS-2CD7A26G0/P-IZHS")
        self.assertEqual(facts[0]["source_excerpt"], source)
        self.assertIn("4K", facts[0]["content"])
        self.assertIn("PoE", facts[0]["content"])
        self.assertEqual(len(llm.calls), 2)

    def test_chinese_fact_is_normalized_without_translating_product_name(self):
        source = "观澜是 Hikvision 的 AI 产品线。"
        llm = RussianPipelineLLM(
            legacy_extraction(source, source, title="产品线", entity_name="观澜"),
            {
                "knowledge_units": [{
                    "index": 0,
                    "title": "Продуктовая линия Hikvision AI",
                    "canonical_fact": "观澜 — продуктовая линия Hikvision для решений на основе AI.",
                }],
            },
        )
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertTrue(_has_russian_prose(facts[0]["content"]))
        self.assertIn("观澜", facts[0]["content"])
        self.assertIn("Hikvision", facts[0]["content"])

    def test_russian_fact_does_not_need_second_call(self):
        source = "F-NR-208E/2 поддерживает установку в стандартную 19-дюймовую стойку."
        llm = RussianPipelineLLM(
            legacy_extraction(source, source, title="Монтаж в стойку", entity_name="F-NR-208E/2"),
            {"knowledge_units": []},
        )
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(facts), 1)
        self.assertEqual(len(llm.calls), 1)

    def test_large_multilingual_source_produces_russian_knowledge_and_keeps_excerpt(self):
        paragraphs = [
            "The RUS-LONG-2026 camera uses a 1/1.8-inch sensor and supports 8 MP resolution.",
            "在低照度场景下，RUS-LONG-2026 可以通过 SmartIR 保持目标可见，同时保留曝光控制条件。",
            "При записи камера поддерживает H.265, H.264 и два независимых потока с разными профилями.",
            "The main stream supports 4K at 25 fps, while the sub-stream supports 1080p at 30 fps.",
            "设备支持 PoE 供电，但在使用加热器和补光灯时必须遵守安装手册中的功率限制。",
            "The camera supports ONVIF Profile S and HTTPS management when the corresponding option is enabled.",
            "При температуре от -30°C до 60°C устройство сохраняет заявленный режим работы; это ограничение относится к данной модели.",
            "This description is intentionally long enough to exercise the real bulk input path without changing the product scope.",
        ]
        source = " ".join(paragraphs * 12)
        self.assertGreater(len(source), 7000)
        proposed = "RUS-LONG-2026 supports the listed sensor, resolution, streams, codecs, power and operating conditions."
        llm = RussianPipelineLLM(
            legacy_extraction(source, proposed, title="Long camera specification", entity_name="RUS-LONG-2026"),
            {
                "knowledge_units": [{
                    "index": 0,
                    "title": "Характеристики RUS-LONG-2026",
                    "canonical_fact": "RUS-LONG-2026 поддерживает указанные сенсор, разрешение, потоки, кодеки, питание и диапазон рабочих температур.",
                }],
            },
        )
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(facts), 1)
        self.assertTrue(_has_russian_prose(facts[0]["content"]))
        self.assertEqual(facts[0]["source_excerpt"], source)
        self.assertGreater(len(source), 7000)

    def test_invalid_russian_normalization_fails_closed(self):
        source = "Model A supports Wi-Fi."
        llm = RussianPipelineLLM(
            legacy_extraction(source, source, title="Wi-Fi", entity_name="Model A"),
            {
                "knowledge_units": [{
                    "index": 0,
                    "title": "Wi-Fi",
                    "canonical_fact": "Model A supports Wi-Fi.",
                }],
            },
        )
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertTrue(fallback)
        self.assertEqual(facts, [])

    def test_invalid_extraction_contract_is_retried_once(self):
        source = "ZH-RETRY-2026 支持 H.265 视频编码。"
        valid = {
            "claims": [{
                "id": "c1",
                "text": source,
            }],
            "knowledge_units": [{
                "title": "Поддержка H.265",
                "canonical_fact": "ZH-RETRY-2026 поддерживает видеокодирование H.265.",
                "entity_name": "ZH-RETRY-2026",
                "supporting_claim_ids": ["c1"],
                "source_excerpt": source,
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
        llm = RetryRussianPipelineLLM([{"knowledge_units": []}, valid])
        facts, fallback = _model_facts(source, llm, normalize_to_russian=True)

        self.assertFalse(fallback)
        self.assertEqual(len(facts), 1)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("上一次输出没有通过", llm.calls[1][0][-2]["content"])

    def test_missing_llm_does_not_save_non_russian_fallback(self):
        facts, fallback = _model_facts(
            "Model A supports Wi-Fi.",
            None,
            normalize_to_russian=True,
        )
        self.assertTrue(fallback)
        self.assertEqual(facts, [])


if __name__ == "__main__":
    unittest.main()
