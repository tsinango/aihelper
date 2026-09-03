import unittest
from unittest.mock import patch

from app import build_decision_messages, llm_decision, model_confidence, no_support_answer, select_scoped_hits
from helpers import apply_scope_to_answer, scope_details


DS_K1T320_EVIDENCE = [{
    "title": "DS-K1T320 Series Face Recognition Terminal_User Manual_V2.2_20260729",
    "product_model": None,
}]


class ScopeAwareAnswerTest(unittest.TestCase):
    def test_concrete_route_without_support_does_not_request_model(self):
        scope = scope_details("какой адрес сервера hikconnect", [])
        answer = no_support_answer({}, scope, "parameter")

        self.assertEqual(answer["answer_status"], "unsupported")
        self.assertIn("подтвердить", answer["unsupported_message"])
        self.assertNotIn("clarifying_question", answer)

    def test_exact_verified_hit_suppresses_unrelated_vector_hits(self):
        hits = [
            {"scope_match": "generic", "exact_match": False, "rrf_score": 0.9, "verified_knowledge_id": 10},
            {"scope_match": "generic", "exact_match": True, "rrf_score": 0.1, "verified_knowledge_id": 11},
        ]

        selected = select_scoped_hits(hits, 3)

        self.assertEqual([item["verified_knowledge_id"] for item in selected], [11])

    def test_unknown_route_still_requests_a_concrete_question(self):
        scope = scope_details("iflow", [])
        answer = no_support_answer({}, scope, "unknown")

        self.assertEqual(answer["answer_status"], "needs_clarification")
        self.assertIn("что именно", answer["clarifying_question"])

    def test_model_confidence_is_normalized_safely(self):
        self.assertEqual(model_confidence("0.75"), 0.75)
        self.assertEqual(model_confidence("high"), 0.85)
        self.assertEqual(model_confidence("unexpected"), 0.0)

    def test_retrieved_model_is_not_assumed_for_unspecified_query(self):
        scope = scope_details("как добавить палец в оборудование", DS_K1T320_EVIDENCE)
        answer = apply_scope_to_answer("Чтобы добавить отпечаток, откройте User.", scope)

        self.assertEqual(scope["explicit_user_models"], [])
        self.assertEqual(scope["retrieved_document_models"], ["DS-K1T320"])
        self.assertEqual(scope["scope_match"], "unspecified")
        self.assertTrue(answer.startswith("Если речь о DS-K1T320,"))
        self.assertIn("Если у вас другая модель, укажите её", answer)

    def test_explicit_model_allows_direct_answer(self):
        scope = scope_details("как добавить палец в DS-K1T320", DS_K1T320_EVIDENCE)
        answer = apply_scope_to_answer("Чтобы добавить отпечаток, откройте User.", scope)

        self.assertEqual(scope["explicit_user_models"], ["DS-K1T320"])
        self.assertEqual(scope["scope_match"], "exact")
        self.assertEqual(answer, "Чтобы добавить отпечаток, откройте User.")

    def test_llm_prompt_keeps_ai_derived_status_and_route(self):
        class FakeLLM:
            def __init__(self):
                self.messages = None

            def judge(self, messages):
                self.messages = messages
                return '{"supported": true, "confidence": 0.8, "source_indexes": [0], "answer": "Проверьте устройство."}'

        fake = FakeLLM()
        evidence = [{
            "source_type": "case_memory",
            "support_case_id": 42,
            "title": "Исторический ответ",
            "page_number": None,
            "language": "ru",
            "content": "проверить устройство",
            "source_status": "ai_derived",
            "source_confidence": 0.8,
            "requires_context": False,
        }]
        with patch("app.llm", fake), patch.dict("app.settings", {"openrouter_api_key": "test-key"}):
            decision = llm_decision("Что сделать?", evidence, "Что сделать?", route="operation")

        self.assertTrue(decision["supported"])
        self.assertIn("ai_derived", fake.messages[1]["content"])
        self.assertIn('"route": "operation"', fake.messages[1]["content"])

    def test_llm_prompt_restricts_facts_and_brand_scope(self):
        evidence = [{
            "source_type": "verified_knowledge",
            "title": "Hik-Connect server",
            "page_number": None,
            "language": "ru",
            "content": '{"answer_text":"Укажите адрес сервера из настроек устройства.","claims":[],"procedure_steps":[]}',
            "answer_text": "Укажите адрес сервера из настроек устройства.",
            "claims": [],
            "procedure_steps": [],
            "scope": {"brands": ["iFlow"]},
        }]
        messages, _scope = build_decision_messages(
            "какой адрес сервера hikconnect", evidence, "какой адрес сервера hikconnect hcserver",
            route="parameter",
        )
        prompt = messages[0]["content"]

        self.assertIn("answer_text, claims, procedure_steps", prompt)
        self.assertIn("scope.brands", prompt)
        self.assertIn("Do not add a URL, DNS, NTP, reboot", prompt)
        self.assertIn('"fact_fields"', messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
