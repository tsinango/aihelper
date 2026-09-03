import unittest

from import_case_memory import build_memory


class CaseMemoryImportTest(unittest.TestCase):
    def test_unreviewed_case_is_recall_only_and_preserves_follow_up(self):
        case = {
            "id": 42,
            "root_author": "Customer",
            "root_question": "Как восстановить пароль?",
            "messages": [
                {"author": "Customer", "text": "Как восстановить пароль?"},
                {"author": "Engineer", "text": "Отправьте запрос в поддержку и фото устройства."},
            ],
            "production_answer_allowed": False,
            "content_hash": "case-hash",
        }
        analysis = {
            "canonical_question": "Как восстановить пароль на устройстве?",
            "knowledge_key": "password_access.reset",
            "knowledge_type": "configuration_howto",
            "models_json": ["DS-K1T320"],
            "context_status": "standalone",
            "extraction_confidence": 0.8,
            "source_content_hash": "analysis-hash",
        }

        memory = build_memory(case, analysis)

        self.assertEqual(memory["source_status"], "ai_derived")
        self.assertFalse(memory["answer_allowed"])
        self.assertFalse(memory["requires_context"])
        self.assertIn("Отправьте запрос", memory["answer_text"])
        self.assertIn("[engineer_instruction] Отправьте запрос", memory["answer_text"])
        self.assertIn("password_access.reset", memory["searchable_text"])

    def test_context_required_case_is_not_silently_presented_as_generic(self):
        case = {
            "id": 43,
            "root_author": "Customer",
            "root_question": "Сколько устройств подключить?",
            "messages": [
                {"author": "Customer", "text": "Сколько устройств подключить?"},
                {"author": "Engineer", "text": "Уточните модель панели."},
            ],
            "production_answer_allowed": False,
        }
        analysis = {
            "canonical_question": "Сколько устройств подключить?",
            "knowledge_key": "capacity_limit.check",
            "models_json": [],
            "context_status": "context_required",
            "extraction_confidence": 0.6,
        }

        memory = build_memory(case, analysis)

        self.assertEqual(memory["source_status"], "needs_context")
        self.assertTrue(memory["requires_context"])
        self.assertFalse(memory["answer_allowed"])


if __name__ == "__main__":
    unittest.main()
