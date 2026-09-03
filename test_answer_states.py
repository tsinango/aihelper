import unittest

from app import FAILURE, SERVICE_ERROR, customer_facing_text


class AnswerStateTest(unittest.TestCase):
    def test_clarification_is_not_overwritten_by_answer_fallback(self):
        self.assertEqual(
            customer_facing_text({
                "answer_status": "needs_clarification",
                "answer": FAILURE,
                "clarifying_question": "Уточните точную модель.",
            }),
            "Уточните точную модель.",
        )

    def test_unsupported_and_service_error_have_separate_outputs(self):
        self.assertEqual(customer_facing_text({"answer_status": "unsupported", "unsupported_message": "Не поддерживается."}), "Не поддерживается.")
        self.assertEqual(customer_facing_text({"answer_status": "service_error", "service_error": SERVICE_ERROR}), SERVICE_ERROR)


if __name__ == "__main__":
    unittest.main()
