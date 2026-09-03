import unittest

from pathlib import Path

from organize_telegram_knowledge import atomic_qa, build, make_case


class TelegramKnowledgeWorkbookTest(unittest.TestCase):
    def test_root_author_resolution_is_kept_as_feedback_evidence(self):
        case = {
            "id": 481,
            "external_thread_id": "4690",
            "root_author": "Customer",
            "root_question": "What server should I use?",
            "content_hash": "hash",
            "models": "[\"DS-N104P\"]",
            "messages": [
                {"author": "Customer", "text": "What server should I use?"},
                {"author": "Engineer", "text": "Please follow the setup guide."},
                {"author": "Customer", "text": "Thanks, it worked. I used dev.example.com."},
            ],
        }
        analysis = {
            "id": 1,
            "prompt_version": "TELEGRAM_QUESTION_EXTRACTION_V2_1",
            "canonical_question": "What server should I use?",
            "question_quality": "good",
            "models_json": '["DS-N104P"]',
            "extraction_confidence": 0.9,
        }

        organized = make_case(case, analysis)

        self.assertEqual(organized["message_count"], 3)
        self.assertEqual(organized["messages"][2]["review_role"], "confirmed_resolution")
        self.assertEqual(organized["answer_candidate"]["confirmation_status"], "confirmed_resolution")
        self.assertEqual(organized["answer_candidate"]["feedback_message_indexes"], [2])
        self.assertIn("dev.example.com", organized["answer_candidate"]["customer_feedback_text"])
        self.assertTrue(organized["answer_candidate"]["customer_feedback_was_included"])
        self.assertEqual(organized["atomic_qa"][0]["feedback_message_indexes"], [2])

    def test_root_follow_up_question_starts_a_second_atomic_qa(self):
        case = {
            "id": 602,
            "root_author": "Customer",
            "messages": [
                {"author": "Customer", "text": "How many panels can I connect?"},
                {"author": "Engineer", "text": "Two panels."},
                {"author": "Customer", "text": "What is the maximum cable distance?"},
                {"author": "Engineer", "text": "150 meters."},
            ],
        }
        analysis = {"canonical_question": "How many panels can I connect?", "models_json": []}

        qas = atomic_qa(case["messages"], case, analysis)

        self.assertEqual(len(qas), 2)
        self.assertEqual(qas[0]["answer_text"], "Two panels.")
        self.assertEqual(qas[1]["answer_text"], "150 meters.")

    def test_export_build_contains_all_cases_and_import_shaped_candidates(self):
        root = Path(__file__).parent
        workbook = build(root / "d1-export" / "support_cases.sql", root / "d1-export" / "support_case_analysis.sql")

        self.assertEqual(workbook["summary"]["support_cases"], 602)
        self.assertEqual(len(workbook["cases"]), 602)
        self.assertEqual(len(workbook["candidates"]), 602)
        self.assertTrue({item["scope_level"] for item in workbook["candidates"]} <= {
            "generic", "brand", "family", "series", "model", "conditional", "unspecified",
        })
        candidate = next(item for item in workbook["candidates"] if item["candidate_id"] == "CASE-000481")
        self.assertEqual(candidate["review_status"], "pending")
        self.assertFalse(candidate["production_answer_allowed"])
        self.assertIn("dev.hik-connectru.com", candidate["answer_text"])
        self.assertEqual(candidate["scope_level"], "conditional")


if __name__ == "__main__":
    unittest.main()
