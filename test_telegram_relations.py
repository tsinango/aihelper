import unittest

from telegram_relations import (
    classify_message,
    infer_message_relations,
    message_evidence_status,
    merge_relation,
)


class TelegramRelationsTest(unittest.TestCase):
    def test_root_author_follow_up_is_result_not_dropped(self):
        case = {
            "root_author": "Customer",
            "messages": [
                {"message_id": 1, "author": "Customer", "text": "Как настроить?"},
                {"message_id": 2, "author": "Engineer", "text": "Укажите модель и обновите прошивку."},
                {"message_id": 3, "author": "Customer", "text": "Спасибо, работает."},
            ],
        }
        self.assertEqual(classify_message(case["messages"][2], case, 2), "confirmed_resolution")
        relations = infer_message_relations(case)
        self.assertIn({
            "source_message_id": "3", "target_message_id": "2",
            "relation_type": "confirm_success", "source": "inferred", "confidence": 0.92,
        }, relations)

    def test_native_reply_is_imported(self):
        case = {
            "root_author": "Customer",
            "messages": [
                {"message_id": 11, "author": "Customer", "text": "Как настроить?"},
                {"message_id": 12, "author": "Engineer", "reply_to_message_id": 11, "text": "Откройте меню."},
            ],
        }
        self.assertIn({
            "source_message_id": "12", "target_message_id": "11",
            "relation_type": "answers", "source": "telegram", "confidence": 1.0,
        }, infer_message_relations(case))

    def test_manual_relation_survives_rebuild(self):
        manual = {"source": "manual", "relation_type": "confirm_failure", "confidence": 1.0, "note": "reviewed"}
        inferred = {"source": "inferred", "relation_type": "confirm_success", "confidence": 0.92}
        result = merge_relation(manual, inferred)
        self.assertEqual(result, manual)

    def test_low_signal_engineer_message_is_unconfirmed(self):
        case = {"root_author": "Customer", "messages": []}
        self.assertEqual(classify_message({"author": "Engineer", "text": "Так."}, case, 1), "unconfirmed_claim")

    def test_success_and_failure_evidence_are_distinct(self):
        self.assertEqual(message_evidence_status({"text": "Спасибо, всё работает", "effective_role": "confirmed_resolution"}), "confirmed_success")
        self.assertEqual(message_evidence_status({"text": "Не помогло, всё ещё не работает", "effective_role": "observed_result"}), "confirmed_failure")


if __name__ == "__main__":
    unittest.main()
