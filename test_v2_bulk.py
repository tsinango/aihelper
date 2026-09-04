import json
import unittest
from unittest.mock import patch

from v2.bulk import (
    classify_input_mode,
    coverage,
    deduplicate_knowledge,
    looks_like_bulk,
    non_exhaustive_semantics,
    parse_batch_confirmation,
    requires_individual_confirmation,
    segment_bulk_text,
)
from v2.compare import compare_and_ask, intrinsic_clarification_question
from v2.learning import (
    _ask,
    _expected_answer_type,
    _learn_bulk_turn,
    _model_facts,
    classify_reply,
    learn_turn,
)


GUANLAN_ARCHITECTURE = (
    "海康观澜大模型整体分为三级架构，分别为基础大模型、行业大模型和任务模型。"
    "基础大模型，包括视觉大模型、语言大模型、多模态大模型，X光大模型、毫米波大模型，"
    "光纤大模型等。基础大模型主要基于海量数据预训练，提供通用基础能力；"
    "行业大模型是在基础大模型的基础上，利用行业数据进一步预训练和微调而成；"
    "任务模型专注于某个具体的场景或业务，是大模型能力落地的重要方式，"
    "如目标识别，周界防范，牲畜检测等。"
)


GUANLAN_21_ITEMS = "\n".join(
    f"Номер {index}. Пакет алгоритмов Guanlan {index:02d}; тип: detection; модель: большой; статус: в разработке."
    for index in range(1, 22)
) + "\nОжидаются другие алгоритмы."


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def extract(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        return self.response


class _Cursor:
    def __init__(self, rows=()):
        self.rows = iter(rows)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.queries.append((str(query), params))

    def fetchone(self):
        return next(self.rows, None)

    def fetchall(self):
        return []


class _Connection:
    def __init__(self, rows=()):
        self.cursor_obj = _Cursor(rows)

    def cursor(self):
        return self.cursor_obj


class BulkIntakeTest(unittest.TestCase):
    def test_bulk_input_detection(self):
        numbered = "\n".join(f"{index}. package {index}" for index in range(1, 4))
        self.assertTrue(looks_like_bulk(numbered))
        self.assertEqual(classify_input_mode(numbered), "bulk_knowledge_payload")
        self.assertEqual(
            classify_input_mode("补充说明", has_pending=True, pending_question="哪个版本？"),
            "answer_to_current_question",
        )
        self.assertEqual(classify_input_mode("对"), "control_reply")

    def test_numbered_list_segmentation(self):
        text = "\n".join(f"Номер {index}. item {index}" for index in range(1, 22))
        segments = segment_bulk_text(text)
        self.assertEqual(len(segments), 21)
        self.assertEqual([item["segment_no"] for item in segments], list(range(1, 22)))
        self.assertEqual(segments[0]["text"], "item 1")
        self.assertEqual(segments[-1]["text"], "item 21")

    def test_bulk_extraction_no_silent_truncation(self):
        facts = [
            {"title": f"事实 {index}", "content": f"事实内容 {index}", "entity_name": "Guanlan"}
            for index in range(1, 26)
        ]
        model = FakeLLM(json.dumps({"facts": facts}, ensure_ascii=False))
        extracted, fallback = _model_facts("长资料", model)
        self.assertFalse(fallback)
        self.assertEqual(len(extracted), 25)
        self.assertEqual(extracted[-1]["content"], "事实内容 25")

    def test_bulk_expected_vs_processed_segments(self):
        snapshot = coverage(
            [
                {"segment_no": 1, "status": "processed"},
                {"segment_no": 2, "status": "processed"},
                {"segment_no": 3, "status": "failed"},
            ],
            facts=[{"content": "a"}, {"content": "b"}],
        )
        self.assertEqual(snapshot["expected_segments"], 3)
        self.assertEqual(snapshot["processed_segments"], 2)
        self.assertEqual(snapshot["failed_segments"], 1)
        self.assertEqual(snapshot["failed_segment_numbers"], [3])

    def test_bulk_processes_every_numbered_segment(self):
        segments = [{"segment_no": index, "text": f"Пакет алгоритмов Guanlan {index:02d}; тип: detection; модель: большой; статус: в разработке."} for index in range(1, 22)]
        batch = {"id": 70, "thread_id": 7, "total_segments": 21, "processed_segments": 0, "failed_segments": 0}

        def extract(segment, _llm):
            number = int(segment.rsplit("Guanlan ", 1)[-1].split(";", 1)[0])
            return ([{"title": f"算法 {number}", "content": segment, "entity_name": "Guanlan"}], False)

        def plan(*_args, **kwargs):
            return {
                "id": kwargs["segment_no"],
                "status": "pending_confirmation",
                "comparison_result": "NEW",
            }

        with patch("v2.learning.segment_bulk_text", return_value=segments), patch(
            "v2.learning._pause_pending_proposals"
        ), patch("v2.learning._create_batch", return_value=batch), patch(
            "v2.learning._model_facts", side_effect=extract
        ) as model, patch("v2.learning._plan_fact", side_effect=plan) as plan_fact, patch(
            "v2.learning._update_batch", return_value=batch
        ) as update, patch(
            "v2.learning._next_question", return_value=({"id": 80, "content": "批量确认"}, "批量确认", {"status": "pending_confirmation", "batch_id": 70})
        ), patch("v2.learning._result", return_value={"status": "awaiting_confirmation"}) as result:
            output = _learn_bulk_turn(
                _Connection(),
                clean=GUANLAN_21_ITEMS,
                thread_id=7,
                session={"id": 9},
                evidence={"id": 10},
                user_message={"id": 11},
                llm_service=object(),
                embedding_client=None,
                had_pending=True,
            )
        self.assertEqual(output, {"status": "awaiting_confirmation"})
        self.assertEqual(model.call_count, 21)
        segment_inputs = [call.args[0] for call in model.call_args_list]
        self.assertTrue(all("Пакет алгоритмов" in value for value in segment_inputs))
        self.assertTrue(all("тип: detection" in value and "модель: большой" in value and "статус: в разработке" in value for value in segment_inputs))
        self.assertEqual(plan_fact.call_count, 21)
        self.assertEqual(update.call_args.kwargs["processed_segments"], 21)
        self.assertEqual(update.call_args.kwargs["failed_segments"], 0)
        self.assertEqual(len(update.call_args.kwargs["clear_facts"]), 21)
        result.assert_called_once()

    def test_pending_question_does_not_swallow_bulk_input(self):
        pending = {
            "id": 30,
            "thread_id": 7,
            "status": "pending_clarification",
            "clarification_question": "哪个硬件 revision？",
        }
        with patch("v2.learning.create_thread", return_value={"id": 7}), patch(
            "v2.learning._lock_thread"
        ), patch("v2.learning._ensure_session", return_value={"id": 9, "question_budget": 5}), patch(
            "v2.learning._pending_proposal", return_value=pending
        ), patch("v2.learning._pending_batch", return_value=None), patch(
            "v2.learning._insert_evidence", return_value={"id": 10}
        ), patch(
            "v2.learning._insert_message", return_value={"id": 11, "thread_id": 7}
        ), patch("v2.learning._learn_bulk_turn", return_value={"status": "bulk"}) as bulk:
            result = learn_turn(object(), GUANLAN_21_ITEMS, llm_service=object())
        self.assertEqual(result["status"], "bulk")
        self.assertTrue(bulk.call_args.kwargs["had_pending"])

    def test_binary_clarification_accepts_yes(self):
        self.assertEqual(classify_reply("是的"), "confirm")
        self.assertEqual(_expected_answer_type("X 是否属于 Y？"), "binary")

    def test_binary_clarification_accepts_no(self):
        self.assertEqual(classify_reply("否"), "negative")
        self.assertEqual(classify_reply("нет"), "negative")

    def test_same_question_not_repeated(self):
        previous = {
            "id": 12,
            "thread_id": 7,
            "sequence_no": 2,
            "role": "assistant",
            "message_type": "clarification",
            "content": "X 是否属于 Y？",
        }
        conn = _Connection([previous])
        with patch("v2.learning._insert_message") as insert:
            message, text = _ask(
                conn,
                {
                    "id": 30,
                    "thread_id": 7,
                    "status": "pending_clarification",
                    "clarification_question": "X 是否属于 Y？",
                },
                {"id": 9, "_unlimited_questions": True},
            )
        self.assertEqual(message["id"], 12)
        self.assertEqual(text, "X 是否属于 Y？")
        insert.assert_not_called()

    def test_no_open_world_questions(self):
        self.assertIsNone(intrinsic_clarification_question({"content": GUANLAN_ARCHITECTURE}))

        class Judge:
            def judge(self, *_args, **_kwargs):
                return json.dumps({
                    "decision": "UNCLEAR",
                    "knowledge_id": None,
                    "question": "是否还有其他模型？",
                    "reason": "补全列表",
                }, ensure_ascii=False)

        result = compare_and_ask({"content": "基础大模型包括视觉、语言、多模态等。", "title": "类型", "entity_name": "Guanlan"}, [], Judge())
        self.assertNotIn("其他模型", result["question"])

    def test_non_exhaustive_list_semantics(self):
        self.assertTrue(non_exhaustive_semantics("视觉、语言、多模态等"))
        self.assertTrue(non_exhaustive_semantics("Ожидаются другие алгоритмы"))
        self.assertFalse(non_exhaustive_semantics("只有这三项"))

    def test_risky_scope_and_negative_facts_stay_individual(self):
        self.assertTrue(requires_individual_confirmation({"content": "整个系列都支持该功能"}))
        self.assertTrue(requires_individual_confirmation({"content": "硬件 revision B 不支持该功能"}))
        self.assertFalse(requires_individual_confirmation({"content": "任务模型用于目标识别"}))

    def test_batch_confirmation(self):
        selected = parse_batch_confirmation("全部确认明确知识", 21)
        self.assertEqual(selected, set(range(1, 22)))

    def test_partial_batch_confirmation(self):
        self.assertEqual(parse_batch_confirmation("确认第1、3、7项", 10), {1, 3, 7})
        self.assertEqual(parse_batch_confirmation("确认前2项", 10), {1, 2})

    def test_summary_deduplicates_knowledge(self):
        rows = deduplicate_knowledge([
            {"id": 4, "content": "Guanlan 有三级架构"},
            {"id": 4, "content": "Guanlan 有三级架构"},
            {"id": 5, "content": "Guanlan 有三级架构"},
            {"id": 6, "content": "任务模型用于具体场景"},
        ])
        self.assertEqual([row["id"] for row in rows], [4, 6])

    def test_guanlan_architecture_regression(self):
        terms = ("三级架构", "基础大模型", "行业大模型", "任务模型", "视觉大模型", "语言大模型", "多模态大模型", "X光大模型", "毫米波大模型", "光纤大模型", "预训练", "微调", "目标识别", "周界防范", "牲畜检测")
        for term in terms:
            self.assertIn(term, GUANLAN_ARCHITECTURE)
        self.assertTrue(non_exhaustive_semantics(GUANLAN_ARCHITECTURE))

    def test_guanlan_21_items_regression(self):
        segments = segment_bulk_text(GUANLAN_21_ITEMS)
        # The trailing non-numbered sentence is retained with item 21 rather
        # than silently dropped, so all source text has a processing owner.
        self.assertEqual(len(segments), 21)
        self.assertIn("Ожидаются другие алгоритмы", segments[-1]["text"])
        self.assertEqual([item["segment_no"] for item in segments], list(range(1, 22)))
        self.assertTrue(all("Пакет алгоритмов" in item["text"] for item in segments))


if __name__ == "__main__":
    unittest.main()
