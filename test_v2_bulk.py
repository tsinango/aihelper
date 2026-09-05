import json
import re
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
    [
        "Описание готовности алгоритмов к использованию (Hikvision / Guanlan)",
        "Номер 1. Пакет «Распознавание лиц». Алгоритм: Распознавание лиц. Тип: Специальный алгоритм. Модель: Малая модель. Состояние: Готов к использованию.",
        "Номер 2. Пакет «Защита периметра». Алгоритмы: Обнаружение вторжения, Обнаружение пересечения линии, Обнаружение входа в область, Обнаружение выхода из области. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 3. Пакет «Структуризация видео». Алгоритм: Структуризация видео. Тип: Специальный алгоритм. Модель: Малая модель. Состояние: Готов к использованию.",
        "Номер 4. Пакет «Рабочие события (внутри помещения)». Алгоритмы: Детекция отсутствия сотрудника / детекция сна на рабочем месте, Детекция длительного нахождения в области, Подсчёт количества сотрудников с настройкой исключений, Детекция использования смартфона. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 5. Пакет «Детекция аномальных событий на улице». Алгоритмы: Детекция скоплений людей, Детекция бегущего человека, Детекция агрессивного поведения, Детекция упавшего человека. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 6. Пакет «Анализ трендов». Алгоритмы: Анализ скопления людей, Статистика подсчёта людей, Подсчёт людей в области. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 7. Алгоритм: Детекция оружия. Тип: AIOP. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 8. Алгоритм: Детекция СИЗ. Тип: AIOP. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 9. Алгоритм: Детекция дыма и огня. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 10. Алгоритм: Обнаружение праздношатания. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 11. Алгоритм: Блокировка пожарного выхода. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: В процессе разработки.",
        "Номер 12. Алгоритм: Детекция курения и использования мобильного телефона. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 13. Алгоритм: Детекция корзины покупок. Тип: AIOP. Модель: Большая модель. Состояние: Настраивается.",
        "Номер 14. Алгоритм: Обнаружение оставленных товаров в корзине покупок. Тип: AIOP. Модель: Большая модель. Состояние: Настраивается.",
        "Номер 15. Алгоритм: Обнаружение свободных мест на вертикальных полках. Тип: AIOP. Модель: Большая модель. Состояние: Настраивается.",
        "Номер 16. Алгоритм: Обнаружение свободных мест в лотке с фруктами. Тип: AIOP. Модель: Большая модель. Состояние: Настраивается.",
        "Номер 17. Алгоритм: Детекция действий с ценными товарами. Тип: AIOP. Модель: Большая модель. Состояние: В процессе разработки.",
        "Номер 18. Алгоритм: Обнаружение открытия / закрытия дверцы холодильника. Тип: AIOP. Модель: Большая модель. Состояние: Настраивается.",
        "Номер 19. Алгоритм: Обнаружение опасного поведения в кассовой зоне. Тип: AIOP. Модель: Большая модель. Состояние: Готов к использованию.",
        "Номер 20. Алгоритм: Детекция несканированных товаров. Тип: AIOP. Модель: Большая модель. Состояние: В процессе разработки.",
        "Номер 21. Алгоритм: Обнаружение сокрытия товара. Тип: Специальный алгоритм. Модель: Большая модель. Состояние: В процессе разработки.",
        "В конце указано, что ожидаются другие алгоритмы.",
        "Краткое резюме:",
        "Большинство алгоритмов, связанных с безопасностью, контролем периметра, аналитикой поведения людей и базовыми детекциями, уже готовы к использованию. Алгоритмы для ритейла (корзины, полки, товары, холодильники) в основном находятся в статусе «Настраивается» или «В процессе разработки».",
    ]
)


GUANLAN_21_EXPECTED = (
    {"number": 1, "names": ("Распознавание лиц",), "type": "Специальный алгоритм", "model": "Малая модель", "status": "Готов к использованию"},
    {"number": 2, "names": ("Защита периметра", "Обнаружение вторжения", "Обнаружение пересечения линии", "Обнаружение входа в область", "Обнаружение выхода из области"), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 3, "names": ("Структуризация видео",), "type": "Специальный алгоритм", "model": "Малая модель", "status": "Готов к использованию"},
    {"number": 4, "names": ("Рабочие события (внутри помещения)", "Детекция отсутствия сотрудника / детекция сна на рабочем месте", "Детекция длительного нахождения в области", "Подсчёт количества сотрудников с настройкой исключений", "Детекция использования смартфона"), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 5, "names": ("Детекция аномальных событий на улице", "Детекция скоплений людей", "Детекция бегущего человека", "Детекция агрессивного поведения", "Детекция упавшего человека"), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 6, "names": ("Анализ трендов", "Анализ скопления людей", "Статистика подсчёта людей", "Подсчёт людей в области"), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 7, "names": ("Детекция оружия",), "type": "AIOP", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 8, "names": ("Детекция СИЗ",), "type": "AIOP", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 9, "names": ("Детекция дыма и огня",), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 10, "names": ("Обнаружение праздношатания",), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 11, "names": ("Блокировка пожарного выхода",), "type": "Специальный алгоритм", "model": "Большая модель", "status": "В процессе разработки"},
    {"number": 12, "names": ("Детекция курения и использования мобильного телефона",), "type": "Специальный алгоритм", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 13, "names": ("Детекция корзины покупок",), "type": "AIOP", "model": "Большая модель", "status": "Настраивается"},
    {"number": 14, "names": ("Обнаружение оставленных товаров в корзине покупок",), "type": "AIOP", "model": "Большая модель", "status": "Настраивается"},
    {"number": 15, "names": ("Обнаружение свободных мест на вертикальных полках",), "type": "AIOP", "model": "Большая модель", "status": "Настраивается"},
    {"number": 16, "names": ("Обнаружение свободных мест в лотке с фруктами",), "type": "AIOP", "model": "Большая модель", "status": "Настраивается"},
    {"number": 17, "names": ("Детекция действий с ценными товарами",), "type": "AIOP", "model": "Большая модель", "status": "В процессе разработки"},
    {"number": 18, "names": ("Обнаружение открытия / закрытия дверцы холодильника",), "type": "AIOP", "model": "Большая модель", "status": "Настраивается"},
    {"number": 19, "names": ("Обнаружение опасного поведения в кассовой зоне",), "type": "AIOP", "model": "Большая модель", "status": "Готов к использованию"},
    {"number": 20, "names": ("Детекция несканированных товаров",), "type": "AIOP", "model": "Большая модель", "status": "В процессе разработки"},
    {"number": 21, "names": ("Обнаружение сокрытия товара",), "type": "Специальный алгоритм", "model": "Большая модель", "status": "В процессе разработки"},
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def extract(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        return self.response


class GuanlanArchitectureExtractor(FakeLLM):
    """Deterministic extraction response; assertions operate on parsed facts."""

    def __init__(self):
        super().__init__(json.dumps({
            "facts": [
                {"title": "架构层级", "content": "海康观澜大模型整体分为三级架构：基础大模型、行业大模型和任务模型。", "entity_name": "Guanlan"},
                {"title": "基础大模型类型", "content": "基础大模型包括视觉大模型、语言大模型、多模态大模型、X光大模型、毫米波大模型和光纤大模型等。", "entity_name": "Guanlan"},
                {"title": "基础大模型作用", "content": "基础大模型主要基于海量数据预训练，提供通用基础能力。", "entity_name": "Guanlan"},
                {"title": "行业大模型构建", "content": "行业大模型在基础大模型的基础上，利用行业数据进一步预训练和微调而成。", "entity_name": "Guanlan"},
                {"title": "任务模型定位", "content": "任务模型专注于某个具体的场景或业务，是大模型能力落地的重要方式。", "entity_name": "Guanlan"},
                {"title": "任务模型示例", "content": "任务模型示例包括目标识别、周界防范和牲畜检测等。", "entity_name": "Guanlan"},
            ]
        }, ensure_ascii=False))


class Guanlan21Extractor:
    """Deterministic extraction mock for the verbatim Russian regression input."""

    def __init__(self):
        self.segments = []

    def extract(self, messages, max_tokens=800):
        source = messages[-1]["content"]
        self.segments.append(source)
        fields = []

        package = re.search(r"Пакет «[^»]+»", source)
        if package:
            fields.append(package.group(0))
        algorithms = re.search(r"Алгоритмы?:\s*(.+?)\. Тип:", source, re.S)
        if algorithms:
            label = "Алгоритмы" if "Алгоритмы:" in algorithms.group(0) else "Алгоритм"
            fields.append(f"{label}: {algorithms.group(1).strip()}")
        for pattern in (
            r"Тип:\s*([^\.]+)",
            r"Модель:\s*([^\.]+)",
            r"Состояние:\s*([^\.]+)",
        ):
            match = re.search(pattern, source, re.S)
            if match:
                fields.append(match.group(0).strip())
        if "ожидаются другие алгоритмы" in source.casefold():
            match = re.search(r"В конце указано, что ожидаются другие алгоритмы\.", source, re.I)
            if match:
                fields.append(match.group(0))
        summary = re.search(r"Краткое резюме:\n(.+)", source, re.S)
        if summary:
            fields.extend(
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", summary.group(1).strip())
                if sentence.strip()
            )
        facts = [
            {
                "title": field[:80],
                "content": field,
                "entity_name": "Guanlan",
                "source_excerpt": field,
            }
            for field in fields
        ]
        return json.dumps({
            "facts": facts,
            "coverage": {
                "complete": True,
                "claims": [
                    {"text": field, "fact_indexes": [index]}
                    for index, field in enumerate(fields)
                ],
                "uncovered_claims": [],
            },
        }, ensure_ascii=False)


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
        batch = {"id": 70, "thread_id": 7, "total_segments": 21, "processed_segments": 0, "failed_segments": 0}
        extractor = Guanlan21Extractor()
        planned_facts = []

        def plan(*_args, **kwargs):
            planned_facts.append((kwargs["segment_no"], kwargs["fact"]))
            return {
                "id": kwargs["segment_no"],
                "status": "pending_confirmation",
                "comparison_result": "NEW",
            }

        with patch("v2.learning._pause_pending_proposals"), patch(
            "v2.learning._create_batch", return_value=batch
        ) as create, patch("v2.learning._plan_fact", side_effect=plan) as plan_fact, patch(
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
                llm_service=extractor,
                embedding_client=None,
                had_pending=True,
            )
        self.assertEqual(output, {"status": "awaiting_confirmation"})
        self.assertEqual(len(extractor.segments), 21)
        self.assertEqual(create.call_args.args[3], GUANLAN_21_ITEMS)
        self.assertEqual(update.call_args.kwargs["processed_segments"], 21)
        self.assertEqual(update.call_args.kwargs["failed_segments"], 0)
        self.assertEqual(plan_fact.call_count, len(planned_facts))
        by_segment = {
            number: "\n".join(fact["content"] for segment, fact in planned_facts if segment == number)
            for number in range(1, 22)
        }
        for expected in GUANLAN_21_EXPECTED:
            with self.subTest(number=expected["number"]):
                extracted = by_segment[expected["number"]]
                for name in expected["names"]:
                    self.assertIn(name, extracted)
                self.assertIn(expected["type"], extracted)
                self.assertIn(expected["model"], extracted)
                self.assertIn(expected["status"], extracted)
        all_extracted = "\n".join(by_segment.values())
        self.assertTrue(non_exhaustive_semantics(all_extracted))
        self.assertIn("ожидаются другие алгоритмы", all_extracted.casefold())
        self.assertGreater(len(update.call_args.kwargs["clear_facts"]), 21)
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

    def test_explicit_limitations_do_not_become_unclear(self):
        self.assertFalse(requires_individual_confirmation({
            "content": "4K доступно до 25 кадров/с, а 1080p — до 60 кадров/с; 4K при 60 кадрах/с не поддерживается.",
            "derived": False,
        }))
        self.assertFalse(requires_individual_confirmation({
            "content": "Фиксированное количество дней хранения нельзя вывести только из модели камеры.",
            "derived": False,
        }))
        self.assertTrue(requires_individual_confirmation({
            "content": "Hardware revision B не поддерживает функцию.",
            "derived": False,
        }))

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
            def __init__(self):
                self.calls = []

            def judge(self, *_args, **_kwargs):
                self.calls.append(True)
                return json.dumps({
                    "decision": "UNCLEAR",
                    "knowledge_id": None,
                    "question": "是否还有其他模型？",
                    "reason": "补全列表",
                }, ensure_ascii=False)

        judge = Judge()
        result = compare_and_ask({"content": "基础大模型包括视觉、语言、多模态等。", "title": "类型", "entity_name": "Guanlan"}, [], judge)
        self.assertEqual(result["decision"], "NEW")
        self.assertIsNone(result["question"])
        self.assertEqual(judge.calls, [])

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
        extractor = GuanlanArchitectureExtractor()
        facts, fallback = _model_facts(GUANLAN_ARCHITECTURE, extractor)
        self.assertFalse(fallback)
        self.assertEqual(extractor.calls[0][0][-1]["content"], GUANLAN_ARCHITECTURE)
        extracted = "\n".join(item["content"] for item in facts)
        expected = (
            "三级架构", "基础大模型、行业大模型和任务模型", "视觉大模型", "语言大模型",
            "多模态大模型", "X光大模型", "毫米波大模型", "光纤大模型", "等",
            "海量数据预训练", "通用基础能力", "行业数据进一步预训练和微调",
            "具体的场景或业务", "目标识别", "周界防范", "牲畜检测",
        )
        for term in expected:
            self.assertIn(term, extracted)
        self.assertEqual(len(facts), 6)

    def test_segment_with_multiple_facts_requires_complete_coverage(self):
        incomplete = FakeLLM(json.dumps({
            "facts": [{
                "title": "算法包",
                "content": "Пакет алгоритмов Guanlan 01",
                "entity_name": "Guanlan",
                "source_excerpt": "Пакет алгоритмов Guanlan 01",
            }],
            "coverage": {
                "complete": True,
                "claims": [{"text": "Пакет алгоритмов Guanlan 01", "fact_indexes": [0]}],
                "uncovered_claims": [],
            },
        }, ensure_ascii=False))
        facts, fallback = _model_facts(
            "Пакет алгоритмов Guanlan 01; тип: detection; модель: большой; статус: в разработке.",
            incomplete,
            require_coverage=True,
        )
        self.assertTrue(fallback)
        self.assertEqual(facts, [])

    def test_segment_coverage_contract_accepts_all_explicit_fields(self):
        source = "Пакет алгоритмов Guanlan 01; тип: detection; модель: большой; статус: в разработке."
        facts = [
            {"title": "пакет", "content": "Пакет алгоритмов Guanlan 01", "source_excerpt": "Пакет алгоритмов Guanlan 01"},
            {"title": "тип", "content": "тип: detection", "source_excerpt": "тип: detection"},
            {"title": "модель", "content": "модель: большой", "source_excerpt": "модель: большой"},
            {"title": "статус", "content": "статус: в разработке", "source_excerpt": "статус: в разработке"},
        ]
        complete = FakeLLM(json.dumps({
            "facts": facts,
            "coverage": {
                "complete": True,
                "claims": [
                    {"text": fact["source_excerpt"], "fact_indexes": [index]}
                    for index, fact in enumerate(facts)
                ],
                "uncovered_claims": [],
            },
        }, ensure_ascii=False))
        extracted, fallback = _model_facts(source, complete, require_coverage=True)
        self.assertFalse(fallback)
        self.assertEqual(len(extracted), 4)

    def test_guanlan_21_items_regression(self):
        segments = segment_bulk_text(GUANLAN_21_ITEMS)
        # The trailing non-numbered sentence is retained with item 21 rather
        # than silently dropped, so all source text has a processing owner.
        self.assertEqual(len(segments), 21)
        self.assertIn("ожидаются другие алгоритмы", segments[-1]["text"])
        self.assertEqual([item["segment_no"] for item in segments], list(range(1, 22)))
        self.assertIn("Описание готовности алгоритмов", segments[0]["text"])
        segmented_source = "\n".join(item["text"] for item in segments)
        for expected in GUANLAN_21_EXPECTED:
            for name in expected["names"]:
                self.assertIn(name, segmented_source)


if __name__ == "__main__":
    unittest.main()
