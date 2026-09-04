import unittest
from unittest.mock import patch

from v2.learning import (
    UNDERSTANDING_SYSTEM_PROMPT,
    _confirm,
    classify_reply,
    _deduplicate_facts,
    _ask,
    learn_turn,
    _model_facts,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def extract(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        return self.response


class FakeConnection:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.executed = []
        self.cursor_context = _FakeCursorContext(self)

    def cursor(self):
        return self.cursor_context


class _FakeCursorContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.connection.executed.append((" ".join(str(query).split()), params))

    def fetchone(self):
        try:
            return next(self.connection.rows)
        except StopIteration:
            return None

    def fetchall(self):
        return []


class LearningLoopTest(unittest.TestCase):
    def _patch_common(self, pending=None):
        patches = [
            patch("v2.learning.create_thread", return_value={"id": 7, "channel": "web", "mode": "learn"}),
            patch("v2.learning.get_thread", return_value={"id": 7, "channel": "web", "mode": "learn"}),
            patch("v2.learning._ensure_session", return_value={"id": 9, "question_budget": 5}),
            patch("v2.learning._pending_proposal", return_value=pending),
            patch("v2.learning._pending_context", return_value=[]),
            patch("v2.learning._update_proposal"),
            patch("v2.learning._insert_evidence", return_value={"id": 10, "content": "input"}),
            patch("v2.learning._insert_message", return_value={"id": 11, "thread_id": 7, "content": "message"}),
            patch("v2.learning.thread_response", return_value={"thread": {"id": 7}, "messages": []}),
            patch("v2.learning._lock_thread"),
        ]
        started = [item.start() for item in patches]
        self.addCleanup(lambda: [item.stop() for item in patches])
        return started

    def test_control_replies_are_exact_and_other_text_is_correction(self):
        self.assertEqual(classify_reply("对。"), "confirm")
        self.assertEqual(classify_reply("YES"), "confirm")
        self.assertEqual(classify_reply("不知道"), "unknown")
        self.assertEqual(classify_reply("以后再说"), "skip")
        self.assertEqual(classify_reply("对，但旧版不同"), "correction")
        self.assertEqual(classify_reply("不是这个意思"), "correction")

    def test_new_input_stays_provisional_and_asks_one_of_many_facts(self):
        pending = None
        common = self._patch_common(pending)
        model = FakeLLM('{"facts":[{"title":"高度","content":"F-NR-208E/2 是 1U","entity_name":"F-NR-208E/2"},{"title":"系列范围","content":"需要另外确认 F-NR 系列范围","entity_name":"F-NR"}]}')
        common[3].return_value = pending
        with patch("v2.learning._model_facts", return_value=([
            {"title": "高度", "content": "F-NR-208E/2 是 1U", "entity_name": "F-NR-208E/2"},
            {"title": "系列范围", "content": "需要另外确认 F-NR 系列范围", "entity_name": "F-NR"},
        ], False)), patch("v2.learning._find_knowledge", return_value=None) as find, patch("v2.learning._create_knowledge", side_effect=[{"id": 20}, {"id": 21}]) as create, patch("v2.learning._insert_proposal", side_effect=[{"id": 30, "thread_id": 7, "fact_text": "F-NR-208E/2 是 1U"}, {"id": 31, "thread_id": 7, "fact_text": "需要另外确认 F-NR 系列范围"}]) as insert, patch("v2.learning._next_question", return_value=({"id": 40, "content": "我理解为：F-NR-208E/2 是 1U。对吗？"}, "我理解为：F-NR-208E/2 是 1U。对吗？", {"id": 30})):
            result = learn_turn(object(), "F-NR-208E/2 是 1U，并且整个系列都是 1U", llm_service=model)

        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertEqual(result["message"]["id"], 40)
        self.assertEqual(create.call_args_list[0].kwargs["trust"], "provisional")
        self.assertEqual(create.call_args_list[1].kwargs["trust"], "provisional")
        self.assertEqual(insert.call_count, 2)
        find.assert_called()
        self.assertNotIn("user_confirmed", str(create.call_args_list))

    def test_identical_model_facts_are_deduplicated(self):
        facts = [
            {"title": "A", "content": "F-X 支持声音阈值", "entity_name": "F-X"},
            {"title": "A again", "content": " f-x 支持声音阈值 ", "entity_name": "f-x"},
            {"title": "B", "content": "F-X 是黑色", "entity_name": "F-X"},
        ]
        result = _deduplicate_facts(facts)
        self.assertEqual([item["content"] for item in result], ["F-X 支持声音阈值", "F-X 是黑色"])

    def test_only_explicit_confirmation_reaches_confirm_handler(self):
        pending = {"id": 30, "thread_id": 7, "fact_text": "F-NR-208E/2 是 1U", "confirmed_knowledge_id": 20}
        common = self._patch_common(pending)
        with patch("v2.learning._confirm", return_value={"id": 20, "trust": "user_confirmed"}) as confirm, patch("v2.learning._next_question", return_value=(None, None, None)), patch("v2.learning._summary", return_value=({"id": 50, "content": "今天我学会了 1 件事"}, "今天我学会了 1 件事")):
            result = learn_turn(object(), "对")
        confirm.assert_called_once()
        self.assertEqual(result["status"], "confirmed")

        common = self._patch_common(pending)
        correction_conn = object()
        with patch("v2.learning._confirm") as confirm, patch("v2.learning._retire_corrected_knowledge") as retire, patch("v2.learning._model_facts", return_value=([{"title": "修正", "content": "F-NR-208E/2 在旧版不是 1U", "entity_name": "F-NR-208E/2"}], False)), patch("v2.learning._find_knowledge", return_value=None), patch("v2.learning._create_knowledge", return_value={"id": 21}), patch("v2.learning._insert_proposal", return_value={"id": 31}), patch("v2.learning._next_question", return_value=({"id": 41, "content": "我理解为：F-NR-208E/2 在旧版不是 1U。对吗？"}, "", {"id": 31})):
            result = learn_turn(correction_conn, "我觉得旧版不是 1U")
        confirm.assert_not_called()
        retire.assert_called_once_with(correction_conn, pending)
        common[5].assert_called_once_with(correction_conn, 30, "corrected", message_id=11)
        self.assertEqual(result["status"], "awaiting_confirmation")

    def test_model_output_cannot_grant_trust_and_fallback_is_safe(self):
        llm = FakeLLM('{"facts":[{"title":"事实","content":"设备支持声音阈值检测","entity_name":"F-X","trust":"user_confirmed"}]}')
        facts, fallback = _model_facts("F-X 支持声音阈值检测", llm)
        self.assertFalse(fallback)
        self.assertEqual(facts[0]["content"], "设备支持声音阈值检测")
        self.assertNotIn("trust", facts[0])
        self.assertIn("没有提到", UNDERSTANDING_SYSTEM_PROMPT)

        facts, fallback = _model_facts("F-X 可能支持声音阈值检测", None)
        self.assertTrue(fallback)
        self.assertEqual(facts[0]["content"], "F-X 可能支持声音阈值检测")

    def test_confirmation_transition_writes_user_confirmed_and_audit_source(self):
        conn = FakeConnection([{
            "id": 20,
            "title": "高度",
            "content": "F-NR-208E/2 是 1U",
            "entity_name": "F-NR-208E/2",
            "trust": "user_confirmed",
            "active": True,
        }, {
            "id": 30,
            "thread_id": 7,
            "source_message_id": 11,
            "question_message_id": 12,
            "fact_text": "高度",
            "entity_name": "F-NR-208E/2",
            "proposed_trust": "provisional",
            "status": "confirmed",
            "confirmed_knowledge_id": 20,
            "resolution_message_id": 12,
        }])
        knowledge = _confirm(
            conn,
            {"id": 30, "confirmed_knowledge_id": 20},
            {"id": 11, "content": "对"},
            {"id": 12, "content": "对"},
        )
        self.assertEqual(knowledge["trust"], "user_confirmed")
        statements = [query for query, _ in conn.executed]
        self.assertTrue(any("SET trust=CASE" in query and "user_confirmed" in query for query in statements))
        self.assertTrue(any("source_kind, relation" in query and "user_confirmation" in str(params) for query, params in conn.executed))
        self.assertTrue(any("SET status=%s" in query and params[0] == "confirmed" for query, params in conn.executed))

    def test_unknown_and_skip_are_legal_without_reasking_same_proposal(self):
        for answer, expected in (("不知道", "unknown"), ("跳过", "skipped")):
            pending = {"id": 30, "thread_id": 7, "fact_text": "F-X 支持某功能", "confirmed_knowledge_id": 20}
            common = self._patch_common(pending)
            with patch("v2.learning._next_question", return_value=(None, None, None)), patch("v2.learning._summary", return_value=(None, None)), patch("v2.learning._model_facts") as model:
                result = learn_turn(object(), answer)
            self.assertEqual(result["status"], expected)
            model.assert_not_called()
            common[5].assert_called_once_with(
                unittest.mock.ANY,
                30,
                expected,
                message_id=11,
            )

    def test_passive_budget_does_not_cap_direct_inbox_questions(self):
        message = {"id": 40, "thread_id": 7, "content": "question"}
        with patch("v2.learning._session_question_count", return_value=(5, 5)) as count, patch(
            "v2.learning._insert_message", return_value=message
        ):
            result, text = _ask(
                FakeConnection([]),
                {"id": 30, "thread_id": 7, "fact_text": "F-X 支持某功能"},
                {"id": 9, "_unlimited_questions": True},
            )
        self.assertEqual(result, message)
        self.assertIn("F-X 支持某功能", text)
        count.assert_not_called()


if __name__ == "__main__":
    unittest.main()
