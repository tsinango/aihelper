import unittest

from v2.compare import COMPARE_SYSTEM_PROMPT, compare_and_ask


class FakeJudge:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def judge(self, messages, max_tokens=1000):
        self.calls.append((messages, max_tokens))
        return self.response


FACT = {
    "title": "机箱高度",
    "content": "F-NR-208E/2 的机箱高度是 1U。",
    "entity_name": "F-NR-208E/2",
}
CANDIDATE = {
    "id": 17,
    "title": "机箱高度",
    "content": "F-NR-208E/2 的机箱高度为 1U。",
    "entity_name": "F-NR-208E/2",
    "trust": "official_source",
}


class CompareAndAskTest(unittest.TestCase):
    def call(self, response, fact=FACT, candidates=(CANDIDATE,)):
        judge = FakeJudge(response)
        result = compare_and_ask(fact, candidates, judge)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(judge.calls[0][1], 1000)
        return result, judge.calls[0][0]

    def test_new_has_no_candidate_id_or_question(self):
        judge = FakeJudge('{"decision":"UNCLEAR","knowledge_id":null,"question":"是否还有其他模型？"}')
        result = compare_and_ask(FACT, [], judge)
        self.assertEqual(result["decision"], "NEW")
        self.assertIsNone(result["knowledge_id"])
        self.assertIsNone(result["question"])
        self.assertEqual(judge.calls, [])

    def test_vague_version_cannot_be_new_even_without_candidates(self):
        vague = {"title": "版本差异", "content": "F-X 新版和以前不一样", "entity_name": "F-X"}
        judge = FakeJudge('{"decision":"NEW","knowledge_id":null,"question":null,"reason":"没有候选"}')
        result = compare_and_ask(vague, [], judge)
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertIn("revision", result["question"])
        self.assertEqual(judge.calls, [])

        unclear_judge = FakeJudge('{"decision":"UNCLEAR","knowledge_id":null,"question":"请说明具体产品信息？","reason":"不清楚"}')
        unclear = compare_and_ask(vague, [], unclear_judge)
        self.assertEqual(unclear["decision"], "UNCLEAR")
        self.assertIn("硬件 revision", unclear["question"])
        self.assertEqual(unclear_judge.calls, [])

    def test_uncertain_wording_cannot_be_confirmed(self):
        uncertain = {"title": "功能", "content": "F-X 可能支持声音检测", "entity_name": "F-X"}
        result, _ = self.call(
            '{"decision":"CONFIRM","knowledge_id":17,"question":null,"reason":"相似"}',
            fact=uncertain,
        )
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertIn("准确结论", result["question"])

    def test_confirm_must_reference_a_retrieved_candidate(self):
        result, _ = self.call('{"decision":"CONFIRM","knowledge_id":17,"question":null,"reason":"同一事实"}')
        self.assertEqual(result["decision"], "CONFIRM")
        self.assertEqual(result["knowledge_id"], 17)

    def test_enrich_requires_one_real_product_question(self):
        result, _ = self.call('{"decision":"ENRICH","knowledge_id":17,"question":"这个补充信息适用于哪个硬件版本？","reason":"补充范围"}')
        self.assertEqual(result["decision"], "ENRICH")
        self.assertEqual(result["knowledge_id"], 17)
        self.assertEqual(result["question"], "这个补充信息适用于哪个硬件版本？")

    def test_precise_enrich_can_continue_to_atomic_confirmation(self):
        result, _ = self.call('{"decision":"ENRICH","knowledge_id":17,"question":null,"reason":"补充信息范围已明确"}')
        self.assertEqual(result["decision"], "ENRICH")
        self.assertEqual(result["knowledge_id"], 17)
        self.assertIsNone(result["question"])

    def test_conflict_keeps_both_sides_and_asks_expert(self):
        result, _ = self.call('{"decision":"CONFLICT","knowledge_id":17,"question":"关于这个型号，哪一条产品结论准确？","reason":"两条信息矛盾"}')
        self.assertEqual(result["decision"], "CONFLICT")
        self.assertEqual(result["knowledge_id"], 17)
        self.assertIn("哪一条", result["question"])

    def test_unclear_requires_question(self):
        result, _ = self.call('{"decision":"UNCLEAR","knowledge_id":null,"question":"这里的新版具体指哪个硬件版本？","reason":"新版含义不明"}')
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertIsNone(result["knowledge_id"])
        self.assertTrue(result["question"].endswith("？"))

    def test_invalid_candidate_id_fails_closed(self):
        result, _ = self.call('{"decision":"CONFIRM","knowledge_id":999,"question":null,"reason":"同一事实"}')
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertIsNone(result["knowledge_id"])
        self.assertIn("F-NR-208E/2", result["question"])

    def test_missing_required_question_fails_closed(self):
        result, _ = self.call('{"decision":"CONFLICT","knowledge_id":17,"question":null,"reason":"矛盾"}')
        self.assertEqual(result["decision"], "CONFLICT")
        self.assertEqual(result["knowledge_id"], 17)
        self.assertTrue(result["question"].endswith("？"))

    def test_unsafe_or_multiple_questions_fail_closed(self):
        unsafe = '{"decision":"UNCLEAR","knowledge_id":null,"question":"请填写 database 的字段？","reason":"不清楚"}'
        result, _ = self.call(unsafe)
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertNotIn("database", result["question"].casefold())

        multiple = '{"decision":"ENRICH","knowledge_id":17,"question":"适用于哪个型号？哪个版本？","reason":"补充"}'
        result, _ = self.call(multiple)
        self.assertEqual(result["decision"], "ENRICH")
        self.assertEqual(result["knowledge_id"], 17)

        placeholder = '{"decision":"UNCLEAR","knowledge_id":null,"question":"为什么？","reason":"不清楚"}'
        result, _ = self.call(placeholder)
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertNotEqual(result["question"], "为什么？")

    def test_parse_failure_fails_closed(self):
        result, _ = self.call("not json")
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertIsNone(result["knowledge_id"])
        self.assertTrue(result["question"])

    def test_unknown_keys_cannot_smuggle_trust_or_conflict_resolution(self):
        response = '{"decision":"CONFLICT","knowledge_id":17,"question":"哪一条产品结论准确？","trust":"official_source","selected_knowledge_id":17}'
        result, _ = self.call(response)
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertNotIn("trust", result)

    def test_prompt_contains_fact_safety_rules_and_candidates_without_trust_output(self):
        _, messages = self.call('{"decision":"NEW","knowledge_id":null,"question":null,"reason":"new"}')
        prompt = messages[0]["content"]
        user_payload = messages[1]["content"]
        self.assertIn("没有写某功能，不能推断为“不支持”", prompt)
        self.assertIn("不能推广到整个系列", prompt)
        self.assertIn("不得选择、覆盖或自动裁决", prompt)
        self.assertIn('"id":17', user_payload)
        self.assertNotIn('"trust"', user_payload)

    def test_invalid_input_is_rejected_before_judge(self):
        judge = FakeJudge('{"decision":"NEW","knowledge_id":null,"question":null}')
        with self.assertRaises(ValueError):
            compare_and_ask([], [], judge)
        with self.assertRaises(ValueError):
            compare_and_ask([FACT, FACT], [], judge)
        with self.assertRaises(ValueError):
            compare_and_ask(FACT, [{"id": 0, "content": "bad"}], judge)
        self.assertEqual(judge.calls, [])

    def test_judge_failure_fails_closed(self):
        class BrokenJudge:
            def judge(self, messages, max_tokens=1000):
                raise RuntimeError("OpenRouter unavailable")

        result = compare_and_ask(FACT, [CANDIDATE], BrokenJudge())
        self.assertEqual(result["decision"], "UNCLEAR")
        self.assertIsNone(result["knowledge_id"])


if __name__ == "__main__":
    unittest.main()
