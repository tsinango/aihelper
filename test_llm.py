import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm import (
    LLMService,
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_DEFAULT_MODEL,
    OPENROUTER_PROVIDER,
    V2_LEARNING_MODEL,
    V2_LEARNING_MODEL_DEFAULT,
    OpenRouterLLM,
    OpenRouterRequestError,
    parse_json_response,
)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class OpenRouterServiceTest(unittest.TestCase):
    def test_internal_interface_and_openrouter_defaults(self):
        client = FakeClient(['{"answer":"ok"}'] * 4)
        service = OpenRouterLLM("test-key", client=client, max_retries=0)

        self.assertIsInstance(service, LLMService)
        self.assertEqual(service.provider, OPENROUTER_PROVIDER)
        self.assertEqual(service.model, OPENROUTER_DEFAULT_MODEL)
        self.assertEqual(service.complete([]), '{"answer":"ok"}')
        self.assertEqual(service.extract([]), '{"answer":"ok"}')
        self.assertEqual(service.judge([]), '{"answer":"ok"}')
        self.assertEqual(
            {request["model"] for request in client.chat.completions.requests},
            {OPENROUTER_DEFAULT_MODEL},
        )
        self.assertNotIn("response_format", client.chat.completions.requests[0])
        self.assertEqual(
            client.chat.completions.requests[0]["extra_body"],
            {"reasoning": {"effort": "none"}},
        )

    def test_learning_extraction_uses_free_structured_output_model(self):
        client = FakeClient(['{"claims":[],"knowledge_units":[],"coverage":{}}'])
        service = OpenRouterLLM("test-key", client=client, max_retries=0)
        schema = {"name": "learning", "strict": True, "schema": {"type": "object"}}

        service.extract_structured([], schema)

        request = client.chat.completions.requests[0]
        self.assertEqual(V2_LEARNING_MODEL, V2_LEARNING_MODEL_DEFAULT)
        self.assertEqual(request["model"], V2_LEARNING_MODEL)
        self.assertEqual(request["response_format"], {
            "type": "json_schema", "json_schema": schema,
        })

    @patch("llm.OpenAI")
    def test_default_client_uses_openrouter_endpoint_and_disables_sdk_retries(self, openai):
        OpenRouterLLM("test-key")

        openai.assert_called_once_with(
            base_url=OPENROUTER_DEFAULT_BASE_URL,
            api_key="test-key",
            timeout=30.0,
            max_retries=0,
        )

    def test_timeout_is_retried_only_within_service_limit(self):
        client = FakeClient([TimeoutError(), TimeoutError(), TimeoutError()])
        service = OpenRouterLLM("test-key", client=client, max_retries=2)

        with patch("llm.time.sleep") as sleep:
            with self.assertRaises(OpenRouterRequestError):
                service.complete([])

        self.assertEqual(len(client.chat.completions.requests), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_empty_assistant_response_is_retried(self):
        client = FakeClient([None, '{"answer":"ok"}'])
        service = OpenRouterLLM("test-key", client=client, max_retries=1)

        with patch("llm.time.sleep"):
            self.assertEqual(service.complete([]), '{"answer":"ok"}')

        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_exhausted_http_failure_keeps_status_without_fallback(self):
        class TransientError(Exception):
            status_code = 503

        error = TransientError("temporarily unavailable")
        client = FakeClient([error])
        service = OpenRouterLLM("test-key", client=client, max_retries=0)

        with self.assertRaises(OpenRouterRequestError) as raised:
            service.complete([])

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_non_transient_error_does_not_switch_or_retry(self):
        error = RuntimeError("invalid request")
        client = FakeClient([error])
        service = OpenRouterLLM("test-key", client=client, max_retries=2)

        with self.assertRaisesRegex(RuntimeError, "invalid request"):
            service.complete([])

        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_alternate_model_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "only nvidia/nemotron-3-ultra-550b-a55b:free is supported"):
            OpenRouterLLM("test-key", model="another-model")

    def test_json_parser_recovers_from_model_commentary(self):
        self.assertEqual(parse_json_response('I will comply. {"ok": true}'), {"ok": True})

    def test_complete_json_requests_openrouter_json_object(self):
        client = FakeClient(['{"action":"NO_CHANGE"}'])
        service = OpenRouterLLM("test-key", client=client, max_retries=0)

        self.assertEqual(service.complete_json([]), '{"action":"NO_CHANGE"}')
        self.assertEqual(
            client.chat.completions.requests[0]["response_format"],
            {"type": "json_object"},
        )


if __name__ == "__main__":
    unittest.main()
