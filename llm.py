from __future__ import annotations

import json
import os
import time
from typing import Protocol, runtime_checkable

from openai import APIConnectionError, APITimeoutError, OpenAI


OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
V2_LEARNING_MODEL_DEFAULT = "openai/gpt-oss-20b:free"
V2_LEARNING_MODEL = os.getenv("V2_LEARNING_MODEL", V2_LEARNING_MODEL_DEFAULT).strip() or V2_LEARNING_MODEL_DEFAULT
OPENROUTER_DEFAULT_TIMEOUT_SECONDS = 30.0
OPENROUTER_MAX_RETRIES = 2
TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


def parse_json_response(content: str):
    """Parse strict JSON or recover the final JSON value from model chatter."""
    text = str(content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        values = []
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            values.append(value)
        if values:
            return values[-1]
        raise


@runtime_checkable
class LLMService(Protocol):
    """Small internal interface shared by chat, extraction, and judging."""

    def complete(self, messages: list[dict], max_tokens: int = 600) -> str: ...

    def complete_json(self, messages: list[dict], max_tokens: int = 600) -> str: ...

    def extract(self, messages: list[dict], max_tokens: int = 600) -> str: ...

    def extract_structured(
        self, messages: list[dict], schema: dict, max_tokens: int = 600
    ) -> str: ...

    def judge(self, messages: list[dict], max_tokens: int = 600) -> str: ...


class OpenRouterRequestError(RuntimeError):
    """A request failed after the bounded OpenRouter retry policy was exhausted."""

    def __init__(self, message: str, status_code: int | None = None):
        if status_code is not None:
            message = f"{message} (status {status_code})"
        super().__init__(message)
        self.status_code = status_code


class OpenRouterLLM:
    """The sole LLM implementation used by this project."""

    provider = OPENROUTER_PROVIDER

    def __init__(
        self,
        api_key: str = "",
        *,
        timeout: float = OPENROUTER_DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = OPENROUTER_MAX_RETRIES,
        model: str = OPENROUTER_DEFAULT_MODEL,
        client=None,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if not model or model.casefold() != OPENROUTER_DEFAULT_MODEL.casefold():
            raise RuntimeError(f"only {OPENROUTER_DEFAULT_MODEL} is supported")
        if timeout <= 0:
            raise ValueError("OpenRouter timeout must be positive")
        self.model = OPENROUTER_DEFAULT_MODEL
        self.max_retries = max(0, int(max_retries))
        self.client = client if client is not None else OpenAI(
            base_url=OPENROUTER_DEFAULT_BASE_URL,
            api_key=api_key,
            timeout=timeout,
            # Retry policy is owned by this service so the total is bounded
            # and never silently multiplied by the SDK's own retries.
            max_retries=0,
        )

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        if status in TRANSIENT_STATUS_CODES:
            return True
        if isinstance(error, (TimeoutError, APITimeoutError, APIConnectionError)):
            return True
        # Keep compatibility with timeout exceptions from the HTTP transport
        # without making the transport an additional production dependency.
        return error.__class__.__name__ in {"TimeoutException", "ConnectTimeout", "ReadTimeout"}

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 600,
        *,
        response_format: dict | None = None,
        model: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = {
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    # OpenRouter's Nemotron endpoint may return a response
                    # with no assistant content when reasoning is left at its
                    # provider default.  Use the native OpenRouter field via
                    # extra_body; the OpenAI-compatible reasoning_effort
                    # parameter is not handled consistently by this endpoint.
                    "extra_body": {"reasoning": {"effort": "none"}},
                }
                if response_format is not None:
                    request["response_format"] = response_format
                response = self.client.chat.completions.create(**request)
                choices = getattr(response, "choices", None)
                if not choices or not getattr(choices[0], "message", None):
                    # OpenRouter can surface a provider-side failure as HTTP
                    # 200 with no choices. Treat it like a transient 503 so
                    # the bounded service retry policy handles it safely.
                    raise OpenRouterRequestError(
                        "OpenRouter returned no assistant message",
                        status_code=503,
                    )
                content = getattr(choices[0].message, "content", None)
                if not content:
                    raise OpenRouterRequestError(
                        "OpenRouter returned empty assistant content",
                        status_code=503,
                    )
                return content
            except Exception as error:
                last_error = error
                if attempt >= self.max_retries or not self._is_retryable(error):
                    if attempt >= self.max_retries and self._is_retryable(error):
                        raise OpenRouterRequestError(
                            f"OpenRouter request failed after {attempt + 1} attempts",
                            status_code=getattr(error, "status_code", None),
                        ) from error
                    raise
                time.sleep(2 ** attempt)
        raise OpenRouterRequestError("OpenRouter request failed") from last_error

    def complete_json(self, messages: list[dict], max_tokens: int = 600) -> str:
        """Request a JSON object while retaining the shared retry policy."""

        return self.complete(
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    def extract(self, messages: list[dict], max_tokens: int = 600) -> str:
        return self.complete(messages, max_tokens=max_tokens)

    def extract_structured(self, messages: list[dict], schema: dict, max_tokens: int = 600) -> str:
        if not V2_LEARNING_MODEL.endswith(":free"):
            raise RuntimeError("V2_LEARNING_MODEL must be a free OpenRouter model")
        return self.complete(
            messages,
            max_tokens=max_tokens,
            model=V2_LEARNING_MODEL,
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
        )

    def judge(self, messages: list[dict], max_tokens: int = 600) -> str:
        return self.complete(messages, max_tokens=max_tokens)
