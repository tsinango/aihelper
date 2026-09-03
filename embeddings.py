"""OpenRouter embedding client used by both the API and offline jobs."""

from __future__ import annotations

import os
import time
import math
from pathlib import Path

import httpx


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
OPENROUTER_EMBEDDING_DIMENSIONS = 2048
OPENROUTER_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 524, 529})


def read_openrouter_token(path: str | Path = "openrouter") -> str:
    configured = os.getenv("OPENROUTER_API_KEY", "").strip()
    if configured:
        return configured
    try:
        return "".join(Path(path).read_text(encoding="utf-8").split())
    except OSError:
        return ""


class OpenRouterEmbeddingError(RuntimeError):
    """Raised when OpenRouter cannot return a valid embedding batch."""

    def __init__(self, message: str, status_code: int | None = None):
        if status_code is not None:
            message = f"{message} (status {status_code})"
        super().__init__(message)
        self.status_code = status_code


class OpenRouterEmbeddingClient:
    """Small OpenAI-compatible client for the OpenRouter embeddings endpoint."""

    model = OPENROUTER_EMBEDDING_MODEL
    dimensions = OPENROUTER_EMBEDDING_DIMENSIONS

    def __init__(self, api_key: str = "", *, token_file: str | Path = "openrouter",
                 timeout: float = 60.0, max_retries: int = 2, client=None):
        self.api_key = api_key.strip() or read_openrouter_token(token_file)
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY or the openrouter token file is not configured")
        if timeout <= 0:
            raise ValueError("OpenRouter embedding timeout must be positive")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.client = client

    def encode(self, texts: list[str] | tuple[str, ...] | str, **kwargs) -> list[list[float]]:
        inputs = [texts] if isinstance(texts, str) else [str(text) for text in texts]
        if not inputs:
            return []
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(inputs)
                status_code = getattr(response, "status_code", None)
                if status_code in OPENROUTER_TRANSIENT_STATUS_CODES:
                    raise OpenRouterEmbeddingError("OpenRouter embedding request is temporarily unavailable", status_code)
                if status_code is not None and status_code >= 400:
                    detail = "request failed"
                    try:
                        detail = str((response.json().get("error") or {}).get("message") or detail)
                    except Exception:
                        pass
                    raise OpenRouterEmbeddingError(f"OpenRouter embedding failed: {detail}", status_code)
                payload = response.json()
                rows = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(rows, list) or len(rows) != len(inputs):
                    raise OpenRouterEmbeddingError("OpenRouter embedding response length does not match request")
                ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
                vectors = []
                for row in ordered:
                    vector = row.get("embedding")
                    if not isinstance(vector, list) or len(vector) != self.dimensions:
                        raise OpenRouterEmbeddingError("OpenRouter returned an unexpected embedding dimension")
                    normalized = [float(value) for value in vector]
                    if kwargs.get("normalize_embeddings"):
                        norm = math.sqrt(sum(value * value for value in normalized))
                        if norm:
                            normalized = [value / norm for value in normalized]
                    vectors.append(normalized)
                return vectors
            except Exception as error:
                last_error = error
                retryable = isinstance(error, (httpx.HTTPError, OpenRouterEmbeddingError)) and (
                    not isinstance(error, OpenRouterEmbeddingError)
                    or error.status_code in OPENROUTER_TRANSIENT_STATUS_CODES
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        raise OpenRouterEmbeddingError("OpenRouter embedding request failed") from last_error

    def _post(self, inputs: list[str]):
        payload = {"model": self.model, "input": inputs}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "AI Sales Engineer",
        }
        if self.client is not None:
            return self.client.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=self.timeout)
        return httpx.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=self.timeout)
