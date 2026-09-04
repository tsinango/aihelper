"""OpenRouter reranking client for the online retrieval second pass."""

from __future__ import annotations

import time

import httpx

from embeddings import read_openrouter_token


OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
OPENROUTER_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 524, 529})


class OpenRouterRerankError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        if status_code is not None:
            message = f"{message} (status {status_code})"
        super().__init__(message)
        self.status_code = status_code


class OpenRouterReranker:
    model = OPENROUTER_RERANK_MODEL

    def __init__(self, api_key: str = "", *, token_file="openrouter", timeout: float = 60.0,
                 max_retries: int = 1, client=None):
        self.api_key = api_key.strip() or read_openrouter_token(token_file)
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if timeout <= 0:
            raise ValueError("OpenRouter rerank timeout must be positive")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.client = client

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": str(query),
            "documents": [str(document) for document in documents],
        }
        if top_n is not None:
            payload["top_n"] = max(1, min(int(top_n), len(documents)))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(payload)
                status_code = getattr(response, "status_code", None)
                if status_code in TRANSIENT_STATUS_CODES:
                    raise OpenRouterRerankError("OpenRouter rerank request is temporarily unavailable", status_code)
                if status_code is not None and status_code >= 400:
                    detail = "request failed"
                    try:
                        detail = str((response.json().get("error") or {}).get("message") or detail)
                    except Exception:
                        pass
                    raise OpenRouterRerankError(f"OpenRouter rerank failed: {detail}", status_code)
                result = response.json().get("results")
                if not isinstance(result, list):
                    raise OpenRouterRerankError("OpenRouter rerank response has no results")
                normalized = []
                for item in result:
                    try:
                        index = int(item["index"])
                        score = float(item["relevance_score"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise OpenRouterRerankError("OpenRouter rerank response contains an invalid result") from error
                    if not 0 <= index < len(documents):
                        raise OpenRouterRerankError("OpenRouter rerank returned an invalid document index")
                    normalized.append({"index": index, "relevance_score": score})
                return normalized
            except Exception as error:
                last_error = error
                retryable = isinstance(error, (httpx.HTTPError, OpenRouterRerankError)) and (
                    not isinstance(error, OpenRouterRerankError)
                    or error.status_code in TRANSIENT_STATUS_CODES
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        raise OpenRouterRerankError("OpenRouter rerank request failed") from last_error

    def _post(self, payload: dict):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "aihelper",
        }
        if self.client is not None:
            return self.client.post(OPENROUTER_RERANK_URL, headers=headers, json=payload, timeout=self.timeout)
        return httpx.post(OPENROUTER_RERANK_URL, headers=headers, json=payload, timeout=self.timeout)
