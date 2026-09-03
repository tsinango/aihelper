import unittest
from types import SimpleNamespace
from unittest.mock import patch

from embeddings import OPENROUTER_EMBEDDING_DIMENSIONS, OpenRouterEmbeddingClient
from rerank import OPENROUTER_RERANK_MODEL, OpenRouterReranker


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class OpenRouterClientTest(unittest.TestCase):
    def test_embedding_batch_is_ordered_and_validated(self):
        vectors = [[float(index)] * OPENROUTER_EMBEDDING_DIMENSIONS for index in (2, 1)]
        calls = []
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeResponse({
                "data": [{"index": 1, "embedding": vectors[0]}, {"index": 0, "embedding": vectors[1]}]
            })
        fake = SimpleNamespace(post=post)
        client = OpenRouterEmbeddingClient("test-key", client=fake, max_retries=0)
        result = client.encode(["first", "second"])

        self.assertEqual(result[0][0], 1.0)
        self.assertEqual(result[1][0], 2.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["json"]["model"], "nvidia/nemotron-3-embed-1b:free")

    def test_reranker_normalizes_results(self):
        fake = SimpleNamespace(post=lambda *args, **kwargs: FakeResponse({
            "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        }))
        client = OpenRouterReranker("test-key", client=fake, max_retries=0)
        result = client.rerank("query", ["one", "two"], top_n=2)

        self.assertEqual(result[0]["index"], 1)
        self.assertEqual(client.model, OPENROUTER_RERANK_MODEL)

    @patch("embeddings.read_openrouter_token", return_value="file-key")
    def test_embedding_uses_token_file_when_environment_is_empty(self, read_token):
        client = OpenRouterEmbeddingClient(token_file="/tmp/openrouter-test-token", max_retries=0)
        self.assertEqual(client.api_key, "file-key")
        read_token.assert_called_once()


if __name__ == "__main__":
    unittest.main()
