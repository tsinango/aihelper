from __future__ import annotations

import unittest
from pathlib import Path

from embeddings import OPENROUTER_EMBEDDING_DIMENSIONS, OPENROUTER_EMBEDDING_MODEL
from v2.retrieval import retrieve_learning_knowledge, store_knowledge_embedding


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.query = " ".join(str(query).split())

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, rows):
        self.rows = rows
        self.cursors = []

    def cursor(self):
        cursor = Cursor(self.rows)
        self.cursors.append(cursor)
        return cursor


def row(identifier, content, entity, *, trust="provisional", embedding=None, model=None):
    return {"id": identifier, "title": content, "content": content, "entity_name": entity,
            "trust": trust, "active": True, "embedding": embedding, "embedding_model": model,
            "created_at": None, "updated_at": None}


class Embedder:
    def __init__(self, vector=None, error=None):
        self.vector, self.error, self.calls = vector, error, []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if self.error:
            raise self.error
        return [self.vector]


class V2RetrievalTest(unittest.TestCase):
    def test_same_model_first_and_lexical_hits_are_returned(self):
        conn = Connection([row(1, "F-X 支持声音阈值检测", "F-X"), row(2, "F-Y 支持声音阈值检测", "F-Y"), row(3, "F-X 机箱高度是 1U", "F-X")])
        result = retrieve_learning_knowledge(conn, "F-X 是否支持声音阈值检测", top_k=3)
        self.assertEqual([item["id"] for item in result], [1, 3, 2])
        self.assertEqual(result[0]["retrieval_sources"], ["lexical"])
        self.assertIn("WHERE active=TRUE", conn.cursors[0].query)

    def test_embedding_top_k_merges_and_ignores_other_model(self):
        zeros = [0.0] * OPENROUTER_EMBEDDING_DIMENSIONS
        query = [1.0] + zeros[1:]
        close = [1.0] + zeros[1:]
        far = [0.0, 1.0] + zeros[2:]
        conn = Connection([row(1, "unrelated wording", "F-X", embedding=close, model=OPENROUTER_EMBEDDING_MODEL), row(2, "another wording", "F-Y", embedding=far, model=OPENROUTER_EMBEDDING_MODEL), row(3, "old model", "F-Z", embedding=close, model="old-model")])
        result = retrieve_learning_knowledge(conn, "F-X", embedder=Embedder(query), top_k=2)
        self.assertEqual([item["id"] for item in result], [1, 2])
        self.assertEqual(result[0]["retrieval_sources"], ["lexical", "embedding"])
        self.assertAlmostEqual(result[0]["embedding_score"], 1.0)

    def test_embedding_failure_keeps_lexical_fail_safe(self):
        conn = Connection([row(1, "F-X 支持声音阈值检测", "F-X")])
        result = retrieve_learning_knowledge(conn, "F-X 支持声音阈值检测", embedder=Embedder(error=RuntimeError("offline")))
        self.assertEqual([item["id"] for item in result], [1])
        self.assertEqual(result[0]["retrieval_sources"], ["lexical"])

    def test_explicit_model_does_not_retrieve_nearby_product_line(self):
        conn = Connection([
            row(1, "TandemVu keeps the wide scene visible.", "TandemVu", trust="user_confirmed"),
            row(2, "TEST-TANDEM-X uses a dual-lens design.", "TEST-TANDEM-X", trust="user_confirmed"),
        ])
        result = retrieve_learning_knowledge(
            conn,
            "TEST-TANDEM-X uses a dual-lens design like TandemVu cameras",
            top_k=3,
            same_model_only=True,
        )
        self.assertEqual([item["id"] for item in result], [2])

    def test_stores_only_a_valid_openrouter_embedding(self):
        conn = Connection([])
        vector = [1.0] + [0.0] * (OPENROUTER_EMBEDDING_DIMENSIONS - 1)
        self.assertTrue(store_knowledge_embedding(conn, 7, "F-X fact", embedder=Embedder(vector)))
        self.assertIn("UPDATE v2_knowledge", conn.cursors[0].query)

        failed = Connection([])
        self.assertFalse(store_knowledge_embedding(failed, 7, "F-X fact", embedder=Embedder(error=RuntimeError("offline"))))
        self.assertEqual(failed.cursors, [])

    def test_migration_has_minimal_compare_columns_and_no_vector_index(self):
        sql = Path("migrations/014_v2_learning_compare.sql").read_text(encoding="utf-8")
        for term in ("embedding vector(2048)", "embedding_model", "comparison_result", "clarification_question", "related_knowledge_ids", "pending_clarification", "NEW", "CONFIRM", "ENRICH", "CONFLICT", "UNCLEAR"):
            self.assertIn(term, sql)
        self.assertNotIn("USING hnsw", sql)
        self.assertNotIn("USING ivfflat", sql)


if __name__ == "__main__":
    unittest.main()
