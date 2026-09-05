from __future__ import annotations

import unittest

from v2.retrieval import (
    _lexical_score,
    _same_model,
    retrieve_learning_knowledge,
)


class _Cursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.connection.query = " ".join(str(query).split())

    def fetchall(self):
        return [self.connection._canonical_row(row) for row in self.connection.rows]


class _JoinConnection:
    def __init__(self, rows, entities):
        self.rows = rows
        self.entities = entities
        self.query = ""

    def cursor(self):
        return _Cursor(self)

    def _canonical_row(self, row):
        result = dict(row)
        current = self.entities.get(row.get("entity_id"))
        result["entity_name"] = current or row.get("entity_name", "")
        result["legacy_entity_name"] = row.get("entity_name", "")
        return result


def _row(identifier, entity_name, *, entity_id=None, title="feature", content="supports it"):
    return {
        "id": identifier,
        "title": title,
        "content": content,
        "entity_name": entity_name,
        "entity_id": entity_id,
        "trust": "user_confirmed",
        "active": True,
        "embedding": None,
        "embedding_model": None,
        "created_at": None,
        "updated_at": None,
    }


class RetrievalEntityConsistencyTest(unittest.TestCase):
    def test_current_entity_name_wins_and_legacy_name_falls_back(self):
        current = _row(1, "LEGACY-MODEL-2024", entity_id=7)
        legacy = _row(2, "LEGACY-ONLY-2024")
        conn = _JoinConnection([current, legacy], {7: "CURRENT-MODEL-2026"})

        result = retrieve_learning_knowledge(conn, "CURRENT-MODEL-2026 feature", top_k=2)

        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["entity_name"], "CURRENT-MODEL-2026")
        self.assertIn("LEFT JOIN v2_entities", conn.query)
        self.assertIn("COALESCE(e.name, k.entity_name)", conn.query)

    def test_same_model_and_lexical_use_the_same_canonical_name(self):
        row = {
            "title": "feature",
            "content": "supports it",
            "entity_name": "CURRENT-MODEL-2026",
            "legacy_entity_name": "LEGACY-MODEL-2024",
        }
        self.assertTrue(_same_model("CURRENT-MODEL-2026 feature", row))
        self.assertFalse(_same_model("LEGACY-MODEL-2024 feature", row))
        self.assertGreater(_lexical_score("CURRENT-MODEL-2026 feature", row), 0)

    def test_same_model_only_uses_current_name_and_preserves_legacy_fallback(self):
        rows = [
            _row(1, "LEGACY-MODEL-2024", entity_id=7),
            _row(2, "LEGACY-ONLY-2024"),
        ]
        conn = _JoinConnection(rows, {7: "CURRENT-MODEL-2026"})

        current_name = retrieve_learning_knowledge(
            conn, "CURRENT-MODEL-2026 feature", same_model_only=True, top_k=3,
        )
        legacy_name = retrieve_learning_knowledge(
            conn, "LEGACY-ONLY-2024 feature", same_model_only=True, top_k=3,
        )
        self.assertEqual([item["id"] for item in current_name], [1])
        self.assertEqual([item["id"] for item in legacy_name], [2])


if __name__ == "__main__":
    unittest.main()
