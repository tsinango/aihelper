from __future__ import annotations

import unittest
from pathlib import Path

from v2.service import V2NotFound, prune_empty_entity_subtree


class _Transaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.transaction_count += 1
        return self

    def __exit__(self, *_):
        return False


class _Cursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        sql = " ".join(str(query).split())
        self.connection.executed.append((sql, params))
        self.rows = self.connection.execute(sql, params)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class MemoryPruningDB:
    """Semantic fake for service-level pruning tests; no PostgreSQL required."""

    def __init__(self):
        self.entities = []
        self.relations = []
        self.knowledge = []
        self.executed = []
        self.transaction_count = 0

    def cursor(self):
        return _Cursor(self)

    def transaction(self):
        return _Transaction(self)

    def execute(self, sql, params):
        if sql.startswith("SELECT id, name, entity_type, active, created_at, updated_at FROM v2_entities WHERE id=%s FOR UPDATE"):
            entity_id = int(params[0])
            return [dict(row) for row in self.entities if row["id"] == entity_id]

        if sql.startswith("WITH RECURSIVE subtree"):
            root_id = int(params[0])
            entity_ids = []
            queue = [root_id]
            while queue:
                current = queue.pop(0)
                if current in entity_ids:
                    continue
                entity = next((row for row in self.entities if row["id"] == current and row["active"]), None)
                if entity is None:
                    continue
                entity_ids.append(current)
                queue.extend(
                    relation["child_entity_id"]
                    for relation in self.relations
                    if relation["active"]
                    and relation["relation_type"] == "belongs_to"
                    and relation["parent_entity_id"] == current
                )
            return [dict(next(row for row in self.entities if row["id"] == entity_id)) for entity_id in sorted(entity_ids)]

        if sql.startswith("SELECT id, name, entity_type, active, created_at, updated_at FROM v2_entities WHERE id=ANY"):
            ids = {int(value) for value in params[0]}
            return [dict(row) for row in sorted(self.entities, key=lambda row: row["id"])
                    if row["id"] in ids and row["active"]]

        if sql.startswith("SELECT id, parent_entity_id, child_entity_id, relation_type,") and "FOR UPDATE" in sql:
            ids = {int(value) for value in params[0]} | {int(value) for value in params[1]}
            return [dict(row) for row in sorted(self.relations, key=lambda row: row["id"])
                    if row["active"] and (row["parent_entity_id"] in ids or row["child_entity_id"] in ids)]

        if sql.startswith("SELECT id, entity_id, active, trust FROM v2_knowledge"):
            ids = {int(value) for value in params[0]}
            return [dict(row) for row in sorted(self.knowledge, key=lambda row: row["id"])
                    if row["entity_id"] in ids]

        if sql.startswith("UPDATE v2_entity_relations"):
            ids = {int(value) for value in params[0]}
            changed = []
            for row in self.relations:
                if row["id"] in ids and row["active"]:
                    row["active"] = False
                    row["deactivated_at"] = "now"
                    changed.append({"id": row["id"]})
            return changed

        if sql.startswith("UPDATE v2_entities"):
            ids = {int(value) for value in params[0]}
            changed = []
            for row in self.entities:
                if row["id"] in ids and row["active"]:
                    row["active"] = False
                    row["deactivated_at"] = "now"
                    changed.append({"id": row["id"]})
            return changed

        raise AssertionError(f"unhandled fake SQL: {sql}")


def entity(entity_id, name, *, active=True):
    return {
        "id": entity_id,
        "name": name,
        "entity_type": "concept",
        "active": active,
        "created_at": None,
        "updated_at": None,
        "deactivated_at": None,
    }


def relation(relation_id, parent, child, *, active=True):
    return {
        "id": relation_id,
        "parent_entity_id": parent,
        "child_entity_id": child,
        "relation_type": "belongs_to",
        "source_id": None,
        "provenance": "confirmed",
        "provenance_kind": "user_confirmed",
        "active": active,
        "created_at": None,
        "updated_at": None,
        "deactivated_at": None,
    }


class EntityPruningTest(unittest.TestCase):
    def test_empty_leaf_is_soft_pruned_inside_transaction(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Empty leaf")]

        result = prune_empty_entity_subtree(db, 1)

        self.assertTrue(result["pruned"])
        self.assertEqual(result["entity_ids"], [1])
        self.assertFalse(db.entities[0]["active"])
        self.assertEqual(db.entities[0]["deactivated_at"], "now")
        self.assertEqual(db.transaction_count, 1)

    def test_empty_subtree_deactivates_entities_and_relations(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Parent"), entity(2, "Child"), entity(3, "Grandchild")]
        db.relations = [relation(1, 1, 2), relation(2, 2, 3)]

        result = prune_empty_entity_subtree(db, 1)

        self.assertTrue(result["pruned"])
        self.assertEqual(result["entity_ids"], [1, 2, 3])
        self.assertEqual(result["relation_ids"], [1, 2])
        self.assertTrue(all(not row["active"] for row in db.entities))
        self.assertTrue(all(not row["active"] for row in db.relations))
        self.assertTrue(all(row["deactivated_at"] == "now" for row in db.relations))

    def test_active_knowledge_reference_blocks_whole_subtree(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Parent"), entity(2, "Child")]
        db.relations = [relation(1, 1, 2)]
        db.knowledge = [{"id": 9, "entity_id": 2, "active": True, "trust": "user_confirmed"}]

        with self.assertRaisesRegex(ValueError, "Knowledge references"):
            prune_empty_entity_subtree(db, 1)
        self.assertTrue(all(row["active"] for row in db.entities))
        self.assertTrue(db.relations[0]["active"])

    def test_inactive_knowledge_reference_also_blocks_pruning(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Historical")]
        db.knowledge = [{"id": 10, "entity_id": 1, "active": False, "trust": "user_confirmed"}]

        with self.assertRaisesRegex(ValueError, "active or deleted Knowledge"):
            prune_empty_entity_subtree(db, 1)
        self.assertTrue(db.entities[0]["active"])

    def test_boundary_parent_relation_is_soft_deactivated(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Outside"), entity(2, "Pruned root")]
        db.relations = [relation(1, 1, 2)]

        result = prune_empty_entity_subtree(db, 2)

        self.assertTrue(result["pruned"])
        self.assertEqual(result["relation_ids"], [1])
        self.assertFalse(db.relations[0]["active"])
        self.assertTrue(db.entities[0]["active"])
        self.assertFalse(db.entities[1]["active"])

    def test_malformed_cycle_is_detected_without_infinite_traversal(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "A"), entity(2, "B")]
        db.relations = [relation(1, 1, 2), relation(2, 2, 1)]

        with self.assertRaisesRegex(ValueError, "active cycle"):
            prune_empty_entity_subtree(db, 1)
        self.assertTrue(all(row["active"] for row in db.entities))
        self.assertTrue(all(row["active"] for row in db.relations))

    def test_inactive_root_is_noop_and_unknown_root_is_not_found(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Already inactive", active=False)]
        with self.assertRaises(ValueError):
            prune_empty_entity_subtree(db, 1)

        with self.assertRaises(V2NotFound):
            prune_empty_entity_subtree(db, 404)

    def test_locks_current_entities_relations_and_all_knowledge_states(self):
        db = MemoryPruningDB()
        db.entities = [entity(1, "Locked")]
        db.knowledge = [{"id": 11, "entity_id": 1, "active": False, "trust": "provisional"}]

        with self.assertRaisesRegex(ValueError, "active or deleted Knowledge"):
            prune_empty_entity_subtree(db, 1)

        statements = [query for query, _ in db.executed]
        self.assertTrue(any("FROM v2_entities WHERE id=%s FOR UPDATE" in query for query in statements))
        self.assertTrue(any("FROM v2_entities WHERE id=ANY" in query and "FOR UPDATE" in query for query in statements))
        self.assertTrue(any("FROM v2_entity_relations" in query and "FOR UPDATE" in query for query in statements))
        self.assertTrue(any("FROM v2_knowledge" in query and "FOR UPDATE" in query for query in statements))

    def test_no_physical_delete_and_migration_is_additive(self):
        source = Path("v2/service.py").read_text(encoding="utf-8").upper()
        migration = Path("migrations/020_v2_entity_pruning.sql").read_text(encoding="utf-8").upper()
        self.assertNotIn("DELETE FROM V2_ENTITIES", source)
        self.assertNotIn("DELETE FROM V2_ENTITY_RELATIONS", source)
        self.assertIn("ADD COLUMN IF NOT EXISTS DEACTIVATED_AT", migration)
        self.assertNotIn("DROP TABLE", migration)
        self.assertNotIn("DROP COLUMN", migration)


if __name__ == "__main__":
    unittest.main()
