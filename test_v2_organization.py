from __future__ import annotations

import unittest
from pathlib import Path

from v2.organization import (
    CycleError,
    ProvenanceError,
    create_relation,
    extract_explicit_chain,
    get_or_create_entity,
    get_relation,
    list_local_context,
    local_organization_review,
    move_relation,
    normalize_entity_name,
)


class _Cursor:
    def __init__(self, db):
        self.db = db
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        sql = " ".join(str(query).split())
        self.db.queries.append((sql, params))
        self.result = self.db.execute(sql, params)

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return list(self.result)


class FakeOrganizationDB:
    """Small semantic fake: these tests never connect to production PostgreSQL."""

    def __init__(self):
        self.entities = []
        self.relations = []
        self.knowledge = {}
        self.next_entity_id = 1
        self.next_relation_id = 1
        self.queries = []

    def cursor(self):
        return _Cursor(self)

    def _entity(self, entity_id):
        return next((row for row in self.entities if row["id"] == int(entity_id)), None)

    def execute(self, sql, params):
        if sql.startswith("INSERT INTO v2_entities"):
            name, normalized, entity_type, active = params
            existing = next((row for row in self.entities if row["normalized_name"] == normalized), None)
            if existing:
                return [dict(existing)]
            row = {"id": self.next_entity_id, "name": name, "normalized_name": normalized,
                   "entity_type": entity_type, "active": active,
                   "created_at": None, "updated_at": None}
            self.next_entity_id += 1
            self.entities.append(row)
            return [dict(row)]

        if "FROM v2_entities" in sql and "WHERE normalized_name=%s" in sql:
            normalized = params[0]
            rows = [row for row in self.entities if row["normalized_name"] == normalized]
            if "active=TRUE" in sql:
                rows = [row for row in rows if row["active"]]
            return [dict(rows[0])] if rows else []

        if "FROM v2_entities" in sql and "WHERE id=%s" in sql:
            row = self._entity(params[0])
            if row and ("active=TRUE" not in sql or row["active"]):
                return [dict(row)]
            return []

        if sql.startswith("SELECT id, trust, active FROM v2_knowledge"):
            row = self.knowledge.get(int(params[0]))
            return [{"id": row["id"], "trust": row["trust"], "active": row.get("active", True)}] if row else []

        if sql.startswith("UPDATE v2_knowledge"):
            row = self.knowledge.get(int(params[1]))
            if row and (row.get("entity_id") is None or row.get("entity_id") == int(params[2])):
                row["entity_id"] = int(params[0])
            return []

        if "WITH RECURSIVE ancestors(entity_id, path)" in sql:
            parent, _ignored, relation_type, child = params
            seen = {int(parent)}
            todo = [int(parent)]
            while todo:
                current = todo.pop()
                for relation in self.relations:
                    if relation["active"] and relation["relation_type"] == relation_type and relation["child_entity_id"] == current:
                        ancestor = relation["parent_entity_id"]
                        if ancestor == int(child):
                            return [{"would_cycle": True}]
                        if ancestor not in seen:
                            seen.add(ancestor)
                            todo.append(ancestor)
            return [{"would_cycle": False}]

        if "SELECT id, parent_entity_id, child_entity_id, relation_type" in sql and "WHERE parent_entity_id=%s" in sql:
            parent, child, relation_type = params
            rows = [row for row in self.relations if row["parent_entity_id"] == int(parent)
                    and row["child_entity_id"] == int(child)
                    and row["relation_type"] == relation_type
                    and ("active=TRUE" not in sql or row["active"])]
            return [dict(rows[-1])] if rows else []

        if "WHERE child_entity_id=%s AND relation_type=%s AND active=TRUE" in sql:
            child, relation_type = params
            rows = [row for row in self.relations if row["child_entity_id"] == int(child)
                    and row["relation_type"] == relation_type and row["active"]]
            return [dict(rows[-1])] if rows else []

        if sql.startswith("INSERT INTO v2_entity_relations"):
            parent, child, relation_type, source_id, provenance, kind = params
            existing = next((row for row in self.relations if row["parent_entity_id"] == int(parent)
                             and row["child_entity_id"] == int(child)
                             and row["relation_type"] == relation_type and row["active"]), None)
            if existing:
                return [dict(existing)]
            row = {"id": self.next_relation_id, "parent_entity_id": int(parent),
                   "child_entity_id": int(child), "relation_type": relation_type,
                   "source_id": source_id, "provenance": provenance,
                   "provenance_kind": kind, "active": True,
                   "created_at": None, "updated_at": None}
            self.next_relation_id += 1
            self.relations.append(row)
            return [dict(row)]

        if sql.startswith("UPDATE v2_entity_relations"):
            relation = next(row for row in self.relations if row["id"] == int(params[0]))
            relation["active"] = False
            return []

        if "FROM v2_entity_relations r" in sql and "JOIN v2_entities p" in sql:
            child = int(params[0])
            rows = [row for row in self.relations if row["child_entity_id"] == child and row["active"]]
            rows = sorted(rows, key=lambda row: row["id"], reverse=True)
            parent = self._entity(rows[0]["parent_entity_id"]) if rows else None
            return [dict(parent)] if parent else []

        if "WITH RECURSIVE ancestors AS" in sql:
            return []

        if "FROM v2_entity_relations r" in sql and "WHERE r.parent_entity_id=%s" in sql:
            parent = int(params[0])
            rows = [row for row in self.relations if row["parent_entity_id"] == parent and row["active"]]
            if "e.id<>%s" in sql:
                rows = [row for row in rows if row["child_entity_id"] != int(params[1])]
            return [dict(self._entity(row["child_entity_id"])) for row in rows]

        if sql.startswith("SELECT id, entity_name FROM v2_knowledge"):
            return [{"id": row["id"], "entity_name": row.get("entity_name", "")}
                    for row in self.knowledge.values() if row.get("entity_id") is None and row.get("entity_name", "").strip()]

        raise AssertionError(f"unhandled fake SQL: {sql}")


class OrganizationTest(unittest.TestCase):
    def setUp(self):
        self.db = FakeOrganizationDB()
        self.db.knowledge[1] = {"id": 1, "trust": "user_confirmed", "active": True,
                                "entity_name": "F-NR-208E/2", "content": "supports rack"}

    def entity(self, name, entity_type="concept"):
        return get_or_create_entity(self.db, name, entity_type=entity_type)

    def entity_by_id(self, entity_id):
        return next(row for row in self.db.entities if row["id"] == entity_id)

    def test_normalization_reuses_exact_model(self):
        self.assertEqual(normalize_entity_name("F-NR-208E/2"), normalize_entity_name("F-NR-208E / 2"))
        first = self.entity("F-NR-208E/2", "model")
        second = self.entity("f-nr-208e / 2", "model")
        self.assertEqual(first["id"], second["id"])

    def test_chain_extractor_ignores_unstructured_model_prose(self):
        self.assertEqual(extract_explicit_chain({"entity_name": "F-NR-208E/2", "content": "is an NVR"}), [])
        self.assertEqual(
            [item["name"] for item in extract_explicit_chain({"explicit_chain": [
                {"name": "NVR", "entity_type": "category"},
                {"name": "F-NR-208E/2", "entity_type": "model"},
            ]})],
            ["NVR", "F-NR-208E/2"],
        )

    def test_chain_extractor_accepts_only_explicit_fnr_relationships(self):
        first = extract_explicit_chain({
            "entity_name": "F-NR-208E/2",
            "content": "F-NR-208E/2 是 iFlow 的 NVR。",
        })
        self.assertEqual([item["name"] for item in first], ["iFlow", "NVR", "F-NR-208E/2"])
        final = extract_explicit_chain({
            "entity_name": "F-NR-208E/2",
            "content": "F-NR-208E/2 属于 F-NR 系列，F-NR 系列属于 iFlow 后端产品中的 NVR。",
        })
        self.assertEqual(
            [item["name"] for item in final],
            ["iFlow", "后端产品", "NVR", "F-NR 系列", "F-NR-208E/2"],
        )
        self.assertEqual(
            extract_explicit_chain({
                "entity_name": "F-NR-208E/2",
                "content": "NR 型号通常属于 NVR。",
            }),
            [],
        )

    def test_confirmed_relation_keeps_provenance(self):
        parent, child = self.entity("NVR"), self.entity("F-NR-208E/2", "model")
        relation = create_relation(self.db, parent["id"], child["id"], source_id=1,
                                   provenance="Knowledge #1 explicitly says it is an NVR",
                                   provenance_kind="user_confirmed")
        self.assertEqual(relation["source_id"], 1)
        self.assertEqual(relation["provenance_kind"], "user_confirmed")
        self.assertEqual(get_relation(self.db, parent["id"], child["id"])["id"], relation["id"])

    def test_provisional_does_not_change_structure(self):
        provisional = {"id": 2, "trust": "provisional", "entity_name": "F-ABC"}
        self.db.knowledge[2] = provisional
        result = local_organization_review(self.db, provisional,
                                           proposed_parent="NVR", provenance="model guess",
                                           provenance_kind="user_confirmed")
        self.assertEqual(result["action"], "NO_CHANGE")
        self.assertEqual(self.db.entities, [])
        self.assertEqual(self.db.relations, [])

    def test_local_review_no_change(self):
        child = self.entity("F-NR-208E/2", "model")
        parent = self.entity("NVR", "category")
        create_relation(self.db, parent["id"], child["id"], source_id=1,
                        provenance="confirmed fact", provenance_kind="user_confirmed")
        result = local_organization_review(self.db, self.db.knowledge[1])
        self.assertEqual(result["action"], "NO_CHANGE")

    def test_local_review_creates_parent_relation(self):
        child = self.entity("F-NR-208E/2", "model")
        parent = self.entity("NVR", "category")
        result = local_organization_review(self.db, self.db.knowledge[1],
                                           proposed_parent=parent["id"],
                                           provenance="Knowledge #1 says this model is an NVR",
                                           provenance_kind="user_confirmed")
        self.assertEqual(result["action"], "CREATE_RELATION")
        self.assertEqual(self.db.relations[0]["source_id"], 1)

    def test_move_preserves_old_relation_inactive(self):
        child = self.entity("F-NR-208E/2", "model")
        old_parent = self.entity("产品知识", "category")
        new_parent = self.entity("NVR", "category")
        create_relation(self.db, old_parent["id"], child["id"], source_id=1,
                        provenance="initial confirmed structure", provenance_kind="user_confirmed")
        moved = move_relation(self.db, child["id"], new_parent["id"], source_id=1,
                              provenance="new confirmed structure", provenance_kind="user_confirmed")
        self.assertTrue(moved["active"])
        self.assertFalse(self.db.relations[0]["active"])
        self.assertEqual(len(self.db.relations), 2)

    def test_local_review_moves_relation_and_keeps_history(self):
        child = self.entity("F-NR-208E/2", "model")
        old_parent = self.entity("产品知识", "category")
        new_parent = self.entity("NVR", "category")
        create_relation(self.db, old_parent["id"], child["id"], source_id=1,
                        provenance="initial confirmed structure", provenance_kind="user_confirmed")
        result = local_organization_review(
            self.db, self.db.knowledge[1], proposed_parent=new_parent["id"],
            provenance="new confirmed structure", provenance_kind="user_confirmed",
        )
        self.assertEqual(result["action"], "MOVE_RELATION")
        self.assertEqual(sum(row["active"] for row in self.db.relations), 1)
        self.assertEqual(self.db.relations[-1]["parent_entity_id"], new_parent["id"])

    def test_single_model_does_not_generalize_to_series(self):
        child = self.entity("F-NR-208E/2", "model")
        series_model = self.entity("F-NR-216E/2", "model")
        parent = self.entity("NVR", "category")
        create_relation(self.db, parent["id"], child["id"], source_id=1,
                        provenance="only this model is confirmed", provenance_kind="user_confirmed")
        self.assertIsNone(get_relation(self.db, parent["id"], series_model["id"]))

    def test_unknown_entity_remains_unorganized(self):
        unknown = {"id": 3, "trust": "user_confirmed", "entity_name": "F-ABC", "content": "one fact"}
        self.db.knowledge[3] = unknown
        result = local_organization_review(self.db, unknown)
        self.assertEqual(result["action"], "CREATE_ENTITY")
        entity = self.entity("F-ABC", "model")
        self.assertEqual(list_local_context(self.db, entity["id"])["current_parent"], None)

    def test_entity_created_from_confirmed_knowledge(self):
        result = local_organization_review(self.db, self.db.knowledge[1])
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(result["entity"]["normalized_name"], "f-nr-208e/2")
        self.assertEqual(self.db.knowledge[1]["entity_id"], result["entity"]["id"])

    def test_explicit_fnr_acceptance_chain_changes_only_organization_layer(self):
        first = local_organization_review(self.db, self.db.knowledge[1])
        model_id = first["entity"]["id"]
        self.db.knowledge[2] = {"id": 2, "trust": "user_confirmed", "active": True,
                                "entity_name": "F-NR-208E/2", "content": "is an iFlow NVR"}
        local_organization_review(
            self.db, self.db.knowledge[2],
            explicit_chain=[
                {"name": "iFlow", "entity_type": "brand"},
                {"name": "NVR", "entity_type": "category"},
                {"name": "F-NR-208E/2", "entity_type": "model"},
            ], provenance="Knowledge #2 explicitly confirms this chain",
            provenance_kind="user_confirmed",
        )
        self.db.knowledge[3] = {"id": 3, "trust": "user_confirmed", "active": True,
                                "entity_name": "F-NR-208E/2", "content": "belongs to F-NR series"}
        local_organization_review(
            self.db, self.db.knowledge[3],
            explicit_chain=[
                {"name": "iFlow", "entity_type": "brand"},
                {"name": "后端产品", "entity_type": "category"},
                {"name": "NVR", "entity_type": "category"},
                {"name": "F-NR Series", "entity_type": "series"},
                {"name": "F-NR-208E/2", "entity_type": "model"},
            ], provenance="Knowledge #3 explicitly confirms the intermediate layers",
            provenance_kind="user_confirmed",
        )
        active_edges = {(self.entity_by_id(row["parent_entity_id"])["name"],
                         self.entity_by_id(row["child_entity_id"])["name"])
                        for row in self.db.relations if row["active"]}
        self.assertEqual(active_edges, {
            ("iFlow", "后端产品"), ("后端产品", "NVR"),
            ("NVR", "F-NR Series"), ("F-NR Series", "F-NR-208E/2"),
        })
        self.assertEqual(self.db.knowledge[1]["content"], "supports rack")
        self.assertEqual(self.db.knowledge[2]["content"], "is an iFlow NVR")
        self.assertEqual(self.db.knowledge[3]["content"], "belongs to F-NR series")
        self.assertEqual(model_id, self.db.knowledge[1]["entity_id"])

    def test_cycle_is_rejected(self):
        a, b, c = self.entity("A"), self.entity("B"), self.entity("C")
        for parent, child in ((a, b), (b, c)):
            create_relation(self.db, parent["id"], child["id"], source_id=1,
                            provenance="confirmed chain", provenance_kind="user_confirmed")
        with self.assertRaises(CycleError):
            create_relation(self.db, c["id"], a["id"], source_id=1,
                            provenance="would close cycle", provenance_kind="user_confirmed")

    def test_untrusted_source_and_missing_provenance_are_rejected(self):
        self.db.knowledge[4] = {"id": 4, "trust": "provisional", "active": True}
        parent, child = self.entity("NVR"), self.entity("F-X", "model")
        with self.assertRaises(ProvenanceError):
            create_relation(self.db, parent["id"], child["id"], source_id=4,
                            provenance="guess", provenance_kind="user_confirmed")
        with self.assertRaises(ProvenanceError):
            create_relation(self.db, parent["id"], child["id"], source_id=1,
                            provenance="", provenance_kind="user_confirmed")

    def test_migration_is_additive_and_minimal(self):
        sql = Path("migrations/018_v2_organization.sql").read_text(encoding="utf-8")
        self.assertIn("CREATE TABLE IF NOT EXISTS v2_entities", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS v2_entity_relations", sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS entity_id", sql)
        self.assertIn("relation_type IN ('belongs_to')", sql)
        self.assertIn("source_id BIGINT REFERENCES v2_knowledge", sql)
        self.assertNotRegex(sql, r"(?i)\bDROP\s+(TABLE|COLUMN)\b")
        self.assertNotIn("ontology", sql.casefold())
        self.assertNotIn("dimension", sql.casefold())

    def test_entity_tree_rendering_and_knowledge_filter_contract(self):
        page = Path("templates/knowledge.html").read_text(encoding="utf-8")
        self.assertIn('id="entity-tree"', page)
        self.assertIn('id="knowledge-list"', page)
        self.assertIn("data.tree", page)
        self.assertIn("/api/v2/knowledge?entity_id=", page)
        self.assertIn("未整理", page)
        self.assertNotIn("drag", page.casefold())
        self.assertNotIn("graph", page.casefold())


if __name__ == "__main__":
    unittest.main()
