from __future__ import annotations

import json
import unittest
from pathlib import Path

from v2.organization import (
    CycleError,
    create_relation,
    extract_explicit_chain,
    get_or_create_entity,
    local_organization_review,
    normalize_entity_name,
    review_local_organization,
    validate_organization_proposal,
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
        self.result = self.db.execute(" ".join(str(query).split()), params)

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return list(self.result)


class MemoryOrganizationDB:
    """Small semantic fake for generalized organization contract tests."""

    def __init__(self):
        self.entities = []
        self.relations = []
        self.knowledge = {}
        self.next_entity_id = 1
        self.next_relation_id = 1

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
            row = {
                "id": self.next_entity_id,
                "name": name,
                "normalized_name": normalized,
                "entity_type": entity_type,
                "active": active,
                "created_at": None,
                "updated_at": None,
            }
            self.next_entity_id += 1
            self.entities.append(row)
            return [dict(row)]

        if "FROM v2_entities" in sql and "WHERE normalized_name=%s" in sql:
            rows = [row for row in self.entities if row["normalized_name"] == params[0]]
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
            row = {
                "id": self.next_relation_id,
                "parent_entity_id": int(parent),
                "child_entity_id": int(child),
                "relation_type": relation_type,
                "source_id": source_id,
                "provenance": provenance,
                "provenance_kind": kind,
                "active": True,
                "created_at": None,
                "updated_at": None,
            }
            self.next_relation_id += 1
            self.relations.append(row)
            return [dict(row)]

        if sql.startswith("UPDATE v2_entity_relations"):
            relation = next(row for row in self.relations if row["id"] == int(params[0]))
            relation["active"] = False
            return []

        if "FROM v2_entity_relations r" in sql and "JOIN v2_entities p" in sql:
            child = int(params[0])
            rows = sorted(
                [row for row in self.relations if row["child_entity_id"] == child and row["active"]],
                key=lambda row: row["id"],
                reverse=True,
            )
            parent = self._entity(rows[0]["parent_entity_id"]) if rows else None
            return [dict(parent)] if parent else []

        if "FROM v2_entity_relations r" in sql and "WHERE r.parent_entity_id=%s" in sql:
            parent = int(params[0])
            rows = [row for row in self.relations if row["parent_entity_id"] == parent and row["active"]]
            if "e.id<>%s" in sql:
                rows = [row for row in rows if row["child_entity_id"] != int(params[1])]
            return [dict(self._entity(row["child_entity_id"])) for row in rows]

        if "WITH RECURSIVE ancestors AS" in sql:
            return []

        raise AssertionError(f"unhandled fake SQL: {sql}")


def _knowledge(entity_name, content, *, trust="user_confirmed", knowledge_id=1):
    return {
        "id": knowledge_id,
        "trust": trust,
        "active": True,
        "entity_name": entity_name,
        "content": content,
    }


class ProposalLLM:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = []

    def complete_json(self, messages, max_tokens=900):
        self.calls.append((messages, max_tokens))
        return json.dumps(self.proposal, ensure_ascii=False)


class FailingProposalLLM:
    def complete_json(self, messages, max_tokens=900):
        raise TimeoutError("test timeout")


class GeneralizedOrganizationTest(unittest.TestCase):
    def test_general_org_review_not_fnr_specific(self):
        chain = extract_explicit_chain(_knowledge(
            "DS-2CD7A26G0/P-IZHS",
            "DS-2CD7A26G0/P-IZHS 属于 Hikvision 的网络摄像机产品。",
        ))
        self.assertEqual([item["name"] for item in chain], [
            "Hikvision", "网络摄像机", "DS-2CD7A26G0/P-IZHS",
        ])

    def test_general_org_review_hikvision_camera(self):
        chain = extract_explicit_chain(_knowledge(
            "DS-2CD7A26G0/P-IZHS",
            "DS-2CD7A26G0/P-IZHS 属于 Hikvision 的网络摄像机产品。",
        ))
        self.assertEqual([item["entity_type"] for item in chain], ["brand", "category", "model"])

    def test_general_org_review_ai_product_line(self):
        chain = extract_explicit_chain(_knowledge("观澜", "观澜是 Hikvision 的 AI 产品线。"))
        self.assertEqual([item["name"] for item in chain], ["Hikvision", "AI 产品线", "观澜"])

    def test_general_org_review_access_controller(self):
        chain = extract_explicit_chain(_knowledge(
            "DS-K2604T", "DS-K2604T 属于 Hikvision 门禁控制器。",
        ))
        self.assertEqual([item["name"] for item in chain], [
            "Hikvision", "门禁控制器", "DS-K2604T",
        ])

    def test_fnr_acceptance_case_grows_only_from_explicit_layers(self):
        db = MemoryOrganizationDB()
        first = _knowledge("F-NR-208E/2", "F-NR-208E/2 支持标准 19 英寸机架安装。")
        db.knowledge[1] = first
        review_local_organization(db, first)

        second = _knowledge("F-NR-208E/2", "F-NR-208E/2 是 iFlow 的 NVR。", knowledge_id=2)
        db.knowledge[2] = second
        review_local_organization(db, second)
        active_edges = lambda: {
            (db._entity(row["parent_entity_id"])["name"], db._entity(row["child_entity_id"])["name"])
            for row in db.relations if row["active"]
        }
        self.assertEqual(active_edges(), {
            ("iFlow", "NVR"), ("NVR", "F-NR-208E/2"),
        })

        third = _knowledge("F-NR-208E/2", "F-NR-208E/2 属于 F-NR Series。", knowledge_id=3)
        db.knowledge[3] = third
        review_local_organization(db, third)
        self.assertEqual(active_edges(), {
            ("iFlow", "NVR"), ("NVR", "F-NR Series"),
            ("F-NR Series", "F-NR-208E/2"),
        })
        self.assertEqual(db.knowledge[1]["content"], "F-NR-208E/2 支持标准 19 英寸机架安装。")

    def test_provisional_does_not_reorganize(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Model A 是 Brand B 的 NVR。", trust="provisional")
        db.knowledge[1] = knowledge
        result = local_organization_review(db, knowledge, explicit_chain=extract_explicit_chain(knowledge))
        self.assertEqual(result["action"], "NO_CHANGE")
        self.assertEqual(db.entities, [])
        self.assertEqual(db.relations, [])

    def test_naming_pattern_does_not_create_relation(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("F-NR-999", "F-NR-999 支持双电源。")
        db.knowledge[1] = knowledge
        result = local_organization_review(db, knowledge, explicit_chain=extract_explicit_chain(knowledge))
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(db.relations, [])

    def test_naming_pattern_cannot_be_promoted_by_llm_cooccurrence(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("F-NR-999", "F-NR-999 在 NVR 场景中支持双电源。")
        db.knowledge[1] = knowledge
        get_or_create_entity(db, "NVR", entity_type="category")
        llm = ProposalLLM({
            "action": "CREATE_RELATION",
            "subject_entity": "F-NR-999",
            "target_parent": "NVR",
            "new_entity": None,
            "entity_type": None,
            "relation_type": "belongs_to",
            "reason": "the model prefix looks similar",
            "evidence_quote": "F-NR-999 在 NVR 场景中支持双电源。",
            "confidence": "explicit",
        })
        result = review_local_organization(db, knowledge, llm_service=llm)
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(db.relations, [])

    def test_no_invented_intermediate_entities(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Model A 是 Brand B 的 NVR。")
        db.knowledge[1] = knowledge
        result = local_organization_review(db, knowledge, explicit_chain=extract_explicit_chain(knowledge))
        self.assertEqual(result["action"], "CREATE_RELATION")
        self.assertEqual({entity["name"] for entity in db.entities}, {"Brand B", "NVR", "Model A"})
        self.assertEqual(len(db.relations), 2)

    def test_general_create_relation_keeps_provenance(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge(
            "观澜", "观澜是 Hikvision 的 AI 产品线。", knowledge_id=7,
        )
        db.knowledge[7] = knowledge
        local_organization_review(
            db, knowledge, explicit_chain=extract_explicit_chain(knowledge),
        )
        self.assertEqual({row["provenance_kind"] for row in db.relations}, {"user_confirmed"})
        self.assertEqual({row["source_id"] for row in db.relations}, {7})

    def test_cycle_is_rejected(self):
        db = MemoryOrganizationDB()
        source = _knowledge("C", "confirmed chain")
        db.knowledge[1] = source
        a, b, c = (get_or_create_entity(db, name) for name in ("A", "B", "C"))
        create_relation(db, a["id"], b["id"], source_id=1, provenance="confirmed", provenance_kind="user_confirmed")
        create_relation(db, b["id"], c["id"], source_id=1, provenance="confirmed", provenance_kind="user_confirmed")
        with self.assertRaises(CycleError):
            create_relation(db, c["id"], a["id"], source_id=1, provenance="cycle", provenance_kind="user_confirmed")

    def test_organization_module_has_no_product_specific_rules(self):
        source = Path("v2/organization.py").read_text(encoding="utf-8").casefold()
        for term in ("f-nr", "iflow", "nvr"):
            self.assertNotIn(term, source)

    def test_normalization_remains_exact_not_alias_matching(self):
        self.assertEqual(normalize_entity_name("DS-2CD7A26G0/P-IZHS"), normalize_entity_name("ds-2cd7a26g0 / p-izhs"))
        self.assertNotEqual(normalize_entity_name("Model A"), normalize_entity_name("Model B"))

    def test_general_review_uses_llm_only_when_shortcut_is_insufficient(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Manual placement: Model A is categorized beneath NVR.")
        db.knowledge[1] = knowledge
        parent = get_or_create_entity(db, "NVR", entity_type="category")
        llm = ProposalLLM({
            "action": "CREATE_RELATION",
            "subject_entity": "Model A",
            "target_parent": "NVR",
            "new_entity": None,
            "entity_type": None,
            "relation_type": "belongs_to",
            "reason": "the confirmed sentence explicitly names the parent",
            "evidence_quote": "Manual placement: Model A is categorized beneath NVR.",
            "confidence": "explicit",
        })
        result = review_local_organization(db, knowledge, llm_service=llm)
        self.assertEqual(result["action"], "CREATE_RELATION")
        self.assertEqual(len(db.relations), 1)
        self.assertEqual(db.relations[0]["parent_entity_id"], parent["id"])
        self.assertEqual(len(llm.calls), 1)

    def test_general_review_can_create_one_explicit_entity_without_relation(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Model A references Brand B in the manual.")
        db.knowledge[1] = knowledge
        llm = ProposalLLM({
            "action": "CREATE_ENTITY",
            "subject_entity": "Model A",
            "target_parent": None,
            "new_entity": {"name": "Brand B", "entity_type": "brand"},
            "entity_type": "brand",
            "relation_type": "belongs_to",
            "reason": "Brand B is explicitly named in the evidence",
            "evidence_quote": "Model A references Brand B in the manual.",
            "confidence": "explicit",
        })
        result = review_local_organization(db, knowledge, llm_service=llm)
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual({row["name"] for row in db.entities}, {"Model A", "Brand B"})
        self.assertEqual(db.relations, [])

    def test_general_review_moves_parent_and_preserves_history(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Manual placement: Model A is categorized beneath New Parent.")
        db.knowledge[1] = knowledge
        child = get_or_create_entity(db, "Model A", entity_type="model")
        old_parent = get_or_create_entity(db, "Old Parent", entity_type="category")
        new_parent = get_or_create_entity(db, "New Parent", entity_type="category")
        source = _knowledge("Old Parent", "Old Parent is the current local parent.", knowledge_id=2)
        db.knowledge[2] = source
        create_relation(
            db, old_parent["id"], child["id"], source_id=2,
            provenance="initial", provenance_kind="user_confirmed",
        )
        llm = ProposalLLM({
            "action": "MOVE_RELATION",
            "subject_entity": "Model A",
            "target_parent": "New Parent",
            "new_entity": None,
            "entity_type": None,
            "relation_type": "belongs_to",
            "reason": "the new confirmed sentence names a different parent",
            "evidence_quote": "Manual placement: Model A is categorized beneath New Parent.",
            "confidence": "explicit",
        })
        result = review_local_organization(db, knowledge, llm_service=llm)
        self.assertEqual(result["action"], "MOVE_RELATION")
        self.assertEqual(sum(row["active"] for row in db.relations), 1)
        self.assertFalse(db.relations[0]["active"])
        self.assertEqual(db.relations[-1]["parent_entity_id"], new_parent["id"])

    def test_structured_proposal_validation_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_organization_proposal({"action": "DROP_DATABASE"})
        with self.assertRaises(ValueError):
            validate_organization_proposal({
                "action": "CREATE_RELATION",
                "subject_entity": "Model A",
                "target_parent": "NVR",
                "new_entity": None,
                "entity_type": None,
                "relation_type": "is_a",
                "reason": "bad relation",
                "evidence_quote": "Model A belongs to NVR.",
                "confidence": "explicit",
            })

    def test_invalid_llm_action_fails_closed_without_a_relation(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Model A supports a rack feature.")
        db.knowledge[1] = knowledge
        llm = ProposalLLM({
            "action": "MOVE_ALL_ENTITIES",
            "subject_entity": "Model A",
            "target_parent": "NVR",
            "new_entity": None,
            "entity_type": None,
            "relation_type": "belongs_to",
            "reason": "invalid action",
            "evidence_quote": "Model A supports a rack feature.",
            "confidence": "explicit",
        })
        result = review_local_organization(db, knowledge, llm_service=llm)
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(db.relations, [])

    def test_evidence_quote_is_required_and_must_match(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Model A supports a rack feature.")
        db.knowledge[1] = knowledge
        get_or_create_entity(db, "NVR", entity_type="category")
        llm = ProposalLLM({
            "action": "CREATE_RELATION",
            "subject_entity": "Model A",
            "target_parent": "NVR",
            "new_entity": None,
            "entity_type": None,
            "relation_type": "belongs_to",
            "reason": "unsupported claim",
            "evidence_quote": "Model A belongs to NVR.",
            "confidence": "explicit",
        })
        result = review_local_organization(db, knowledge, llm_service=llm)
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(db.relations, [])

    def test_llm_failure_does_not_erase_confirmed_knowledge(self):
        db = MemoryOrganizationDB()
        knowledge = _knowledge("Model A", "Model A supports a rack feature.")
        db.knowledge[1] = knowledge
        result = review_local_organization(db, knowledge, llm_service=FailingProposalLLM())
        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(db.knowledge[1]["content"], "Model A supports a rack feature.")
        self.assertEqual(db.relations, [])


if __name__ == "__main__":
    unittest.main()
