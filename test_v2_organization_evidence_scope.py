from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from test_organization_generalized import MemoryOrganizationDB
from v2.learning import (
    _insert_proposal,
    _organization_review_context,
    _run_local_organization_review,
)


class _ContextCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.query = " ".join(str(query).split())
        self.params = params

    def fetchall(self):
        return list(self.rows)


class _ContextDB:
    def __init__(self, rows):
        self.cursor_obj = _ContextCursor(rows)

    def cursor(self):
        return self.cursor_obj


class _EvidenceScopeDB(MemoryOrganizationDB):
    def __init__(self, source_links):
        super().__init__()
        self.source_links = source_links

    def execute(self, sql, params):
        if sql.startswith("SELECT s.excerpt FROM v2_knowledge_sources"):
            return [
                {"excerpt": row["excerpt"]}
                for row in self.source_links
                if row.get("knowledge_id", params[0]) == params[0]
                if row["active"]
                and row["relation"] == "supports"
                and row["resolution"] == "accepted"
                and row["excerpt"].strip()
            ]
        return super().execute(sql, params)


class _ProposalLLM:
    def __init__(self, proposal):
        self.proposal = proposal

    def complete_json(self, _messages, max_tokens=900):
        return json.dumps(self.proposal, ensure_ascii=False)


def _relation_proposal(evidence_quote):
    return {
        "action": "CREATE_RELATION",
        "subject_entity": "Model X",
        "target_parent": "NVR",
        "new_entity": None,
        "entity_type": None,
        "relation_type": "belongs_to",
        "reason": "claimed parent relation",
        "evidence_quote": evidence_quote,
        "confidence": "explicit",
    }


class OrganizationEvidenceScopeTest(unittest.TestCase):
    def test_proposal_source_persists_atomic_source_excerpt(self):
        conn = MagicMock()
        cursor = conn.cursor.return_value
        cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = {"id": 9}
        fact = {
            "title": "rack support",
            "content": "Model X supports rack",
            "source_excerpt": "Model X supports standard rack mounting",
            "entity_name": "Model X",
        }

        with patch("v2.learning._link_source") as link_source:
            _insert_proposal(conn, 1, 2, 3, fact, 4)

        self.assertEqual(link_source.call_args.kwargs["excerpt"], fact["source_excerpt"])

    def test_context_keeps_knowledge_content_and_uses_accepted_excerpt_only(self):
        db = _ContextDB([{"excerpt": "Model X supports rack"}])
        knowledge = {
            "id": 7,
            "content": "Model X supports rack",
            "entity_name": "Model X",
            "trust": "user_confirmed",
        }

        context = _organization_review_context(db, knowledge)

        self.assertEqual(context["content"], "Model X supports rack")
        self.assertEqual(context["accepted_source_excerpts"], ["Model X supports rack"])
        self.assertNotIn("Model X belongs to NVR", context["accepted_source_excerpts"])
        self.assertNotIn("source_content", context)
        self.assertNotIn("v2_raw_evidence", db.cursor_obj.query)
        self.assertIn("SELECT s.excerpt", db.cursor_obj.query)

    def test_shared_raw_evidence_unconfirmed_claim_cannot_create_relation(self):
        # The raw input was "Model X supports rack. Model X belongs to NVR.".
        # Only the first atomic Knowledge is confirmed and its accepted
        # excerpt is the only source text exposed to organization review.
        knowledge = {
            "id": 1,
            "trust": "user_confirmed",
            "active": True,
            "entity_name": "Model X",
            "content": "Model X supports rack",
        }
        db = _EvidenceScopeDB([
            {
                "excerpt": "Model X supports rack",
                "active": True,
                "relation": "supports",
                "resolution": "accepted",
            },
            {
                "excerpt": "Model X belongs to NVR",
                "active": True,
                "relation": "supports",
                "resolution": "unresolved",
            },
        ])
        db.knowledge[1] = knowledge

        result = _run_local_organization_review(
            db,
            knowledge,
            llm_service=_ProposalLLM(_relation_proposal("Model X belongs to NVR")),
        )

        self.assertEqual(result["action"], "CREATE_ENTITY")
        self.assertEqual(db.relations, [])
        self.assertNotIn("NVR", {entity["name"] for entity in db.entities})

    def test_relation_appears_only_after_b_is_confirmed(self):
        knowledge_a = {
            "id": 1,
            "trust": "user_confirmed",
            "active": True,
            "entity_name": "Model X",
            "content": "Model X supports rack",
        }
        knowledge_b = {
            "id": 2,
            "trust": "user_confirmed",
            "active": True,
            "entity_name": "Model X",
            "content": "Model X 的产品分类已确认",
        }
        db = _EvidenceScopeDB([
            {
                "knowledge_id": 1,
                "excerpt": "Model X supports rack",
                "active": True,
                "relation": "supports",
                "resolution": "accepted",
            },
            {
                "knowledge_id": 2,
                "excerpt": "Model X belongs to NVR",
                "active": True,
                "relation": "supports",
                "resolution": "unresolved",
            },
        ])
        db.knowledge[1] = knowledge_a
        db.knowledge[2] = knowledge_b

        first = _run_local_organization_review(db, knowledge_a)
        self.assertEqual(first["action"], "CREATE_ENTITY")
        self.assertEqual(db.relations, [])

        db.source_links[1]["resolution"] = "accepted"
        second = _run_local_organization_review(db, knowledge_b)
        self.assertEqual(second["action"], "CREATE_RELATION")
        self.assertEqual(len(db.relations), 1)
        self.assertEqual(
            (db._entity(db.relations[0]["parent_entity_id"])["name"],
             db._entity(db.relations[0]["child_entity_id"])["name"]),
            ("NVR", "Model X"),
        )


if __name__ == "__main__":
    unittest.main()
