import unittest
from pathlib import Path

from v2.service import (
    V2NotFound,
    deactivate_knowledge,
    edit_knowledge,
    list_knowledge_history,
    list_knowledge_sources,
    list_knowledge,
    restore_knowledge,
)


class _Cursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.connection.executed.append((" ".join(str(query).split()), params))

    def fetchone(self):
        if self.connection.rows:
            return self.connection.rows.pop(0)
        return None

    def fetchall(self):
        rows = self.connection.fetchall_rows
        self.connection.fetchall_rows = []
        return rows


class _Connection:
    def __init__(self, rows=None, fetchall_rows=None):
        self.rows = list(rows or [])
        self.fetchall_rows = list(fetchall_rows or [])
        self.executed = []

    def cursor(self):
        return _Cursor(self)


def _knowledge(*, content="old text", entity_id=1, active=True, trust="user_confirmed"):
    return {
        "id": 42,
        "title": "Test knowledge",
        "content": content,
        "entity_name": "Model A",
        "entity_id": entity_id,
        "trust": trust,
        "active": active,
        "created_at": None,
        "updated_at": None,
    }


class KnowledgeMaintenanceTest(unittest.TestCase):
    def test_edit_keeps_id_and_source_links_are_not_touched(self):
        current = _knowledge()
        updated = _knowledge(content="new text", entity_id=2)
        conn = _Connection([current, {"id": 2}, updated])

        result = edit_knowledge(conn, 42, "new text", 2)

        self.assertEqual(result["id"], 42)
        self.assertEqual(result["content"], "new text")
        statements = [query for query, _ in conn.executed]
        self.assertTrue(any(query.startswith("UPDATE v2_knowledge") for query in statements))
        self.assertTrue(any("embedding=NULL" in query and "embedding_model=NULL" in query for query in statements))
        self.assertEqual(sum("INSERT INTO v2_knowledge_history" in query for query in statements), 2)
        self.assertEqual(sum("v2_knowledge_sources" in query for query in statements), 0)
        actions = [params[1] for query, params in conn.executed if "INSERT INTO v2_knowledge_history" in query]
        self.assertEqual(actions, ["edit", "move"])

    def test_delete_is_soft_and_restore_reactivates_same_object(self):
        deleted = _knowledge(active=False)
        conn = _Connection([_knowledge(), deleted])
        result = deactivate_knowledge(conn, 42)
        self.assertFalse(result["active"])
        self.assertTrue(any("SET active=FALSE" in query for query, _ in conn.executed))
        self.assertFalse(any(query.startswith("DELETE") for query, _ in conn.executed))
        self.assertTrue(any(params[1] == "deactivate" for query, params in conn.executed if "INSERT INTO v2_knowledge_history" in query))

        restored = _knowledge(active=True)
        restore_conn = _Connection([deleted, restored])
        result = restore_knowledge(restore_conn, 42)
        self.assertEqual(result["id"], 42)
        self.assertTrue(result["active"])
        self.assertTrue(any("SET active=TRUE" in query for query, _ in restore_conn.executed))
        self.assertTrue(any(params[1] == "restore" for query, params in restore_conn.executed if "INSERT INTO v2_knowledge_history" in query))

    def test_source_and_history_reads_preserve_audit_material(self):
        source = {"id": 7, "raw_content": "Original evidence", "excerpt": "Original evidence"}
        source_conn = _Connection([_knowledge()], [source])
        self.assertEqual(list_knowledge_sources(source_conn, 42), [source])
        self.assertIn("JOIN v2_raw_evidence", source_conn.executed[-1][0])

        history = {"id": 8, "action": "edit", "before_json": {"content": "old"}, "after_json": {"content": "new"}}
        history_conn = _Connection([_knowledge()], [history])
        self.assertEqual(list_knowledge_history(history_conn, 42), [history])
        self.assertIn("FROM v2_knowledge_history", history_conn.executed[-1][0])

    def test_knowledge_list_supports_deleted_and_entity_or_content_search(self):
        conn = _Connection(fetchall_rows=[{"id": 42, "active": False}])
        result = list_knowledge(conn, active=False, search="TandemVu")
        self.assertEqual(result, [{"id": 42, "active": False}])
        query, params = conn.executed[0]
        self.assertIn("k.active=%s", query)
        self.assertIn("ILIKE", query)
        self.assertEqual(params[0], False)
        self.assertIn("%TandemVu%", params)

    def test_deleted_knowledge_cannot_be_edited(self):
        with self.assertRaises(ValueError):
            edit_knowledge(_Connection([_knowledge(active=False)]), 42, "new", 1)

    def test_unknown_knowledge_is_not_mutated(self):
        with self.assertRaises(V2NotFound):
            deactivate_knowledge(_Connection([]), 42)

    def test_history_migration_is_compact_and_additive(self):
        migration = Path("migrations/019_v2_knowledge_history.sql").read_text(encoding="utf-8")
        self.assertIn("CREATE TABLE IF NOT EXISTS v2_knowledge_history", migration)
        for action in ("edit", "deactivate", "restore", "move"):
            self.assertIn(f"'{action}'", migration)
        self.assertIn("before_json JSONB", migration)
        self.assertIn("after_json JSONB", migration)
        self.assertIn("REFERENCES v2_knowledge(id) ON DELETE RESTRICT", migration)
        self.assertNotIn("DROP TABLE", migration.upper())

    def test_learning_sql_has_manual_edit_protection(self):
        learning = Path("v2/learning.py").read_text(encoding="utf-8")
        self.assertIn("FROM v2_knowledge_history h", learning)
        self.assertIn("h.action='edit'", learning)
        self.assertIn("THEN v2_knowledge.content", learning)


if __name__ == "__main__":
    unittest.main()
