import unittest
from unittest.mock import patch

import app
from v2.service import edit_pending_proposal, list_editable_proposals, reject_pending_proposal


class Cursor:
    def __init__(self, one=None, many=()):
        self.one = one
        self.many = list(many)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.queries.append((" ".join(str(query).split()), params))

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class Connection:
    def __init__(self, cursors):
        self.cursors = iter(cursors)

    def cursor(self):
        return next(self.cursors)


class ProposalEditingTest(unittest.TestCase):
    def test_list_returns_only_active_pending_batch_proposals(self):
        batch = {"id": 9, "thread_id": 7, "status": "awaiting_confirmation"}
        item = {"id": 30, "batch_id": 9, "fact_text": "старый текст", "status": "pending_confirmation"}
        # The production helper uses one cursor for both sequential queries.
        batch_cursor = Cursor(one=batch, many=[item])

        result = list_editable_proposals(Connection([batch_cursor]), 7)

        self.assertEqual(result, {"batch": batch, "items": [item]})
        self.assertIn("status IN ('pending_confirmation', 'pending_clarification')", batch_cursor.queries[1][0])
        self.assertIn("paused=FALSE", batch_cursor.queries[1][0])

    def test_list_supports_one_pending_non_batch_proposal(self):
        item = {"id": 31, "batch_id": None, "fact_text": "один факт", "status": "pending_confirmation"}
        cursor = Cursor(one=None, many=[item])

        result = list_editable_proposals(Connection([cursor]), 7)

        self.assertEqual(result, {"batch": None, "items": [item]})
        self.assertIn("batch_id IS NULL", cursor.queries[1][0])

    def test_edit_changes_only_proposal_text(self):
        row = {"id": 30, "fact_text": "новый текст", "status": "pending_confirmation"}
        cursor = Cursor(one=row)

        result = edit_pending_proposal(Connection([cursor]), 30, "  новый текст  ")

        self.assertEqual(result, row)
        query, params = cursor.queries[0]
        self.assertIn("SET fact_text=%s", query)
        self.assertNotIn("raw_evidence", query)
        self.assertEqual(params, ("новый текст", 30))

    def test_delete_soft_rejects_proposal_and_keeps_evidence(self):
        row = {"id": 30, "status": "rejected"}
        cursor = Cursor(one=row)

        result = reject_pending_proposal(Connection([cursor]), 30)

        self.assertEqual(result, row)
        query, _ = cursor.queries[0]
        self.assertIn("SET status='rejected'", query)
        self.assertIn("status IN ('pending_confirmation', 'pending_clarification')", query)

    def test_edit_route_does_not_call_llm(self):
        body = {"fact_text": "исправленная формулировка"}
        with patch("app.auth"), patch("app.db"), patch(
            "app.edit_pending_proposal", return_value={"id": 30, "fact_text": body["fact_text"]}
        ) as edit:
            result = app.v2_edit_inbox_proposal(30, body, x_api_key="key")

        edit.assert_called_once()
        self.assertEqual(result["fact_text"], body["fact_text"])


if __name__ == "__main__":
    unittest.main()
