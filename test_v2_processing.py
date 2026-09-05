import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from fastapi import BackgroundTasks

import app
from v2.processing import process_inbox_job


class DbContext:
    def __init__(self, conn=None):
        self.conn = conn or Mock()

    def __enter__(self):
        return self.conn

    def __exit__(self, *_):
        return False


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


class V2InboxProcessingTest(unittest.TestCase):
    def test_submit_returns_before_learning_finishes(self):
        job = {"id": 41, "thread_id": 7, "status": "queued"}
        tasks = BackgroundTasks()
        with patch("app.auth"), patch("app.db", return_value=DbContext()), patch(
            "app.enqueue_inbox_job", return_value=job
        ), patch("app._process_v2_inbox_job") as process:
            result = app.v2_inbox_message(
                app.V2InboxMessageIn(content="一批知识"),
                tasks,
                x_api_key="key",
                idempotency_key="client-41",
            )

        self.assertEqual(result.status_code, 202)
        self.assertEqual(response_json(result), {"thread_id": 7, "job_id": 41, "status": "queued"})
        process.assert_not_called()
        self.assertEqual(len(tasks.tasks), 0)

    def test_submit_passes_idempotency_key_to_durable_enqueue(self):
        job = {"id": 42, "thread_id": 7, "status": "queued"}
        with patch("app.auth"), patch("app.db", return_value=DbContext()), patch(
            "app.enqueue_inbox_job", return_value=job
        ) as enqueue:
            app.v2_inbox_message(
                app.V2InboxMessageIn(content="原始资料"),
                BackgroundTasks(),
                x_api_key="key",
                idempotency_key="same-client-request",
            )

        self.assertEqual(enqueue.call_args.kwargs["idempotency_key"], "same-client-request")

    def test_job_progresses_processing_to_completed(self):
        conn = Mock()
        cursor = Mock()
        cursor.fetchone.side_effect = [
            {"id": 100, "content": "完整原文"},
            {"id": 101, "thread_id": 7, "content": "完整原文", "raw_evidence_id": 100},
        ]
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = cursor
        conn.cursor.return_value = cursor_context
        claimed = {
            "id": 55,
            "thread_id": 7,
            "raw_evidence_id": 100,
            "user_message_id": 101,
            "status": "processing",
        }
        with patch("v2.processing.claim_processing_job", return_value=claimed) as claim, patch(
            "v2.processing.get_processing_job", return_value={"id": 55, "assistant_message_id": None}
        ), patch("v2.processing.learn_turn") as learn, patch(
            "v2.processing.complete_processing_job", return_value={"id": 55, "status": "completed"}
        ) as complete, patch("v2.processing.fail_processing_job") as fail:
            process_inbox_job(55, db_factory=lambda: DbContext(conn), llm_service=Mock())

        claim.assert_called_once_with(conn, 55)
        learn.assert_called_once()
        self.assertTrue(learn.call_args.kwargs["normalize_to_russian"])
        complete.assert_called_once_with(conn, 55)
        fail.assert_not_called()
        self.assertGreaterEqual(conn.commit.call_count, 2)

    def test_learning_failure_marks_same_job_failed_for_safe_retry(self):
        conn = Mock()
        cursor = MagicMock()
        cursor.fetchone.side_effect = [
            {"id": 100, "content": "原始资料"},
            {"id": 101, "thread_id": 7, "content": "原始资料", "raw_evidence_id": 100},
        ]
        cursor.__enter__.return_value = cursor
        conn.cursor.return_value = cursor
        claimed = {"id": 59, "thread_id": 7, "raw_evidence_id": 100, "user_message_id": 101, "status": "processing"}
        with patch("v2.processing.claim_processing_job", return_value=claimed), patch(
            "v2.processing.get_processing_job", return_value={"id": 59, "assistant_message_id": None}
        ), patch("v2.processing.learn_turn", side_effect=RuntimeError("temporary LLM failure")), patch(
            "v2.processing.complete_processing_job"
        ) as complete, patch("v2.processing.fail_processing_job") as fail:
            process_inbox_job(59, db_factory=lambda: DbContext(conn))

        complete.assert_not_called()
        fail.assert_called_once()
        self.assertEqual(fail.call_args.args[:2], (conn, 59))
        conn.rollback.assert_called_once()

    def test_completed_job_exposes_assistant_response(self):
        job = {
            "id": 56,
            "thread_id": 7,
            "raw_evidence_id": 100,
            "user_message_id": 101,
            "status": "completed",
            "attempts": 1,
            "assistant_message_id": 102,
            "assistant_message": "我已理解这批资料。",
            "assistant_message_type": "summary",
        }
        with patch("app.auth"), patch("app.db", return_value=DbContext()), patch(
            "app.get_processing_job", return_value=job
        ):
            result = app.v2_inbox_job(56, x_api_key="key")

        payload = result
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["assistant_message"]["content"], "我已理解这批资料。")
        self.assertEqual(payload["assistant_message"]["message_type"], "summary")

    def test_failed_job_can_be_retried_without_new_evidence(self):
        failed = {
            "id": 57,
            "thread_id": 7,
            "raw_evidence_id": 100,
            "user_message_id": 101,
            "status": "queued",
            "attempts": 1,
        }
        tasks = BackgroundTasks()
        with patch("app.auth"), patch("app.db", return_value=DbContext()), patch(
            "app.retry_inbox_job", return_value=failed
        ), patch("app.enqueue_inbox_job") as enqueue:
            result = app.v2_retry_inbox_job(57, tasks, x_api_key="key")

        self.assertEqual(result.status_code, 202)
        self.assertEqual(response_json(result)["job_id"], 57)
        enqueue.assert_not_called()
        self.assertEqual(len(tasks.tasks), 0)

    def test_same_idempotency_key_returns_same_job(self):
        job = {"id": 58, "thread_id": 7, "status": "queued"}
        tasks = BackgroundTasks()
        with patch("app.auth"), patch("app.db", return_value=DbContext()), patch(
            "app.enqueue_inbox_job", return_value=job
        ) as enqueue:
            first = app.v2_inbox_message(
                app.V2InboxMessageIn(content="同一份资料"), tasks, x_api_key="key", idempotency_key="request-58"
            )
            second = app.v2_inbox_message(
                app.V2InboxMessageIn(content="同一份资料"), tasks, x_api_key="key", idempotency_key="request-58"
            )

        self.assertEqual(response_json(first), response_json(second))
        self.assertEqual(enqueue.call_count, 2)
        self.assertEqual(
            [call.kwargs["idempotency_key"] for call in enqueue.call_args_list],
            ["request-58", "request-58"],
        )


if __name__ == "__main__":
    unittest.main()
