import json
import unittest
from pathlib import Path
from unittest.mock import patch

import app


ROOT = Path(__file__).parent


class _CursorContext:
    def __init__(self, row=None, error=None):
        self.row = row
        self.error = error

    def __enter__(self):
        if self.error:
            raise self.error
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_args, **_kwargs):
        return None

    def fetchone(self):
        return self.row


class _ConnectionContext:
    def __init__(self, row=None, error=None):
        self.cursor_context = _CursorContext(row=row, error=error)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cursor_context


class ProductionHardeningTest(unittest.TestCase):
    def test_health_is_only_web_process_liveness(self):
        with patch("app.db", side_effect=AssertionError("health must not query DB")):
            self.assertEqual(app.health(), {"status": "ok", "service": "aihelper"})

    def test_ready_reports_worker_unavailable_without_secrets(self):
        schema = {
            "questions": "questions", "v2_knowledge": "v2_knowledge", "jobs": "jobs",
            "workers": "workers", "entities": "v2_entities",
            "entity_relations": "v2_entity_relations",
            "knowledge_history": "v2_knowledge_history",
        }
        with patch("app.db", return_value=_ConnectionContext(row=schema)), patch(
            "app.worker_health", return_value={"worker_name": "aihelper-inbox-worker", "healthy": False}
        ), patch.object(app, "embedder", object()), patch.object(app, "llm", object()):
            response = app.ready()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body), {"ready": False, "reason": "inbox_worker_unavailable"})

    def test_ready_reports_database_failure_without_details(self):
        with patch("app.db", return_value=_ConnectionContext(error=RuntimeError("connection details"))):
            response = app.ready()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body), {"ready": False, "reason": "database_unavailable"})
        self.assertNotIn("connection details", response.body.decode())

    def test_new_units_use_aihelper_and_hardened_defaults(self):
        for name, description in (
            ("aihelper.service", "Description=aihelper FastAPI backend"),
            ("aihelper-inbox-worker.service", "Description=aihelper Inbox Worker"),
        ):
            unit = (ROOT / "deploy" / name).read_text()
            self.assertIn(description, unit)
            self.assertIn("User=ubuntu", unit)
            self.assertIn("Group=ubuntu", unit)
            self.assertIn("WorkingDirectory=/opt/aihelper", unit)
            self.assertIn("EnvironmentFile=/etc/aihelper.env", unit)
            self.assertIn("RestartSec=5", unit)
            self.assertIn("NoNewPrivileges=true", unit)
            self.assertIn("PrivateTmp=true", unit)

    def test_inbox_worker_unavailable_copy_and_polling_message_are_present(self):
        content = (ROOT / "templates" / "inbox.html").read_text()
        self.assertIn("后台知识处理服务暂时不可用", content)
        self.assertIn("内容已经保存，后台处理恢复后会继续", content)
        self.assertIn("worker_healthy", content)
        self.assertNotIn('href="/chat"', content)


if __name__ == "__main__":
    unittest.main()
