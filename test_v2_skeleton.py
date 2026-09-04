import unittest
from pathlib import Path
import re

import app


ROOT = Path(__file__).parent


class V2SkeletonTest(unittest.TestCase):
    def test_migration_is_additive_and_closes_trust_vocabulary(self):
        migration = (ROOT / "migrations" / "013_v2_skeleton.sql").read_text()
        for table in (
            "v2_raw_evidence",
            "v2_knowledge",
            "v2_knowledge_sources",
            "v2_inbox_threads",
            "v2_inbox_messages",
            "v2_learning_proposals",
            "v2_learning_sessions",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", migration)
        for trust in ("official_source", "user_confirmed", "provisional", "conflicted"):
            self.assertIn(trust, migration)
        self.assertIn("CHECK (trust <> 'conflicted' OR active = FALSE)", migration)
        self.assertIn("REFERENCES v2_knowledge(id) ON DELETE RESTRICT", migration)
        self.assertIn("relation = 'supports' AND source_role IN ('primary', 'supporting')", migration)
        self.assertIn("relation = 'contradicts' AND source_role = 'contradicting'", migration)
        self.assertIn("'active_inbox'", migration)
        self.assertIn("ux_v2_learning_sessions_active_thread", migration)
        self.assertIn("ux_v2_inbox_threads_external", migration)
        self.assertNotRegex(migration, r"(?i)\b(?:DROP|ALTER)\s+TABLE\b")
        self.assertNotIn("CREATE TABLE IF NOT EXISTS dimension", migration.casefold())
        self.assertNotIn("CREATE TABLE IF NOT EXISTS ontology", migration.casefold())

    def test_service_queries_match_thread_schema_and_are_idempotent(self):
        service = (ROOT / "v2" / "service.py").read_text()
        self.assertNotIn("FROM v2_inbox_threads\n            FROM v2_inbox_threads", service)
        self.assertIn("external_thread_id", service)
        self.assertIn("ON CONFLICT (origin, external_thread_id)", service)
        self.assertIn("WHERE external_thread_id IS NOT NULL", service)
        self.assertIn("RETURNING id, origin AS channel, status, thread_type AS mode", service)

    def test_inbox_processing_jobs_are_durable_and_idempotent(self):
        migration = (ROOT / "migrations" / "016_v2_inbox_processing_jobs.sql").read_text()
        self.assertIn("CREATE TABLE IF NOT EXISTS v2_inbox_processing_jobs", migration)
        for status in ("queued", "processing", "completed", "failed"):
            self.assertIn(f"'{status}'", migration)
        self.assertIn("idempotency_key TEXT NOT NULL UNIQUE", migration)
        self.assertIn("REFERENCES v2_raw_evidence(id) ON DELETE RESTRICT", migration)
        self.assertIn("REFERENCES v2_inbox_messages(id) ON DELETE RESTRICT", migration)

    def test_v2_pages_and_routes_exist_without_replacing_v1(self):
        for page in ("inbox.html", "knowledge.html", "documents.html", "chat.html"):
            self.assertTrue((ROOT / "templates" / page).is_file(), page)
        paths = {route.path for route in app.app.routes}
        for path in ("/inbox", "/knowledge", "/documents", "/chat", "/api/v2/inbox", "/api/v2/knowledge"):
            self.assertIn(path, paths)
        self.assertIn("/api/v1/query", paths)
        self.assertIn("/telegram/webhook", paths)
        self.assertIn("/review", paths)

    def test_v2_pages_keep_technical_workflow_out_of_user_copy(self):
        forbidden = ("candidate", "knowledge_key", "taxonomy", "publish", "review")
        for page in ("inbox.html", "knowledge.html", "documents.html", "chat.html"):
            content = (ROOT / "templates" / page).read_text().casefold()
            for word in forbidden:
                self.assertIsNone(re.search(rf"\\b{re.escape(word)}\\b", content), f"{word} leaked into {page}")


if __name__ == "__main__":
    unittest.main()
