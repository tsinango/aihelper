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
        self.assertNotIn("CREATE TABLE IF NOT EXISTS dimension", migration.casefold())
        self.assertNotIn("CREATE TABLE IF NOT EXISTS ontology", migration.casefold())

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
