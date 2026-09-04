import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
PAGES = ("inbox.html", "knowledge.html", "documents.html", "chat.html")


class V2UiTest(unittest.TestCase):
    def test_pages_use_v2_contract_and_preserve_key_flow(self):
        for page in PAGES:
            content = (ROOT / "templates" / page).read_text()
            self.assertIn("sessionStorage.getItem('x-api-key')", content, page)
            self.assertIn("x-api-key", content, page)
            self.assertRegex(content, r"/api/v2/(inbox|knowledge|documents)", page)
            self.assertIn("API Key", content, page)

    def test_inbox_reads_actual_summary_keys_and_handles_zero(self):
        content = (ROOT / "templates" / "inbox.html").read_text()
        self.assertIn("week_new_count", content)
        self.assertIn("unresolved_gap_count", content)
        self.assertIn("!= null", content)

    def test_list_and_source_rendering_are_safe_for_api_shapes(self):
        for page in PAGES:
            content = (ROOT / "templates" / page).read_text()
            self.assertIn("Array.isArray", content, page)
        for page in ("inbox.html", "chat.html"):
            content = (ROOT / "templates" / page).read_text()
            self.assertIn("sourceLabels", content, page)
        knowledge = (ROOT / "templates" / "knowledge.html").read_text()
        self.assertIn("Array.isArray(sources)", knowledge)
        self.assertIn("[sources]", knowledge)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_all_inline_scripts_parse(self):
        checker = (
            "const fs=require('fs');"
            "const source=fs.readFileSync(process.argv[1],'utf8');"
            "for(const match of source.matchAll(/<script>([\\s\\S]*?)<\\/script>/gi))"
            "new Function(match[1]);"
        )
        for page in PAGES:
            completed = subprocess.run(
                ["node", "-e", checker, str(ROOT / "templates" / page)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
