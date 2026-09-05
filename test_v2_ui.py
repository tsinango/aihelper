import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
PAGES = ("inbox.html", "knowledge.html", "documents.html", "chat.html")
INTERACTIVE_PAGES = ("inbox.html", "chat.html")
KEY_SETTINGS_PAGES = PAGES


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

    def test_learning_quick_replies_only_follow_latest_assistant_question(self):
        for page in INTERACTIVE_PAGES:
            content = (ROOT / "templates" / page).read_text()
            for label in ("对", "查看并编辑", "不知道", "跳过"):
                self.assertIn(label, content, page)
            self.assertIn("quick-replies", content, page)
            self.assertIn("message_type", content, page)
            self.assertIn("clarification", content, page)
            self.assertIn("lastIndex", content, page)
            self.assertIn("label === '查看并编辑'", content, page)
            self.assertTrue("input.focus()" in content or "$('message-input').focus()" in content, page)
        self.assertIn("window.location.assign(`/inbox?thread=", (ROOT / "templates" / "chat.html").read_text())
        inbox = (ROOT / "templates" / "inbox.html").read_text()
        self.assertIn("/api/v2/inbox/threads/${encodeURIComponent(state.threadId)}/proposals", inbox)
        self.assertIn("/api/v2/inbox/proposals/${encodeURIComponent(proposal.id)}", inbox)
        self.assertIn("proposal-editor-close", inbox)

    def test_api_key_is_a_compact_session_storage_setting(self):
        for page in KEY_SETTINGS_PAGES:
            content = (ROOT / "templates" / page).read_text()
            self.assertIn('id="key-settings"', content, page)
            self.assertIn('id="save-key"', content, page)
            self.assertIn("sessionStorage.setItem('x-api-key'", content, page)
            self.assertIn("请先在设置中保存 API Key", content, page)
            self.assertNotIn('<label class="keybox"', content, page)

    def test_inbox_chat_reliability_contract(self):
        content = (ROOT / "templates" / "inbox.html").read_text()
        self.assertIn("cache: CACHE_MODE", content)
        self.assertIn("no-store", content)
        self.assertIn("AbortController", content)
        self.assertIn("SUBMIT_TIMEOUT_MS", content)
        self.assertIn("POLL_TIMEOUT_MS", content)
        self.assertIn("POLL_INTERVAL_MS", content)
        self.assertIn("Idempotency-Key", content)
        self.assertIn("连接暂时中断，正在重新连接", content)
        self.assertIn("正在理解你的内容", content)
        self.assertIn("请重试", content)
        self.assertIn('id="chat-error-retry"', content)
        self.assertIn("我来回答", content)
        self.assertIn("inbox-thread-id", content)
        self.assertIn("state.sending", content)
        self.assertIn("pointer: coarse", content)
        self.assertIn("matchMedia", content)
        self.assertIn("100dvh", content)
        self.assertIn("quickReplyLabels", content)
        self.assertIn('id="new-thread"', content)

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
