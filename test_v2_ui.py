import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
PAGES = ("inbox.html", "knowledge.html", "documents.html", "chat.html")
LEARNING_REPLY_PAGES = ("inbox.html",)
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
        for page in LEARNING_REPLY_PAGES:
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

    def test_main_nav_links_to_chat_from_every_page(self):
        for page in PAGES:
            content = (ROOT / "templates" / page).read_text()
            self.assertIn('href="/chat"', content, page)
        chat = (ROOT / "templates" / "chat.html").read_text()
        self.assertIn('class="active" href="/chat"', chat)

    def test_chat_correction_loop_contract(self):
        content = (ROOT / "templates" / "chat.html").read_text()
        # Correction widgets: edit, kind, submit, explicit confirm, retest,
        # before/after compare card, human verdict buttons, source check.
        for marker in (
            "correct-toggle", "correct-text", "correct-kind",
            "correct-submit", "confirm-submit", "retest-submit",
            "retest-card", "verdict-pass", "verdict-fail",
            "feedback-history", "feedback-line", "check-sources",
        ):
            self.assertIn(marker, content)
        for kind in (
            "reply_only", "save_experience", "missing_information",
            "retrieval_failure", "generation_failure",
            "field_result_success", "field_result_failure",
        ):
            self.assertIn(kind, content)
        for endpoint in (
            "/feedback", "/confirm", "/retest", "/verdict",
        ):
            self.assertIn(endpoint, content)
        # Still exactly one learning entry: forwarding material to the Inbox.
        self.assertEqual(content.count("/api/v2/inbox/messages"), 1)

    def test_inbox_gap_queue_contract(self):
        content = (ROOT / "templates" / "inbox.html").read_text()
        self.assertIn("gaps-toggle", content)
        self.assertIn("drawer-title", content)
        self.assertIn("/api/v2/feedback/unresolved", content)
        self.assertIn("/close", content)

    def test_document_learning_contract(self):
        content = (ROOT / "templates" / "documents.html").read_text()
        for marker in (
            "learn-start", "proposals", "confirmProposal", "ordered_steps",
            "/learn", "/proposals", "/document-proposals/", "/confirm",
            "coverage", "with_destination",
        ):
            self.assertIn(marker, content)

    def test_knowledge_unit_rendering_contract(self):
        content = (ROOT / "templates" / "knowledge.html").read_text()
        for marker in ("showUnit", "ordered_steps", "origin_document_version_id"):
            self.assertIn(marker, content)

    def test_documents_upload_and_structure_contract(self):
        content = (ROOT / "templates" / "documents.html").read_text()
        for marker in (
            "doc-key", "doc-label", "doc-title", "doc-auth",
            "upload-status", "versions", "version-detail", "blocks",
            "parse_job", "download", "job-retry",
        ):
            self.assertIn(marker, content)
        self.assertIn("/api/v2/documents", content)
        self.assertIn("/api/v2/document-jobs/", content)
        # Legacy read-only table stays on the page.
        self.assertIn("没有找到资料", content)

    def test_chat_is_a_read_only_internal_qa_page(self):
        content = (ROOT / "templates" / "chat.html").read_text()
        # Questions go to the answer service, never to learning ingestion.
        self.assertIn("api('/api/v2/answers'", content)
        self.assertIn("Idempotency-Key", content)
        self.assertIn("answer_status", content)
        self.assertIn("needs_clarification", content)
        self.assertIn("clarifying_question", content)
        self.assertIn("service_error", content)
        self.assertIn("citations", content)
        self.assertIn("Internal engineer draft", content)
        # Exactly one learning entry remains: forwarding material to the Inbox.
        self.assertEqual(content.count("/api/v2/inbox/messages"), 1)
        self.assertIn("转入 Inbox 学习", content)

    def test_api_key_is_a_compact_session_storage_setting(self):
        for page in KEY_SETTINGS_PAGES:
            content = (ROOT / "templates" / page).read_text()
            self.assertIn('id="key-settings"', content, page)
            self.assertIn('id="save-key"', content, page)
            self.assertIn("sessionStorage.setItem('x-api-key'", content, page)
            self.assertIn("请先在设置中保存 API Key", content, page)
            self.assertNotIn('<label class="keybox"', content, page)

    def test_knowledge_page_has_maintenance_lite_controls(self):
        content = (ROOT / "templates" / "knowledge.html").read_text()
        for label in ("Active", "Deleted", "编辑", "删除", "恢复", "来源", "历史", "归属 Entity"):
            self.assertIn(label, content)
        for contract in (
            "method:'PATCH'",
            "method:'DELETE'",
            "/restore",
            "/sources",
            "/history",
            "/api/v2/entities/${encodeURIComponent(node.id)}/prune",
            "删除空分支",
            "node.prune_allowed",
            "textarea.value=String(item.content||'')",
            "entity_id:select.value",
        ):
            self.assertIn(contract, content, contract)

    def test_entity_tree_summary_selects_without_blocking_details_toggle(self):
        content = (ROOT / "templates" / "knowledge.html").read_text()
        self.assertIn("summary.dataset.entityId=String(node.id)", content)
        self.assertIn("summary.onclick=()=>{orgSelect(node.id)}", content)
        self.assertNotIn("summary.onclick=event=>{event.preventDefault();orgSelect(node.id)}", content)
        self.assertIn("function orgMarkSelected()", content)
        self.assertIn("orgState.selected=entityId;orgMarkSelected()", content)
        self.assertNotIn("orgState.selected=entityId;orgTree(orgState.tree)", content)

    def test_empty_branch_confirmation_counts_descendants_not_self(self):
        content = (ROOT / "templates" / "knowledge.html").read_text()
        self.assertIn("function orgDescendantCount(node)", content)
        self.assertIn("Math.max(0,total-1)", content)
        self.assertIn("const descendantCount=orgDescendantCount(node)", content)
        self.assertIn("其下还有 ${descendantCount} 个空节点", content)

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
