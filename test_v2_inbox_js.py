import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parent

# A tiny fake DOM + fetch harness that executes the real inline script from
# templates/inbox.html and drives the chat flow end to end. It verifies the
# reliability contract that string-level tests cannot: no-store fetches,
# thinking indicator lifecycle, POST -> GET thread refresh, error bubble +
# retry, duplicate-send guard, clarification quick replies, and resilience
# when optional elements are missing.
HARNESS = r"""
const assert = require('assert');
const source = process.argv[1];

function makeStorage() {
  const map = new Map();
  return {
    getItem: key => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => { map.set(key, String(value)); },
    removeItem: key => { map.delete(key); },
  };
}
globalThis.sessionStorage = makeStorage();
globalThis.localStorage = makeStorage();
sessionStorage.setItem('x-api-key', 'test-key');

function makeEl(tag) {
  const el = {
    tagName: tag || 'div', children: [], _listeners: {}, style: {}, dataset: {},
    value: '', hidden: false, disabled: false, open: false, scrollHeight: 100, scrollTop: 0,
    _text: '', _html: '', className: '',
    classList: (() => {
      const set = new Set();
      return {
        add: (...cls) => cls.forEach(name => set.add(name)),
        remove: (...cls) => cls.forEach(name => set.delete(name)),
        toggle: (name, force) => {
          const on = force === undefined ? !set.has(name) : Boolean(force);
          if (on) set.add(name); else set.delete(name);
        },
        contains: name => set.has(name),
      };
    })(),
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = [...kids]; },
    addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); },
    removeEventListener() {},
    querySelector() { return makeEl('span'); },
    querySelectorAll() { return []; },
    setAttribute() {}, getAttribute() { return null; },
    focus() { this.focused = true; },
    requestSubmit() { (this._listeners.submit || []).forEach(fn => fn({ preventDefault() {} })); },
    click() { (this._listeners.click || []).forEach(fn => fn({ preventDefault() {} })); },
    dispatchEvent(event) { (this._listeners[event.type] || []).forEach(fn => fn(event)); return true; },
  };
  Object.defineProperty(el, 'textContent', {
    get() { return this._text; },
    set(value) { this._text = String(value); },
  });
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._html; },
    set(value) { this._html = String(value); this.children = []; },
  });
  return el;
}

const byId = {};
globalThis.document = {
  getElementById(id) { return byId[id] || (byId[id] = makeEl('div#' + id)); },
  createElement(tag) { return makeEl(tag); },
  addEventListener() {},
  body: makeEl('body'),
};
// Elements that carry a `hidden` attribute in the real markup start hidden.
for (const id of ['page-error', 'thinking', 'chat-error', 'drawer-backdrop']) {
  const el = makeEl('div#' + id);
  el.hidden = true;
  byId[id] = el;
}

const calls = [];
globalThis.fetch = (url, options = {}) => {
  const call = { url: String(url), options: options || {}, done: false };
  calls.push(call);
  return new Promise((resolve, reject) => {
    call.resolve = response => { call.done = true; resolve(response); };
    call.reject = error => { call.done = true; reject(error); };
  });
};
const jsonResponse = (data, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => data,
});
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const pending = () => calls.filter(call => !call.done);
const answer = (call, data, status = 200) => call.resolve(jsonResponse(data, status));
const answerAll = data => pending().forEach(call => answer(call, data));
const posts = () => calls.filter(call => call.options.method === 'POST');

let failures = 0;
const ok = (condition, message) => {
  if (condition) { console.log('ok - ' + message); }
  else { failures += 1; console.error('FAIL: ' + message); }
};
const quickReplyTexts = () => {
  const rows = byId['messages'].children;
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    const wrap = rows[i].children && rows[i].children[0];
    if (!wrap) continue;
    const replies = (wrap.children || []).find(child => child.className === 'quick-replies');
    if (replies) return replies.children.map(button => button._text);
  }
  return null;
};

const exportLine = ';globalThis.__inboxTest = { state, sendMessage, openThread, loadInbox, renderMessages, quickReplyLabels, pollJob, resumeActiveJob, clearPollTimer, showThinking, hideThinking };';
eval(source + exportLine);
const T = globalThis.__inboxTest;

const THREAD_SNAPSHOT = {
  summary: { knowledge_count: 86, week_new_count: 12, pending_count: 5, unresolved_gap_count: 0 },
  threads: [{ id: 7, preview: 'hello', updated_at: '2026-09-04T10:00:00', message_count: 2 }],
};

(async () => {
  await sleep(0);
  ok(calls.length === 1 && calls[0].url === '/api/v2/inbox', 'boot loads inbox snapshot');
  ok(calls.every(call => call.options.cache === 'no-store'), 'every fetch uses no-store');
  answer(calls[0], THREAD_SNAPSHOT);
  await sleep(0);
  ok(calls.length === 2 && calls[1].url === '/api/v2/inbox/threads/7', 'opens the latest thread');
  answer(calls[1], {
    thread: { id: 7 },
    messages: [
      { role: 'user', content: 'hi' },
      { role: 'assistant', message_type: 'question', content: '我理解为：X。对吗？' },
    ],
  });
  await sleep(0);
  ok(localStorage.getItem('inbox-thread-id') === '7', 'current thread persisted across reloads');
  ok(JSON.stringify(quickReplyTexts()) === JSON.stringify(['对', '不对，我来修正', '不知道', '跳过']),
    'confirmation question offers confirm/correct/unknown/skip');
  ok(JSON.stringify(T.quickReplyLabels('batch_confirmation')) === JSON.stringify(['全部确认明确知识', '查看明细', '我来修正']),
    'bulk confirmation offers one-click review controls');
  ok(byId['count-week']._text === '+12', 'week count renders with + prefix');

  // Successful send: optimistic message, short 202 submit, job polling, then
  // one explicit GET refresh when the durable job is complete.
  byId['message-input'].value = '新知识';
  byId['composer']._listeners.submit[0]({ preventDefault() {} });
  await sleep(0);
  ok(posts().length === 1, 'submit issues exactly one POST');
  ok(posts()[0].options.cache === 'no-store', 'POST uses no-store');
  ok(JSON.parse(posts()[0].options.body).content === '新知识', 'POST body carries the message');
  ok(String(JSON.parse(posts()[0].options.body).thread_id) === '7', 'POST keeps the current thread');
  const firstRequestKey = posts()[0].options.headers['Idempotency-Key'];
  ok(Boolean(firstRequestKey), 'POST sends an Idempotency-Key');
  ok(byId['thinking'].hidden === false, 'thinking indicator shown while waiting');
  ok(byId['thinking-text']._text.includes('已收到') || byId['thinking-text']._text.includes('正在理解'), 'thinking text explains the wait');
  ok(byId['send-button'].disabled === true, 'send button disabled while sending');
  ok(quickReplyTexts() === null, 'quick replies hidden while sending');
  answer(posts()[0], {thread_id: 7, job_id: 101, status: 'queued'});
  await sleep(0);
  const jobCall = pending().find(call => call.url === '/api/v2/inbox/jobs/101');
  ok(Boolean(jobCall), '202 response starts polling the durable job');
  answer(jobCall, {job_id: 101, thread_id: 7, status: 'queued'});
  await sleep(0);
  ok(T.state.activeJob.status === 'queued', 'queued job remains active');
  ok(T.state.pollTimer !== 0, 'queued job schedules the next poll');
  T.pollJob(T.state.activeJob);
  await sleep(0);
  const processingJobCall = pending().find(call => call.url === '/api/v2/inbox/jobs/101');
  ok(Boolean(processingJobCall), 'queued job continues polling');
  answer(processingJobCall, {job_id: 101, thread_id: 7, status: 'processing'});
  await sleep(0);
  ok(T.state.activeJob.status === 'processing', 'processing job remains active');
  ok(T.state.pollTimer !== 0, 'processing job continues polling');
  ok(byId['send-button'].disabled === true && byId['message-input'].disabled === true, 'composer stays disabled while job is active');
  T.state.thinkingStartedAt = Date.now() - 10001;
  T.showThinking('processing');
  ok(byId['thinking-text']._text === '正在查看已有知识，还需要一点时间……', 'long processing gets the slow-state message');
  T.pollJob(T.state.activeJob);
  await sleep(0);
  const completedJobCall = pending().find(call => call.url === '/api/v2/inbox/jobs/101');
  answer(completedJobCall, {job_id: 101, thread_id: 7, status: 'completed'});
  await sleep(0);
  const refreshCall = pending().find(call => call.url === '/api/v2/inbox/threads/7');
  ok(Boolean(refreshCall), 'thread is refreshed after the job completes');
  answer(refreshCall, {
    thread: { id: 7 },
    messages: [
      { role: 'user', content: 'hi' },
      { role: 'assistant', message_type: 'question', content: '我理解为：X。对吗？' },
      { role: 'user', content: '新知识' },
      { role: 'assistant', message_type: 'question', content: '我理解为：Y。对吗？' },
    ],
  });
  await sleep(0);
  await sleep(0);
  ok(byId['thinking'].hidden === true, 'thinking indicator hidden after reply');
  ok(byId['send-button'].disabled === false, 'send button re-enabled after reply');
  ok(byId['message-input'].value === '', 'input cleared after send');
  const qr = quickReplyTexts();
  ok(qr && qr.includes('对') && qr.includes('不对，我来修正'), 'quick replies reappear after AI question');

  // Failure path: error bubble, re-enabled button, retry resends the same content.
  byId['message-input'].value = '失败内容';
  byId['composer']._listeners.submit[0]({ preventDefault() {} });
  await sleep(0);
  const failedPost = posts().pop();
  const failedRequestKey = failedPost.options.headers['Idempotency-Key'];
  failedPost.reject(new TypeError('network down'));
  await sleep(0);
  ok(byId['chat-error'].hidden === false, 'chat error bubble shown on failure');
  ok(byId['chat-error-text']._text.includes('连接暂时中断'), 'network failure surfaces a reconnecting message');
  ok(byId['send-button'].disabled === false, 'send button re-enabled after failure');
  ok(T.state.failedContent === '失败内容', 'failed content kept for retry');
  const postCount = posts().length;
  byId['chat-error-retry']._listeners.click[0]({ preventDefault() {} });
  await sleep(0);
  ok(posts().length === postCount + 1, 'retry issues a new POST');
  const retryPost = posts().pop();
  ok(JSON.parse(retryPost.options.body).content === '失败内容', 'retry resends the failed content');
  ok(retryPost.options.headers['Idempotency-Key'] === failedRequestKey, 'transport retry reuses the same Idempotency-Key');
  answer(retryPost, {thread_id: 7, job_id: 102, status: 'queued'});
  await sleep(0);
  const retryJobGet = pending().find(call => call.url === '/api/v2/inbox/jobs/102');
  answer(retryJobGet, {job_id: 102, thread_id: 7, status: 'completed'});
  await sleep(0);
  const retryGet = pending().find(call => call.url === '/api/v2/inbox/threads/7');
  answer(retryGet, {
    thread: { id: 7 },
    messages: [
      { role: 'user', content: '失败内容' },
      { role: 'assistant', message_type: 'text', content: '好。' },
    ],
  });
  await sleep(0);
  await sleep(0);
  ok(byId['chat-error'].hidden === true, 'chat error cleared after retry succeeds');
  const dupRows = byId['messages'].children.filter(row => {
    const bubble = row.children[0] && row.children[0].children[1];
    return bubble && bubble._text === '失败内容';
  });
  ok(dupRows.length === 1, 'retry does not duplicate the optimistic message');

  // A real job failure is retried through the job endpoint only; the raw
  // /messages submission is never posted a second time.
  T.sendMessage('会失败的资料');
  await sleep(0);
  const failedJobSubmit = posts().filter(call => call.url === '/api/v2/inbox/messages').pop();
  answer(failedJobSubmit, {thread_id: 9, job_id: 105, status: 'queued'});
  await sleep(0);
  const failedJobPoll = pending().find(call => call.url === '/api/v2/inbox/jobs/105');
  answer(failedJobPoll, {job_id: 105, thread_id: 9, status: 'failed', error_message: 'OpenRouter timeout'});
  await sleep(0);
  ok(byId['chat-error-text']._text === 'OpenRouter timeout', 'failed job shows the actual job failure');
  ok(T.state.activeJob === null && byId['send-button'].disabled === false, 'failed job stops polling and re-enables composer');
  const messagePostCountBeforeJobRetry = calls.filter(call => call.url === '/api/v2/inbox/messages').length;
  byId['chat-error-retry']._listeners.click[0]({ preventDefault() {} });
  await sleep(0);
  const jobRetryPost = pending().find(call => call.url === '/api/v2/inbox/jobs/105/retry');
  ok(Boolean(jobRetryPost), 'failed Retry posts to the existing job endpoint');
  ok(calls.filter(call => call.url === '/api/v2/inbox/messages').length === messagePostCountBeforeJobRetry,
    'failed Retry does not repost the original input');
  answer(jobRetryPost, {job_id: 105, thread_id: 9, status: 'queued'});
  await sleep(0);
  const retriedJobPoll = pending().find(call => call.url === '/api/v2/inbox/jobs/105');
  answer(retriedJobPoll, {job_id: 105, thread_id: 9, status: 'completed'});
  await sleep(0);
  answer(pending().find(call => call.url === '/api/v2/inbox/threads/9'), {thread: {id: 9}, messages: []});
  await sleep(0);

  // Duplicate-send guard.
  T.state.lastMessages = [];
  T.sendMessage('dup-a');
  T.sendMessage('dup-b');
  await sleep(0);
  const dupPosts = posts().filter(call => call.url === '/api/v2/inbox/messages').filter(call => {
    const body = JSON.parse(call.options.body);
    return body.content === 'dup-a' || body.content === 'dup-b';
  });
  ok(dupPosts.length === 1, 'second send is blocked while one is in flight');
  answer(dupPosts[0], {thread_id: 9, job_id: 103, status: 'queued'});
  await sleep(0);
  answer(pending().find(call => call.url === '/api/v2/inbox/jobs/103'), {job_id: 103, thread_id: 9, status: 'completed'});
  await sleep(0);
  answer(pending().find(call => call.url === '/api/v2/inbox/threads/9'), {thread: {id: 9}, messages: []});
  await sleep(0);

  // Clarification state gets answer-oriented quick replies.
  T.renderMessages({ messages: [{ role: 'assistant', message_type: 'clarification', content: '具体指哪一款型号？' }] });
  ok(JSON.stringify(quickReplyTexts()) === JSON.stringify(['我来回答', '不知道', '跳过']),
    'clarification offers answer/unknown/skip without a bare confirm');

  // sendMessage must not crash when optional elements are missing.
  const realGetElementById = document.getElementById;
  document.getElementById = id => (id === 'thinking' ? null : realGetElementById(id));
  let crashed = null;
  try {
    T.sendMessage('safety-check');
    await sleep(0);
  } catch (error) {
    crashed = error;
  }
  ok(!crashed && posts().filter(call => call.url === '/api/v2/inbox/messages').some(call => JSON.parse(call.options.body).content === 'safety-check'),
    'sendMessage survives a missing thinking element');
  const safetyPost = posts().filter(call => call.url === '/api/v2/inbox/messages').filter(call => JSON.parse(call.options.body).content === 'safety-check').pop();
  answer(safetyPost, {thread_id: 9, job_id: 104, status: 'queued'});
  await sleep(0);
  answer(pending().find(call => call.url === '/api/v2/inbox/jobs/104'), {job_id: 104, thread_id: 9, status: 'completed'});
  await sleep(0);
  answer(pending().find(call => call.url === '/api/v2/inbox/threads/9'), {thread: {id: 9}, messages: []});
  await sleep(0);
  document.getElementById = realGetElementById;

  // A refreshed thread advertises unfinished durable work and resumes it;
  // a temporary polling outage never creates another POST.
  T.openThread(9);
  await sleep(0);
  const refreshedWithJob = pending().find(call => call.url === '/api/v2/inbox/threads/9');
  answer(refreshedWithJob, {thread: {id: 9}, messages: [], jobs: [{job_id: 200, thread_id: 9, status: 'processing'}]});
  await sleep(0);
  ok(T.state.activeJob && T.state.activeJob.job_id === 200, 'refresh resumes an unfinished processing job');
  ok(byId['thinking'].hidden === false, 'refresh shows processing state');
  const postsBeforePollFailure = posts().length;
  T.pollJob(T.state.activeJob);
  await sleep(0);
  const failedPoll = pending().find(call => call.url === '/api/v2/inbox/jobs/200');
  failedPoll.reject(new TypeError('temporary network outage'));
  await sleep(0);
  ok(byId['chat-error-text']._text.includes('连接暂时中断'), 'temporary polling failure shows reconnecting state');
  ok(posts().length === postsBeforePollFailure, 'temporary polling failure does not create duplicate input');
  T.clearPollTimer();
  T.state.activeJob = null;
  T.hideThinking();

  // Drain any fire-and-forget refresh fetches so their timeout timers clear.
  answerAll({ summary: {}, threads: [] });
  await sleep(0);
  answerAll({ summary: {}, threads: [] });
  await sleep(0);

  if (failures) {
    console.error(failures + ' check(s) failed');
    process.exit(1);
  }
  console.log('ALL PASS');
})().catch(error => {
  console.error('HARNESS ERROR', error);
  process.exit(2);
});
"""


class V2InboxJsTest(unittest.TestCase):
    def test_inbox_inline_script_chat_flow(self):
        if not shutil.which("node"):
            self.skipTest("node is not installed")
        content = (ROOT / "templates" / "inbox.html").read_text()
        match = re.search(r"<script>([\s\S]*?)</script>", content)
        self.assertIsNotNone(match, "inbox.html must contain an inline script")
        completed = subprocess.run(
            ["node", "-e", HARNESS, match.group(1)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("ALL PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
