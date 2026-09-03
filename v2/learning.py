"""The small, text-only V2 learning loop.

The database stores raw input, a provisional interpretation and one atomic
proposal at a time.  The only operation that changes a Knowledge row to
``user_confirmed`` is an explicit confirmation reply to the assistant's
recap.  This module is intentionally independent from the V1 review code.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from llm import parse_json_response

from v2.service import (
    V2NotFound,
    create_thread,
    get_thread,
    json_safe,
    thread_response,
)
from psycopg.types.json import Jsonb


log = logging.getLogger("ai-sales-engineer.v2.learning")

TRUST_VALUES = frozenset({
    "official_source", "user_confirmed", "provisional", "conflicted",
})
ANSWER_TRUST_VALUES = frozenset({"official_source", "user_confirmed"})

CONFIRM_WORDS = frozenset({
    "对", "对的", "正确", "没错", "是的", "确认", "是", "yes", "true", "y",
})
UNKNOWN_WORDS = frozenset({"不知道", "不清楚", "不确定", "не знаю", "неизвестно"})
SKIP_WORDS = frozenset({"跳过", "以后再说", "稍后再说", "先不说", "skip", "later"})

UNDERSTANDING_SYSTEM_PROMPT = """
你是一个产品知识学习助理。你正在和产品专家聊天，不是在维护数据库。
从用户刚刚提供的文字中提炼一个或多个原子事实，返回严格 JSON：
{"facts":[{"title":"简短标题","content":"一个可独立确认的事实","entity_name":"明确型号或产品名，没有就空字符串"}]}

硬规则：
- 每个 facts 项只能包含一个事实；型号、系列范围、hardware revision、firmware、条件、例外和否定事实必须拆成不同项。
- 不要把单型号推广到系列，不要推断用户没有写出的范围。
- 手册或文字没有提到某功能，不等于不支持；只有明确写出不支持或用户明确确认，才可表达否定。
- “大概”“可能”“应该”等不确定说法仍然只能作为待确认理解。
- 不要回答客户问题，不要使用训练知识，不要询问数据库字段或技术实现。
- 只返回 JSON，不要 markdown，不要解释。
""".strip()


def _clean(value: Any, limit: int = 12000) -> str:
    return str(value or "").strip()[:limit]


def classify_reply(content: str) -> str:
    """Classify only exact short control replies; everything else is a correction."""

    normalized = re.sub(r"[\s。.!！?？,，]+", "", _clean(content, 200).casefold())
    if normalized in {item.casefold() for item in CONFIRM_WORDS}:
        return "confirm"
    if normalized in {item.casefold() for item in UNKNOWN_WORDS}:
        return "unknown"
    if normalized in {item.casefold() for item in SKIP_WORDS}:
        return "skip"
    return "correction"


def _entity_name(text: str) -> str:
    # Keep this deliberately conservative.  An identifier is context, not a
    # license to infer a family or a broader product scope.
    match = re.search(r"(?=[A-Za-z0-9./()\-]*\d)[A-Za-z0-9][A-Za-z0-9./()\-]{2,}", text)
    return match.group(0).upper() if match else ""


def _fallback_facts(content: str) -> list[dict]:
    return [{
        "title": "待确认的产品信息",
        "content": content,
        "entity_name": _entity_name(content),
    }]


def _model_facts(content: str, llm_service=None, context: list[dict] | None = None) -> tuple[list[dict], bool]:
    if llm_service is None:
        return _fallback_facts(content), True
    messages = [{"role": "system", "content": UNDERSTANDING_SYSTEM_PROMPT}]
    if context:
        messages.append({
            "role": "user",
            "content": "此前待确认理解：\n" + "\n".join(_clean(item.get("fact_text")) for item in context),
        })
    messages.append({"role": "user", "content": content})
    try:
        parsed = parse_json_response(llm_service.extract(messages, max_tokens=800))
        raw_facts = parsed.get("facts") if isinstance(parsed, dict) else None
        if not isinstance(raw_facts, list):
            raw_facts = [parsed] if isinstance(parsed, dict) and parsed.get("content") else []
        facts = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_content = _clean(item.get("content") or item.get("fact") or item.get("text"))
            if not fact_content:
                continue
            facts.append({
                "title": _clean(item.get("title"), 500) or fact_content[:120],
                "content": fact_content,
                "entity_name": _clean(item.get("entity_name") or item.get("entity"), 500),
            })
        if facts:
            return facts[:20], False
    except Exception:
        log.exception("V2 learning understanding failed; keeping a provisional raw interpretation")
    return _fallback_facts(content), True


def _insert_evidence(conn, content: str, thread_id: int, *, label: str = "Inbox") -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_raw_evidence(
                evidence_type, author_role, content, raw_payload,
                source_label, source_locator
            )
            VALUES('user_input', 'product_expert', %s, %s, %s, %s)
            RETURNING id, evidence_type, author_role, content, raw_payload,
                      source_label, source_locator, created_at
            """,
            (
                content,
                Jsonb({"thread_id": thread_id, "channel": "inbox"}),
                label,
                f"v2-thread:{thread_id}",
            ),
        )
        return dict(cur.fetchone())


def _insert_message(
    conn,
    thread_id: int,
    role: str,
    message_type: str,
    content: str,
    raw_evidence_id: int | None = None,
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_messages(
                thread_id, sequence_no, role, message_type, content, raw_evidence_id
            )
            VALUES(
                %s,
                (SELECT COALESCE(MAX(sequence_no), 0) + 1
                 FROM v2_inbox_messages WHERE thread_id=%s),
                %s, %s, %s, %s
            )
            RETURNING id, thread_id, sequence_no, role, message_type, content,
                      raw_evidence_id, created_at
            """,
            (thread_id, thread_id, role, message_type, content, raw_evidence_id),
        )
        row = dict(cur.fetchone())
        cur.execute(
            "UPDATE v2_inbox_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (thread_id,),
        )
    return row


def _ensure_session(conn, thread_id: int, question_budget: int) -> dict:
    budget = max(0, int(question_budget))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, session_type, status, question_budget,
                   questions_asked, summary, started_at, completed_at,
                   created_at, updated_at
            FROM v2_learning_sessions
            WHERE thread_id=%s AND status='active'
            ORDER BY id DESC LIMIT 1
            """,
            (thread_id,),
        )
        current = cur.fetchone()
        if current:
            return dict(current)
        cur.execute(
            """
            INSERT INTO v2_learning_sessions(
                thread_id, session_type, status, question_budget
            )
            VALUES(%s, 'passive', 'active', %s)
            RETURNING id, thread_id, session_type, status, question_budget,
                      questions_asked, summary, started_at, completed_at,
                      created_at, updated_at
            """,
            (thread_id, budget),
        )
        return dict(cur.fetchone())


def _pending_proposal(conn, thread_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, source_message_id, question_message_id,
                   fact_text, entity_name, proposed_trust, status,
                   confirmed_knowledge_id, resolution_message_id,
                   created_at, updated_at
            FROM v2_learning_proposals
            WHERE thread_id=%s AND status='pending_confirmation'
            ORDER BY id LIMIT 1
            """,
            (thread_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _pending_context(conn, thread_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fact_text FROM v2_learning_proposals WHERE thread_id=%s AND status='pending_confirmation' ORDER BY id",
            (thread_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def _find_knowledge(conn, fact: dict) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, entity_name, trust, active,
                   created_at, updated_at
            FROM v2_knowledge
            WHERE active=TRUE AND lower(content)=lower(%s)
              AND lower(entity_name)=lower(%s)
            ORDER BY CASE trust
                WHEN 'official_source' THEN 0
                WHEN 'user_confirmed' THEN 1
                WHEN 'provisional' THEN 2
                ELSE 3 END, id
            LIMIT 1
            """,
            (fact["content"], fact.get("entity_name") or ""),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _create_knowledge(conn, fact: dict, *, trust: str = "provisional") -> dict:
    if trust not in TRUST_VALUES:
        raise ValueError("unknown V2 trust value")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge(title, content, entity_name, trust, active)
            VALUES(%s, %s, %s, %s, %s)
            RETURNING id, title, content, entity_name, trust, active,
                      created_at, updated_at
            """,
            (
                fact["title"], fact["content"], fact.get("entity_name") or "",
                trust, trust != "conflicted",
            ),
        )
        return dict(cur.fetchone())


def _link_source(
    conn,
    knowledge_id: int,
    evidence_id: int,
    *,
    source_kind: str,
    relation: str = "supports",
    source_role: str = "supporting",
    resolution: str = "accepted",
    excerpt: str = "",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge_sources(
                knowledge_id, raw_evidence_id, source_kind, relation,
                source_role, excerpt, active, resolution
            )
            VALUES(%s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT (knowledge_id, raw_evidence_id, relation) DO NOTHING
            """,
            (knowledge_id, evidence_id, source_kind, relation, source_role, excerpt[:4000], resolution),
        )


def _insert_proposal(conn, thread_id: int, message_id: int, evidence_id: int, fact: dict, knowledge_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_learning_proposals(
                thread_id, source_message_id, fact_text, entity_name,
                proposed_trust, status, confirmed_knowledge_id
            )
            VALUES(%s, %s, %s, %s, 'provisional', 'pending_confirmation', %s)
            RETURNING id, thread_id, source_message_id, question_message_id,
                      fact_text, entity_name, proposed_trust, status,
                      confirmed_knowledge_id, resolution_message_id,
                      created_at, updated_at
            """,
            (thread_id, message_id, fact["content"], fact.get("entity_name") or "", knowledge_id),
        )
        proposal = dict(cur.fetchone())
    _link_source(
        conn,
        knowledge_id,
        evidence_id,
        source_kind="user_input",
        source_role="primary",
        excerpt=fact["content"],
    )
    return proposal


def _session_question_count(conn, session_id: int) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT questions_asked, question_budget FROM v2_learning_sessions WHERE id=%s", (session_id,))
        row = cur.fetchone()
    return int(row["questions_asked"]), int(row["question_budget"])


def _ask(conn, proposal: dict, session: dict) -> tuple[dict | None, str | None]:
    asked, budget = _session_question_count(conn, int(session["id"]))
    if asked >= budget:
        return None, None
    text = f"我理解为：{proposal['fact_text']}。对吗？"
    message = _insert_message(conn, int(proposal["thread_id"]), "assistant", "question", text)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET question_message_id=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (message["id"], proposal["id"]),
        )
        cur.execute(
            """
            UPDATE v2_learning_sessions
            SET questions_asked=questions_asked + 1, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (session["id"],),
        )
    return message, text


def _update_proposal(conn, proposal_id: int, status: str, *, knowledge_id: int | None = None, message_id: int | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET status=%s, confirmed_knowledge_id=COALESCE(%s, confirmed_knowledge_id),
                resolution_message_id=COALESCE(%s, resolution_message_id),
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (status, knowledge_id, message_id, proposal_id),
        )


def _confirm(conn, proposal: dict, evidence: dict, message: dict) -> dict:
    knowledge_id = proposal.get("confirmed_knowledge_id")
    if not knowledge_id:
        raise ValueError("pending proposal has no provisional Knowledge")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET trust='user_confirmed', active=TRUE, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            RETURNING id, title, content, entity_name, trust, active,
                      created_at, updated_at
            """,
            (knowledge_id,),
        )
        knowledge = dict(cur.fetchone())
    _link_source(
        conn,
        int(knowledge_id),
        int(evidence["id"]),
        source_kind="user_confirmation",
        source_role="primary",
        excerpt=evidence["content"],
    )
    _update_proposal(conn, int(proposal["id"]), "confirmed", knowledge_id=int(knowledge_id), message_id=int(message["id"]))
    return knowledge


def _summary(conn, thread_id: int, session: dict) -> tuple[dict | None, str | None]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT k.title, k.content
            FROM v2_learning_proposals p
            JOIN v2_knowledge k ON k.id=p.confirmed_knowledge_id
            WHERE p.thread_id=%s AND p.status='confirmed'
              AND p.updated_at >= %s
            ORDER BY p.id
            """,
            (thread_id, session["started_at"]),
        )
        facts = [dict(row) for row in cur.fetchall()]
    if not facts:
        return None, None
    lines = [f"今天我学会了 {len(facts)} 件事：", *[f"* {item['content']}" for item in facts]]
    text = "\n".join(lines)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_sessions
            SET status='completed', summary=%s, completed_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (text, session["id"]),
        )
    message = _insert_message(conn, thread_id, "assistant", "summary", text)
    return message, text


def _next_question(conn, thread_id: int, session: dict) -> tuple[dict | None, str | None, dict | None]:
    proposal = _pending_proposal(conn, thread_id)
    if not proposal:
        return None, None, None
    if proposal.get("question_message_id"):
        with conn.cursor() as cur:
            cur.execute("SELECT id, thread_id, sequence_no, role, message_type, content, raw_evidence_id, created_at FROM v2_inbox_messages WHERE id=%s", (proposal["question_message_id"],))
            message = cur.fetchone()
        return dict(message) if message else None, message["content"] if message else None, proposal
    message, text = _ask(conn, proposal, session)
    return message, text, proposal


def _result(conn, thread_id: int, *, status: str, message: dict | None = None, proposal: dict | None = None, fallback: bool = False, summary: str | None = None) -> dict:
    payload = {
        "thread_id": thread_id,
        "status": status,
        "message": message,
        "proposal": proposal,
        "summary": summary,
        "fallback": fallback,
        **thread_response(conn, thread_id),
    }
    return json_safe(payload)


def learn_turn(
    conn,
    content: str,
    *,
    thread_id: int | None = None,
    channel: str = "inbox",
    llm_service=None,
    question_budget: int = 5,
) -> dict:
    """Persist one expert turn and advance exactly one atomic confirmation."""

    clean = _clean(content)
    if not clean:
        raise ValueError("content must not be empty")
    if thread_id is None:
        thread = create_thread(conn, channel=channel, mode="learn")
    else:
        thread = get_thread(conn, int(thread_id))
    current_thread_id = int(thread["id"])
    session = _ensure_session(conn, current_thread_id, question_budget)
    pending = _pending_proposal(conn, current_thread_id)
    evidence = _insert_evidence(conn, clean, current_thread_id)
    reply_kind = classify_reply(clean) if pending else "new"
    message_type = {
        "confirm": "confirmation",
        "unknown": "unknown",
        "skip": "skip",
        "correction": "correction",
    }.get(reply_kind, "evidence")
    user_message = _insert_message(conn, current_thread_id, "user", message_type, clean, int(evidence["id"]))

    if pending and reply_kind == "confirm":
        knowledge = _confirm(conn, pending, evidence, user_message)
        next_message, next_text, next_proposal = _next_question(conn, current_thread_id, session)
        if next_message:
            return _result(conn, current_thread_id, status="awaiting_confirmation", message=next_message, proposal=next_proposal)
        summary_message, summary_text = _summary(conn, current_thread_id, session)
        return _result(conn, current_thread_id, status="confirmed", message=summary_message, summary=summary_text)

    if pending and reply_kind in {"unknown", "skip"}:
        _update_proposal(conn, int(pending["id"]), reply_kind)
        acknowledgement = (
            "好的，我先记下你现在不知道。以后有新的资料或上下文时再回来。"
            if reply_kind == "unknown" else
            "好的，先跳过这个问题。我不会反复追问。"
        )
        acknowledgement_message = _insert_message(conn, current_thread_id, "assistant", "text", acknowledgement)
        next_message, _, next_proposal = _next_question(conn, current_thread_id, session)
        if next_message:
            return _result(conn, current_thread_id, status="awaiting_confirmation", message=next_message, proposal=next_proposal)
        return _result(conn, current_thread_id, status=reply_kind, message=acknowledgement_message)

    facts, fallback = _model_facts(clean, llm_service, _pending_context(conn, current_thread_id) if pending else None)
    if pending:
        _update_proposal(conn, int(pending["id"]), "corrected", message_id=int(user_message["id"]))

    first_proposal = None
    for fact in facts:
        existing = _find_knowledge(conn, fact)
        if existing and existing["trust"] in ANSWER_TRUST_VALUES:
            _link_source(conn, int(existing["id"]), int(evidence["id"]), source_kind="user_input", source_role="supporting", excerpt=fact["content"])
            continue
        knowledge = existing or _create_knowledge(conn, fact, trust="provisional")
        proposal = _insert_proposal(conn, current_thread_id, int(user_message["id"]), int(evidence["id"]), fact, int(knowledge["id"]))
        if first_proposal is None:
            first_proposal = proposal

    if first_proposal is None:
        response = _insert_message(
            conn,
            current_thread_id,
            "assistant",
            "text",
            "这条信息和已有的已确认知识一致，我已保留这次原始来源。",
        )
        return _result(conn, current_thread_id, status="reused", message=response, fallback=fallback)

    question_message, _, active_proposal = _next_question(conn, current_thread_id, session)
    if not question_message:
        response = _insert_message(conn, current_thread_id, "assistant", "text", "我先记下这条待确认信息，稍后再问你。")
        return _result(conn, current_thread_id, status="waiting", message=response, proposal=active_proposal, fallback=fallback)
    return _result(conn, current_thread_id, status="awaiting_confirmation", message=question_message, proposal=active_proposal, fallback=fallback)
