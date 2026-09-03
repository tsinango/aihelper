"""Persistence helpers for the Phase 1 V2 Inbox skeleton.

This module is intentionally boring: SQL is visible, transactions belong to
the caller, and the V2 tables are never mixed with V1 candidate/review tables.
"""

from __future__ import annotations

import json
from typing import Any

from psycopg.types.json import Jsonb


class V2NotFound(LookupError):
    """Raised when a requested V2 thread does not exist."""


def _dict(row: Any) -> dict:
    return dict(row) if row is not None else {}


def create_thread(conn, *, channel: str = "inbox", mode: str = "learn") -> dict:
    origin = {
        "inbox": "web",
        "chat": "web",
        "telegram": "telegram",
        "import": "import",
    }.get(channel, "web")
    thread_type = "learning" if mode == "learn" else "general"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_threads(thread_type, origin, status)
            VALUES(%s, %s, 'open')
            RETURNING id, origin AS channel, status, thread_type AS mode,
                      created_at, updated_at
            """,
            (thread_type, origin),
        )
        return _dict(cur.fetchone())


def get_thread(conn, thread_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, origin AS channel, status, thread_type AS mode,
                   created_at, updated_at
            FROM v2_inbox_threads
            WHERE id=%s
            """,
            (thread_id,),
        )
        row = cur.fetchone()
    if not row:
        raise V2NotFound(f"V2 thread {thread_id} was not found")
    return _dict(row)


def ensure_thread(conn, thread_id: int | None, *, channel: str = "inbox", mode: str = "learn") -> dict:
    return get_thread(conn, thread_id) if thread_id is not None else create_thread(conn, channel=channel, mode=mode)


def add_message(
    conn,
    *,
    thread_id: int,
    role: str,
    content: str,
    message_type: str,
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_messages(thread_id, sequence_no, role, content, message_type)
            VALUES(
                %s,
                (SELECT COALESCE(MAX(sequence_no), 0) + 1
                 FROM v2_inbox_messages WHERE thread_id=%s),
                %s, %s, %s
            )
            RETURNING id, thread_id, sequence_no, role, content, message_type,
                      raw_evidence_id, created_at
            """,
            (thread_id, thread_id, role, content, message_type),
        )
        return _dict(cur.fetchone())


def add_raw_evidence(
    conn,
    *,
    evidence_type: str,
    source_label: str,
    source_locator: str,
    content: str,
    metadata: dict | None = None,
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_raw_evidence(
                evidence_type, author_role, content, raw_payload,
                source_label, source_locator
            )
            VALUES(%s, 'product_expert', %s, %s, %s, %s)
            RETURNING id, evidence_type, author_role, content, raw_payload,
                      source_label, source_locator, created_at
            """,
            (evidence_type, content, Jsonb(metadata or {}), source_label, source_locator),
        )
        return _dict(cur.fetchone())


def append_user_message(
    conn,
    content: str,
    *,
    thread_id: int | None = None,
    channel: str = "inbox",
    mode: str = "learn",
) -> dict:
    """Persist a user turn and its immutable raw evidence.

    Phase 1 deliberately does not infer or confirm knowledge.  The returned
    status is only receipt acknowledgement; Phase 2 adds the learning reply.
    """

    clean = str(content or "").strip()
    if not clean:
        raise ValueError("content must not be empty")
    thread = ensure_thread(conn, thread_id, channel=channel, mode=mode)
    evidence = add_raw_evidence(
        conn,
        evidence_type="user_input",
        source_label="Inbox",
        source_locator=f"v2-thread:{thread['id']}",
        content=clean,
        metadata={"thread_id": thread["id"], "channel": channel},
    )
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
                'user', 'evidence', %s, %s
            )
            RETURNING id, thread_id, sequence_no, role, content, message_type,
                      raw_evidence_id, created_at
            """,
            (int(thread["id"]), int(thread["id"]), clean, evidence["id"]),
        )
        message = _dict(cur.fetchone())
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE v2_inbox_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (int(thread["id"]),),
        )
    return {"thread": thread, "message": message, "evidence": evidence}


def list_thread_messages(conn, thread_id: int) -> list[dict]:
    get_thread(conn, thread_id)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, sequence_no, role, content, message_type,
                   raw_evidence_id, created_at
            FROM v2_inbox_messages
            WHERE thread_id=%s
            ORDER BY sequence_no, id
            """,
            (thread_id,),
        )
        return [_dict(row) for row in cur.fetchall()]


def thread_response(conn, thread_id: int) -> dict:
    return {"thread": get_thread(conn, thread_id), "messages": list_thread_messages(conn, thread_id)}


def list_threads(conn, *, limit: int = 30) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.id, t.origin AS channel, t.status,
                   t.thread_type AS mode, t.created_at, t.updated_at,
                   (SELECT m.content FROM v2_inbox_messages m
                    WHERE m.thread_id=t.id ORDER BY m.id DESC LIMIT 1) AS preview,
                   (SELECT count(*) FROM v2_inbox_messages m
                    WHERE m.thread_id=t.id) AS message_count
            FROM v2_inbox_threads t
            ORDER BY t.updated_at DESC, t.id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [_dict(row) for row in cur.fetchall()]


def summary(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS count FROM v2_knowledge WHERE active=TRUE")
        knowledge_count = int(cur.fetchone()["count"])
        cur.execute(
            "SELECT count(*) AS count FROM v2_knowledge WHERE active=TRUE AND created_at >= CURRENT_DATE - INTERVAL '7 days'"
        )
        week_count = int(cur.fetchone()["count"])
        cur.execute(
            "SELECT count(*) AS count FROM v2_learning_proposals WHERE status='pending_confirmation'"
        )
        pending_count = int(cur.fetchone()["count"])
    return {
        "knowledge_count": knowledge_count,
        "week_new_count": week_count,
        "pending_count": pending_count,
        # Gaps are introduced in Phase 3; keeping this explicit avoids
        # pretending that Phase 1 already has a refusal loop.
        "unresolved_gap_count": 0,
    }


def inbox_snapshot(conn) -> dict:
    return {"summary": summary(conn), "threads": list_threads(conn)}


def list_knowledge(conn, *, limit: int = 100) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT k.id, k.title, k.content, k.entity_name, k.trust,
                   k.active, k.created_at, k.updated_at,
                   count(s.raw_evidence_id) AS source_count
            FROM v2_knowledge k
            LEFT JOIN v2_knowledge_sources s ON s.knowledge_id=k.id
            WHERE k.active=TRUE
            GROUP BY k.id
            ORDER BY k.updated_at DESC, k.id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [_dict(row) for row in cur.fetchall()]


def list_documents(conn, *, limit: int = 100) -> list[dict]:
    """Expose existing document metadata without creating a second document store."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, title, version, status, created_at, updated_at
            FROM documents
            ORDER BY updated_at DESC, id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [_dict(row) for row in cur.fetchall()]


def json_safe(value: Any) -> Any:
    """Make service payloads convenient for small unit tests and logs."""

    return json.loads(json.dumps(value, default=str, ensure_ascii=False))
