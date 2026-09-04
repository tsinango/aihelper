"""Persistence helpers for the Phase 2 V2 Inbox data layer.

This module is intentionally boring: SQL is visible, transactions belong to
the caller, and the V2 tables are never mixed with V1 candidate/review tables.
"""

from __future__ import annotations

import json
from typing import Any


class V2NotFound(LookupError):
    """Raised when a requested V2 thread does not exist."""


def _dict(row: Any) -> dict:
    return dict(row) if row is not None else {}


def create_thread(
    conn,
    *,
    channel: str = "inbox",
    mode: str = "learn",
    external_thread_id: str | None = None,
) -> dict:
    origin = {
        "inbox": "web",
        "chat": "web",
        "telegram": "telegram",
        "import": "import",
    }.get(channel, "web")
    thread_type = "learning" if mode == "learn" else "general"
    external_thread_id = str(external_thread_id or "").strip() or None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_threads(thread_type, origin, status, external_thread_id)
            VALUES(%s, %s, 'open', %s)
            ON CONFLICT (origin, external_thread_id)
              WHERE external_thread_id IS NOT NULL
              DO UPDATE SET updated_at=CURRENT_TIMESTAMP
            RETURNING id, origin AS channel, status, thread_type AS mode,
                      external_thread_id, created_at, updated_at
            """,
            (thread_type, origin, external_thread_id),
        )
        return _dict(cur.fetchone())


def get_thread(conn, thread_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, origin AS channel, status, thread_type AS mode,
                   external_thread_id, created_at, updated_at
            FROM v2_inbox_threads
            WHERE id=%s
            """,
            (thread_id,),
        )
        row = cur.fetchone()
    if not row:
        raise V2NotFound(f"V2 thread {thread_id} was not found")
    return _dict(row)


def ensure_thread(
    conn,
    thread_id: int | None,
    *,
    channel: str = "inbox",
    mode: str = "learn",
    external_thread_id: str | None = None,
) -> dict:
    if thread_id is not None:
        return get_thread(conn, thread_id)
    return create_thread(
        conn,
        channel=channel,
        mode=mode,
        external_thread_id=external_thread_id,
    )


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
                   t.thread_type AS mode, t.external_thread_id,
                   t.created_at, t.updated_at,
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
            """
            SELECT count(*) AS count
            FROM v2_learning_proposals
            WHERE status IN ('pending_clarification', 'pending_confirmation')
            """
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
