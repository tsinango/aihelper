"""Persistence helpers for the Phase 2 V2 Inbox data layer.

This module is intentionally boring: SQL is visible, transactions belong to
the caller, and the V2 tables are never mixed with V1 candidate/review tables.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any


class V2NotFound(LookupError):
    """Raised when a requested V2 thread does not exist."""


# Keep worker liveness policy in one place so the API, worker, and Inbox UI
# agree on what "healthy" means.
INBOX_WORKER_NAME = os.getenv("INBOX_WORKER_NAME", "aihelper-inbox-worker")
INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS = max(
    5.0, float(os.getenv("INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS", "10"))
)
INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS = max(
    INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS * 2,
    float(os.getenv("INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS", "45")),
)


def _dict(row: Any) -> dict:
    return dict(row) if row is not None else {}


def record_worker_heartbeat(conn, worker_name: str = INBOX_WORKER_NAME) -> dict:
    """Record one worker liveness pulse; the caller owns the transaction."""

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_workers(worker_name, last_seen_at)
            VALUES(%s, CURRENT_TIMESTAMP)
            ON CONFLICT (worker_name) DO UPDATE
              SET last_seen_at=CURRENT_TIMESTAMP
            RETURNING worker_name, last_seen_at
            """,
            (str(worker_name),),
        )
        return _dict(cur.fetchone())


def worker_health(conn, worker_name: str = INBOX_WORKER_NAME) -> dict:
    """Return a non-secret liveness snapshot for the dedicated Inbox worker."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT worker_name, last_seen_at,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen_at))
                     AS age_seconds
            FROM v2_inbox_workers
            WHERE worker_name=%s
            """,
            (str(worker_name),),
        )
        row = cur.fetchone()
    if not row:
        return {
            "worker_name": str(worker_name),
            "last_seen_at": None,
            "healthy": False,
        }
    result = _dict(row)
    age_seconds = result.get("age_seconds")
    result["healthy"] = age_seconds is not None and float(age_seconds) <= INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS
    result.pop("age_seconds", None)
    return result


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
    return {
        "thread": get_thread(conn, thread_id),
        "messages": list_thread_messages(conn, thread_id),
        "jobs": list_active_jobs(conn, thread_id),
    }


def list_active_jobs(conn, thread_id: int) -> list[dict]:
    """Return durable work that a refreshed Inbox should continue polling."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, raw_evidence_id, user_message_id,
                   idempotency_key, status, error_message, attempts,
                   created_at, started_at, completed_at, updated_at
            FROM v2_inbox_processing_jobs
            WHERE thread_id=%s AND status IN ('queued', 'processing')
            ORDER BY id
            """,
            (thread_id,),
        )
        return [_dict(row) for row in cur.fetchall()]


def create_processing_job(
    conn,
    *,
    thread_id: int,
    raw_evidence_id: int,
    user_message_id: int,
    idempotency_key: str,
) -> dict:
    """Create one queued job; the caller owns the surrounding transaction."""

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_processing_jobs(
                thread_id, raw_evidence_id, user_message_id, idempotency_key,
                status
            )
            VALUES(%s, %s, %s, %s, 'queued')
            RETURNING id, thread_id, raw_evidence_id, user_message_id,
                      idempotency_key, status, error_message, attempts,
                      created_at, started_at, completed_at, updated_at
            """,
            (thread_id, raw_evidence_id, user_message_id, idempotency_key),
        )
        return _dict(cur.fetchone())


def get_processing_job(conn, job_id: int) -> dict | None:
    """Load a job and the first assistant response produced after its input."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.id, j.thread_id, j.raw_evidence_id, j.user_message_id,
                   j.idempotency_key, j.status, j.error_message, j.attempts,
                   j.created_at, j.started_at, j.completed_at, j.updated_at,
                   (w.last_seen_at IS NOT NULL AND
                    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - w.last_seen_at)) <= %s)
                     AS worker_healthy,
                   m.id AS assistant_message_id,
                   m.content AS assistant_message,
                   m.message_type AS assistant_message_type
            FROM v2_inbox_processing_jobs j
            LEFT JOIN v2_inbox_workers w ON w.worker_name=%s
            LEFT JOIN LATERAL (
                SELECT id, content, message_type
                FROM v2_inbox_messages
                WHERE thread_id=j.thread_id AND role='assistant'
                  AND sequence_no > (
                      SELECT sequence_no FROM v2_inbox_messages WHERE id=j.user_message_id
                  )
                ORDER BY sequence_no, id
                LIMIT 1
            ) m ON TRUE
            WHERE j.id=%s
            """,
            (INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS, INBOX_WORKER_NAME, job_id),
        )
        row = cur.fetchone()
    return _dict(row) if row else None


def claim_processing_job(conn, job_id: int) -> dict | None:
    """Atomically let one background invocation own a queued job."""

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_inbox_processing_jobs
            SET status='processing', attempts=attempts + 1,
                started_at=COALESCE(started_at, CURRENT_TIMESTAMP),
                error_message=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status='queued'
            RETURNING id, thread_id, raw_evidence_id, user_message_id,
                      idempotency_key, status, error_message, attempts,
                      created_at, started_at, completed_at, updated_at
            """,
            (job_id,),
        )
        row = cur.fetchone()
    return _dict(row) if row else None


def complete_processing_job(conn, job_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_inbox_processing_jobs
            SET status='completed', completed_at=CURRENT_TIMESTAMP,
                error_message=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status='processing'
            RETURNING id, thread_id, raw_evidence_id, user_message_id,
                      idempotency_key, status, error_message, attempts,
                      created_at, started_at, completed_at, updated_at
            """,
            (job_id,),
        )
        row = cur.fetchone()
    return _dict(row) if row else None


def fail_processing_job(conn, job_id: int, error_message: str = "处理失败") -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_inbox_processing_jobs
            SET status='failed', error_message=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status='processing'
            RETURNING id, thread_id, raw_evidence_id, user_message_id,
                      idempotency_key, status, error_message, attempts,
                      created_at, started_at, completed_at, updated_at
            """,
            (str(error_message or "处理失败")[:1000], job_id),
        )
        row = cur.fetchone()
    return _dict(row) if row else None


def retry_processing_job(conn, job_id: int) -> dict | None:
    """Requeue only a failed job; its evidence and user message stay intact."""

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_inbox_processing_jobs
            SET status='queued', error_message=NULL, completed_at=NULL,
                started_at=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status='failed'
            RETURNING id, thread_id, raw_evidence_id, user_message_id,
                      idempotency_key, status, error_message, attempts,
                      created_at, started_at, completed_at, updated_at
            """,
            (job_id,),
        )
        row = cur.fetchone()
    return _dict(row) if row else None


def list_unfinished_job_ids(conn) -> list[int]:
    """Expose only ids for optional process recovery at application startup."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM v2_inbox_processing_jobs
            WHERE status IN ('queued', 'processing') ORDER BY id
            """
        )
        return [int(row["id"]) for row in cur.fetchall()]


def list_queued_job_ids(conn, *, limit: int = 1) -> list[int]:
    """Return a bounded FIFO slice for the dedicated Inbox worker."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM v2_inbox_processing_jobs
            WHERE status='queued'
            ORDER BY id
            LIMIT %s
            """,
            (max(1, int(limit)),),
        )
        return [int(row["id"]) for row in cur.fetchall()]


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
            WHERE paused=FALSE
              AND status IN ('pending_clarification', 'pending_confirmation')
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
    return {
        "summary": summary(conn),
        "threads": list_threads(conn),
        "worker": worker_health(conn),
    }


def list_knowledge(conn, *, limit: int = 100) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT k.id, k.title, k.content, k.entity_name, k.entity_id, k.trust,
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


def list_knowledge_for_entity(conn, entity_id: int, *, limit: int = 100) -> list[dict]:
    """Return facts linked to one entity without changing the fact layer."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT k.id, k.title, k.content, k.entity_name, k.entity_id, k.trust,
                   k.active, k.created_at, k.updated_at,
                   count(s.raw_evidence_id) AS source_count
            FROM v2_knowledge k
            LEFT JOIN v2_knowledge_sources s ON s.knowledge_id=k.id
            WHERE k.active=TRUE AND k.entity_id=%s
            GROUP BY k.id
            ORDER BY k.updated_at DESC, k.id DESC
            LIMIT %s
            """,
            (int(entity_id), limit),
        )
        return [_dict(row) for row in cur.fetchall()]


def list_editable_proposals(conn, thread_id: int) -> dict:
    """Return pending proposals for the Inbox editor without touching evidence."""

    proposal_columns = """
        SELECT id, batch_id, segment_no, fact_text, entity_name,
               status, comparison_result, source_message_id,
               confirmed_knowledge_id, created_at, updated_at
        FROM v2_learning_proposals
        WHERE thread_id=%s AND paused=FALSE
          AND status IN ('pending_confirmation', 'pending_clarification')
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, raw_evidence_id, total_segments,
                   processed_segments, failed_segments, status
            FROM v2_learning_batches
            WHERE thread_id=%s
              AND status IN ('processing', 'awaiting_confirmation',
                             'awaiting_clarification', 'partial')
            ORDER BY id DESC LIMIT 1
            """,
            (int(thread_id),),
        )
        batch = _dict(cur.fetchone())
        if not batch:
            # The same minimal editor is useful for a single pending
            # proposal. It still exposes only fact_text, never raw input.
            cur.execute(
                proposal_columns + " AND batch_id IS NULL ORDER BY id LIMIT 1",
                (int(thread_id),),
            )
            return {"batch": None, "items": [_dict(row) for row in cur.fetchall()]}
        cur.execute(
            proposal_columns + " AND batch_id=%s ORDER BY segment_no NULLS LAST, id",
            (int(thread_id), int(batch["id"])),
        )
        return {"batch": batch, "items": [_dict(row) for row in cur.fetchall()]}


def edit_pending_proposal(conn, proposal_id: int, fact_text: str) -> dict:
    """Edit only the proposal text; source evidence and links stay untouched."""

    clean = str(fact_text or "").strip()
    if not clean:
        raise ValueError("proposal text must not be empty")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET fact_text=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND paused=FALSE
              AND status IN ('pending_confirmation', 'pending_clarification')
            RETURNING id, thread_id, batch_id, segment_no, fact_text,
                      entity_name, status, comparison_result,
                      source_message_id, confirmed_knowledge_id,
                      created_at, updated_at
            """,
            (clean, int(proposal_id)),
        )
        row = cur.fetchone()
    if not row:
        raise V2NotFound("editable V2 proposal was not found")
    return _dict(row)


def reject_pending_proposal(conn, proposal_id: int) -> dict:
    """Soft-delete a proposal while retaining its raw evidence provenance."""

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET status='rejected', updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND paused=FALSE
              AND status IN ('pending_confirmation', 'pending_clarification')
            RETURNING id, thread_id, batch_id, segment_no, fact_text,
                      entity_name, status, comparison_result,
                      source_message_id, confirmed_knowledge_id,
                      created_at, updated_at
            """,
            (int(proposal_id),),
        )
        row = cur.fetchone()
    if not row:
        raise V2NotFound("deletable V2 proposal was not found")
    return _dict(row)


def list_entity_tree(conn) -> dict:
    """Build a small read-only tree for the Knowledge page.

    Organization review is local; this read endpoint may assemble the whole
    display tree because it does not call an LLM or mutate any relation.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.name, e.entity_type, e.active,
                   e.created_at, e.updated_at, count(k.id) AS knowledge_count
            FROM v2_entities e
            LEFT JOIN v2_knowledge k
              ON k.entity_id=e.id AND k.active=TRUE
            WHERE e.active=TRUE
            GROUP BY e.id
            ORDER BY lower(e.name), e.id
            """
        )
        entities = [_dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, parent_entity_id, child_entity_id, relation_type,
                   source_id, provenance, provenance_kind, active,
                   created_at, updated_at
            FROM v2_entity_relations
            WHERE active=TRUE
            ORDER BY parent_entity_id, child_entity_id, id
            """
        )
        relations = [_dict(row) for row in cur.fetchall()]

    nodes = {
        int(entity["id"]): {
            "id": int(entity["id"]),
            "name": entity["name"],
            "entity_type": entity["entity_type"],
            "knowledge_count": int(entity.get("knowledge_count") or 0),
            "children": [],
        }
        for entity in entities
    }
    children: dict[int, list[int]] = defaultdict(list)
    parents: dict[int, list[int]] = defaultdict(list)
    for relation in relations:
        parent_id = int(relation["parent_entity_id"])
        child_id = int(relation["child_entity_id"])
        if parent_id not in nodes or child_id not in nodes:
            continue
        children[parent_id].append(child_id)
        parents[child_id].append(parent_id)

    def render(entity_id: int, path: frozenset[int] = frozenset()) -> dict:
        if entity_id in path:
            return dict(nodes[entity_id], children=[])
        node = dict(nodes[entity_id])
        node["children"] = [render(child_id, path | {entity_id}) for child_id in children.get(entity_id, [])]
        return node

    root_ids = [entity_id for entity_id in nodes if not parents.get(entity_id)]
    roots = [render(entity_id) for entity_id in root_ids if children.get(entity_id)]
    unorganized = [render(entity_id) for entity_id in root_ids if not children.get(entity_id)]
    return {"roots": roots, "unorganized": unorganized}


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
