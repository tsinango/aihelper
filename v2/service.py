"""Persistence helpers for the Phase 2 V2 Inbox data layer.

This module is intentionally boring: SQL is visible, transactions belong to
the caller, and the V2 tables are never mixed with V1 candidate/review tables.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

from psycopg.types.json import Jsonb


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

# Phase 3.0 organization closure: confirmed Knowledge is no longer sent to the
# LLM for automatic relation/structure review.  Deterministic exact Entity
# linking still runs on every confirmation.  This switch exists only as an
# internal deployment/rollback lever for verifying the old behavior; it is not
# a product setting and defaults to off.
V2_ORGANIZATION_LLM_ENABLED = (
    os.getenv("V2_ORGANIZATION_LLM_ENABLED", "").strip().lower()
    in {"1", "true", "yes", "on"}
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
        # The feedback table arrives with migration 022; to_regclass keeps
        # pre-022 databases working without poisoning the transaction.
        cur.execute("SELECT to_regclass('public.v2_answer_feedback') AS name")
        has_feedback = (cur.fetchone() or {}).get("name") is not None
        gap_count = 0
        if has_feedback:
            cur.execute(
                """
                SELECT count(*) AS count
                FROM v2_answer_feedback
                WHERE status='open'
                """
            )
            gap_count = int(cur.fetchone()["count"])
    return {
        "knowledge_count": knowledge_count,
        "week_new_count": week_count,
        "pending_count": pending_count,
        "unresolved_gap_count": gap_count,
    }


def inbox_snapshot(conn) -> dict:
    return {
        "summary": summary(conn),
        "threads": list_threads(conn),
        "worker": worker_health(conn),
    }


def _knowledge_query_filters(*, active: bool, entity_id: int | None, search: str) -> tuple[str, list[Any]]:
    clauses = ["k.active=%s"]
    params: list[Any] = [bool(active)]
    if entity_id is not None:
        clauses.append("k.entity_id=%s")
        params.append(int(entity_id))
    clean_search = str(search or "").strip()
    if clean_search:
        clauses.append(
            "(k.content ILIKE %s OR k.title ILIKE %s "
            "OR k.entity_name ILIKE %s OR COALESCE(e.name, '') ILIKE %s)"
        )
        pattern = f"%{clean_search}%"
        params.extend([pattern, pattern, pattern, pattern])
    return " AND ".join(clauses), params


def _list_knowledge(
    conn,
    *,
    limit: int = 100,
    active: bool = True,
    entity_id: int | None = None,
    search: str = "",
) -> list[dict]:
    where, params = _knowledge_query_filters(active=active, entity_id=entity_id, search=search)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT k.id, k.title, k.content, k.entity_name, k.entity_id,
                   e.name AS entity_display_name, k.trust,
                   k.active, k.created_at, k.updated_at,
                   k.unit_kind, k.applicability, k.revision, k.details_json,
                   k.origin_document_version_id, k.validation_status,
                   count(s.raw_evidence_id) AS source_count
            FROM v2_knowledge k
            LEFT JOIN v2_entities e ON e.id=k.entity_id
            LEFT JOIN v2_knowledge_sources s ON s.knowledge_id=k.id
            WHERE {where}
            GROUP BY k.id, e.name
            ORDER BY k.updated_at DESC, k.id DESC
            LIMIT %s
            """,
            tuple(params + [max(1, int(limit))]),
        )
        return [_dict(row) for row in cur.fetchall()]


def list_knowledge(
    conn,
    *,
    limit: int = 100,
    active: bool = True,
    search: str = "",
) -> list[dict]:
    return _list_knowledge(conn, limit=limit, active=active, search=search)


def list_knowledge_for_entity(
    conn,
    entity_id: int,
    *,
    limit: int = 100,
    active: bool = True,
    search: str = "",
) -> list[dict]:
    """Return facts linked to one entity without changing the fact layer."""
    return _list_knowledge(
        conn,
        limit=limit,
        active=active,
        entity_id=entity_id,
        search=search,
    )


def _knowledge_snapshot(row: dict) -> dict:
    return {
        "content": str(row.get("content") or ""),
        "entity_id": row.get("entity_id"),
        "active": bool(row.get("active")),
        "trust": str(row.get("trust") or ""),
        "unit_kind": str(row.get("unit_kind") or ""),
        "applicability": row.get("applicability") or {},
        "revision": row.get("revision"),
        "details_json": row.get("details_json") or {},
        "origin_document_version_id": row.get("origin_document_version_id"),
        "validation_status": row.get("validation_status"),
    }


def _write_knowledge_history(
    conn,
    knowledge_id: int,
    action: str,
    before: dict,
    after: dict,
) -> None:
    if action not in {"edit", "deactivate", "restore", "move", "confirm", "revalidate"}:
        raise ValueError(f"unknown Knowledge history action: {action}")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge_history(
                knowledge_id, action, before_json, after_json
            ) VALUES(%s, %s, %s, %s)
            """,
            (int(knowledge_id), action, Jsonb(before), Jsonb(after)),
        )


def _load_knowledge_for_maintenance(conn, knowledge_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, entity_name, entity_id, trust, active,
                   unit_kind, applicability, revision, details_json,
                   origin_document_version_id, validation_status,
                   created_at, updated_at
            FROM v2_knowledge
            WHERE id=%s
            """,
            (int(knowledge_id),),
        )
        row = cur.fetchone()
    if not row:
        raise V2NotFound(f"V2 Knowledge {knowledge_id} was not found")
    return _dict(row)


def _validate_entity_for_maintenance(conn, entity_id: int | None) -> None:
    if entity_id is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM v2_entities WHERE id=%s AND active=TRUE",
            (int(entity_id),),
        )
        if cur.fetchone() is None:
            raise V2NotFound(f"active V2 entity {entity_id} was not found")


def edit_knowledge(
    conn,
    knowledge_id: int,
    content: str,
    entity_id: int | None,
    applicability: dict | None = None,
    details_json: dict | None = None,
    validation_status: str | None = None,
) -> dict:
    """Deterministically edit one Knowledge row; no LLM or embedding call.

    Any content, applicability, details, entity, or validation change bumps
    ``revision`` and records history; content/applicability/details changes
    also clear the embedding so the stale vector can never support an
    answer.  Validation transitions audit as ``revalidate``, other changes
    as ``edit``/``move``.
    """

    clean = str(content or "").strip()
    if not clean or len(clean) > 12000:
        raise ValueError("Knowledge content must contain 1-12000 characters")
    if applicability is not None and not isinstance(applicability, dict):
        raise ValueError("Knowledge applicability must be a JSON object")
    if details_json is not None and not isinstance(details_json, dict):
        raise ValueError("Knowledge details must be a JSON object")
    if validation_status is not None and validation_status not in (
        "pending", "validated", "needs_revalidation",
    ):
        raise ValueError("unknown validation status")
    current = _load_knowledge_for_maintenance(conn, int(knowledge_id))
    if validation_status is not None and current.get("origin_document_version_id") is None:
        raise ValueError("validation status applies to document-learned Knowledge only")
    if not current.get("active"):
        raise ValueError("deleted Knowledge must be restored before editing")
    _validate_entity_for_maintenance(conn, entity_id)
    old_snapshot = _knowledge_snapshot(current)
    content_changed = clean != old_snapshot["content"]
    entity_changed = entity_id != old_snapshot["entity_id"]
    applicability_changed = (
        applicability is not None
        and {str(key): applicability[key] for key in applicability}
        != dict(old_snapshot.get("applicability") or {})
    )
    details_changed = (
        details_json is not None
        and {str(key): details_json[key] for key in details_json}
        != dict(old_snapshot.get("details_json") or {})
    )
    validation_changed = (
        validation_status is not None
        and validation_status != old_snapshot.get("validation_status")
    )
    if not content_changed and not entity_changed and not applicability_changed and not details_changed and not validation_changed:
        return current
    new_applicability = (
        {str(key): applicability[key] for key in applicability}
        if applicability is not None
        else dict(old_snapshot.get("applicability") or {})
    )
    new_details = (
        {str(key): details_json[key] for key in details_json}
        if details_json is not None
        else dict(old_snapshot.get("details_json") or {})
    )
    new_validation = validation_status if validation_status is not None else old_snapshot.get("validation_status")
    with conn.cursor() as cur:
        if content_changed or applicability_changed or details_changed or validation_changed:
            cur.execute(
                """
                UPDATE v2_knowledge
                SET content=%s, entity_id=%s, applicability=%s, details_json=%s,
                    validation_status=%s,
                    embedding=NULL, embedding_model=NULL,
                    revision=revision+1, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s AND active=TRUE
                RETURNING id, title, content, entity_name, entity_id, trust, active,
                          unit_kind, applicability, revision, details_json,
                          origin_document_version_id, validation_status,
                          created_at, updated_at
                """,
                (clean, entity_id, Jsonb(new_applicability), Jsonb(new_details),
                 new_validation, int(knowledge_id)),
            )
        else:
            cur.execute(
                """
                UPDATE v2_knowledge
                SET entity_id=%s, revision=revision+1, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s AND active=TRUE
                RETURNING id, title, content, entity_name, entity_id, trust, active,
                          unit_kind, applicability, revision, details_json,
                          origin_document_version_id, validation_status,
                          created_at, updated_at
                """,
                (entity_id, int(knowledge_id)),
            )
        updated = _dict(cur.fetchone())
    if not updated:
        raise V2NotFound(f"active V2 Knowledge {knowledge_id} was not found")
    if content_changed or applicability_changed or details_changed:
        after = _knowledge_snapshot(updated)
        after["entity_id"] = old_snapshot["entity_id"]
        _write_knowledge_history(conn, int(knowledge_id), "edit", old_snapshot, after)
    if validation_changed:
        after = _knowledge_snapshot(updated)
        after["entity_id"] = old_snapshot["entity_id"]
        _write_knowledge_history(conn, int(knowledge_id), "revalidate", old_snapshot, after)
    if entity_changed:
        before = _knowledge_snapshot(updated)
        before["entity_id"] = old_snapshot["entity_id"]
        after = _knowledge_snapshot(updated)
        _write_knowledge_history(conn, int(knowledge_id), "move", before, after)
    return updated


def deactivate_knowledge(conn, knowledge_id: int) -> dict:
    current = _load_knowledge_for_maintenance(conn, int(knowledge_id))
    if not current.get("active"):
        return current
    before = _knowledge_snapshot(current)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET active=FALSE, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND active=TRUE
            RETURNING id, title, content, entity_name, entity_id, trust, active,
                      created_at, updated_at
            """,
            (int(knowledge_id),),
        )
        updated = _dict(cur.fetchone())
    if not updated:
        raise V2NotFound(f"active V2 Knowledge {knowledge_id} was not found")
    _write_knowledge_history(conn, int(knowledge_id), "deactivate", before, _knowledge_snapshot(updated))
    return updated


def restore_knowledge(conn, knowledge_id: int) -> dict:
    current = _load_knowledge_for_maintenance(conn, int(knowledge_id))
    if current.get("active"):
        return current
    if current.get("trust") == "conflicted":
        raise ValueError("conflicted Knowledge must be resolved before restoration")
    before = _knowledge_snapshot(current)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET active=TRUE, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND active=FALSE
              AND trust IN ('official_source', 'user_confirmed', 'provisional')
            RETURNING id, title, content, entity_name, entity_id, trust, active,
                      created_at, updated_at
            """,
            (int(knowledge_id),),
        )
        updated = _dict(cur.fetchone())
    if not updated:
        raise ValueError("Knowledge cannot be restored")
    _write_knowledge_history(conn, int(knowledge_id), "restore", before, _knowledge_snapshot(updated))
    return updated


def list_knowledge_sources(conn, knowledge_id: int) -> list[dict]:
    _load_knowledge_for_maintenance(conn, int(knowledge_id))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.knowledge_id, s.raw_evidence_id, s.source_kind,
                   s.relation, s.source_role, s.excerpt, s.active, s.resolution,
                   s.created_at, r.evidence_type, r.source_label,
                   r.source_locator, r.content AS raw_content,
                   r.evidence_status
            FROM v2_knowledge_sources s
            JOIN v2_raw_evidence r ON r.id=s.raw_evidence_id
            WHERE s.knowledge_id=%s
            ORDER BY s.id
            """,
            (int(knowledge_id),),
        )
        return [_dict(row) for row in cur.fetchall()]


def list_knowledge_history(conn, knowledge_id: int) -> list[dict]:
    _load_knowledge_for_maintenance(conn, int(knowledge_id))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, knowledge_id, action, before_json, after_json, created_at
            FROM v2_knowledge_history
            WHERE knowledge_id=%s
            ORDER BY created_at DESC, id DESC
            """,
            (int(knowledge_id),),
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
                   e.created_at, e.updated_at,
                   count(k.id) FILTER (WHERE k.active=TRUE) AS knowledge_count,
                   count(k.id) AS knowledge_reference_count
            FROM v2_entities e
            LEFT JOIN v2_knowledge k
              ON k.entity_id=e.id
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
            "knowledge_reference_count": int(entity.get("knowledge_reference_count") or 0),
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
            node = dict(nodes[entity_id], children=[])
            node["subtree_knowledge_reference_count"] = node["knowledge_reference_count"]
            node["subtree_entity_count"] = 1
            node["prune_allowed"] = node["subtree_knowledge_reference_count"] == 0
            return node
        node = dict(nodes[entity_id])
        node["children"] = [render(child_id, path | {entity_id}) for child_id in children.get(entity_id, [])]
        node["subtree_knowledge_reference_count"] = node["knowledge_reference_count"] + sum(
            child["subtree_knowledge_reference_count"] for child in node["children"]
        )
        node["subtree_entity_count"] = 1 + sum(
            child["subtree_entity_count"] for child in node["children"]
        )
        node["prune_allowed"] = node["subtree_knowledge_reference_count"] == 0
        return node

    root_ids = [entity_id for entity_id in nodes if not parents.get(entity_id)]
    roots = [render(entity_id) for entity_id in root_ids if children.get(entity_id)]
    unorganized = [render(entity_id) for entity_id in root_ids if not children.get(entity_id)]
    return {"roots": roots, "unorganized": unorganized}


def _active_entity_subtree(conn, root_entity_id: int) -> list[dict]:
    """Read an active subtree with path protection against malformed cycles."""

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE subtree(entity_id, path) AS (
              SELECT %s::BIGINT, ARRAY[%s::BIGINT]
              UNION ALL
              SELECT r.child_entity_id, s.path || r.child_entity_id
              FROM subtree s
              JOIN v2_entity_relations r
                ON r.parent_entity_id=s.entity_id
               AND r.relation_type='belongs_to'
               AND r.active=TRUE
              JOIN v2_entities child
                ON child.id=r.child_entity_id AND child.active=TRUE
              WHERE NOT r.child_entity_id = ANY(s.path)
            )
            SELECT DISTINCT e.id, e.name, e.entity_type, e.active,
                            e.created_at, e.updated_at
            FROM subtree s
            JOIN v2_entities e ON e.id=s.entity_id AND e.active=TRUE
            ORDER BY e.id
            """,
            (int(root_entity_id), int(root_entity_id)),
        )
        return [_dict(row) for row in cur.fetchall()]


def _lock_active_entities(conn, entity_ids: list[int]) -> list[dict]:
    if not entity_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, entity_type, active, created_at, updated_at
            FROM v2_entities
            WHERE id=ANY(%s) AND active=TRUE
            ORDER BY id
            FOR UPDATE
            """,
            (entity_ids,),
        )
        return [_dict(row) for row in cur.fetchall()]


def _lock_touching_relations(conn, entity_ids: list[int]) -> list[dict]:
    if not entity_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, parent_entity_id, child_entity_id, relation_type,
                   source_id, provenance, provenance_kind, active,
                   created_at, updated_at
            FROM v2_entity_relations
            WHERE active=TRUE
              AND (parent_entity_id=ANY(%s) OR child_entity_id=ANY(%s))
            ORDER BY id
            FOR UPDATE
            """,
            (entity_ids, entity_ids),
        )
        return [_dict(row) for row in cur.fetchall()]


def _lock_knowledge_references(conn, entity_ids: list[int]) -> list[dict]:
    if not entity_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, entity_id, active, trust
            FROM v2_knowledge
            WHERE entity_id=ANY(%s)
            ORDER BY id
            FOR UPDATE
            """,
            (entity_ids,),
        )
        return [_dict(row) for row in cur.fetchall()]


def _contains_active_cycle(entity_ids: list[int], relations: list[dict]) -> bool:
    """Reject malformed active cycles instead of treating them as a tree."""

    allowed = set(entity_ids)
    children: dict[int, list[int]] = defaultdict(list)
    for relation in relations:
        parent_id = int(relation["parent_entity_id"])
        child_id = int(relation["child_entity_id"])
        if parent_id in allowed and child_id in allowed:
            children[parent_id].append(child_id)

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(entity_id: int) -> bool:
        if entity_id in visiting:
            return True
        if entity_id in visited:
            return False
        visiting.add(entity_id)
        if any(visit(child_id) for child_id in children.get(entity_id, [])):
            return True
        visiting.remove(entity_id)
        visited.add(entity_id)
        return False

    return any(visit(entity_id) for entity_id in entity_ids)


def prune_empty_entity_subtree(conn, entity_id: int) -> dict:
    """Human-initiated soft-prune of an entirely empty active entity subtree.

    This helper is intentionally not called by learning or organization
    review. It locks and re-checks the current database state in one
    transaction. Any Knowledge reference blocks the whole operation, including
    inactive Knowledge that may later be restored. Entities and relations are
    only deactivated.
    """

    root_id = int(entity_id)
    transaction = (
        conn.transaction()
        if callable(getattr(conn, "transaction", None))
        else nullcontext()
    )
    with transaction:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, entity_type, active, created_at, updated_at
                FROM v2_entities
                WHERE id=%s
                FOR UPDATE
                """,
                (root_id,),
            )
            root = _dict(cur.fetchone()) or None
        if root is None:
            raise V2NotFound(f"V2 entity {root_id} was not found")
        if not root.get("active"):
            raise ValueError("entity structure is already inactive")

        # Refresh after locking the discovered rows.  The bounded second pass
        # keeps the operation local while avoiding a stale first traversal.
        subtree: list[dict] = []
        relations: list[dict] = []
        for _ in range(2):
            subtree = _active_entity_subtree(conn, root_id)
            entity_ids = sorted({int(row["id"]) for row in subtree})
            if root_id not in entity_ids:
                raise ValueError("entity structure is no longer active")
            _lock_active_entities(conn, entity_ids)
            relations = _lock_touching_relations(conn, entity_ids)
            refreshed_ids = sorted({int(row["id"]) for row in _active_entity_subtree(conn, root_id)})
            if refreshed_ids == entity_ids:
                break

        entity_ids = sorted({int(row["id"]) for row in subtree})
        knowledge = _lock_knowledge_references(conn, entity_ids)
        knowledge_ids = [int(row["id"]) for row in knowledge]
        if _contains_active_cycle(entity_ids, relations):
            raise ValueError("cannot prune malformed entity structure with an active cycle")
        if knowledge:
            raise ValueError(
                "cannot prune entity structure with active or deleted Knowledge references"
            )

        relation_ids = [int(row["id"]) for row in relations]
        deactivated_relation_ids: list[int] = []
        if relation_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE v2_entity_relations
                    SET active=FALSE, deactivated_at=COALESCE(deactivated_at, CURRENT_TIMESTAMP),
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=ANY(%s) AND active=TRUE
                    RETURNING id
                    """,
                    (relation_ids,),
                )
                deactivated_relation_ids = [int(row["id"]) for row in cur.fetchall()]

        deactivated_entity_ids: list[int] = []
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE v2_entities
                SET active=FALSE, deactivated_at=COALESCE(deactivated_at, CURRENT_TIMESTAMP),
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=ANY(%s) AND active=TRUE
                RETURNING id
                """,
                (entity_ids,),
            )
            deactivated_entity_ids = [int(row["id"]) for row in cur.fetchall()]

        return {
            "pruned": True,
            "reason": "empty_subtree",
            "entity_id": root_id,
            "entity_ids": deactivated_entity_ids,
            "relation_ids": deactivated_relation_ids,
            "knowledge_ids": knowledge_ids,
            "active_knowledge_ids": [],
        }


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
