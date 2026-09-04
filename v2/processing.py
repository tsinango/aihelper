"""Minimal durable processing boundary for V2 Inbox submissions."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from v2.bulk import classify_input_mode
from v2.learning import (
    _insert_evidence,
    _insert_message,
    _lock_thread,
    _pending_batch,
    _pending_proposal,
    classify_reply,
    learn_turn,
)
from v2.service import (
    claim_processing_job,
    complete_processing_job,
    create_processing_job,
    create_thread,
    get_processing_job,
    get_thread,
    retry_processing_job,
    fail_processing_job,
    list_unfinished_job_ids,
)


log = logging.getLogger("ai-sales-engineer.v2.processing")


def _job_key(value: str | None) -> str:
    clean = str(value or "").strip()
    return clean[:200] or str(uuid.uuid4())


def _message_type(content: str, has_pending: bool) -> str:
    if not has_pending:
        return "evidence"
    return {
        "confirm": "confirmation",
        "negative": "correction",
        "unknown": "unknown",
        "skip": "skip",
        "correction": "correction",
    }.get(classify_reply(content), "evidence")


def enqueue_inbox_job(
    conn,
    content: str,
    *,
    thread_id: int | None,
    channel: str,
    idempotency_key: str | None,
) -> dict[str, Any]:
    """Persist the input envelope in one transaction and return its job."""

    key = _job_key(idempotency_key)
    with conn.cursor() as cur:
        # Serialize same-key submissions before checking and inserting the
        # evidence/message pair, avoiding orphan rows on a concurrent retry.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
        cur.execute(
            """
            SELECT id, thread_id, raw_evidence_id, user_message_id,
                   idempotency_key, status, error_message, attempts,
                   created_at, started_at, completed_at, updated_at
            FROM v2_inbox_processing_jobs WHERE idempotency_key=%s
            """,
            (key,),
        )
        existing = cur.fetchone()
    if existing:
        return dict(existing)

    if thread_id is None:
        thread = create_thread(conn, channel=channel, mode="learn")
    else:
        thread = get_thread(conn, int(thread_id))
    current_thread_id = int(thread["id"])
    _lock_thread(conn, current_thread_id)
    pending = _pending_proposal(conn, current_thread_id)
    pending_batch = _pending_batch(conn, current_thread_id)
    has_pending = bool(pending or pending_batch)
    mode = classify_input_mode(
        content,
        pending_question=pending.get("clarification_question") if pending else None,
        has_pending=has_pending,
    )
    evidence = _insert_evidence(
        conn,
        content,
        current_thread_id,
        channel=channel,
        input_mode=mode,
    )
    message = _insert_message(
        conn,
        current_thread_id,
        "user",
        _message_type(content, has_pending),
        content,
        int(evidence["id"]),
    )
    job = create_processing_job(
        conn,
        thread_id=current_thread_id,
        raw_evidence_id=int(evidence["id"]),
        user_message_id=int(message["id"]),
        idempotency_key=key,
    )
    return dict(job)


def process_inbox_job(
    job_id: int,
    *,
    db_factory,
    llm_service=None,
    embedding_client=None,
    question_budget: int = 5,
) -> None:
    """Claim and execute one job using an independent database connection."""

    with db_factory() as conn:
        job = claim_processing_job(conn, int(job_id))
        if not job:
            return
        # Make the visible processing state durable before the potentially
        # long LLM call.  Pollers must never wait on that transaction.
        conn.commit()
        try:
            # If a worker crashed after learn_turn committed but before the
            # status update, completing the same job is safe and avoids a
            # second ingestion on recovery.
            existing = get_processing_job(conn, int(job_id))
            if existing and existing.get("assistant_message_id"):
                complete_processing_job(conn, int(job_id))
                return
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content FROM v2_raw_evidence WHERE id=%s
                    """,
                    (job["raw_evidence_id"],),
                )
                evidence = dict(cur.fetchone())
                cur.execute(
                    """
                    SELECT id, thread_id, content, raw_evidence_id
                    FROM v2_inbox_messages WHERE id=%s
                    """,
                    (job["user_message_id"],),
                )
                user_message = dict(cur.fetchone())
            learn_turn(
                conn,
                evidence["content"],
                thread_id=int(job["thread_id"]),
                channel="inbox",
                llm_service=llm_service,
                embedding_client=embedding_client,
                question_budget=question_budget,
                persisted_evidence=evidence,
                persisted_user_message=user_message,
            )
            # The assistant response is durable before the final status flip;
            # startup recovery can safely finish a job interrupted between the
            # two commits.
            conn.commit()
            complete_processing_job(conn, int(job_id))
        except Exception as exc:
            log.exception("V2 Inbox job failed job_id=%s", job_id)
            conn.rollback()
            fail_processing_job(conn, int(job_id), str(exc)[:1000])


def recover_inbox_jobs(*, db_factory, process_job) -> None:
    """Requeue jobs left processing by a process crash, then hand them off."""

    try:
        with db_factory() as conn:
            job_ids = list_unfinished_job_ids(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE v2_inbox_processing_jobs
                    SET status='queued', started_at=NULL, updated_at=CURRENT_TIMESTAMP
                    WHERE status='processing'
                    """
                )
        for job_id in job_ids:
            process_job(int(job_id))
    except Exception:
        log.exception("V2 Inbox job recovery failed")


def retry_inbox_job(conn, job_id: int) -> dict[str, Any] | None:
    return retry_processing_job(conn, int(job_id))


__all__ = [
    "enqueue_inbox_job",
    "process_inbox_job",
    "recover_inbox_jobs",
    "retry_inbox_job",
]
