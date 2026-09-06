"""Background execution for V2 document jobs (Phase 4.1 parse, 4.2 learn).

The single worker claims one due job at a time and never holds a database
lock across model calls.  Parsing is local CPU work over bounded files;
each learn step runs exactly one bounded extraction for one section
context.  Terminal content errors fail the job and mark the version;
transient errors come back with backoff, at most five attempts.
"""

from __future__ import annotations

import logging
from typing import Any

from v2.document_learning import (
    DocumentLearnError,
    extract_units,
    learn_job_context,
    save_unit_proposals,
    validate_units,
)
from v2.documents import (
    DocumentError,
    claim_document_job,
    complete_document_job,
    fail_document_job,
    fail_version,
    get_document_job,
    get_version,
    parse_file,
    save_parsed_blocks,
    version_file_path,
)

log = logging.getLogger("aihelper.v2.document_processing")

MAX_JOB_ATTEMPTS = 5


def process_document_job(
    job_id: int, *, db_factory, base_dir, stages: tuple[str, ...] = ("parse", "learn"),
    llm_service=None,
) -> None:
    """Run one claimed-or-due document job to completion or a clean failure."""

    with db_factory() as conn:
        job = get_document_job(conn, int(job_id))
        if not job or job.get("status") not in ("queued", "processing"):
            return
        if job.get("stage") not in stages:
            return
        if job.get("status") == "queued":
            claimed = claim_document_job(conn, stages)
            conn.commit()
            if not claimed or int(claimed["id"]) != int(job_id):
                return
            job = claimed
        else:
            conn.commit()
        version_id = int(job["version_id"])
        attempts = int(job.get("attempts") or 0)
    try:
        if str(job.get("stage") or "") == "learn":
            summary = _process_learn_step(
                job, db_factory=db_factory, llm_service=llm_service,
            )
        else:
            summary = _process_parse_step(version_id, db_factory=db_factory, base_dir=base_dir)
        with db_factory() as conn:
            complete_document_job(conn, int(job_id), summary)
            conn.commit()
        if str(job.get("stage") or "") == "learn":
            _maybe_complete_version(version_id, db_factory=db_factory)
    except Exception as exc:
        log.exception("V2 document job failed job_id=%s", job_id)
        terminal = isinstance(exc, (DocumentError, DocumentLearnError)) or attempts + 1 >= MAX_JOB_ATTEMPTS
        with db_factory() as conn:
            fail_document_job(conn, int(job_id), str(exc)[:2000], retryable=not terminal)
            if terminal and str(job.get("stage") or "") == "parse":
                fail_version(conn, version_id, str(exc)[:1000])
            if terminal and str(job.get("stage") or "") == "learn":
                _mark_version_learn_failed(conn, version_id)
            conn.commit()


def _process_parse_step(version_id: int, *, db_factory, base_dir) -> dict:
    with db_factory() as conn:
        version = get_version(conn, version_id)
        if version is None:
            raise DocumentError(f"document version {version_id} is gone")
        path = version_file_path(base_dir, version)
        blocks = parse_file(path, str(version.get("file_type") or ""))
    with db_factory() as conn:
        counts = save_parsed_blocks(conn, version_id, blocks, base_dir=base_dir)
        conn.commit()
    return {
        "blocks": counts["blocks"],
        "evidence": counts["evidence"],
        "assets": counts["assets"],
        "needs_review": counts["needs_review"],
    }


def _process_learn_step(job: dict, *, db_factory, llm_service) -> dict:
    with db_factory() as conn:
        version, context = learn_job_context(conn, job)
    if not context.get("blocks"):
        return {"context": context.get("context_key"), "proposals": 0, "empty": True}
    units = extract_units(llm_service, version, context)
    with db_factory() as conn:
        valid, errors = validate_units(conn, int(version["id"]), units)
        proposals = save_unit_proposals(
            conn, int(version["id"]), str(context.get("context_key") or ""), valid,
        )
        _mark_unused_blocks_evidence_only(
            conn, int(version["id"]), context.get("blocks") or [], proposals,
        )
        conn.commit()
    for error in errors:
        log.warning("document extraction rejected %s: %s",
                    error.get("title"), error.get("error"))
    if errors and not valid:
        raise DocumentLearnError(
            f"context {context.get('context_key')!r} produced no valid units: "
            + "; ".join(str(item.get("error") or "") for item in errors[:3])
        )
    return {
        "context": context.get("context_key"),
        "proposals": len(proposals),
        "rejected": len(errors),
    }


def _mark_unused_blocks_evidence_only(conn, version_id: int, context_blocks: list,
                                        proposals: list) -> None:
    """Blocks that taught nothing stay as evidence with a reason.

    Contexts partition blocks, so an uncited block will never meet another
    extraction; leaving it pending would fake unfinished work forever.
    """

    cited = set()
    for proposal in proposals:
        details = proposal.get("details_json") or {}
        for source in details.get("sources") or []:
            try:
                cited.add(int(source.get("block_id")))
            except (TypeError, ValueError):
                continue
    unused = [
        int(item["block_id"]) for item in context_blocks
        if int(item["block_id"]) not in cited and str(item.get("text") or "").strip()
    ]
    if not unused:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_document_blocks
            SET processing_state='evidence_only',
                state_reason='no unit extracted; kept as evidence',
                updated_at=CURRENT_TIMESTAMP
            WHERE version_id=%s AND id = ANY(%s)
              AND processing_state='pending'
            """,
            (int(version_id), unused),
        )


def _maybe_complete_version(version_id: int, *, db_factory) -> None:
    with db_factory() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) AS n FROM v2_document_jobs
                WHERE version_id=%s AND stage='learn'
                  AND status IN ('queued', 'processing')
                """,
                (int(version_id),),
            )
            open_jobs = int((cur.fetchone() or {}).get("n") or 0)
            cur.execute(
                """
                SELECT count(*) AS n FROM v2_document_blocks
                WHERE version_id=%s AND processing_state='pending'
                """,
                (int(version_id),),
            )
            pending = int((cur.fetchone() or {}).get("n") or 0)
            if not open_jobs and not pending:
                cur.execute(
                    """
                    UPDATE v2_document_versions
                    SET status='complete', updated_at=CURRENT_TIMESTAMP
                    WHERE id=%s AND status='learning'
                    """,
                    (int(version_id),),
                )
        conn.commit()


def _mark_version_learn_failed(conn, version_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_document_versions
            SET status='learning_failed', updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status='learning'
            """,
            (int(version_id),),
        )


def recover_document_jobs(*, db_factory, process_job) -> None:
    """Requeue jobs left processing by a crash, then hand them off."""

    try:
        with db_factory() as conn:
            from v2.documents import unfinished_document_job_ids

            job_ids = unfinished_document_job_ids(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE v2_document_jobs
                    SET status='queued', started_at=NULL,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE status='processing'
                    """
                )
            conn.commit()
        for job_id in job_ids:
            process_job(int(job_id))
    except Exception:
        log.exception("V2 document job recovery failed")


__all__ = [
    "MAX_JOB_ATTEMPTS",
    "process_document_job",
    "recover_document_jobs",
]
