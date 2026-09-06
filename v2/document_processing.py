"""Background execution for V2 document jobs (Phase 4.1: parse stage).

The single worker claims one due job at a time and never holds a database
lock across parsing.  Parsing is local CPU work over bounded files; each
job finishes its version in one step and commits before the status flip.
Terminal content errors (bad bytes, over-limit files) fail the job and mark
the version; transient errors come back with backoff, at most five attempts.
"""

from __future__ import annotations

import logging
from typing import Any

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
    job_id: int, *, db_factory, base_dir, stages: tuple[str, ...] = ("parse",),
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
        with db_factory() as conn:
            version = get_version(conn, version_id)
            if version is None:
                raise DocumentError(f"document version {version_id} is gone")
            path = version_file_path(base_dir, version)
            blocks = parse_file(path, str(version.get("file_type") or ""))
        with db_factory() as conn:
            counts = save_parsed_blocks(conn, version_id, blocks, base_dir=base_dir)
            complete_document_job(conn, int(job_id), {
                "blocks": counts["blocks"],
                "evidence": counts["evidence"],
                "assets": counts["assets"],
                "needs_review": counts["needs_review"],
            })
            conn.commit()
    except Exception as exc:
        log.exception("V2 document job failed job_id=%s", job_id)
        terminal = isinstance(exc, DocumentError) or attempts + 1 >= MAX_JOB_ATTEMPTS
        with db_factory() as conn:
            fail_document_job(conn, int(job_id), str(exc)[:2000], retryable=not terminal)
            if terminal and str(job.get("stage") or "") == "parse":
                fail_version(conn, version_id, str(exc)[:1000])
            conn.commit()


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
