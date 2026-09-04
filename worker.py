"""Dedicated worker for durable V2 Inbox processing jobs."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from embeddings import (
    OPENROUTER_EMBEDDING_MODEL,
    OpenRouterEmbeddingClient,
    read_openrouter_token,
)
from llm import OPENROUTER_DEFAULT_MODEL, OpenRouterLLM
from v2.processing import process_inbox_job, recover_inbox_jobs
from v2.service import list_queued_job_ids


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("ai-sales-engineer.inbox-worker")


def db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def main() -> None:
    token_file = Path(os.getenv("OPENROUTER_TOKEN_FILE", "openrouter"))
    token = os.getenv("OPENROUTER_API_KEY", "").strip() or read_openrouter_token(token_file)
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120"))
    question_budget = int(os.getenv("V2_PASSIVE_QUESTION_BUDGET", "5"))

    embedder = None
    llm = None
    if token:
        try:
            embedder = OpenRouterEmbeddingClient(
                token,
                token_file=token_file,
                timeout=timeout,
            )
            llm = OpenRouterLLM(token, timeout=timeout)
            log.info(
                "OpenRouter clients configured embedding=%s llm=%s",
                OPENROUTER_EMBEDDING_MODEL,
                OPENROUTER_DEFAULT_MODEL,
            )
        except Exception:
            log.exception("OpenRouter clients failed to initialize")
    else:
        log.warning("OpenRouter token is not configured; Inbox jobs will fail safely")

    def run_job(job_id: int) -> None:
        process_inbox_job(
            int(job_id),
            db_factory=db,
            llm_service=llm,
            embedding_client=embedder,
            question_budget=question_budget,
        )

    # Requeue jobs left in processing by a worker or host restart before
    # entering the normal FIFO polling loop.
    recover_inbox_jobs(db_factory=db, process_job=run_job)
    while True:
        try:
            with db() as conn:
                job_ids = list_queued_job_ids(conn, limit=1)
            if job_ids:
                run_job(job_ids[0])
            else:
                time.sleep(1)
        except Exception:
            log.exception("Inbox worker loop failed; retrying")
            time.sleep(5)


if __name__ == "__main__":
    main()
