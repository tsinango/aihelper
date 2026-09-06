# Repository Guidelines

## Project Overview

This is **aihelper**, a Python
FastAPI service backed by PostgreSQL (with pgvector) that answers customer
questions about Hikvision-family security products in Russian. OpenRouter is
the sole model provider: Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`)
for generation, Nemotron 3 Embed 1B (`nvidia/nemotron-3-embed-1b:free`, 2048
dimensions) for embeddings, and an optional Llama Nemotron Rerank model for the
retrieval second pass. Models are fixed in source; there is deliberately no
provider or model fallback — the service is fail-closed on persistent LLM
failures.

The codebase is in transition between two generations:

- **V1** (production): grounded QA at `/api/v1/query`, a Telegram webhook
  (`/telegram/webhook`), and a human knowledge-review UI (`/review`,
  `/review/published`) where reviewers approve Telegram-derived support-case
  knowledge into published `verified_knowledge`. Only published knowledge may
  answer customers; historical case memory is recall/reviewer evidence only.
- **V2** (Phases 1–4.1 implemented): a new Inbox-first
  learning loop under `v2/` with its own `v2_`-prefixed tables, pages
  (`/inbox`, `/knowledge`, `/documents`, `/chat`) and `/api/v2/*` routes.
  Knowledge carries one of four trust values (`official_source`,
  `user_confirmed`, `provisional`, `conflicted`) and only confirmed/official
  knowledge may eventually pass the customer-answer gate. V2 is isolated from
  V1 tables and code paths and must stay additive. The current V2 main
  navigation exposes Inbox, Knowledge, and Documents; Chat is the internal
  engineer QA page (`POST /api/v2/answers`, read-only over trusted Knowledge;
  not a customer-answer gate).

Key design docs: `README.md` (usage and pipeline), `OPERATIONS.md` (runbook:
systemd, batch jobs, review workflow), `TECHNICAL_STATUS_AND_REMEDIATION.md`
(architecture status and data snapshot, in Chinese), `V2_REFACTOR_PLAN.md`
(V2 phases, KEEP/MIGRATE/REWRITE/DEPRECATE, in Chinese),
`V2_FUTURE_DESIGN_NOTES.md` (explicitly forbidden V2 features).

## Project Structure & Module Organization

- `app.py` (~4,000 lines) — the FastAPI application: `/health`, `/ready`,
  `/telegram/webhook`, `/api/v1/query`, document/support-case APIs, the full
  `/api/review/*` surface, V1 page routes, and the `/api/v2/*` and V2 page
  routes. Query pipeline: deterministic model/alias recognition and routing
  (`helpers.py`), hybrid retrieval (structured facts, published
  `verified_knowledge`, document chunks, approved learning examples, optional
  rerank), scope conflict checks, then Nemotron composes the Russian answer;
  requests and retrieval traces are written to the `questions` table.
- `llm.py` — the only LLM interface: `LLMService` protocol and
  `OpenRouterLLM` with a bounded retry policy (`OPENROUTER_MAX_RETRIES`,
  transient status codes), `temperature=0`, and `reasoning.effort=none`.
- `embeddings.py` / `rerank.py` — OpenRouter embedding (2048-dim) and rerank
  clients with their own bounded retries.
- `helpers.py` — deterministic, model-free text utilities: language detection,
  product-model identifier extraction, alias expansion, scope matching
  (`exact`/`family`/`conflict`), question routing.
- `telegram_relations.py` — conservative Telegram message role/relation
  classification; manual relations outrank inferred ones.
- `logging_security.py` — in-memory Telegram bot token redaction for logs.
- `v2/` — `service.py` (SQL persistence and Knowledge maintenance/pruning),
  `learning.py` (text-only learning state machine; only an explicit user
  confirmation moves knowledge to `user_confirmed`), `compare.py` (LLM-assisted
  NEW/CONFIRM/ENRICH/CONFLICT/UNCLEAR decisions with safety checks owned by
  Python), `processing.py` (durable PostgreSQL-backed Inbox jobs),
  `bulk.py` (session-based bulk intake), `answering.py` (Phase 3.1 read-only
  internal QA: states, grounded drafts, citation validation, answer-run
  persistence; never writes learning tables), `feedback.py` (Phase 3.2
  correction loop: reply_only/save_experience/gap kinds, idempotent explicit
  confirm with revision checks, retest-as-new-run, human verdicts; confirm
  is pure database work with no LLM calls), `documents.py` (Phase 4.1
  structured PDF/PPTX intake: immutable versions, page/slide/table/image/
  notes blocks over raw evidence, parse jobs), `document_processing.py`
  (single-worker document job steps, inbox-first interleave), `organization.py` (small local
  Entity organization; automatic LLM organization is off by default), and
  `retrieval.py` (small-corpus lexical + vector retrieval over `v2_knowledge`;
  `retrieve_for_answer()` is the separate eligibility-gated answer entry
  point — never reuse `retrieve_learning_knowledge()` for answers).
- `worker.py` — Inbox worker entrypoint polled by `aihelper-inbox-worker`; it
  claims durable `v2_inbox_processing_jobs` rows only.
- `templates/` — V2 pages (`inbox.html`, `knowledge.html`, `documents.html`,
  `chat.html`, Chinese UI). `review.html` and `published.html` at repo root
  serve the V1 review UI.
- `schema.sql` + `migrations/` — additive SQL migrations `001`–`020`;
  `013_v2_skeleton.sql` through `017_v2_inbox_worker_heartbeat.sql` are the V2
  learning/job/worker tables, `018_v2_organization.sql` is the lightweight
  entity/relation layer,   `019_v2_knowledge_history.sql` is the Knowledge audit
  history, and `020_v2_entity_pruning.sql` adds soft-pruning timestamps.
  `021_v2_answer_runs.sql` adds the Phase 3.1 answer-run table (idempotency
  key unique, immutable evidence snapshots).
  `022_v2_feedback.sql` adds the Phase 3.2 correction loop (`v2_answer_feedback`,
  Knowledge `unit_kind`/`applicability`/`revision`, proposal unit metadata,
  run `retest_of`/`feedback_id`/human-verdict columns, `confirm` history action).
  `023_v2_documents.sql` adds Phase 4.1 intake (`v2_document_versions`,
  `v2_document_blocks`, `v2_document_jobs`, Knowledge
  `origin_document_version_id`/`validation_status` for later phases).
  `apply_migration.py` records SHA-256 checksums in `schema_migrations` and
  rejects changed or out-of-tree migration files.
- Batch/offline scripts (top level): `organize_telegram_knowledge.py`
  (deterministic review workbook from Telegram exports, no LLM/DB writes),
  `import_review_candidates.py`, `import_case_memory.py`, `reembed.py`
  (backfills embeddings asynchronously), `run_knowledge_intents_v1_1.py` /
  `build_openrouter_intent_artifact.py` (offline intent batch with resume
  checkpoints and 429 rate-limit markers), `run_topic_abstraction.py`,
  `build_verified_knowledge_pilot.py`, `import_d1.py`, `evaluate_local_qwen.py`
  (shadow-only local Qwen3.5 2B/4B GGUF benchmark against
  `data/golden_set.json`; never serves production traffic), `evaluate_v2.py`
  (Phase 3.0 evaluation baseline: offline completeness/fixed-selection checks
  and optional retrieval/trust baseline; run without arguments, `--database-url`
  adds a DB baseline — never fabricates answer accuracy).
- `data/` — ignored review artifacts, golden set, local documents, batch
  checkpoints. `data/golden_set.json` (135 samples) is immutable and must not
  be edited; V2 eval inputs live only in the sidecar `data/v2_eval_cases.json`
  (fixed 30-case selection, currently `pending_expert_mapping`).
  `d1-export/` — ignored source SQL exports.
- `deploy/` — `Caddyfile` (web proxy) plus the two checked-in systemd unit
  files (`aihelper.service`, `aihelper-inbox-worker.service`).

## Build, Test, and Development Commands

Use the project virtual environment (Python 3.10; the bare `python` command
does not exist on this host):

```bash
python3 -m venv .venv           # if needed
.venv/bin/python -m pip install -r requirements.txt
```

Run the complete test suite (current baseline: 341 tests; 327 pass and 14
skip without a database — the PostgreSQL integration tests — or all 341 pass
with `V2_TEST_DATABASE_URL` set. Everything else requires no database
or network). Both invocations collect the same suite; `pytest` is primary per
README, and `compileall` is a cheap import sanity check:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m unittest discover -p 'test_*.py'
.venv/bin/python -m compileall -q app.py llm.py reembed.py worker.py v2
```

Start the API locally (requires PostgreSQL and environment configuration):

```bash
DATABASE_URL=postgresql://... .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Apply schema changes (migrations must live in `migrations/`):

```bash
.venv/bin/python apply_migration.py migrations/<file>.sql
```

Production runs as the systemd units `aihelper.service` and
`aihelper-inbox-worker.service` (working directory
`/opt/aihelper`, uvicorn on `127.0.0.1:8000`); see `OPERATIONS.md` for
start/stop/log commands.

## Configuration & Security

Non-secret configuration lives in `/etc/aihelper.env` (root:ubuntu, mode 640 so
the systemd service user can read it):

```dotenv
OPENROUTER_TOKEN_FILE=/opt/aihelper/openrouter
OPENROUTER_TIMEOUT_SECONDS=120
OPENROUTER_RERANK_ENABLED=true
V2_LEARNING_MODEL=openai/gpt-oss-20b:free
INBOX_WORKER_NAME=aihelper-inbox-worker
INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS=10
INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS=45
```

Secrets are the OpenRouter token (`/opt/aihelper/openrouter`, mode 600, or
`OPENROUTER_API_KEY`) and the Telegram bot token (`/opt/aihelper/tgtoken`,
mode 600). Never commit tokens, `DATABASE_URL`, SQL exports, or `data/`
artifacts — all are gitignored. Review APIs require the same `x-api-key`
authentication as the service; the review page keeps the key in browser
session storage only. Do not add provider fallback, model selection, or
alternative LLM providers; the LLM batch scripts require the protected token
— do not run them against production data without confirming output and
checkpoint behavior (`OPERATIONS.md` documents 429/quota resume behavior).

## Coding Style & Naming Conventions

Python 3, four-space indentation, `snake_case` functions/variables,
`PascalCase` classes, descriptive `UPPER_SNAKE_CASE` constants. Keep provider
and model behavior centralized in `llm.py`, `embeddings.py`, and `rerank.py`.
Docstrings/comments are in English (some product-facing strings and docs are
Russian or Chinese). No formatter or linter is configured — keep changes
small and readable and run the test suite.

## Testing Guidelines

Regression tests live in `test_*.py` at the repo root using
`unittest.TestCase` with `test_`-prefixed methods. Mock OpenRouter and other
external services instead of making network calls; the suite must pass
without a database. PostgreSQL-dependent tests skip unless
`V2_TEST_DATABASE_URL` points at an initialized schema (run them with the
env var set; they use outer transactions and roll back their writes). Preserve
coverage for:
bounded retry limits and fail-closed behavior, scope matching, migration
checksum helpers, Telegram token log redaction, and the V2 learning/compare
state machine. `test_v2_postgres.py` covers the `v2_` tables against a real
database when one is available. The repo-root `app/` directory is an empty
leftover; the application source is `app.py`.

## Commit & Pull Request Guidelines

Git history is available in this checkout. Use short imperative commit
subjects (recent examples: `Add V2 learning retrieval comparison`,
`Complete Phase 2 compare and clarification loop`). Before each commit run
the full test suite (above) and check `git diff --check` and `git status` — do
not commit ignored secrets, data exports, or temp files. Pull requests should
explain behavior and data/schema impact, list test commands run, and include
screenshots for `/review` or V2 page UI changes; call out required
environment variables, migrations, deployment steps, and rollback
considerations. V2 work follows the phase discipline in
`V2_REFACTOR_PLAN.md` and `astra.md`: keep V1 routes and tests passing, keep
migrations additive. Phase 2.2 is closed; Phase 3.0 (organization closure +
UX gate + evaluation baseline) is implemented — automatic LLM organization
after confirmation is off by default (`V2_ORGANIZATION_LLM_ENABLED`). Phase
3.1 (read-only internal QA: `retrieve_for_answer()`, `v2/answering.py`,
`v2_answer_runs`, `POST/GET /api/v2/answers`, Chat QA page) is implemented;
Phase 3.2 (correction loop: `v2/feedback.py`, feedback/confirm/retest/verdict
APIs, Chat correction UI, Inbox gap filter) is implemented; production
acceptance evidence lives in gitignored `data/phase32_acceptance.json`.
There is still no
customer-answer gate: only the internal engineer draft endpoint may read
trusted Knowledge. Before committing, also run `git diff --check`.
