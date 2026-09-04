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
- **V2** (in progress, Phases 1–2 of `V2_REFACTOR_PLAN.md`): a new Inbox-first
  learning loop under `v2/` with its own `v2_`-prefixed tables, pages
  (`/inbox`, `/knowledge`, `/documents`, `/chat`) and `/api/v2/*` routes.
  Knowledge carries one of four trust values (`official_source`,
  `user_confirmed`, `provisional`, `conflicted`) and only confirmed/official
  knowledge may eventually pass the customer-answer gate. V2 is isolated from
  V1 tables and code paths and must stay additive.

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
- `v2/` — `service.py` (SQL persistence), `learning.py` (text-only learning
  state machine; only an explicit user confirmation moves knowledge to
  `user_confirmed`), `compare.py` (LLM-assisted NEW/CONFIRM/ENRICH/CONFLICT/
  UNCLEAR decisions with safety checks owned by Python), `retrieval.py`
  (small-corpus lexical + vector retrieval over `v2_knowledge`).
- `templates/` — V2 pages (`inbox.html`, `knowledge.html`, `documents.html`,
  `chat.html`, Chinese UI). `review.html` and `published.html` at repo root
  serve the V1 review UI.
- `schema.sql` + `migrations/` — additive SQL migrations `001`–`018`;
  `013_v2_skeleton.sql` through `016_v2_inbox_processing_jobs.sql` are the V2
  learning/job tables, `017_v2_inbox_worker_heartbeat.sql` is operational
  worker liveness state, and `018_v2_organization.sql` is the lightweight
  entity/relation layer.
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
  `data/golden_set.json`; never serves production traffic).
- `data/` — ignored review artifacts, golden set, local documents, batch
  checkpoints. `d1-export/` — ignored source SQL exports.
- `deploy/Caddyfile` — web-proxy config for the migration deployment.

## Build, Test, and Development Commands

Use the project virtual environment (Python 3.10; the bare `python` command
does not exist on this host):

```bash
python3 -m venv .venv           # if needed
.venv/bin/python -m pip install -r requirements.txt
```

Run the complete test suite (baseline: 124 tests, all passing, 8 skipped —
requires no database or network):

```bash
.venv/bin/python -m unittest discover -p 'test_*.py'
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
without a database (PostgreSQL-dependent tests skip). Preserve coverage for:
bounded retry limits and fail-closed behavior, scope matching, migration
checksum helpers, Telegram token log redaction, and the V2 learning/compare
state machine. `test_v2_postgres.py` covers the `v2_` tables against a real
database when one is available.

## Commit & Pull Request Guidelines

Git history is available in this checkout. Use short imperative commit
subjects (recent examples: `Add V2 learning retrieval comparison`,
`Complete Phase 2 compare and clarification loop`). Before each commit run
the full unittest command and check `git diff --check` and `git status` — do
not commit ignored secrets, data exports, or temp files. Pull requests should
explain behavior and data/schema impact, list test commands run, and include
screenshots for `/review` or V2 page UI changes; call out required
environment variables, migrations, deployment steps, and rollback
considerations. V2 work follows the phase discipline in
`V2_REFACTOR_PLAN.md`: keep V1 routes and tests passing, keep migrations
additive, and stop at the Phase 2 UX gate without manual sign-off.
