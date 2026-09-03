# Repository Guidelines

## Project Structure & Module Organization

This is a Python FastAPI service backed by PostgreSQL and OpenRouter Nemotron. The HTTP
application is in `app.py`; shared LLM behavior and provider constraints are in
`llm.py`, with reusable helpers in `helpers.py`. Batch and migration utilities
are top-level scripts such as `run_knowledge_intents_v1_1.py`,
`build_verified_knowledge_pilot.py`, `import_d1.py`, and `apply_migration.py`.
Tests are `test_*.py`. Database definitions live in `schema.sql` and
`migrations/`; source exports are under `d1-export/`. Review artifacts and
documents are kept in `data/`, while deployment configuration is in `deploy/`.

## Build, Test, and Development Commands

Create/use the local virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete test suite:

```bash
python -m unittest discover -p 'test_*.py'
```

Start the API locally (requires PostgreSQL and environment configuration):

```bash
DATABASE_URL=postgresql://... python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Apply schema changes with `python apply_migration.py migrations/<file>.sql`.
The LLM batch scripts require the protected OpenRouter token; do not run them against
production data without confirming their output and checkpoint behavior.

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, `snake_case` for functions and variables,
`PascalCase` for classes, and descriptive `UPPER_SNAKE_CASE` constants. Keep
provider/model behavior centralized in `llm.py`. Match the existing standard
library style; no repository formatter or linter is configured, so run the test
suite and keep changes small and readable.

## Testing Guidelines

Add regression tests in the relevant `test_*.py` module using `unittest.TestCase`
and names beginning with `test_`. Mock OpenRouter and external services rather
than making network calls. Preserve coverage for retry limits, fail-closed
configuration, scope handling, and migration helpers.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout, so use short imperative commit
subjects, for example `Add verified knowledge publish migration`. Pull requests
should explain the behavior and data/schema impact, list test commands run, and
include screenshots for `/review` UI changes. Call out required environment
variables, migrations, deployment steps, or rollback considerations.

## Security & Configuration Tips

Keep the OpenRouter token and `DATABASE_URL` in protected configuration (mode
600), never in source or committed data. OpenRouter models are intentionally
fixed in source; do not add provider fallback or model-selection configuration.
Treat SQL exports and review artifacts as
potentially sensitive.
