#!/usr/bin/env python3
"""Phase 3.1 first real V2 end-to-end QA baseline.

Runs the fixed 30-case sidecar selection through the read-only Answer Service
(``v2.answering.answer_question``) with the production LLM and embedder, and
records statuses, evidence snapshots, latencies, and mechanical critical-error
flags.  It deliberately does NOT grade answers:

- every draft, citation, and evidence snapshot is stored in the report for
  human review (``human_verdict`` stays null until a person fills it in);
- the model never scores its own output; ``critical_flags`` are deterministic
  text/trust checks only (unconfirmed citation, wrong-model token in the
  draft, cited id outside the run's retrieval candidates, forbidden
  assertion substring).

Golden-set and sidecar files are read-only inputs.  Sidecar cases without
expert V2 Knowledge mapping are still executed: with no covering trusted
Knowledge the honest result is ``unsupported``/``needs_clarification``, which
the report shows instead of fabricating coverage.

Each case persists one tagged ``v2_answer_runs`` row (read-only towards
Knowledge; the service never writes learning tables).

Usage:
    .venv/bin/python evaluate_v2_answers.py \
      --database-url postgresql://... --env-file /etc/aihelper.env \
      --report data/v2_eval_phase31_report.json
    # retrieval triage only, no model calls:
    .venv/bin/python evaluate_v2_answers.py --database-url ... --retrieval-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_GOLDEN_SET = Path("data/golden_set.json")
DEFAULT_SIDECAR = Path("data/v2_eval_cases.json")
DEFAULT_REPORT = Path("data/v2_eval_phase31_report.json")

TRUSTED_FOR_ANSWER = ("official_source", "user_confirmed")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_openrouter_key(env_file: Path | None) -> str:
    import os

    if env_file and env_file.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from build_verified_knowledge_pilot import load_env_file

        load_env_file(env_file)
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    from embeddings import read_openrouter_token

    token_file = os.getenv("OPENROUTER_TOKEN_FILE", "openrouter")
    return read_openrouter_token(token_file).strip()


def _answer_models(text: str) -> list[str]:
    import re

    from helpers import identifiers

    return [str(item).upper() for item in identifiers(text)]


def _critical_flags(question: str, run: dict, forbidden: list[str]) -> list[str]:
    """Deterministic flags only; a human still owns the verdict."""

    from helpers import scope_match

    flags: list[str] = []
    snapshot = run.get("evidence_snapshot") or []
    if run.get("answer_status") == "answered":
        if not snapshot:
            flags.append("answered_without_evidence_snapshot")
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            if str(item.get("trust") or "") not in TRUSTED_FOR_ANSWER:
                flags.append(f"unconfirmed_knowledge_cited:{item.get('knowledge_id')}")
            if not (item.get("sources") or []):
                flags.append(f"citation_without_accepted_source:{item.get('knowledge_id')}")
        candidate_ids = {
            int(item) for item in (run.get("retrieval_trace") or {}).get("candidate_knowledge_ids", [])
        }
        for item in snapshot:
            if isinstance(item, dict) and candidate_ids and int(item.get("knowledge_id", -1)) not in candidate_ids:
                flags.append(f"cited_outside_retrieval_candidates:{item.get('knowledge_id')}")
        question_models = _answer_models(question)
        answer_models = _answer_models(run.get("answer_text") or "")
        if question_models and answer_models:
            if scope_match(question_models, answer_models) == "conflict":
                flags.append("conflicting_model_token_in_answer")
        lowered = str(run.get("answer_text") or "").casefold()
        for assertion in forbidden:
            if assertion and str(assertion).casefold() in lowered:
                flags.append(f"forbidden_assertion:{assertion[:80]}")
    return sorted(set(flags))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--database-url", default="", help="PostgreSQL URL (required)")
    parser.add_argument("--env-file", type=Path, default=Path("/etc/aihelper.env"))
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip model calls; record retrieval triage only")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--tag", default="",
                        help="idempotency key prefix (default: phase31-<UTC date>)")
    parser.add_argument("--extra-questions", type=Path, default=None,
                        help="optional JSON list of {key, question} supplementary prompts "
                             "authored against already-confirmed Knowledge; reported "
                             "separately and never counted in the fixed 30")
    args = parser.parse_args()

    if not args.database_url:
        print("error: --database-url is required", file=sys.stderr)
        return 2

    golden = _load_json(args.golden)
    sidecar = _load_json(args.sidecar)
    samples_by_key = {s["sample_key"]: s for s in golden.get("samples", [])}
    cases = list(sidecar.get("cases", []))
    if len(cases) != 30:
        print(f"error: sidecar must hold the fixed 30-case selection, found {len(cases)}",
              file=sys.stderr)
        return 2

    import psycopg
    from psycopg.rows import dict_row

    from v2.answering import V2_ANSWER_PROMPT_VERSION, answer_question

    llm = embedder = None
    if not args.retrieval_only:
        api_key = _resolve_openrouter_key(args.env_file)
        if not api_key:
            print("error: no OpenRouter key (env OPENROUTER_API_KEY or token file)",
                  file=sys.stderr)
            return 2
        from embeddings import OpenRouterEmbeddingClient
        from llm import OpenRouterLLM

        llm = OpenRouterLLM(api_key, timeout=args.timeout)
        embedder = OpenRouterEmbeddingClient(api_key, timeout=args.timeout)

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    tag = args.tag or f"phase31-{day}"
    # Retrieval-only triage uses its own key namespace so a later full run
    # with the same tag never collides with these model-less runs.
    key_prefix = f"{tag}-retrievalonly" if args.retrieval_only else tag

    def db_factory():
        return psycopg.connect(args.database_url, row_factory=dict_row)

    def _run_one(position: int | str, sample_key: str, question: str, *,
                 category: str, expected: str, mapping: str,
                 sidecar_ids: list, forbidden: list) -> dict:
        entry: dict[str, Any] = {
            "position": position,
            "sample_key": sample_key,
            "category": category,
            "expected_answer_status": expected,
            "mapping_status": mapping,
            "sidecar_knowledge_ids": list(sidecar_ids or []),
            "question": question,
        }
        if not question:
            entry.update({
                "answer_status": "skipped", "reason_code": "no_question_text",
                "critical_flags": [], "human_verdict": None,
            })
            return entry
        t0 = time.monotonic()
        try:
            run = answer_question(
                question,
                context={"eval": "phase3.1", "sample_key": sample_key,
                         "retrieval_only": bool(args.retrieval_only)},
                idempotency_key=f"{key_prefix}-{sample_key}",
                db_factory=db_factory,
                llm_service=llm,
                embedding_client=embedder,
            )
        except Exception as exc:  # harness must report, never crash mid-run
            entry.update({
                "answer_status": "harness_error",
                "reason_code": type(exc).__name__,
                "detail": str(exc)[:500],
                "critical_flags": ["harness_error"],
                "human_verdict": None,
            })
            return entry
        wall_ms = int((time.monotonic() - t0) * 1000)
        snapshot = run.get("evidence_snapshot") or []
        entry.update({
            "run_id": run.get("run_id"),
            "answer_status": run.get("answer_status"),
            "expected_match": run.get("answer_status") == expected,
            "reason_code": run.get("reason_code"),
            "answer_text": run.get("answer_text", ""),
            "clarifying_question": run.get("clarifying_question", ""),
            "evidence_knowledge_ids": [
                item.get("knowledge_id") for item in snapshot if isinstance(item, dict)
            ],
            "evidence_snapshot": snapshot,
            "retrieval_trace": run.get("retrieval_trace", {}),
            "model": run.get("model", ""),
            "prompt_version": run.get("prompt_version") or V2_ANSWER_PROMPT_VERSION,
            "llm_requests": run.get("llm_requests", 0),
            "run_latency_ms": run.get("latency_ms", 0),
            "wall_ms": wall_ms,
            "duplicate": bool(run.get("duplicate", False)),
            "critical_flags": _critical_flags(question, run, list(forbidden or [])),
            "human_verdict": None,
        })
        return entry

    results: list[dict] = []
    llm_calls = 0
    for position, case in enumerate(cases, start=1):
        sample_key = case.get("sample_key", "")
        sample = samples_by_key.get(sample_key, {})
        question = str(sample.get("question") or "").strip()
        expected = case.get("expected_answer_status", "")
        entry = _run_one(
            position, sample_key, question,
            category=case.get("category", ""), expected=expected,
            mapping=case.get("mapping_status", ""),
            sidecar_ids=list(case.get("v2_knowledge_ids") or []),
            forbidden=list(case.get("forbidden_assertions") or []),
        )
        llm_calls += int(entry.get("llm_requests") or 0)
        results.append(entry)
        print(
            f"[{position:02d}/30] {sample_key} expected={expected} "
            f"actual={entry.get('answer_status')} reason={entry.get('reason_code')} "
            f"flags={entry['critical_flags'] or 'none'}",
            flush=True,
        )

    by_category: dict[str, dict[str, int]] = {}
    for entry in results:
        bucket = by_category.setdefault(entry.get("category", "?"), {})
        bucket[entry.get("answer_status", "?")] = bucket.get(entry.get("answer_status", "?"), 0) + 1

    supplementary: list[dict] = []
    if args.extra_questions:
        extra = _load_json(args.extra_questions)
        for item in extra if isinstance(extra, list) else []:
            key = str(item.get("key") or "").strip()
            question = str(item.get("question") or "").strip()
            if not key or not question:
                continue
            entry = _run_one(
                key, key, question,
                category="supplementary", expected="",
                mapping="", sidecar_ids=[], forbidden=[],
            )
            llm_calls += int(entry.get("llm_requests") or 0)
            supplementary.append(entry)
            print(
                f"[extra] {key} actual={entry.get('answer_status')} "
                f"reason={entry.get('reason_code')} "
                f"flags={entry.get('critical_flags') or 'none'}",
                flush=True,
            )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "3.1",
        "retrieval_only": bool(args.retrieval_only),
        "idempotency_tag": tag,
        "counts_by_category": by_category,
        "status_match": sum(1 for e in results if e.get("expected_match")),
        "total": len(results),
        "critical_flag_cases": [
            {"sample_key": e["sample_key"], "flags": e["critical_flags"]}
            for e in results if e.get("critical_flags")
        ],
        "total_llm_requests": llm_calls,
        "cases": results,
        "supplementary": supplementary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {args.report} status_match={report['status_match']}/{len(results)} "
          f"flagged={len(report['critical_flag_cases'])} llm_requests={llm_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
