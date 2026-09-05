#!/usr/bin/env python3
"""Phase 3.0 V2 evaluation-readiness runner.

This runner establishes the evaluation baseline contract for Phase 3.1; it
deliberately does NOT measure answer accuracy.  There is no V2 Answer Service
yet, so any "V2 answer baseline" would be fabricated.  What this script does,
without any LLM call or network access:

1. Validates the integrity of the immutable golden set (``data/golden_set.json``):
   existence, sample count, required fields, allowed labels, unique keys.
2. Validates the V2 sidecar (``data/v2_eval_cases.json``): schema, sample-key
   cross references, agreement with golden labels, and the fixed quota of
   15 answerable / 5 clarify / 5 unsupported / 5 boundary cases.
3. Optionally, with ``--database-url``, checks V2 Knowledge readiness and a
   lexical retrieval baseline over ``v2_knowledge`` (trust gate, accepted
   sources, expected-knowledge recall) — the metrics that can be truthfully
   reported before an Answer Service exists.

The sidecar only carries V2-specific information (knowledge mapping, human
expectations, applicability, paraphrases, forbidden assertions).  It never
modifies or overrides the golden set.  Cases whose trusted V2 Knowledge does
not exist yet stay ``pending_expert_mapping`` and are reported as gaps, never
silently filled.

Usage:
    .venv/bin/python evaluate_v2.py
    .venv/bin/python evaluate_v2.py --database-url postgresql://... \
        --report data/v2_eval_report.json

Exit status is non-zero when any integrity check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_GOLDEN_SET = Path("data/golden_set.json")
DEFAULT_SIDECAR = Path("data/v2_eval_cases.json")
DEFAULT_REPORT = Path("data/v2_eval_report.json")

ANSWER_STATUSES = ("answered", "needs_clarification", "unsupported", "service_error")
TRUSTED_FOR_ANSWER = ("official_source", "user_confirmed")
CATEGORIES = ("answerable", "clarify", "unsupported", "boundary")
QUOTA = {"answerable": 15, "clarify": 5, "unsupported": 5, "boundary": 5}

GOLDEN_FALLBACK_REQUIRED_FIELDS = (
    "sample_key",
    "expected_answer_status",
    "expected_scope",
    "expected_knowledge_keys",
    "evidence_case_ids",
    "must_clarify",
    "must_refuse",
)

SIDECAR_REQUIRED_FIELDS = ("sample_key", "category", "expected_answer_status")


def _load_json(path: Path, errors: list[str]) -> Any:
    if not path.exists():
        errors.append(f"missing file: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None


def check_golden_set(golden: dict | None, errors: list[str], warnings: list[str]) -> dict:
    """Validate the immutable golden set; return samples keyed by sample_key."""

    samples_by_key: dict[str, dict] = {}
    if not isinstance(golden, dict):
        if golden is not None:
            errors.append("golden set root must be a JSON object")
        return samples_by_key
    samples = golden.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("golden set has no samples list")
        return samples_by_key
    labeling = golden.get("labeling") or {}
    required_fields = labeling.get("required_fields")
    if not isinstance(required_fields, list) or not required_fields:
        warnings.append("golden set declares no labeling.required_fields; using fallback set")
        required_fields = list(GOLDEN_FALLBACK_REQUIRED_FIELDS)
    declared = golden.get("coverage", {}).get("sample_count")
    if declared is not None and int(declared) != len(samples):
        warnings.append(
            f"golden coverage declares {declared} samples but file holds {len(samples)}"
        )
    for index, sample in enumerate(samples):
        label = sample.get("sample_key") if isinstance(sample, dict) else None
        label = label or f"index {index}"
        if not isinstance(sample, dict):
            errors.append(f"golden sample {label} is not an object")
            continue
        for field in required_fields:
            if field not in sample:
                errors.append(f"golden sample {label} missing field {field}")
        key = sample.get("sample_key")
        if not isinstance(key, str) or not key.strip():
            errors.append(f"golden sample {label} has an empty sample_key")
            continue
        if key in samples_by_key:
            errors.append(f"duplicate golden sample_key: {key}")
        samples_by_key[key] = sample
        status = sample.get("expected_answer_status")
        if status not in ANSWER_STATUSES:
            errors.append(
                f"golden sample {key} has unexpected expected_answer_status: {status!r}"
            )
        if not str(sample.get("question") or "").strip():
            warnings.append(f"golden sample {key} has an empty question")
    return samples_by_key


def check_sidecar(
    sidecar: dict | None,
    samples_by_key: dict[str, dict],
    errors: list[str],
    warnings: list[str],
) -> list[dict]:
    """Validate the V2 sidecar and return its cases."""

    cases: list[dict] = []
    if not isinstance(sidecar, dict):
        if sidecar is not None:
            errors.append("sidecar root must be a JSON object")
        return cases
    raw_cases = sidecar.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        errors.append("sidecar has no cases list")
        return cases
    seen: set[str] = set()
    for index, case in enumerate(raw_cases):
        label = case.get("sample_key") if isinstance(case, dict) else None
        label = label or f"index {index}"
        if not isinstance(case, dict):
            errors.append(f"sidecar case {label} is not an object")
            continue
        for field in SIDECAR_REQUIRED_FIELDS:
            if field not in case:
                errors.append(f"sidecar case {label} missing field {field}")
        key = case.get("sample_key")
        sample = samples_by_key.get(key)
        if sample is None:
            errors.append(f"sidecar case {label} references unknown golden sample_key")
            continue
        if key in seen:
            errors.append(f"duplicate sidecar sample_key: {key}")
        seen.add(key)
        category = case.get("category")
        if category not in CATEGORIES:
            errors.append(f"sidecar case {key} has unknown category: {category!r}")
        status = case.get("expected_answer_status")
        if status not in ANSWER_STATUSES:
            errors.append(f"sidecar case {key} has unexpected expected_answer_status: {status!r}")
        elif status != sample.get("expected_answer_status"):
            errors.append(
                f"sidecar case {key} disagrees with golden label: "
                f"{status} != {sample.get('expected_answer_status')}"
            )
        if category == "answerable" and status != "answered":
            errors.append(f"sidecar case {key} is answerable but labelled {status}")
        if category == "clarify" and status != "needs_clarification":
            errors.append(f"sidecar case {key} is clarify but labelled {status}")
        if category == "unsupported" and status != "unsupported":
            errors.append(f"sidecar case {key} is unsupported but labelled {status}")
        for field in ("v2_knowledge_ids", "paraphrases", "forbidden_assertions"):
            if field in case and not isinstance(case[field], list):
                errors.append(f"sidecar case {key} field {field} must be a list")
        if not str(sample.get("question") or "").strip():
            warnings.append(
                f"sidecar case {key}: golden sample has no question text; "
                "not usable as a Phase 3.1 QA prompt until one is supplied"
            )
        if not case.get("v2_knowledge_ids"):
            warnings.append(f"sidecar case {key} has no V2 Knowledge mapping yet")
        cases.append(case)
    counts = {category: 0 for category in CATEGORIES}
    for case in cases:
        if case.get("category") in counts:
            counts[case["category"]] += 1
    for category, quota in QUOTA.items():
        if counts[category] < quota:
            errors.append(
                f"fixed selection quota not met: {category} has {counts[category]}, needs {quota}"
            )
    return cases


def check_knowledge_readiness(conn, cases: list[dict], samples_by_key: dict[str, dict], errors: list[str]) -> dict:
    """Check V2 Knowledge eligibility and lexical recall for mapped cases."""

    from v2.retrieval import retrieve_learning_knowledge

    metrics: dict[str, Any] = {
        "mapped_cases": 0,
        "knowledge_rows_checked": 0,
        "knowledge_eligibility_failures": [],
        "retrieval_top5_recall": None,
        "retrieval_hits": 0,
        "retrieval_attempted": 0,
    }
    for case in cases:
        knowledge_ids = [int(item) for item in case.get("v2_knowledge_ids") or []]
        if not knowledge_ids:
            continue
        metrics["mapped_cases"] += 1
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT k.id, k.trust, k.active,
                       EXISTS (
                           SELECT 1 FROM v2_knowledge_sources s
                           WHERE s.knowledge_id = k.id
                             AND s.active = TRUE
                             AND s.relation = 'supports'
                             AND s.resolution = 'accepted'
                       ) AS has_accepted_source
                FROM v2_knowledge k
                WHERE k.id = ANY(%s)
                """,
                (knowledge_ids,),
            )
            rows = {int(row["id"]): dict(row) for row in cur.fetchall()}
        for knowledge_id in knowledge_ids:
            metrics["knowledge_rows_checked"] += 1
            row = rows.get(knowledge_id)
            if row is None:
                metrics["knowledge_eligibility_failures"].append(
                    f"case {case['sample_key']}: knowledge {knowledge_id} not found"
                )
            elif row["trust"] not in TRUSTED_FOR_ANSWER:
                metrics["knowledge_eligibility_failures"].append(
                    f"case {case['sample_key']}: knowledge {knowledge_id} trust={row['trust']}"
                )
            elif not row["active"]:
                metrics["knowledge_eligibility_failures"].append(
                    f"case {case['sample_key']}: knowledge {knowledge_id} inactive"
                )
            elif not row["has_accepted_source"]:
                metrics["knowledge_eligibility_failures"].append(
                    f"case {case['sample_key']}: knowledge {knowledge_id} lacks an accepted supports source"
                )
    answerable = [
        case for case in cases
        if case.get("category") == "answerable" and case.get("v2_knowledge_ids")
    ]
    hits = 0
    for case in answerable:
        sample = samples_by_key.get(case["sample_key"]) or {}
        question = str(sample.get("question") or "").strip()
        if not question:
            continue
        metrics["retrieval_attempted"] += 1
        expected = {int(item) for item in case.get("v2_knowledge_ids") or []}
        hits_now = retrieve_learning_knowledge(conn, question, embedder=None, top_k=5)
        if expected & {int(item["id"]) for item in hits_now}:
            hits += 1
    if metrics["retrieval_attempted"]:
        metrics["retrieval_top5_recall"] = round(hits / metrics["retrieval_attempted"], 4)
        metrics["retrieval_hits"] = hits
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--database-url",
        default="",
        help="optional PostgreSQL URL for V2 Knowledge readiness and lexical retrieval baseline",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    golden = _load_json(args.golden, errors)
    samples_by_key = check_golden_set(golden, errors, warnings)
    sidecar = _load_json(args.sidecar, errors)
    cases = check_sidecar(sidecar, samples_by_key, errors, warnings)

    gaps = [
        case["sample_key"]
        for case in cases
        if isinstance(case.get("sample_key"), str) and not case.get("v2_knowledge_ids")
    ]

    database_metrics: dict[str, Any] | None = None
    if args.database_url:
        try:
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(args.database_url, row_factory=dict_row) as conn:
                database_metrics = check_knowledge_readiness(conn, cases, samples_by_key, errors)
        except Exception as exc:  # readiness must never crash the runner
            errors.append(f"database readiness check failed: {exc}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "3.0",
        "answer_service": "not_implemented_no_answer_accuracy_is_reported",
        "golden_set": {
            "path": str(args.golden),
            "samples": len(samples_by_key),
        },
        "sidecar": {
            "path": str(args.sidecar),
            "cases": len(cases),
            "counts_by_category": {
                category: sum(1 for case in cases if case.get("category") == category)
                for category in CATEGORIES
            },
            "pending_expert_mapping": len(gaps),
        },
        "database_metrics": database_metrics,
        "warnings": warnings,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        print(f"\nevaluation readiness check FAILED ({len(errors)} errors)", file=sys.stderr)
        return 1
    print("\nevaluation readiness check passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
