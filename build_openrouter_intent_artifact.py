"""Materialize an OpenRouter-only view of the resumable V1.1 checkpoint.

The checkpoint may contain pre-migration rows for audit purposes.  Runtime
imports must never consume those rows as current intent data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, output: Path) -> int:
    latest = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("provider") != "openrouter":
            continue
        case_id = item.get("support_case_id")
        if case_id is not None:
            latest[int(case_id)] = item
    intents = [latest[key] for key in sorted(latest)]
    payload = {
        "schema_version": "v1.1-openrouter-only",
        "artifact_type": "openrouter_intents",
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "source_checkpoint": str(source),
        "processed_cases": len(intents),
        "intents": intents,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint = output.with_name("knowledge_intents_v1_1_openrouter.jsonl")
    checkpoint.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in intents),
        encoding="utf-8",
    )
    return len(intents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path("data/knowledge_intents_v1_1_openrouter.jsonl"))
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/knowledge_intents_v1_1_openrouter.json"))
    args = parser.parse_args()
    print(json.dumps({"openrouter_intents": build(args.source, args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
