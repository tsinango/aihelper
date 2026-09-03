#!/usr/bin/env python3
"""Materialize human-review Topic Candidates from the existing offline state."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from embeddings import OPENROUTER_EMBEDDING_DIMENSIONS, OPENROUTER_EMBEDDING_MODEL


DEFAULT_INPUT = Path("/home/ubuntu/ai-sales-engineer-knowledge/input/topic_questions_v2_1.jsonl")
DEFAULT_STATE = Path("/home/ubuntu/ai-sales-engineer-knowledge/reports/topic_cluster_calibration_20260826T062527Z.json")
DEFAULT_JSON = Path("/opt/aihelper/data/topic_candidates.json")
DEFAULT_MARKDOWN = Path("/opt/aihelper/TOPIC_REVIEW.md")
THRESHOLD = 0.80
METHOD = "complete"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def unique_models(members: list[dict]) -> list[str]:
    return sorted({str(model) for member in members for model in member.get("models", []) if str(model)}, key=str.casefold)


def candidate_from_cluster(cluster: dict, candidate_id: str) -> dict:
    members_by_case = {}
    for member in cluster["members"]:
        members_by_case.setdefault(int(member["support_case_id"]), member)
    members = [members_by_case[case_id] for case_id in sorted(members_by_case)]
    case_ids = [int(member["support_case_id"]) for member in members]
    if not case_ids:
        raise ValueError(f"cluster {cluster['cluster_index']} has no cases")
    if cluster["representative_support_case_id"] not in case_ids:
        raise ValueError(f"cluster {cluster['cluster_index']} representative is not a member")

    domain_distribution = distribution([member["domain"] for member in members])
    knowledge_type_distribution = distribution([member["knowledge_type"] for member in members])
    return {
        "topic_candidate_id": candidate_id,
        "frequency": len(case_ids),
        "case_ids": case_ids,
        "representative_case_id": int(cluster["representative_support_case_id"]),
        "representative_question": cluster["representative_canonical_question"],
        "canonical_questions": [member["canonical_question"] for member in members],
        "domain_distribution": domain_distribution,
        "knowledge_type_distribution": knowledge_type_distribution,
        "models": unique_models(members),
        "cluster_min_similarity": float(cluster["min_similarity"]),
        "cluster_mean_similarity": float(cluster["mean_similarity"]),
        "taxonomy_mixed": len(domain_distribution) > 1 or len(knowledge_type_distribution) > 1,
        "review_status": "pending",
        "review_note": "",
        "clustering": {
            "method": METHOD,
            "cosine_similarity_threshold": THRESHOLD,
            "semantic_node_count": int(cluster["semantic_node_count"]),
        },
    }


def load_complete_state(path: Path) -> tuple[dict, list[dict]]:
    state = json.loads(path.read_text(encoding="utf-8"))
    runs = [
        run for run in state["calibration_runs"]
        if run["summary"]["method"] == METHOD
        and abs(float(run["summary"]["similarity_threshold"]) - THRESHOLD) < 1e-9
    ]
    if len(runs) != 1:
        raise ValueError(f"expected one {METHOD} run at {THRESHOLD:.2f}, got {len(runs)}")
    run = runs[0]
    return state, run["clusters"]


def build_candidates(input_path: Path, state_path: Path) -> tuple[dict, list[dict]]:
    rows = load_jsonl(input_path)
    state, clusters = load_complete_state(state_path)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if state.get("input_sha256") != input_hash:
        raise ValueError("calibration state does not match the current V2.1 input")
    if state.get("eligible_cases") != len(rows) or state.get("mapped_support_cases") != len(rows):
        raise ValueError("calibration state/input case count mismatch")
    if state.get("embedding_model") != OPENROUTER_EMBEDDING_MODEL or state.get("dimensions") != OPENROUTER_EMBEDDING_DIMENSIONS:
        raise ValueError("unexpected embedding configuration")
    if state.get("cache_validation", {}).get("new_embeddings", 0) != 0:
        raise ValueError("existing cache state reports newly generated embeddings")

    candidates = [candidate_from_cluster(cluster, "") for cluster in clusters]
    candidates.sort(key=lambda item: (-item["frequency"], item["representative_case_id"], item["case_ids"]))
    for index, candidate in enumerate(candidates, 1):
        candidate["topic_candidate_id"] = f"TC-{index:03d}"
    all_case_ids = [case_id for candidate in candidates for case_id in candidate["case_ids"]]
    if len(all_case_ids) != len(set(all_case_ids)) or len(set(all_case_ids)) != len(rows):
        raise ValueError("topic candidates do not provide one vote per distinct support case")
    metadata = {
        "schema_version": 1,
        "artifact_type": "topic_candidate",
        "selection": {
            "source": str(input_path),
            "analysis_generation": "V2.1",
            "question_quality": "good",
            "primary_question_only": True,
            "secondary_questions_used": False,
            "distinct_support_case_vote": True,
        },
        "embedding": {"model": OPENROUTER_EMBEDDING_MODEL, "dimensions": OPENROUTER_EMBEDDING_DIMENSIONS, "cache_reused": True, "new_embeddings": 0},
        "clustering": {"algorithm": "Agglomerative Hierarchical Clustering", "linkage": METHOD, "cosine_similarity_threshold": THRESHOLD},
        "summary": {
            "good_primary_cases": len(rows),
            "topic_candidate_count": len(candidates),
            "singleton_count": sum(candidate["frequency"] == 1 for candidate in candidates),
            "frequency_ge_2_count": sum(candidate["frequency"] >= 2 for candidate in candidates),
            "frequency_ge_3_count": sum(candidate["frequency"] >= 3 for candidate in candidates),
            "max_frequency": max(candidate["frequency"] for candidate in candidates),
        },
        "topic_candidates": candidates,
    }
    return metadata, candidates


def markdown_distribution(values: dict[str, int], frequency: int) -> str:
    return "\n".join(f"{key} {100.0 * count / frequency:.0f}%" for key, count in values.items()) or "-"


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(metadata: dict, candidates: list[dict], top_n: int = 50) -> str:
    summary = metadata["summary"]
    lines = [
        "# Topic Candidate Review",
        "",
        "> Human-review candidates only. These are not approved Topics, Verified Knowledge, or answers.",
        "> SAME_TOPIC does not imply SAME_ANSWER; model, family, revision, firmware, and operating mode may change the answer.",
        "",
        "## Run Summary",
        "",
        f"- good primary cases: {summary['good_primary_cases']}",
        f"- Topic Candidates: {summary['topic_candidate_count']}",
        f"- singleton candidates: {summary['singleton_count']}",
        f"- frequency >= 2: {summary['frequency_ge_2_count']}",
        f"- frequency >= 3: {summary['frequency_ge_3_count']}",
        f"- maximum frequency: {summary['max_frequency']}",
        f"- clustering: Agglomerative Hierarchical Clustering, {METHOD} linkage, cosine similarity threshold {THRESHOLD:.2f}",
        "- review_status default: pending",
        "",
        f"## Top {min(top_n, len(candidates))}",
        "",
    ]
    for candidate in candidates[:top_n]:
        lines.extend([
            f"## {candidate['topic_candidate_id']} — frequency {candidate['frequency']}",
            "",
            f"Representative: {md(candidate['representative_question'])}",
            "",
            "Canonical questions:",
            "",
        ])
        for case_id, question in zip(candidate["case_ids"], candidate["canonical_questions"], strict=True):
            lines.append(f"- #{case_id} {md(question)}")
        lines.extend([
            "",
            "Cases:",
            "",
            "- " + ", ".join(f"#{case_id}" for case_id in candidate["case_ids"]),
            "",
            "Models:",
            "",
            "- " + (", ".join(md(model) for model in candidate["models"]) or "-"),
            "",
            "Domain:",
            "",
            markdown_distribution(candidate["domain_distribution"], candidate["frequency"]),
            "",
            "Knowledge type:",
            "",
            markdown_distribution(candidate["knowledge_type_distribution"], candidate["frequency"]),
            "",
            "Similarity:",
            "",
            f"min: {candidate['cluster_min_similarity']:.2f}",
            f"mean: {candidate['cluster_mean_similarity']:.2f}",
            f"taxonomy_mixed: {'true' if candidate['taxonomy_mixed'] else 'false'}",
            "",
            "Review:",
            "",
            "[ ] approved",
            "[ ] split_required",
            "[ ] merge_required",
            "[ ] rejected",
            "",
            "Note:",
            "",
            candidate["review_note"],
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    metadata, candidates = build_candidates(args.input, args.state)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(metadata, candidates), encoding="utf-8")
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
