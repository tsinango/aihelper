#!/usr/bin/env python3
"""Create Topic Abstractions and V2 Topic Candidates offline.

The input contains only the already approved V2.1 primary canonical-question
rows. This script never reads support-case messages or engineer answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from embeddings import (
    OPENROUTER_EMBEDDING_DIMENSIONS,
    OPENROUTER_EMBEDDING_MODEL,
    OpenRouterEmbeddingClient,
    read_openrouter_token,
)
from llm import OPENROUTER_DEFAULT_MODEL, OPENROUTER_PROVIDER, LLMService, OpenRouterLLM, parse_json_response

EMBEDDING_MODEL = OPENROUTER_EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = OPENROUTER_EMBEDDING_DIMENSIONS
LLM_MODEL = OPENROUTER_DEFAULT_MODEL
MAX_RETRIES = 2
CLUSTER_THRESHOLD = 0.80
CLUSTER_METHOD = "complete"

DEFAULT_INPUT = Path("/home/ubuntu/ai-sales-engineer-knowledge/input/topic_questions_v2_1.jsonl")
DEFAULT_DATA_DIR = Path("/opt/aihelper/data")

SYSTEM_PROMPT = """Ты анализируешь вопросы технической поддержки систем безопасности.

Твоя задача — НЕ отвечать на вопрос.

Определи абстрактный повторно используемый тип знания, который нужен для ответа на вопрос.

Удали конкретные модели, версии, IP, серийные номера и детали конкретного объекта.
Сохрани технические функции, протоколы и различия, которые меняют смысл вопроса.
Не добавляй требования, которых нет во входном вопросе.
Не используй и не предполагай факты из ответов инженеров.

Верни только строгий JSON без markdown и reasoning:
{
  "topic_signature": "..."
}

topic_signature должен быть кратким, естественным и на русском языке."""

REPAIR_PROMPT = """Предыдущий JSON нарушил правило абстракции.
Повторите ответ строго как JSON с единственным ключом topic_signature.
Критически важно: topic_signature не должен содержать ни одной конкретной модели
или другого идентификатора из исходного вопроса. Если исходный вопрос упоминает
только конкретную модель, опишите общий тип функции, совместимости или настройки
без копирования этой модели. Не добавляйте новые требования и не отвечайте на вопрос."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except PermissionError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(name, value)


def load_input(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("support_case_id", "analysis_id", "source_content_hash", "canonical_question", "domain", "knowledge_type", "models")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"input line {line_number} missing: {', '.join(missing)}")
        case_id = int(row["support_case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate support_case_id in input: {case_id}")
        seen.add(case_id)
        models = row["models"] if isinstance(row["models"], list) else []
        question = str(row["canonical_question"]).strip()
        if not question:
            raise ValueError(f"input line {line_number} has an empty canonical_question")
        rows.append({
            "support_case_id": case_id,
            "analysis_id": int(row["analysis_id"]),
            "source_content_hash": str(row["source_content_hash"]),
            "canonical_question": question,
            "domain": str(row["domain"]),
            "knowledge_type": str(row["knowledge_type"]),
            "models": [str(model) for model in models if str(model)],
        })
    return rows


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def parse_signature(content: str, row: dict) -> str:
    value = parse_json_response(content)
    if not isinstance(value, dict) or set(value) != {"topic_signature"}:
        raise ValueError("response must contain only topic_signature")
    signature = value.get("topic_signature")
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError("topic_signature is empty")
    signature = re.sub(r"\s+", " ", signature).strip()
    if len(signature) > 240:
        raise ValueError("topic_signature is too long")
    for model in row["models"]:
        if model and re.search(re.escape(model), signature, re.IGNORECASE):
            raise ValueError("topic_signature retains a case-specific model")
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", signature):
        raise ValueError("topic_signature retains an IP address")
    return signature


def abstract_one(llm: LLMService, row: dict) -> tuple[str, int]:
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["canonical_question"]},
    ]
    messages = base_messages
    last_error = None
    for retry in range(MAX_RETRIES + 1):
        try:
            content = llm.extract(messages, max_tokens=160)
            return parse_signature(content, row), retry + 1
        except ValueError as error:  # Repair malformed OpenRouter JSON within the batch limit.
            last_error = error
            if retry >= MAX_RETRIES:
                break
            messages = base_messages + [{"role": "user", "content": REPAIR_PROMPT}]
        except Exception as error:
            # OpenRouterLLM owns network/timeout retries. Only malformed output
            # gets a bounded repair request from the batch parser.
            last_error = error
            break
    raise RuntimeError(f"abstraction failed after {MAX_RETRIES + 1} attempts: {last_error}")


def abstract_cases(rows: list[dict], output_path: Path, failures_path: Path, model: str) -> tuple[list[dict], int]:
    default_token_file = str(Path(__file__).with_name("openrouter"))
    api_key = read_openrouter_token(os.environ.get("OPENROUTER_TOKEN_FILE", default_token_file))
    if not api_key:
        raise RuntimeError("OpenRouter token is not available; refusing to fabricate topic signatures")
    llm = OpenRouterLLM(
        api_key=api_key,
        model=model,
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120")),
    )
    existing = load_jsonl(output_path)
    completed = {
        (int(item["support_case_id"]), str(item["source_content_hash"]), str(item["canonical_question"])): item
        for item in existing
        if item.get("topic_signature")
    }
    input_by_key = {
        (row["support_case_id"], row["source_content_hash"], row["canonical_question"]): row
        for row in rows
    }
    for key, item in completed.items():
        row = input_by_key.get(key)
        if row is None:
            continue
        item.setdefault("domain", row["domain"])
        item.setdefault("knowledge_type", row["knowledge_type"])
        item.setdefault("models", row["models"])
    for row in rows:
        key = (row["support_case_id"], row["source_content_hash"], row["canonical_question"])
        if key in completed:
            continue
        try:
            signature, attempts = abstract_one(llm, row)
            item = {
                "support_case_id": row["support_case_id"],
                "analysis_id": row["analysis_id"],
                "source_content_hash": row["source_content_hash"],
                "canonical_question": row["canonical_question"],
                "topic_signature": signature,
                "domain": row["domain"],
                "knowledge_type": row["knowledge_type"],
                "models": row["models"],
                "model": model,
                "attempt_count": attempts,
                "created_at": utc_now(),
            }
            append_jsonl(output_path, item)
            completed[key] = item
        except Exception as error:
            append_jsonl(failures_path, {
                "support_case_id": row["support_case_id"],
                "analysis_id": row["analysis_id"],
                "source_content_hash": row["source_content_hash"],
                "model": model,
                "attempt_count": MAX_RETRIES + 1,
                "error_message": str(error)[:1000],
                "created_at": utc_now(),
            })
    input_keys = {
        (row["support_case_id"], row["source_content_hash"], row["canonical_question"])
        for row in rows
    }
    current = [item for key, item in completed.items() if key in input_keys]
    return current, len(rows) - len(current)


def signature_key(signature: str) -> str:
    return hashlib.sha256((EMBEDDING_MODEL + signature).encode("utf-8")).hexdigest()


def load_embedding_cache(npz_path: Path, meta_path: Path) -> dict[str, dict]:
    import numpy as np

    if not npz_path.exists() and not meta_path.exists():
        return {}
    if not npz_path.exists() or not meta_path.exists():
        raise RuntimeError("topic signature embedding cache is incomplete")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL or int(metadata.get("dimensions", -1)) != EMBEDDING_DIMENSIONS:
        raise RuntimeError("topic signature embedding cache configuration mismatch")
    with np.load(npz_path, allow_pickle=False) as data:
        keys = [str(value) for value in data["keys"].tolist()]
        vectors = np.asarray(data["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(keys), EMBEDDING_DIMENSIONS):
        raise RuntimeError("topic signature embedding cache shape mismatch")
    entries = metadata.get("entries", {})
    return {key: {"text": (entries.get(key) or {}).get("text", ""), "embedding": vectors[index]} for index, key in enumerate(keys)}


def save_embedding_cache(cache: dict[str, dict], npz_path: Path, meta_path: Path) -> None:
    import numpy as np

    keys = sorted(cache)
    vectors = np.vstack([np.asarray(cache[key]["embedding"], dtype=np.float32) for key in keys]) if keys else np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
    npz_tmp = npz_path.with_name(f".{npz_path.name}.tmp")
    meta_tmp = meta_path.with_name(f".{meta_path.name}.tmp")
    with npz_tmp.open("wb") as handle:
        np.savez_compressed(handle, keys=np.asarray(keys, dtype="U64"), embeddings=vectors)
    metadata = {
        "schema_version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "dtype": "float32",
        "count": len(keys),
        "entries": {key: {"text": str(cache[key]["text"])} for key in keys},
    }
    meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(npz_tmp, npz_path)
    os.replace(meta_tmp, meta_path)


def embed_signatures(signatures: list[str], npz_path: Path, meta_path: Path, embedder: OpenRouterEmbeddingClient) -> tuple[dict[str, dict], int]:
    import numpy as np

    cache = load_embedding_cache(npz_path, meta_path)
    unique = sorted(set(signatures), key=str.casefold)
    missing = [(signature_key(signature), signature) for signature in unique if signature_key(signature) not in cache]
    if missing:
        batch_size = max(1, int(os.environ.get("OPENROUTER_EMBED_BATCH_SIZE", "32")))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = embedder.encode([text for _, text in batch], batch_size=batch_size, normalize_embeddings=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(batch), EMBEDDING_DIMENSIONS):
                raise RuntimeError(f"unexpected embedding shape: {vectors.shape}")
            for (key, text), vector in zip(batch, vectors, strict=True):
                cache[key] = {"text": text, "embedding": vector}
            save_embedding_cache(cache, npz_path, meta_path)
    return cache, len(missing)


def cosine_similarity_matrix(vectors):
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise RuntimeError("zero topic signature embedding")
    return np.clip((vectors / norms[:, None]) @ (vectors / norms[:, None]).T, -1.0, 1.0).astype(np.float32, copy=False)


def cluster_nodes(vectors):
    import inspect
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    params = {"n_clusters": None, "distance_threshold": 1.0 - CLUSTER_THRESHOLD, "linkage": CLUSTER_METHOD}
    if "metric" in inspect.signature(AgglomerativeClustering).parameters:
        params["metric"] = "cosine"
    else:
        params["affinity"] = "cosine"
    labels = AgglomerativeClustering(**params).fit_predict(vectors)
    groups = [sorted(np.flatnonzero(labels == label).tolist()) for label in sorted(set(labels))]
    return sorted(groups, key=lambda group: group[0])


def distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def cluster_candidates(abstractions: list[dict], cache: dict[str, dict]) -> list[dict]:
    import numpy as np

    by_signature = defaultdict(list)
    for item in abstractions:
        by_signature[item["topic_signature"]].append(item)
    signatures = sorted(by_signature, key=str.casefold)
    vectors = np.vstack([cache[signature_key(signature)]["embedding"] for signature in signatures]).astype(np.float32, copy=False)
    similarity = cosine_similarity_matrix(vectors)
    groups = cluster_nodes(vectors)
    candidates = []
    for node_indices in groups:
        node_indices = sorted(node_indices)
        members = sorted((item for index in node_indices for item in by_signature[signatures[index]]), key=lambda item: int(item["support_case_id"]))
        case_ids = sorted({int(item["support_case_id"]) for item in members})
        node_values = [signatures[index] for index in node_indices]
        if len(node_indices) == 1:
            pair_values = np.asarray([1.0], dtype=np.float32)
            medoid_index = node_indices[0]
        else:
            pair_values = similarity[np.ix_(node_indices, node_indices)][np.triu_indices(len(node_indices), 1)]
            medoid_index = max(
                node_indices,
                key=lambda index: (float(np.mean([similarity[index, other] for other in node_indices if other != index])), -index),
            )
        medoid_members = by_signature[signatures[medoid_index]]
        representative = min(medoid_members, key=lambda item: int(item["support_case_id"]))
        domain_distribution = distribution([item["domain"] for item in members])
        knowledge_distribution = distribution([item["knowledge_type"] for item in members])
        candidates.append({
            "frequency": len(case_ids),
            "case_ids": case_ids,
            "representative_case_id": int(representative["support_case_id"]),
            "representative_question": representative["canonical_question"],
            "topic_signature": signatures[medoid_index],
            "canonical_questions": [item["canonical_question"] for item in members],
            "models": sorted({model for item in members for model in item["models"] if model}, key=str.casefold),
            "domain_distribution": domain_distribution,
            "knowledge_type_distribution": knowledge_distribution,
            "cluster_min_similarity": float(np.min(pair_values)),
            "cluster_mean_similarity": float(np.mean(pair_values)),
            "taxonomy_mixed": len(domain_distribution) > 1 or len(knowledge_distribution) > 1,
            "review_status": "pending",
            "review_note": "",
            "clustering": {
                "algorithm": "Agglomerative Hierarchical Clustering",
                "linkage": CLUSTER_METHOD,
                "cosine_similarity_threshold": CLUSTER_THRESHOLD,
                "semantic_node_count": len(node_values),
            },
        })
    candidates.sort(key=lambda item: (-item["frequency"], item["representative_case_id"], item["case_ids"]))
    for index, candidate in enumerate(candidates, 1):
        candidate["topic_candidate_id"] = f"TC-{index:03d}"
    return candidates


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent_distribution(values: dict[str, int], total: int) -> str:
    return "\n".join(f"{key} {100.0 * value / total:.0f}%" for key, value in values.items()) or "-"


def render_review(candidates: list[dict], summary: dict, top_n: int = 50) -> str:
    lines = [
        "# Telegram Topic Candidate Review V2",
        "",
        "> Human-review candidates only. No answers or Verified Knowledge were generated.",
        "> SAME_TOPIC means the same reusable knowledge dimension; it does not mean SAME_ANSWER.",
        "",
        "## Run Summary",
        "",
        f"- successfully abstracted cases: {summary['successfully_abstracted_cases']}",
        f"- OpenRouter failures: {summary['provider_failures']}",
        f"- unique topic_signatures: {summary['unique_topic_signatures']}",
        f"- Topic Candidates: {summary['topic_candidate_count']}",
        f"- singletons: {summary['singleton_count']}",
        f"- frequency >= 2: {summary['frequency_ge_2_count']}",
        f"- frequency >= 3: {summary['frequency_ge_3_count']}",
        f"- frequency >= 5: {summary['frequency_ge_5_count']}",
        f"- maximum frequency: {summary['max_frequency']}",
        f"- clustering: {CLUSTER_METHOD} linkage, cosine similarity threshold {CLUSTER_THRESHOLD:.2f}",
        "- review_status default: pending",
        "",
        f"## Top {min(top_n, len(candidates))}",
        "",
    ]
    for candidate in candidates[:top_n]:
        lines.extend([
            f"## {candidate['topic_candidate_id']} — frequency {candidate['frequency']}",
            "",
            f"Topic Signature: {md(candidate['topic_signature'])}",
            "",
            "Original canonical questions:",
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
            percent_distribution(candidate["domain_distribution"], candidate["frequency"]),
            "",
            "Knowledge type:",
            "",
            percent_distribution(candidate["knowledge_type_distribution"], candidate["frequency"]),
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


def run(args) -> dict:
    load_env_file(args.env_file)
    rows = load_input(args.input)
    if not rows:
        raise ValueError("input contains no question rows")
    case_ids = [str(row.get("support_case_id")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("input contains duplicate support_case_id values")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    abstraction_path = args.data_dir / "topic_abstractions.jsonl"
    failures_path = args.data_dir / "topic_abstraction_failures.jsonl"
    abstractions, _ = abstract_cases(rows, abstraction_path, failures_path, LLM_MODEL)
    abstraction_by_case = {int(item["support_case_id"]): item for item in abstractions}
    failures = len(rows) - len(abstraction_by_case)
    if not abstractions:
        raise RuntimeError("no topic abstractions were generated; signatures and candidates were not generated")
    embedder = OpenRouterEmbeddingClient(
        read_openrouter_token(os.environ.get("OPENROUTER_TOKEN_FILE", default_token_file)),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120")),
    )
    cache, new_embeddings = embed_signatures(
        [item["topic_signature"] for item in abstractions],
        args.data_dir / "topic_signature_embeddings.npz",
        args.data_dir / "topic_signature_embeddings_meta.json",
        embedder,
    )
    candidates = cluster_candidates(abstractions, cache)
    summary = {
        "schema_version": 1,
        "artifact_type": "topic_candidate_v2",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "provider": OPENROUTER_PROVIDER,
        "model": LLM_MODEL,
        "successfully_abstracted_cases": len(abstractions),
        "provider_failures": failures,
        "unique_topic_signatures": len({item["topic_signature"] for item in abstractions}),
        "topic_candidate_count": len(candidates),
        "singleton_count": sum(candidate["frequency"] == 1 for candidate in candidates),
        "frequency_ge_2_count": sum(candidate["frequency"] >= 2 for candidate in candidates),
        "frequency_ge_3_count": sum(candidate["frequency"] >= 3 for candidate in candidates),
        "frequency_ge_5_count": sum(candidate["frequency"] >= 5 for candidate in candidates),
        "max_frequency": max(candidate["frequency"] for candidate in candidates),
        "embedding": {"model": EMBEDDING_MODEL, "dimensions": EMBEDDING_DIMENSIONS, "new_embeddings": new_embeddings},
        "clustering": {"algorithm": "Agglomerative Hierarchical Clustering", "linkage": CLUSTER_METHOD, "cosine_similarity_threshold": CLUSTER_THRESHOLD},
        "created_at": utc_now(),
        "topic_candidates": candidates,
    }
    json_path = args.output_json
    markdown_path = args.output_markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_review(candidates, summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-json", type=Path, default=Path("/opt/aihelper/data/topic_candidates_v2.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("/opt/aihelper/TOPIC_REVIEW_V2.md"))
    parser.add_argument("--env-file", type=Path, default=Path("/etc/ai-sales-engineer.env"))
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:
        print(f"topic abstraction failed: {error}")
        return 1
    print(json.dumps({key: summary[key] for key in (
        "successfully_abstracted_cases", "provider_failures", "unique_topic_signatures",
        "topic_candidate_count", "singleton_count", "frequency_ge_2_count",
        "frequency_ge_3_count", "frequency_ge_5_count", "max_frequency",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
