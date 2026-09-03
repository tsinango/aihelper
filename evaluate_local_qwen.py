#!/usr/bin/env python3
"""Run an evaluation-only comparison of the local Qwen GGUF models.

The runner deliberately stays outside the production FastAPI provider path.
It reads the labelled golden set and its real review-artifact references,
sends the same prompt/evidence to each local model in sequence, and writes
results to a dedicated table and JSON artifact. No production question,
knowledge, or case-memory rows are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import psycopg
from psycopg.types.json import Jsonb

from build_verified_knowledge_pilot import load_env_file


DEFAULT_ENV_FILE = Path("/etc/ai-sales-engineer.env")
DEFAULT_OUTPUT = Path("data/local_qwen_smoke.json")
DEFAULT_GOLDEN_SET = Path("data/golden_set.json")
DEFAULT_GOLDEN_SOURCE = Path("data/telegram_knowledge_review.json")
DEFAULT_LLAMA_BIN = Path("/home/ubuntu/.local/bin/llama")
DEFAULT_CONTEXT_SIZE = 16384
DEFAULT_MAX_TOKENS = 600
DEFAULT_STARTUP_TIMEOUT = 180.0
DEFAULT_REQUEST_TIMEOUT = 180.0
DEFAULT_PORT = 18902
MODEL_SPECS = {
    "2b": {
        "filename": "Qwen3.5-2B-Q4_K_M.gguf",
        "repo_dir": "models--unsloth--Qwen3.5-2B-GGUF",
        "alias": "qwen3.5-2b-q4-k-m",
        "quantization": "Q4_K_M",
    },
    "4b": {
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "repo_dir": "models--unsloth--Qwen3.5-4B-GGUF",
        "alias": "qwen3.5-4b-q4-k-m",
        "quantization": "Q4_K_M",
    },
}
GOLDEN_ANSWER_STATUS_VALUES = frozenset({"answered", "needs_clarification", "unsupported", "service_error"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_model_path(label: str, override: Path | None = None) -> Path:
    if label not in MODEL_SPECS:
        raise ValueError(f"unknown local model: {label}")
    if override is not None:
        path = override.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    spec = MODEL_SPECS[label]
    cache_root = Path("/home/ubuntu/.cache/huggingface/hub") / spec["repo_dir"] / "snapshots"
    matches = sorted(cache_root.glob(f"*/{spec['filename']}"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"cached model not found: {spec['filename']} under {cache_root}; "
        "pass an explicit --model-2b or --model-4b path"
    )


def build_llama_command(
    llama_bin: Path,
    model_path: Path,
    alias: str,
    port: int,
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> list[str]:
    """Build the deterministic, evaluation-only llama server command."""
    return [
        str(llama_bin),
        "serve",
        "--model", str(model_path),
        "--alias", alias,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-ui",
        "--offline",
        "--parallel", "1",
        "--threads", "2",
        "--threads-batch", "2",
        "--ctx-size", str(context_size),
        "--reasoning", "off",
    ]


class LocalLlamaServer:
    def __init__(
        self,
        command: list[str],
        port: int,
        *,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        poll_interval: float = 0.5,
    ):
        self.command = command
        self.port = port
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.process: subprocess.Popen | None = None
        self.load_ms: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _ready(self, client: httpx.Client) -> bool:
        try:
            response = client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def start(self) -> int:
        if self.process is not None:
            raise RuntimeError("local llama server is already started")
        started = time.monotonic()
        try:
            with httpx.Client(timeout=1.5) as client:
                if self._ready(client):
                    raise RuntimeError(f"evaluation port is already in use: {self.port}")
        except RuntimeError:
            raise
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        with httpx.Client(timeout=1.5) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama server exited during startup with code {self.process.returncode}")
                if self._ready(client):
                    self.load_ms = int((time.monotonic() - started) * 1000)
                    return self.load_ms
                time.sleep(self.poll_interval)
        self.stop()
        raise TimeoutError(f"llama server did not become ready within {self.startup_timeout:.0f}s")

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class LocalQwenClient:
    def __init__(self, base_url: str, model_alias: str, timeout: float = DEFAULT_REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model_alias = model_alias
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def complete(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
        started = time.monotonic()
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model_alias,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        raw = response.json()
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("local llama response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("local llama response has no text content")
        timings = raw.get("timings") if isinstance(raw, dict) else {}
        timings = timings if isinstance(timings, dict) else {}
        usage = raw.get("usage") if isinstance(raw, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = _first_int(timings.get("prompt_n"), usage.get("prompt_tokens"))
        completion_tokens = _first_int(timings.get("predicted_n"), usage.get("completion_tokens"))
        tokens_per_second = _first_float(
            timings.get("predicted_per_second"),
            (completion_tokens / max((time.monotonic() - started), 0.001)) if completion_tokens else None,
        )
        generation_ms = _first_int(
            timings.get("predicted_ms"),
            int((time.monotonic() - started) * 1000),
        )
        return {
            "content": content,
            "raw": raw,
            "generation_ms": generation_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
        }


def _first_int(*values) -> int | None:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_float(*values) -> float | None:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _case_question(case: dict, atomic_qa_id: str | None = None) -> str:
    atomic_qas = case.get("atomic_qa") if isinstance(case.get("atomic_qa"), list) else []
    if atomic_qa_id:
        for item in atomic_qas:
            if str(item.get("atomic_qa_id")) == str(atomic_qa_id):
                question = str(item.get("question") or "").strip()
                if question:
                    return question
    analysis = case.get("analysis") if isinstance(case.get("analysis"), dict) else {}
    return str(analysis.get("canonical_question") or case.get("root_question") or "").strip()


def _case_answer(case: dict, atomic_qa_id: str | None = None) -> str:
    atomic_qas = case.get("atomic_qa") if isinstance(case.get("atomic_qa"), list) else []
    if atomic_qa_id:
        for item in atomic_qas:
            if str(item.get("atomic_qa_id")) == str(atomic_qa_id):
                answer = str(item.get("answer_text") or "").strip()
                if answer:
                    return answer
    candidate = case.get("answer_candidate") if isinstance(case.get("answer_candidate"), dict) else {}
    return str(candidate.get("text") or "").strip()


def _case_analysis(case: dict) -> dict:
    return case.get("analysis") if isinstance(case.get("analysis"), dict) else {}


def _golden_reference_evidence(case: dict, sample: dict, question: str, answer_text: str, app_module) -> dict:
    """Create evaluation-only evidence from the real review artifact.

    This source is intentionally marked as ``golden_reference``.  It is never
    inserted into production tables and is used only when a matching published
    Verified Knowledge row is unavailable for an offline benchmark sample.
    """
    analysis = _case_analysis(case)
    scope = case.get("scope") if isinstance(case.get("scope"), dict) else {}
    models = [str(item).strip() for item in scope.get("models", []) if str(item).strip()]
    content = {
        "source": "data/telegram_knowledge_review.json",
        "support_case_id": case.get("support_case_id"),
        "canonical_question": analysis.get("canonical_question") or case.get("root_question") or "",
        "knowledge_key": analysis.get("knowledge_key"),
        "knowledge_type": analysis.get("knowledge_type"),
        "context_status": analysis.get("context_status"),
        "scope": scope,
        "answer_text": answer_text[:4000],
    }
    return {
        "source_type": "golden_reference",
        "title": "Offline golden reference",
        "page_number": None,
        "language": "ru",
        "product_model": " ".join(models) or None,
        "retrieved_document_models": models,
        "content": json.dumps(content, ensure_ascii=False),
        "knowledge_key": analysis.get("knowledge_key"),
        "verified_knowledge_id": None,
        "support_case_id": case.get("support_case_id"),
        "source_status": "golden_reference",
        "source_confidence": 1.0,
        "requires_context": bool(sample.get("must_clarify") or analysis.get("context_status") == "context_required"),
        "evidence": [{"source": "telegram_review_artifact", "support_case_id": case.get("support_case_id")}],
        "scope": scope,
        "scope_match": app_module.verified_scope_match(question, scope),
        "rrf_score": 0.0,
        "exact_match": True,
    }


def load_golden_set(path: Path, source_path: Path) -> tuple[dict, dict[int, dict]]:
    """Load labels plus the real source threads referenced by those labels."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"could not read golden set or source artifact: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"golden set or source artifact is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("golden set must contain a samples array")
    if not isinstance(source, dict) or not isinstance(source.get("cases"), list):
        raise ValueError("golden source artifact must contain a cases array")
    cases = {}
    for case in source["cases"]:
        try:
            cases[int(case["support_case_id"])] = case
        except (KeyError, TypeError, ValueError):
            continue
    return payload, cases


def _published_by_key(conn, limit: int = 100000) -> dict[str, dict]:
    if conn is None:
        return {}
    rows = load_published_samples(conn, limit)
    indexed = {}
    for row in rows:
        key = str(row.get("knowledge_key") or "").strip()
        if key:
            indexed.setdefault(key, row)
    return indexed


def prepare_golden_samples(
    conn,
    golden_path: Path,
    source_path: Path,
    limit: int,
    app_module,
) -> tuple[list[dict], dict]:
    """Prepare deterministic prompts from explicitly labelled golden samples."""
    golden, cases = load_golden_set(golden_path, source_path)
    entries = golden["samples"][:limit]
    if not entries:
        raise RuntimeError("golden set contains no samples")
    published = _published_by_key(conn)
    aliases = app_module.load_alias_rows(conn) if conn is not None else []
    prepared = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every golden sample must be an object")
        sample_key = str(entry.get("sample_key") or "").strip()
        if not sample_key:
            raise ValueError("every golden sample needs sample_key")
        expected_status = str(entry.get("expected_answer_status") or "").strip()
        if expected_status not in GOLDEN_ANSWER_STATUS_VALUES:
            raise ValueError(f"invalid expected_answer_status for {sample_key}: {expected_status}")
        base_case_id = entry.get("case_id")
        try:
            base_case = cases[int(base_case_id)] if base_case_id is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unknown case_id for {sample_key}: {base_case_id}") from exc
        if base_case is None and not entry.get("question"):
            raise ValueError(f"{sample_key} needs case_id or question")
        question = str(entry.get("question") or _case_question(base_case, entry.get("atomic_qa_id"))).strip()
        if len(question) < 2:
            raise ValueError(f"golden sample {sample_key} has an empty question")

        raw_keys = entry.get("expected_knowledge_keys")
        if raw_keys is None:
            raw_keys = [entry.get("expected_knowledge_key")] if entry.get("expected_knowledge_key") else []
        expected_keys = [str(key).strip() for key in raw_keys if str(key).strip()]
        source_case_ids = entry.get("evidence_case_ids")
        if source_case_ids is None:
            source_case_ids = [base_case_id] if base_case_id is not None else []
        try:
            source_case_ids = [int(case_id) for case_id in source_case_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid evidence_case_ids for {sample_key}") from exc

        evidence = []
        resolved_vk_ids = []
        for source_case_id in source_case_ids:
            source_case = cases.get(source_case_id)
            if source_case is None:
                raise ValueError(f"unknown evidence case {source_case_id} for {sample_key}")
            analysis = _case_analysis(source_case)
            key = str(analysis.get("knowledge_key") or "").strip()
            row = published.get(key) if key else None
            if row is not None:
                item = _published_row_evidence(row, question)
                item["requires_context"] = bool(entry.get("must_clarify"))
                resolved_vk_ids.append(row["verified_knowledge_id"])
            else:
                item = _golden_reference_evidence(
                    source_case,
                    entry,
                    question,
                    _case_answer(source_case, entry.get("atomic_qa_id") if source_case_id == base_case_id else None),
                    app_module,
                )
            evidence.append(item)
        scope = app_module.scope_details(question, evidence)
        route = app_module.route_question(question, app_module.identifiers(question))
        messages, prompt_scope = app_module.build_decision_messages(
            question,
            evidence,
            app_module.expanded(question, aliases),
            scope,
            route=route,
            learning_examples=[],
        )
        expected_scope = entry.get("expected_scope")
        if not isinstance(expected_scope, dict):
            expected_scope = base_case.get("scope", {}) if base_case else {}
        expected_answer = str(entry.get("expected_answer") or (_case_answer(base_case, entry.get("atomic_qa_id")) if base_case else ""))
        prepared.append({
            "sample_key": sample_key,
            "source_candidate_id": None,
            "expected_verified_knowledge_id": resolved_vk_ids[0] if len(resolved_vk_ids) == 1 else None,
            "expected_knowledge_key": expected_keys[0] if len(expected_keys) == 1 else None,
            "expected_knowledge_keys": expected_keys,
            "expected_source_case_ids": source_case_ids,
            "golden_tags": [str(tag) for tag in entry.get("tags", [])],
            "must_clarify": bool(entry.get("must_clarify")),
            "must_refuse": bool(entry.get("must_refuse")),
            "question": question,
            "expected_answer_status": expected_status,
            "expected_scope": expected_scope,
            "expected_answer": expected_answer,
            "evidence": evidence,
            "retrieval_mode": (
                "published_vk_by_knowledge_key"
                if any(item.get("source_type") == "verified_knowledge" for item in evidence)
                else "golden_reference_snapshot"
            ),
            "retrieved_verified_knowledge_ids": [item.get("verified_knowledge_id") for item in evidence if item.get("verified_knowledge_id") is not None],
            "retrieved_knowledge_keys": [item.get("knowledge_key") for item in evidence if item.get("knowledge_key")],
            "retrieval_trace": {
                "golden_set_version": golden.get("version"),
                "source_artifact": str(source_path),
                "expected_knowledge_keys": expected_keys,
                "evidence_case_ids": source_case_ids,
            },
            "messages": messages,
            "scope": prompt_scope,
            "route": route,
            "prompt_sha256": _prompt_hash(messages),
            "golden_set_version": golden.get("version"),
        })
    return prepared, golden


def _published_row_evidence(row: dict, question: str) -> dict:
    from helpers import verified_scope_match

    scope = row.get("scope") if isinstance(row.get("scope"), dict) else {}
    content = {
        "title": row.get("title") or row.get("knowledge_key"),
        "knowledge_key": row.get("knowledge_key"),
        "answer_text": row.get("answer_text") or "",
        "scope_level": row.get("scope_level") or "unspecified",
        "claims": row.get("claims") or [],
        "conditions": row.get("conditions") or [],
        "procedure_steps": row.get("procedure_steps") or [],
        "exceptions": row.get("exceptions") or [],
        "warnings": row.get("warnings") or [],
        "question_patterns": row.get("question_patterns") or [],
        "aliases": row.get("aliases") or [],
        "evidence": row.get("evidence") or [],
    }
    return {
        "source_type": "verified_knowledge",
        "title": content["title"],
        "page_number": None,
        "language": "ru",
        "product_model": None,
        "retrieved_document_models": [],
        "content": json.dumps(content, ensure_ascii=False),
        "knowledge_key": row.get("knowledge_key"),
        "verified_knowledge_id": row.get("verified_knowledge_id"),
        "support_case_id": None,
        "source_status": "verified",
        "source_confidence": 1.0,
        "requires_context": False,
        "evidence": row.get("evidence") or [],
        "scope": scope,
        "scope_match": verified_scope_match(question, scope),
        "rrf_score": 0.0,
        "exact_match": True,
    }


def load_published_samples(conn, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT verified_knowledge_id,source_candidate_id,knowledge_key,title,
                   knowledge_type,answer_text,scope_level,scope,claims,
                   procedure_steps,conditions,exceptions,warnings,question_patterns,
                   evidence,aliases
            FROM verified_knowledge
            WHERE publication_status='published' AND production_answer_allowed=TRUE
            ORDER BY verified_knowledge_id
            LIMIT %s
            """,
            (limit,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    return rows


def _fallback_evidence(rows: list[dict], question: str) -> list[dict]:
    """Build a deterministic evidence snapshot when embedding is unavailable."""
    normalized_question = question.casefold()
    scored = []
    for row in rows:
        title = str(row.get("title") or "").casefold()
        score = 1 if title and title in normalized_question else 0
        scored.append((score, int(row["verified_knowledge_id"]), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [_published_row_evidence(row, question) for _, _, row in scored[:12]]


def retrieve_sample_evidence(conn, rows: list[dict], question: str, aliases: list[dict], app_module) -> tuple[list[dict], str, dict]:
    """Prefer the production VK retriever, but never block a local smoke test."""
    retrieval_question = app_module.expanded(question, aliases)
    try:
        token = app_module.settings["openrouter_api_key"] or app_module.read_openrouter_token(app_module.settings["openrouter_token_file"])
        if token:
            embedder = app_module.OpenRouterEmbeddingClient(
                token,
                token_file=app_module.settings["openrouter_token_file"],
                timeout=min(float(app_module.settings["openrouter_timeout"]), 30.0),
                max_retries=0,
            )
            query_embedding = embedder.encode([retrieval_question], normalize_embeddings=True, show_progress_bar=False)[0]
            evidence, trace = app_module.retrieve_verified_knowledge(
                conn, question, query_embedding, limit=12, alias_rows=aliases,
            )
            if evidence:
                return evidence, "production_verified_retrieval", trace
    except Exception as error:
        return _fallback_evidence(rows, question), f"published_vk_snapshot:{type(error).__name__}", {
            "retrieval_query": retrieval_question,
            "verified_knowledge_ids": [row["verified_knowledge_id"] for row in rows],
        }
    return _fallback_evidence(rows, question), "published_vk_snapshot", {
        "retrieval_query": retrieval_question,
        "verified_knowledge_ids": [row["verified_knowledge_id"] for row in rows],
    }


def _actual_answer_status(decision: dict, question: str, scope: dict) -> str:
    if decision.get("supported"):
        return "answered"
    if not app_identifiers(question):
        return "needs_clarification"
    if scope.get("scope_match") == "unspecified":
        # A known model with no retrieved document is an unsupported request;
        # an identified document with no user model still needs clarification.
        return "unsupported" if not scope.get("retrieved_document_models") else "needs_clarification"
    return "unsupported"


def app_identifiers(question: str) -> list[str]:
    from helpers import identifiers

    return identifiers(question)


def _prompt_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_samples(conn, limit: int, app_module) -> list[dict]:
    rows = load_published_samples(conn, limit)
    if not rows:
        raise RuntimeError("no published production-enabled Verified Knowledge is available")
    aliases = app_module.load_alias_rows(conn)
    prepared = []
    for row in rows:
        question = str(row.get("title") or row.get("knowledge_key") or "").strip()
        if len(question) < 2:
            continue
        evidence, retrieval_mode, trace = retrieve_sample_evidence(conn, rows, question, aliases, app_module)
        scope = app_module.scope_details(question, evidence)
        route = app_module.route_question(question, app_identifiers(question))
        messages, prompt_scope = app_module.build_decision_messages(
            question,
            evidence,
            app_module.expanded(question, aliases),
            scope,
            route=route,
            learning_examples=[],
        )
        prepared.append({
            "sample_key": f"vk-{row['verified_knowledge_id']}",
            "source_candidate_id": row.get("source_candidate_id"),
            "expected_verified_knowledge_id": row["verified_knowledge_id"],
            "expected_knowledge_key": row.get("knowledge_key"),
            "expected_knowledge_keys": [row.get("knowledge_key")],
            "expected_source_case_ids": [],
            "golden_tags": ["published_knowledge_smoke"],
            "must_clarify": False,
            "must_refuse": False,
            "question": question,
            "expected_answer_status": "answered",
            "expected_scope": row.get("scope") or {},
            "expected_answer": row.get("answer_text") or "",
            "evidence": evidence,
            "retrieval_mode": retrieval_mode,
            "retrieved_verified_knowledge_ids": [item.get("verified_knowledge_id") for item in evidence],
            "retrieval_trace": trace,
            "messages": messages,
            "scope": prompt_scope,
            "route": route,
            "prompt_sha256": _prompt_hash(messages),
        })
    return prepared


def _result_row(sample: dict, model: dict, server: LocalLlamaServer, response: dict | None, error: Exception | None, app_module) -> dict:
    base = {
        "run_id": model["run_id"],
        "model_name": model["label"],
        "model_path": model["path"],
        "quantization": model["quantization"],
        "sample_key": sample["sample_key"],
        "source_candidate_id": sample["source_candidate_id"],
        "expected_verified_knowledge_id": sample["expected_verified_knowledge_id"],
        "expected_knowledge_key": sample.get("expected_knowledge_key"),
        "expected_knowledge_keys": sample.get("expected_knowledge_keys", []),
        "expected_source_case_ids": sample.get("expected_source_case_ids", []),
        "golden_tags": sample.get("golden_tags", []),
        "must_clarify": sample.get("must_clarify", False),
        "must_refuse": sample.get("must_refuse", False),
        "question": sample["question"],
        "expected_answer_status": sample["expected_answer_status"],
        "expected_scope": sample["expected_scope"],
        "expected_answer": sample["expected_answer"],
        "retrieved_verified_knowledge_ids": sample["retrieved_verified_knowledge_ids"],
        "retrieved_knowledge_keys": sample.get("retrieved_knowledge_keys", []),
        "retrieval_mode": sample["retrieval_mode"],
        "retrieval_hit_at_5": (
            sample["expected_verified_knowledge_id"] in sample["retrieved_verified_knowledge_ids"][:5]
            if sample["expected_verified_knowledge_id"] is not None
            else (
                set(sample.get("expected_knowledge_keys", [])).issubset(set(sample.get("retrieved_knowledge_keys", [])[:5]))
                if sample.get("expected_knowledge_keys") else None
            )
        ),
        "retrieval_hit_at_10": (
            sample["expected_verified_knowledge_id"] in sample["retrieved_verified_knowledge_ids"][:10]
            if sample["expected_verified_knowledge_id"] is not None
            else (
                set(sample.get("expected_knowledge_keys", [])).issubset(set(sample.get("retrieved_knowledge_keys", [])[:10]))
                if sample.get("expected_knowledge_keys") else None
            )
        ),
        "prompt_sha256": sample["prompt_sha256"],
        "answer_text": "",
        "actual_answer_status": "service_error",
        "actual_source_indexes": [],
        "actual_knowledge_keys": [],
        "actual_source_case_ids": [],
        "structure_pass": False,
        "applicability_pass": False,
        "status_pass": False,
        "source_selection_pass": False,
        "golden_pass": False,
        "answer_pass": None,
        "reviewer_verdict": "pending",
        "reviewer_note": "Smoke test result requires human answer review; no automatic semantic judge is used.",
        "raw_response": "",
        "error_message": str(error) if error else "",
        "model_load_ms": server.load_ms,
        "generation_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "tokens_per_second": None,
    }
    if error is not None or response is None:
        return base
    raw = response.get("raw") or {}
    try:
        decision = app_module.normalize_llm_decision(response["content"], sample["evidence"])
        actual_status = _actual_answer_status(decision, sample["question"], sample["scope"])
        selected_items = [sample["evidence"][index] for index in decision["source_indexes"]]
        selected_ids = [
            item.get("verified_knowledge_id") for item in selected_items
        ]
        selected_keys = sorted({
            str(item.get("knowledge_key")).strip()
            for item in selected_items
            if str(item.get("knowledge_key") or "").strip()
        })
        selected_case_ids = sorted({
            int(item["support_case_id"])
            for item in selected_items
            if item.get("support_case_id") is not None
        })
        expected_keys = set(sample.get("expected_knowledge_keys", []))
        if sample.get("must_refuse"):
            source_selection_pass = not selected_items or all(item.get("scope_match") == "conflict" for item in selected_items)
        elif sample.get("must_clarify"):
            source_selection_pass = not selected_items
        elif expected_keys:
            source_selection_pass = expected_keys.issubset(set(selected_keys))
        elif sample.get("expected_verified_knowledge_id") is not None:
            source_selection_pass = sample["expected_verified_knowledge_id"] in selected_ids
        else:
            source_selection_pass = True
        status_pass = actual_status == sample["expected_answer_status"]
        golden_pass = bool(decision["supported"] and status_pass and source_selection_pass) if sample["expected_answer_status"] == "answered" else bool(status_pass and source_selection_pass)
        base.update({
            "answer_text": decision["answer"],
            "actual_answer_status": actual_status,
            "actual_source_indexes": decision["source_indexes"],
            "actual_knowledge_keys": selected_keys,
            "actual_source_case_ids": selected_case_ids,
            "structure_pass": True,
            "applicability_pass": (
                actual_status == "answered"
                and source_selection_pass
                and all(
                    sample["evidence"][index].get("scope_match") != "conflict"
                    for index in decision["source_indexes"]
                )
            ),
            "status_pass": status_pass,
            "source_selection_pass": source_selection_pass,
            "golden_pass": golden_pass,
            "raw_response": json.dumps(raw, ensure_ascii=False),
            "generation_ms": response.get("generation_ms"),
            "prompt_tokens": response.get("prompt_tokens"),
            "completion_tokens": response.get("completion_tokens"),
            "tokens_per_second": response.get("tokens_per_second"),
        })
    except Exception as parse_error:
        base["raw_response"] = json.dumps(raw, ensure_ascii=False)
        base["error_message"] = f"response normalization failed: {parse_error}"
    return base


def insert_run(conn, run_id: str, mode: str, samples: list[dict], models: list[dict], config: dict, output_path: str) -> None:
    conn.execute(
        """
        INSERT INTO local_model_eval_runs
          (run_id,mode,sample_count,model_names,config,output_path,status)
        VALUES (%s,%s,%s,%s,%s,%s,'running')
        """,
        (run_id, mode, len(samples), Jsonb([model["label"] for model in models]), Jsonb(config), output_path),
    )


def insert_result(conn, row: dict) -> None:
    columns = (
        "run_id,model_name,model_path,quantization,sample_key,source_candidate_id,"
        "expected_verified_knowledge_id,question,expected_answer_status,expected_scope,"
        "expected_answer,retrieved_verified_knowledge_ids,retrieval_mode,"
        "retrieval_hit_at_5,retrieval_hit_at_10,prompt_sha256,answer_text,"
        "actual_answer_status,actual_source_indexes,structure_pass,applicability_pass,"
        "answer_pass,reviewer_verdict,reviewer_note,raw_response,error_message,"
        "model_load_ms,generation_ms,prompt_tokens,completion_tokens,tokens_per_second"
    )
    values = [
        row["run_id"], row["model_name"], row["model_path"], row["quantization"], row["sample_key"],
        row["source_candidate_id"], row["expected_verified_knowledge_id"], row["question"],
        row["expected_answer_status"], Jsonb(_json_value(row["expected_scope"])), row["expected_answer"],
        Jsonb(_json_value(row["retrieved_verified_knowledge_ids"])), row["retrieval_mode"],
        row["retrieval_hit_at_5"], row["retrieval_hit_at_10"], row["prompt_sha256"], row["answer_text"],
        row["actual_answer_status"], Jsonb(_json_value(row["actual_source_indexes"])), row["structure_pass"],
        row["applicability_pass"], row["answer_pass"], row["reviewer_verdict"], row["reviewer_note"],
        row["raw_response"], row["error_message"], row["model_load_ms"], row["generation_ms"],
        row["prompt_tokens"], row["completion_tokens"], row["tokens_per_second"],
    ]
    placeholders = ",".join(["%s"] * len(values))
    conn.execute(
        f"INSERT INTO local_model_eval_results ({columns}) VALUES ({placeholders}) "
        "ON CONFLICT (run_id,model_name,sample_key) DO UPDATE SET "
        "answer_text=EXCLUDED.answer_text,actual_answer_status=EXCLUDED.actual_answer_status,"
        "actual_source_indexes=EXCLUDED.actual_source_indexes,structure_pass=EXCLUDED.structure_pass,"
        "applicability_pass=EXCLUDED.applicability_pass,raw_response=EXCLUDED.raw_response,"
        "error_message=EXCLUDED.error_message,model_load_ms=EXCLUDED.model_load_ms,"
        "generation_ms=EXCLUDED.generation_ms,prompt_tokens=EXCLUDED.prompt_tokens,"
        "completion_tokens=EXCLUDED.completion_tokens,tokens_per_second=EXCLUDED.tokens_per_second",
        values,
    )


def finish_run(conn, run_id: str, status: str, error_message: str = "") -> None:
    conn.execute(
        "UPDATE local_model_eval_runs SET status=%s,error_message=%s,completed_at=CURRENT_TIMESTAMP WHERE run_id=%s",
        (status, error_message, run_id),
    )


def summarize(results: list[dict]) -> dict:
    summary = {}
    for label in sorted({row["model_name"] for row in results}):
        rows = [row for row in results if row["model_name"] == label]
        speeds = [float(row["tokens_per_second"]) for row in rows if row.get("tokens_per_second")]
        expected_statuses = {}
        for row in rows:
            status = row.get("expected_answer_status", "unknown")
            expected_statuses[status] = expected_statuses.get(status, 0) + 1
        summary[label] = {
            "count": len(rows),
            "structure_pass": sum(bool(row.get("structure_pass")) for row in rows),
            "applicability_pass": sum(bool(row.get("applicability_pass")) for row in rows),
            "status_pass": sum(bool(row.get("status_pass")) for row in rows),
            "source_selection_pass": sum(bool(row.get("source_selection_pass")) for row in rows),
            "golden_pass": sum(bool(row.get("golden_pass")) for row in rows),
            "expected_status_counts": expected_statuses,
            "median_tokens_per_second": median(speeds) if speeds else None,
            "generation_ms": [row.get("generation_ms") for row in rows],
        }
    return summary


def parse_models(value: str) -> list[str]:
    labels = [item.strip().casefold() for item in value.split(",") if item.strip()]
    if not labels or any(label not in MODEL_SPECS for label in labels) or len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError("models must be a comma-separated subset of 2b,4b")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--models", type=parse_models, default=["2b", "4b"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--mode", choices=("smoke", "golden", "benchmark"), default="smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--golden-source", type=Path, default=DEFAULT_GOLDEN_SOURCE)
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="run from the local golden/reference artifacts and write JSON only",
    )
    parser.add_argument("--llama-bin", type=Path, default=DEFAULT_LLAMA_BIN)
    parser.add_argument("--model-2b", type=Path)
    parser.add_argument("--model-4b", type=Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed JSON-only golden results from --output",
    )
    return parser.parse_args()


def write_output(
    output_path: Path,
    run_id: str,
    mode: str,
    models: list[dict],
    samples: list[dict],
    golden_metadata: dict | None,
    results: list[dict],
    run_error: str,
    golden_path: Path = DEFAULT_GOLDEN_SET,
    source_path: Path = DEFAULT_GOLDEN_SOURCE,
) -> None:
    output = {
        "schema_version": "local-qwen-evaluation-v1",
        "run_id": run_id,
        "mode": mode,
        "created_at": utc_now(),
        "models": models,
        "sample_count": len({row["sample_key"] for row in results}),
        "result_count": len(results),
        "summary": summarize(results),
        "results": [_json_value(row) for row in results],
        "error": run_error,
        "golden_set": {
            "path": str(golden_path),
            "source_artifact": str(source_path),
            "sample_count": len(samples),
            "version": golden_metadata.get("version") if isinstance(golden_metadata, dict) else None,
        } if mode != "smoke" else None,
        "decision": (
            "golden benchmark; review golden_pass and human answer labels before any model decision"
            if mode != "smoke" else "smoke test only; do not use this run to accept or delete 4B"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    load_env_file(args.env_file)
    use_database = bool(os.getenv("DATABASE_URL")) and not args.no_database
    if args.mode == "smoke" and not use_database:
        raise SystemExit("smoke mode requires DATABASE_URL; use --mode golden --no-database for local artifact evaluation")

    # Importing app after loading the protected env file gives the evaluator
    # the same database and retrieval configuration as the running service.
    import app as app_module

    run_id = f"qwen-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output_path = args.output.resolve()
    model_paths = {
        "2b": args.model_2b,
        "4b": args.model_4b,
    }
    models = []
    for label in args.models:
        path = resolve_model_path(label, model_paths[label])
        models.append({
            "label": label,
            "path": str(path),
            "alias": MODEL_SPECS[label]["alias"],
            "quantization": MODEL_SPECS[label]["quantization"],
            "run_id": run_id,
        })

    samples = []
    golden_metadata = None
    results = []
    run_error = ""
    checkpoint = lambda: write_output(
        output_path, run_id, args.mode, models, samples, golden_metadata, results, run_error,
        args.golden_set, args.golden_source,
    )
    try:
        if use_database:
            with app_module.db() as conn:
                if args.mode == "smoke":
                    samples = prepare_samples(conn, args.limit, app_module)
                    golden_metadata = None
                else:
                    samples, golden_metadata = prepare_golden_samples(
                        conn, args.golden_set, args.golden_source, args.limit, app_module,
                    )
                config = {
                    "runner": "evaluate_local_qwen.py",
                    "reasoning": "off",
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "context_size": args.context_size,
                    "threads": 2,
                    "threads_batch": 2,
                    "parallel": 1,
                    "golden_set": str(args.golden_set) if args.mode != "smoke" else None,
                    "sample_keys": [sample["sample_key"] for sample in samples],
                }
                if args.resume:
                    raise SystemExit("--resume is supported only with --no-database")
                insert_run(conn, run_id, args.mode, samples, models, config, str(output_path))
                conn.commit()
        else:
            samples, golden_metadata = prepare_golden_samples(
                None, args.golden_set, args.golden_source, args.limit, app_module,
            )
            config = {
                "runner": "evaluate_local_qwen.py",
                "reasoning": "off",
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "context_size": args.context_size,
                "threads": 2,
                "threads_batch": 2,
                "parallel": 1,
                "database": False,
                "golden_set": str(args.golden_set),
                "sample_keys": [sample["sample_key"] for sample in samples],
            }
            if args.resume and output_path.is_file():
                try:
                    previous = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"cannot resume from {output_path}: {exc}") from exc
                if previous.get("mode") != args.mode:
                    raise RuntimeError("resume output mode does not match current mode")
                valid_keys = {sample["sample_key"] for sample in samples}
                valid_models = {model["label"] for model in models}
                results = []
                for row in previous.get("results", []):
                    if row.get("model_name") not in valid_models or row.get("sample_key") not in valid_keys:
                        continue
                    if row.get("error_message"):
                        continue
                    resumed_row = dict(row)
                    resumed_row["run_id"] = run_id
                    results.append(resumed_row)

        completed = {(row.get("model_name"), row.get("sample_key")) for row in results}
        checkpoint = lambda: write_output(
            output_path, run_id, args.mode, models, samples, golden_metadata, results, run_error,
            args.golden_set, args.golden_source,
        )

        for position, model in enumerate(models):
            if all((model["label"], sample["sample_key"]) in completed for sample in samples):
                continue
            port = args.port + position
            command = build_llama_command(
                args.llama_bin,
                Path(model["path"]),
                model["alias"],
                port,
                args.context_size,
            )
            server = LocalLlamaServer(command, port, startup_timeout=args.startup_timeout)
            try:
                with server:
                    client = LocalQwenClient(server.base_url, model["alias"], args.request_timeout)
                    try:
                        for sample in samples:
                            if (model["label"], sample["sample_key"]) in completed:
                                continue
                            error = None
                            response = None
                            try:
                                response = client.complete(sample["messages"], max_tokens=args.max_tokens)
                            except Exception as exc:
                                error = exc
                            result = _result_row(sample, model, server, response, error, app_module)
                            result["command"] = command
                            results.append(result)
                            completed.add((model["label"], sample["sample_key"]))
                            checkpoint()
                            if use_database:
                                with app_module.db() as result_conn:
                                    insert_result(result_conn, result)
                                    result_conn.commit()
                    finally:
                        client.close()
            except Exception as exc:
                run_error = f"{model['label']} server failed: {exc}"
                for sample in samples:
                    result = _result_row(sample, model, server, None, exc, app_module)
                    result["command"] = command
                    results.append(result)
                    completed.add((model["label"], sample["sample_key"]))
                    checkpoint()
                    if use_database:
                        with app_module.db() as result_conn:
                            insert_result(result_conn, result)
                            result_conn.commit()

        if use_database:
            with app_module.db() as conn:
                finish_run(conn, run_id, "failed" if run_error else "completed", run_error)
                conn.commit()
    except Exception as exc:
        run_error = str(exc)
        try:
            if use_database:
                with app_module.db() as conn:
                    finish_run(conn, run_id, "failed", run_error)
                    conn.commit()
        except Exception:
            pass

    write_output(
        output_path, run_id, args.mode, models, samples, golden_metadata, results, run_error,
        args.golden_set, args.golden_source,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps({"run_id": run_id, "output": str(output_path), "summary": output["summary"], "error": run_error}, ensure_ascii=False))
    return 1 if run_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
