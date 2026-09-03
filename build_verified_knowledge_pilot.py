#!/usr/bin/env python3
"""Build pending Verified Knowledge Candidates for the five pilot packages.

This is a review-artifact builder only. It never writes PostgreSQL and never
changes the production answer path. Existing taxonomy records select
the cases; this script does not reclassify the historical question corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import psycopg
from psycopg.rows import dict_row

from embeddings import OpenRouterEmbeddingClient, read_openrouter_token
from llm import OPENROUTER_DEFAULT_MODEL, OPENROUTER_DEFAULT_TIMEOUT_SECONDS, OPENROUTER_PROVIDER, OpenRouterLLM, parse_json_response


DEFAULT_DATA_DIR = Path("/opt/aihelper/data")
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "verified_knowledge_candidates_openrouter.json"
DEFAULT_REVIEW = Path("/opt/aihelper/VERIFIED_KNOWLEDGE_PILOT_REVIEW_OPENROUTER.md")
TAXONOMY_PATH = DEFAULT_DATA_DIR / "knowledge_intents_v1_1_openrouter.json"
MAX_PACKAGE_CANDIDATES = 3

PACKAGE_CONFIG = {
    "A": {
        "name": "Password Reset",
        "knowledge_keys": "password_access.reset*",
        "instruction": (
            "Keep reset procedures separate by brand, device type, model, and "
            "software/tool. Do not create a universal password-reset method. "
            "Distinguish camera, NVR, access terminal, iFlow, SADP, and "
            "email/file support processes when the thread supports that split."
        ),
        "keys": None,
    },
    "B": {
        "name": "Firmware Acquisition",
        "knowledge_keys": "firmware.find_or_get or firmware.download",
        "instruction": (
            "Cover only where firmware is obtained: manufacturer portal, support "
            "portal, support email, internal source, public download, or a "
            "model-specific source. Do not process firmware.check_latest and do "
            "not turn a freshness-sensitive source into a permanent fact."
        ),
        "keys": {"firmware.find_or_get", "firmware.download"},
    },
    "C": {
        "name": "Rack Ears",
        "knowledge_keys": "accessory.check_bundle.rack_ears",
        "instruction": (
            "Keep rack-ear claims model-scoped. F-NR-232X/2 and F-HR-2164/2 "
            "must not share one unscoped bundle fact."
        ),
        "keys": {"accessory.check_bundle.rack_ears"},
    },
    "D": {
        "name": "Intercom Capacity",
        "knowledge_keys": "monitor-to-door-station and door-station-to-monitor limits",
        "instruction": (
            "Keep directions separate. A monitor maximum number of connected "
            "door stations is not the same fact as a door station maximum number "
            "of connected monitors. Preserve model-scoped limits."
        ),
        "keys": {
            "compatibility.check_limit.intercom_monitor_to_door_station",
            "capacity_limit.check_limit.intercom_monitor_to_door_station",
            "configuration.check_limit.door_station_to_intercom_monitor",
            "capacity_limit.check_limit.max_subscribers.door_station_to_intercom_monitor",
        },
    },
    "E": {
        "name": "Autotracking",
        "knowledge_keys": "autotracking capability",
        "instruction": (
            "Separate capability by model or product family. A tracking "
            "configuration problem is not proof that every model supports the "
            "feature."
        ),
        "keys": None,
    },
}

PACKAGE_E_TOKENS = (
    "autotrack",
    "auto_tracking",
    "автотрекинг",
    "слежение за объектом",
    "tracking",
)

THREAD_ROLES = {
    "user_report",
    "engineer_hypothesis",
    "engineer_instruction",
    "observed_result",
    "confirmed_resolution",
    "unconfirmed_claim",
}

ANALYSIS_SYSTEM_PROMPT = """You organize one Telegram support thread for a review-only knowledge pilot.

Use only the supplied thread. Do not use pretrained product knowledge, do not
invent a model, procedure, number, source, or resolution, and do not answer
the support question. Classify every message. User messages are normally
user_report, but a later user statement describing an observed outcome may be
observed_result or confirmed_resolution. Engineer statements that say maybe,
probably, or speculate are engineer_hypothesis. Instructions are
engineer_instruction. A result is confirmed only when the thread explicitly
supports that the change worked or an engineer explicitly confirms it; otherwise
use unconfirmed_claim.

Return JSON only:
{
  "message_roles": [{"message_index": 0, "role": "user_report", "reason": "..."}],
  "scope": {"brands": [], "product_families": [], "models": [], "hardware_revisions": [], "firmware_versions": [], "software_versions": [], "operating_modes": []},
  "claim_candidates": [{"claim": "...", "claim_type": "...", "status": "confirmed_resolution|observed_result|instruction|hypothesis|unconfirmed", "message_indexes": []}],
  "question_patterns": [],
  "procedure_steps": [],
  "conditions": [],
  "exceptions": [],
  "warnings": [],
  "open_questions": [],
  "resolution_confirmed": false
}

Keep claims concise and quote no more than necessary. An engineer hypothesis
must never be marked confirmed_resolution."""

SYNTHESIS_SYSTEM_PROMPT = """You create review-only Verified Knowledge Candidate drafts from supplied Telegram analyses and official evidence.

Use only the supplied material. Do not use pretrained knowledge. Do not fill
missing product facts. Do not approve anything. Engineer hypotheses and
unconfirmed claims cannot support a factual claim by themselves. If Telegram
and official evidence disagree, preserve the disagreement in conflicts and do
not choose a winner. If only Telegram supports a claim, confidence cannot be
high. If evidence is insufficient, use confidence low and add open_questions.

Every official citation must use an evidence_id supplied in official_evidence.
Never invent document_id, document_title, page, or chunk_id. Every candidate
must remain pending and production_answer_allowed must be false. Return JSON
only in the form {"candidates": [...]} with the candidate fields requested by
the pilot. Generate exactly one concise candidate unless separate scopes
clearly require a split; generate no more than two candidates. Keep each
candidate concise and keep the complete JSON response below 1800 tokens."""


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json(content: str) -> dict[str, Any]:
    result = parse_json_response(content)
    if not isinstance(result, dict):
        raise ValueError("OpenRouter response must be a JSON object")
    return result


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("OPENROUTER_REQUESTS_PER_MINUTE must be positive")
        self.interval = 60.0 / requests_per_minute
        self.next_start = time.monotonic()
        self.lock = Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if wait:
            time.sleep(wait)


def call_json(llm: OpenRouterLLM, method: str, messages: list[dict], limiter: RequestRateLimiter, max_tokens: int) -> dict[str, Any]:
    repair = {
        "role": "user",
        "content": "The previous output was not valid JSON. Return the requested JSON object only; do not add commentary.",
    }
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            limiter.acquire()
            content = getattr(llm, method)(messages, max_tokens=max_tokens)
            return parse_json(content)
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 0:
                messages = messages + [repair]
        except Exception as error:
            # OpenRouterLLM owns finite network/timeout retries. Do not call a
            # different provider or model after an OpenRouter failure.
            last_error = error
            break
    raise RuntimeError(f"OpenRouter {method} JSON call failed: {type(last_error).__name__}") from last_error


def taxonomy_items() -> list[dict]:
    value = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    items = value.get("intents")
    if not isinstance(items, list):
        raise ValueError("knowledge intent taxonomy is missing intents")
    return [item for item in items if isinstance(item, dict)]


def select_cases(items: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for code, config in PACKAGE_CONFIG.items():
        selected = []
        for item in items:
            key = str(item.get("knowledge_key", ""))
            question = str(item.get("canonical_question", "")).casefold()
            if code == "A":
                include = key.startswith("password_access.reset")
            elif code == "B":
                include = key in config["keys"]
            elif code == "C":
                include = key in config["keys"]
            elif code == "D":
                include = key in config["keys"]
            else:
                include = any(token in key.casefold() or token in question for token in PACKAGE_E_TOKENS)
            if include:
                selected.append(item)
        result[code] = sorted(selected, key=lambda item: int(item["support_case_id"]))
    return result


def load_case_rows(conn, case_ids: list[int]) -> dict[int, dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, root_question, root_author, messages, media, models, verification_status, production_answer_allowed "
            "FROM support_cases WHERE id=ANY(%s)",
            (case_ids,),
        )
        rows = {int(row["id"]): dict(row) for row in cur.fetchall()}
    missing = sorted(set(case_ids) - set(rows))
    if missing:
        raise ValueError(f"selected support cases missing from PostgreSQL: {missing}")
    return rows


def attachment_metadata(message: dict) -> list[dict[str, Any]]:
    result = []
    for key in ("file", "photo"):
        if message.get(key):
            result.append({"kind": key, "present": True})
    return result


def thread_payload(row: dict) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    root_author = str(row.get("root_author") or "")
    payload = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        author = str(message.get("author") or "")
        actor = "user" if (author and author == root_author) or (not root_author and index == 0) else "engineer"
        payload.append({
            "message_index": index,
            "actor": actor,
            "date": message.get("date"),
            "message_id": message.get("message_id"),
            "reply_to_message_id": message.get("reply_to_message_id"),
            "text": str(message.get("text") or ""),
            "attachments": attachment_metadata(message),
        })
    if not payload and row.get("root_question"):
        payload.append({
            "message_index": 0,
            "actor": "user",
            "date": None,
            "message_id": None,
            "reply_to_message_id": None,
            "text": str(row["root_question"]),
            "attachments": [],
        })
    return payload


def clean_list(value: Any, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def clean_scope(value: Any) -> dict[str, list[str]]:
    source = value if isinstance(value, dict) else {}
    fields = (
        "brands", "product_families", "models", "hardware_revisions",
        "firmware_versions", "software_versions", "operating_modes",
    )
    return {field: clean_list(source.get(field)) for field in fields}


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def analyze_case(llm: OpenRouterLLM, package_code: str, item: dict, row: dict, limiter: RequestRateLimiter) -> dict[str, Any]:
    thread = thread_payload(row)
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "package": package_code,
            "package_name": PACKAGE_CONFIG[package_code]["name"],
            "package_instruction": PACKAGE_CONFIG[package_code]["instruction"],
            "support_case_id": int(item["support_case_id"]),
            "taxonomy_knowledge_key": item.get("knowledge_key"),
            "canonical_question": item.get("canonical_question"),
            "taxonomy_scope_models": item.get("scope_models", item.get("models", [])),
            "thread": thread,
        }, ensure_ascii=False)},
    ]
    result = call_json(llm, "extract", messages, limiter, max_tokens=1200)
    roles = []
    for entry in result.get("message_roles", []):
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role", "unconfirmed_claim"))
        roles.append({
            "message_index": safe_int(entry.get("message_index", -1)),
            "role": role if role in THREAD_ROLES else "unconfirmed_claim",
            "reason": str(entry.get("reason", "")).strip()[:400],
        })
    claims = []
    for entry in result.get("claim_candidates", []):
        if not isinstance(entry, dict) or not str(entry.get("claim", "")).strip():
            continue
        claims.append({
            "claim": str(entry["claim"]).strip()[:1000],
            "claim_type": str(entry.get("claim_type", "other")).strip() or "other",
            "status": str(entry.get("status", "unconfirmed")).strip() or "unconfirmed",
            "message_indexes": [safe_int(index) for index in entry.get("message_indexes", []) if str(index).lstrip("-").isdigit()],
        })
    return {
        "case_id": int(item["support_case_id"]),
        "knowledge_key": item.get("knowledge_key"),
        "canonical_question": item.get("canonical_question", row.get("root_question", "")),
        "taxonomy_scope_models": clean_list(item.get("scope_models", item.get("models", []))),
        "thread_message_count": len(thread),
        "message_roles": roles,
        "scope": clean_scope(result.get("scope")),
        "claim_candidates": claims,
        "question_patterns": clean_list(result.get("question_patterns")),
        "procedure_steps": clean_list(result.get("procedure_steps")),
        "conditions": clean_list(result.get("conditions")),
        "exceptions": clean_list(result.get("exceptions")),
        "warnings": clean_list(result.get("warnings")),
        "open_questions": clean_list(result.get("open_questions")),
        "resolution_confirmed": bool(result.get("resolution_confirmed")),
    }


def official_citation(evidence_id: str, hit: dict) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_type": "official_document",
        "document_id": int(hit["document_id"]),
        "document_title": str(hit["title"]),
        "page": int(hit["page_number"]) if hit.get("page_number") is not None else None,
        "chunk_id": int(hit["id"]),
    }


def retrieve_official(conn, package_cases: list[dict], case_rows: dict[int, dict], embedder) -> tuple[dict[int, list[dict]], dict[str, dict]]:
    from app import retrieve

    import app
    app.embedder = embedder
    by_case: dict[int, list[dict]] = {}
    source_by_id: dict[str, dict] = {}
    for item in package_cases:
        case_id = int(item["support_case_id"])
        hits, _trace = retrieve(conn, str(item.get("canonical_question") or case_rows[case_id].get("root_question") or ""), limit=5)
        citations = []
        for hit in hits:
            evidence_id = f"official:{int(hit['document_id'])}:{int(hit['id'])}"
            source_by_id.setdefault(evidence_id, {
                **official_citation(evidence_id, hit),
                "content": str(hit.get("content") or "")[:1800],
            })
            citations.append(evidence_id)
        by_case[case_id] = citations
    return by_case, source_by_id


def compact_synthesis_analyses(analyses: list[dict]) -> list[dict]:
    """Keep package synthesis within the model context window.

    Full Telegram messages are sent to OpenRouter during per-case extraction. The
    package pass only needs the resulting role/claim summaries, not every role
    rationale again.
    """
    compact = []
    for analysis in analyses:
        role_counts = Counter(entry.get("role") for entry in analysis.get("message_roles", []) if entry.get("role"))
        claims = []
        for claim in analysis.get("claim_candidates", [])[:8]:
            if not isinstance(claim, dict):
                continue
            claims.append({
                "claim": str(claim.get("claim", ""))[:700],
                "claim_type": str(claim.get("claim_type", "other")),
                "status": str(claim.get("status", "unconfirmed")),
                "message_indexes": [safe_int(index) for index in claim.get("message_indexes", [])[:12] if str(index).lstrip("-").isdigit()],
            })
        compact.append({
            "case_id": analysis.get("case_id"),
            "knowledge_key": analysis.get("knowledge_key"),
            "canonical_question": str(analysis.get("canonical_question", ""))[:700],
            "taxonomy_scope_models": analysis.get("taxonomy_scope_models", []),
            "scope": analysis.get("scope", {}),
            "message_role_counts": dict(role_counts),
            "claim_candidates": claims,
            "question_patterns": analysis.get("question_patterns", [])[:8],
            "procedure_steps": analysis.get("procedure_steps", [])[:8],
            "conditions": analysis.get("conditions", [])[:8],
            "exceptions": analysis.get("exceptions", [])[:8],
            "warnings": analysis.get("warnings", [])[:8],
            "open_questions": analysis.get("open_questions", [])[:8],
            "resolution_confirmed": bool(analysis.get("resolution_confirmed")),
            "official_evidence_ids": analysis.get("official_evidence_ids", []),
        })
    return compact


def compact_synthesis_sources(sources: list[dict], limit: int = 5, snippet_limit: int = 3) -> list[dict]:
    """Pass a bounded official catalog plus a small set of searchable snippets."""
    compact = []
    for index, source in enumerate(sources[:limit]):
        compact.append({
            key: value for key, value in source.items()
            if key != "content"
        } | {"content": str(source.get("content", ""))[:400] if index < snippet_limit else ""})
    return compact


def package_prompt(package_code: str, analyses: list[dict], sources: list[dict]) -> list[dict]:
    config = PACKAGE_CONFIG[package_code]
    return [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "package": package_code,
            "package_name": config["name"],
            "package_instruction": config["instruction"],
            "allowed_taxonomy_keys": sorted(config["keys"] or []),
            "case_analyses": compact_synthesis_analyses(analyses),
            "official_evidence": compact_synthesis_sources(sources),
            "generation_constraints": {
                "default_candidate_count": 1,
                "maximum_candidate_count": 2,
                "maximum_response_tokens": 1800,
            },
            "candidate_schema": {
                "candidate_id": "temporary; the builder assigns the final ID",
                "knowledge_key": "package-appropriate existing key",
                "title": "review title",
                "knowledge_type": "procedure|capability|compatibility|limit|product_fact|troubleshooting|support_process|other",
                "scope": {"brands": [], "product_families": [], "models": [], "hardware_revisions": [], "firmware_versions": [], "software_versions": [], "operating_modes": []},
                "question_patterns": [],
                "claims": [{"claim": "", "claim_type": "", "evidence": []}],
                "procedure_steps": [], "conditions": [], "exceptions": [], "warnings": [],
                "telegram_cases": [], "official_sources": [], "conflicts": [], "open_questions": [],
                "confidence": "low|medium|high",
                "freshness_sensitive": package_code == "B",
                "last_verified_at": None,
                "verification_status": "pending",
                "review_note": "",
                "production_answer_allowed": False,
            },
        }, ensure_ascii=False)},
    ]


def package_fallback(package_code: str, cases: list[dict], analyses: list[dict], sources: dict[str, dict], failure: str) -> list[dict]:
    key = {
        "A": "password_access.reset",
        "B": "firmware.find_or_get",
        "C": "accessory.check_bundle.rack_ears",
        "D": "compatibility.check_limit.intercom_monitor_to_door_station",
        "E": "feature_capability.check_support.autotracking",
    }[package_code]
    return [{
        "knowledge_key": key,
        "title": f"{PACKAGE_CONFIG[package_code]['name']} — engineer review required",
        "knowledge_type": "support_process" if package_code in {"A", "B"} else "other",
        "scope": {}, "question_patterns": [], "claims": [], "procedure_steps": [],
        "conditions": [], "exceptions": [], "warnings": [],
        "telegram_cases": [int(item["support_case_id"]) for item in cases],
        "official_sources": [], "conflicts": [],
        "open_questions": ["OpenRouter synthesis failed; inspect the complete threads and official citations manually."],
        "confidence": "low", "review_note": f"needs_engineer: {failure}",
    }]


def valid_confidence(value: Any) -> str:
    return value if value in {"low", "medium", "high"} else "low"


def normalized_evidence(value: Any, source_by_id: dict[str, dict], case_ids: set[int]) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        if isinstance(entry, str) and entry in source_by_id:
            result.append({key: item for key, item in source_by_id[entry].items() if key != "content"})
        elif isinstance(entry, dict):
            evidence_id = str(entry.get("evidence_id", ""))
            if evidence_id in source_by_id:
                result.append({key: item for key, item in source_by_id[evidence_id].items() if key != "content"})
                continue
            try:
                case_id = int(entry.get("case_id"))
            except (TypeError, ValueError):
                continue
            if case_id in case_ids:
                result.append({
                    "source_type": "telegram",
                    "case_id": case_id,
                    "message_indexes": [int(index) for index in entry.get("message_indexes", []) if str(index).lstrip("-").isdigit()],
                    "role": str(entry.get("role", "unconfirmed_claim")),
                })
    unique = {}
    for item in result:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        unique[key] = item
    return list(unique.values())


def normalize_candidate(package_code: str, draft: dict, package_cases: list[dict], source_by_id: dict[str, dict], analyses: list[dict]) -> dict:
    case_ids = {int(item["support_case_id"]) for item in package_cases}
    scope = clean_scope(draft.get("scope"))
    if not scope["models"]:
        scope["models"] = sorted({model for item in package_cases for model in item.get("scope_models", item.get("models", [])) if model}, key=str.casefold)
    official_sources = normalized_evidence(draft.get("official_sources"), source_by_id, case_ids)
    official_sources = [item for item in official_sources if item.get("source_type") == "official_document"]
    raw_telegram_cases = draft.get("telegram_cases", [])
    if not isinstance(raw_telegram_cases, list):
        raw_telegram_cases = []
    telegram_cases = sorted({int(value) for value in raw_telegram_cases if str(value).lstrip("-").isdigit() and int(value) in case_ids})
    if not telegram_cases:
        telegram_cases = sorted(case_ids)
    claims = []
    for raw_claim in draft.get("claims", []) if isinstance(draft.get("claims"), list) else []:
        if not isinstance(raw_claim, dict) or not str(raw_claim.get("claim", "")).strip():
            continue
        claims.append({
            "claim": str(raw_claim["claim"]).strip()[:1200],
            "claim_type": str(raw_claim.get("claim_type", "other")).strip() or "other",
            "evidence": normalized_evidence(raw_claim.get("evidence"), source_by_id, case_ids),
        })
    confidence = valid_confidence(draft.get("confidence"))
    if not official_sources and confidence == "high":
        confidence = "medium"
    if not claims:
        confidence = "low"
    open_questions = clean_list(draft.get("open_questions"))
    if not official_sources and "Official document evidence is still required." not in open_questions:
        open_questions.append("Official document evidence is still required.")
    if package_code in {"C", "D", "E"} and not scope["models"]:
        confidence = "low"
        open_questions.append("Confirm the exact model or family scope before approval.")
    if package_code == "D":
        direction_text = " ".join([str(draft.get("title", ""))] + [claim["claim"] for claim in claims]).casefold()
        if "monitor" not in direction_text or "door" not in direction_text:
            open_questions.append("Confirm whether the limit direction is monitor→door station or door station→monitor.")
    if package_code == "B":
        open_questions.append("Re-check the source and version at approval time; firmware sources are freshness-sensitive.")
    telegram_evidence = []
    analysis_by_id = {int(item["case_id"]): item for item in analyses}
    for case_id in telegram_cases:
        analysis = analysis_by_id.get(case_id)
        if not analysis:
            continue
        telegram_evidence.append({
            "case_id": case_id,
            "resolution_confirmed": bool(analysis.get("resolution_confirmed")),
            "message_roles": analysis.get("message_roles", []),
        })
    return {
        "candidate_id": "",
        "knowledge_key": str(draft.get("knowledge_key", "")),
        "title": str(draft.get("title", "")).strip() or f"{PACKAGE_CONFIG[package_code]['name']} candidate",
        "knowledge_type": str(draft.get("knowledge_type", "other")).strip() or "other",
        "scope": scope,
        "question_patterns": clean_list(draft.get("question_patterns")),
        "claims": claims,
        "procedure_steps": clean_list(draft.get("procedure_steps")),
        "conditions": clean_list(draft.get("conditions")),
        "exceptions": clean_list(draft.get("exceptions")),
        "warnings": clean_list(draft.get("warnings")),
        "telegram_cases": telegram_cases,
        "telegram_evidence": telegram_evidence,
        "official_sources": official_sources,
        "conflicts": clean_list(draft.get("conflicts")),
        "open_questions": list(dict.fromkeys(open_questions)),
        "confidence": confidence,
        "freshness_sensitive": package_code == "B",
        "last_verified_at": None,
        "verification_status": "pending",
        "review_note": str(draft.get("review_note", "")).strip() or "Pending human review; OpenRouter did not approve this candidate.",
        "production_answer_allowed": False,
    }


def force_package_key(package_code: str, candidate: dict) -> None:
    allowed = PACKAGE_CONFIG[package_code]["keys"]
    if package_code == "A":
        candidate["knowledge_key"] = "password_access.reset"
    elif package_code == "B":
        if candidate["knowledge_key"] not in {"firmware.find_or_get", "firmware.download"}:
            candidate["knowledge_key"] = "firmware.find_or_get"
    elif package_code == "C":
        candidate["knowledge_key"] = "accessory.check_bundle.rack_ears"
    elif package_code == "D":
        if candidate["knowledge_key"] not in allowed:
            candidate["knowledge_key"] = "compatibility.check_limit.intercom_monitor_to_door_station"
    elif package_code == "E":
        candidate["knowledge_key"] = "feature_capability.check_support.autotracking"


def render_scope(scope: dict[str, list[str]]) -> str:
    labels = {
        "brands": "brands", "product_families": "product families", "models": "models",
        "hardware_revisions": "hardware revisions", "firmware_versions": "firmware versions",
        "software_versions": "software versions", "operating_modes": "operating modes",
    }
    return "; ".join(f"{labels[key]}: {', '.join(value) or '—'}" for key, value in scope.items())


def render_evidence(evidence: list[dict]) -> list[str]:
    lines = []
    for item in evidence:
        if item.get("source_type") == "official_document":
            lines.append(
                f"- `{item.get('evidence_id')}` — document_id={item.get('document_id')}, "
                f"{item.get('document_title')}, page={item.get('page')}, chunk_id={item.get('chunk_id')}"
            )
        else:
            lines.append(
                f"- Telegram case #{item.get('case_id')}, messages "
                f"{item.get('message_indexes', [])}, role={item.get('role', 'unconfirmed_claim')}"
            )
    return lines or ["- —"]


def render_review(candidates: list[dict], summary: dict) -> str:
    lines = [
        "# Verified Knowledge Candidate Pilot Review",
        "",
        "> Review-only drafts. Nothing here is approved Verified Knowledge or available to production answers.",
        "> All candidates are `verification_status=pending` and `production_answer_allowed=false`.",
        "",
        "## Run Summary",
        "",
        f"- processed work packages: {', '.join(summary['processed_work_packages'])}",
        f"- support cases inspected: {summary['support_cases_inspected']}",
        f"- candidates generated: {summary['candidates_generated']}",
        f"- candidates with official evidence: {summary['candidates_with_official_evidence']}",
        f"- Telegram-only candidates: {summary['telegram_only_candidates']}",
        f"- candidates with conflicts: {summary['candidates_with_conflicts']}",
        f"- candidates needing engineer review: {summary['candidates_needing_engineer_review']}",
        "",
    ]
    for candidate in candidates:
        lines.extend([
            f"## {candidate['candidate_id']} — {candidate['title']}",
            "",
            f"- Knowledge Key: `{candidate['knowledge_key']}`",
            f"- Knowledge type: `{candidate['knowledge_type']}`",
            f"- Confidence: **{candidate['confidence']}**",
            f"- Freshness-sensitive: `{str(candidate['freshness_sensitive']).lower()}`",
            f"- Last verified at: `{candidate['last_verified_at'] or 'null'}`",
            f"- Verification status: `{candidate['verification_status']}`",
            f"- Production answer allowed: `{str(candidate['production_answer_allowed']).lower()}`",
            "",
            "### Scope",
            "",
            render_scope(candidate["scope"]),
            "",
            "### Triggered Questions",
            "",
            *[f"- {value}" for value in candidate["question_patterns"] or ["—"]],
            "",
            "### Proposed Claims",
            "",
        ])
        if candidate["claims"]:
            for claim in candidate["claims"]:
                lines.append(f"- **{claim['claim_type']}**: {claim['claim']}")
                lines.extend([f"  {line}" for line in render_evidence(claim["evidence"])])
        else:
            lines.append("- —")
        lines.extend(["", "### Procedure / Facts", ""])
        lines.extend([f"- {value}" for value in candidate["procedure_steps"] or ["—"]])
        for label, key in (("Conditions", "conditions"), ("Exceptions", "exceptions"), ("Warnings", "warnings")):
            lines.extend(["", f"### {label}", "", *[f"- {value}" for value in candidate[key] or ["—"]]])
        lines.extend(["", "### Official Evidence", "", *render_evidence(candidate["official_sources"])])
        lines.extend(["", "### Telegram Evidence", ""])
        for item in candidate["telegram_evidence"] or [{"case_id": case_id, "message_roles": []} for case_id in candidate["telegram_cases"]]:
            role_counts = Counter(entry.get("role") for entry in item.get("message_roles", []) if entry.get("role"))
            role_text = ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items())) or "roles unavailable"
            lines.append(f"- case #{item['case_id']}: resolution_confirmed={str(bool(item.get('resolution_confirmed'))).lower()}; {role_text}")
        lines.extend(["", "### Conflicts", "", *[f"- {value}" for value in candidate["conflicts"] or ["—"]]])
        lines.extend(["", "### Open Questions", "", *[f"- {value}" for value in candidate["open_questions"] or ["—"]]])
        lines.extend([
            "",
            "### Review",
            "",
            "[ ] approve",
            "[ ] edit",
            "[ ] reject",
            "[ ] needs_engineer",
            "",
            "Review Note:",
            "",
            candidate["review_note"],
            "",
        ])
    return "\n".join(lines)


def build(args) -> dict:
    load_env_file(args.env_file)
    api_key = read_openrouter_token(os.environ.get("OPENROUTER_TOKEN_FILE", str(Path(__file__).with_name("openrouter"))))
    if not api_key:
        raise RuntimeError("OpenRouter token is not configured")
    rpm = int(os.environ.get("OPENROUTER_REQUESTS_PER_MINUTE", "20"))
    limiter = RequestRateLimiter(rpm)
    items = taxonomy_items()
    selected = select_cases(items)
    all_ids = sorted({int(item["support_case_id"]) for package in selected.values() for item in package})
    print(f"selected_cases={len(all_ids)} package_counts=" + json.dumps({key: len(value) for key, value in selected.items()}, sort_keys=True))

    failures = []
    candidates = []
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        rows = load_case_rows(conn, all_ids)
        embedder = OpenRouterEmbeddingClient(
            api_key,
            timeout=float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120")),
        )
        llm = OpenRouterLLM(
            api_key,
            timeout=float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", str(OPENROUTER_DEFAULT_TIMEOUT_SECONDS))),
        )
        for package_code, package_items in selected.items():
            print(f"package={package_code} cases={len(package_items)}")
            analyses = []
            for item in package_items:
                try:
                    analysis = analyze_case(llm, package_code, item, rows[int(item["support_case_id"])], limiter)
                except Exception as error:
                    failures.append({"stage": "thread_analysis", "package": package_code, "case_id": int(item["support_case_id"]), "error_type": type(error).__name__})
                    analysis = {
                        "case_id": int(item["support_case_id"]),
                        "knowledge_key": item.get("knowledge_key"),
                        "canonical_question": item.get("canonical_question", ""),
                        "taxonomy_scope_models": clean_list(item.get("scope_models", item.get("models", []))),
                        "thread_message_count": len(thread_payload(rows[int(item["support_case_id"])])),
                        "message_roles": [], "scope": {}, "claim_candidates": [],
                        "question_patterns": [], "procedure_steps": [], "conditions": [],
                        "exceptions": [], "warnings": [],
                        "open_questions": ["OpenRouter thread analysis failed; inspect this complete thread manually."],
                        "resolution_confirmed": False,
                    }
                analyses.append(analysis)
            case_evidence, source_by_id = retrieve_official(conn, package_items, rows, embedder)
            for analysis in analyses:
                analysis["official_evidence_ids"] = case_evidence.get(int(analysis["case_id"]), [])
            source_list = list(source_by_id.values())
            try:
                result = call_json(llm, "complete", package_prompt(package_code, analyses, source_list), limiter, max_tokens=2400)
                drafts = result.get("candidates")
                if not isinstance(drafts, list):
                    raise ValueError("OpenRouter synthesis did not return candidates")
                drafts = [draft for draft in drafts if isinstance(draft, dict)][:MAX_PACKAGE_CANDIDATES]
                if not drafts:
                    raise ValueError("OpenRouter synthesis returned no candidate")
            except Exception as error:
                failures.append({"stage": "package_synthesis", "package": package_code, "error_type": type(error).__name__})
                drafts = package_fallback(package_code, package_items, analyses, source_by_id, type(error).__name__)
            for draft in drafts:
                candidate = normalize_candidate(package_code, draft, package_items, source_by_id, analyses)
                force_package_key(package_code, candidate)
                candidates.append(candidate)

    for index, candidate in enumerate(candidates, 1):
        package_code = next(code for code, package in selected.items() if set(candidate["telegram_cases"]) & {int(item["support_case_id"]) for item in package})
        candidate["candidate_id"] = f"VKP-{package_code}-{index:03d}"

    summary = {
        "schema_version": 1,
        "artifact_type": "verified_knowledge_candidates_pilot",
        "provider": OPENROUTER_PROVIDER,
        "model": OPENROUTER_DEFAULT_MODEL,
        "processed_work_packages": [f"{code} — {PACKAGE_CONFIG[code]['name']}" for code in PACKAGE_CONFIG],
        "support_cases_inspected": len(all_ids),
        "package_case_counts": {code: len(value) for code, value in selected.items()},
        "candidates_generated": len(candidates),
        "candidates_with_official_evidence": sum(bool(candidate["official_sources"]) for candidate in candidates),
        "telegram_only_candidates": sum(bool(candidate["telegram_cases"]) and not candidate["official_sources"] for candidate in candidates),
        "candidates_with_conflicts": sum(bool(candidate["conflicts"]) for candidate in candidates),
        "candidates_needing_engineer_review": sum(candidate["confidence"] == "low" or bool(candidate["conflicts"]) for candidate in candidates),
        "failures": failures,
        "created_at": utc_now(),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review.write_text(render_review(candidates, summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("/etc/ai-sales-engineer.env"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    try:
        summary = build(args)
    except Exception as error:
        print(f"verified knowledge pilot failed: {type(error).__name__}")
        return 1
    print(json.dumps({key: summary[key] for key in (
        "processed_work_packages", "support_cases_inspected", "candidates_generated",
        "candidates_with_official_evidence", "telegram_only_candidates",
        "candidates_with_conflicts", "candidates_needing_engineer_review",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
