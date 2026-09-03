#!/usr/bin/env python3
"""Build a complete, review-only knowledge workbook from Telegram exports.

The existing intent pipeline keeps one canonical question per case.  That is
useful for retrieval, but it is not enough for human review: a support thread
can contain follow-up questions and a customer can confirm that an engineer's
answer worked.  This script creates a lossless review artifact for *every*
support case.  It deliberately uses the checked-in SQLite export and existing
V2.1 analysis, so generating the workbook is offline, repeatable, and does not
write PostgreSQL or call an LLM.

The most important invariant is ``root_author`` handling.  Every message is
preserved.  Messages by root_author after the first question are classified as
customer feedback/results, never silently discarded from the answer evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telegram_relations import (
    classify_message as infer_message_role,
    infer_message_relations,
    is_customer_message,
    is_question_text,
)


DEFAULT_CASES_SQL = Path(__file__).with_name("d1-export") / "support_cases.sql"
DEFAULT_ANALYSIS_SQL = Path(__file__).with_name("d1-export") / "support_case_analysis.sql"
DEFAULT_OUTPUT = Path(__file__).with_name("data") / "telegram_knowledge_review.json"
DEFAULT_REVIEW = Path(__file__).with_name("TELEGRAM_KNOWLEDGE_REVIEW.md")
DEFAULT_TAXONOMY = Path(__file__).with_name("data") / "knowledge_intents_v1_1_openrouter.json"
ANALYSIS_PROMPT = "TELEGRAM_QUESTION_EXTRACTION_V2_1"

SCOPE_FIELDS = (
    "brands", "product_families", "series", "models", "hardware_revisions",
    "firmware_versions", "software_versions", "operating_modes",
)

# A customer confirmation is evidence about the proposed answer, not a reason
# to promote it automatically.  The reviewer still decides whether it is
# reusable and whether the scope is safe.
POSITIVE_FEEDBACK = re.compile(
    r"(?:помогло|помог|заработал[оаи]?|работа(?:ет|ло|ет)|получилось|решено|всё работает|все работает|спасибо.*(?:работ|помог)|worked|works|fixed|solved)",
    re.IGNORECASE,
)
NEGATIVE_FEEDBACK = re.compile(
    r"(?:не помог|не заработ|не работает|не получилось|не реш|ошибк|still|doesn't work|not working)",
    re.IGNORECASE,
)
VERSION_RE = re.compile(r"\b(?:v(?:ersion)?\s*)?\d+(?:\.\d+){1,4}(?:\s*(?:build|билд)\s*\d+)?\b", re.IGNORECASE)
MODEL_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:[-/][A-Z0-9]+)+(?:\([A-Z0-9-]+\))?\b", re.IGNORECASE)


def parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default
        return parsed
    return default


def clean_list(value: Any, limit: int = 40) -> list[str]:
    values = parse_json(value, value if isinstance(value, list) else [])
    if not isinstance(values, list):
        return []
    result = []
    seen = set()
    for item in values:
        text = str(item).strip()
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result[:limit]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_export(cases_sql: Path, analysis_sql: Path) -> tuple[list[dict], list[dict]]:
    """Load the SQLite-compatible D1 exports without needing a live DB."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(cases_sql.read_text(encoding="utf-8"))
        connection.executescript(analysis_sql.read_text(encoding="utf-8"))
        connection.row_factory = sqlite3.Row
        cases = [dict(row) for row in connection.execute("SELECT * FROM support_cases ORDER BY id")]
        analyses = [dict(row) for row in connection.execute(
            "SELECT * FROM support_case_analysis ORDER BY support_case_id, id"
        )]
        return cases, analyses
    finally:
        connection.close()


def choose_analysis(rows: list[dict]) -> dict[int, dict]:
    """Prefer V2.1, then the newest available analysis as a safe fallback."""
    selected: dict[int, dict] = {}
    for row in rows:
        case_id = int(row["support_case_id"])
        current = selected.get(case_id)
        is_preferred = str(row.get("prompt_version", "")) == ANALYSIS_PROMPT
        current_preferred = current and str(current.get("prompt_version", "")) == ANALYSIS_PROMPT
        if current is None or (is_preferred and not current_preferred) or (
            is_preferred == current_preferred and int(row.get("id", 0)) > int(current.get("id", 0))
        ):
            selected[case_id] = row
    return selected


def messages_for(case: dict) -> list[dict]:
    messages = parse_json(case.get("messages"), [])
    return [dict(message) for message in messages if isinstance(message, dict)]


def is_root_author(message: dict, case: dict, index: int) -> bool:
    """Compatibility helper; message decisions use the richer role model."""
    return is_customer_message(message, case, index)


def classify_message(message: dict, case: dict, index: int) -> str:
    return infer_message_role(message, case, index, messages_for(case))


def is_question_message(message: dict, case: dict, index: int) -> bool:
    """Detect a root-author follow-up question versus result/feedback.

    Telegram exports do not carry an explicit message role.  A root author can
    therefore appear several times in one thread.  Only the first root turn
    is unconditionally a question; subsequent turns containing a question
    mark or a common Russian/English interrogative are follow-up questions.
    Plain statements such as ``Спасибо, помогло`` remain feedback.
    """
    if not is_customer_message(message, case, index):
        return False
    if index == 0:
        return True
    text = str(message.get("text") or "").strip()
    return is_question_text(text)


def feedback_status(messages: list[dict], case: dict) -> tuple[str, list[int]]:
    feedback = []
    for index, message in enumerate(messages):
        if index == 0 or not is_customer_message(message, case, index):
            continue
        text = str(message.get("text") or "")
        feedback.append(index)
        if POSITIVE_FEEDBACK.search(text) and not NEGATIVE_FEEDBACK.search(text):
            return "confirmed_resolution", feedback
    if feedback:
        return "observed_result", feedback
    if any(str(message.get("text") or "").strip() for index, message in enumerate(messages) if not is_customer_message(message, case, index)):
        return "engineer_answer", []
    return "unanswered", []


def question_parts(text: str) -> list[str]:
    """Expose obvious multi-question parts while retaining the raw message."""
    text = str(text or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in re.findall(r"[^?\n]*\?", text) if part.strip()]
    if len(parts) <= 1:
        return [text]
    return parts[:12]


def atomic_qa(messages: list[dict], case: dict, analysis: dict) -> list[dict]:
    """Pair each customer turn with replies until the next customer turn."""
    user_indexes = [i for i, m in enumerate(messages) if is_question_message(m, case, i)]
    result = []
    for position, question_index in enumerate(user_indexes):
        next_question = user_indexes[position + 1] if position + 1 < len(user_indexes) else len(messages)
        reply_indexes = [
            i for i in range(question_index + 1, next_question)
            if not is_customer_message(messages[i], case, i) and str(messages[i].get("text") or "").strip()
        ]
        feedback_indexes = [
            i for i in range(question_index + 1, next_question)
            if is_customer_message(messages[i], case, i) and not is_question_message(messages[i], case, i) and str(messages[i].get("text") or "").strip()
        ]
        if feedback_indexes and any(
            POSITIVE_FEEDBACK.search(str(messages[i].get("text") or ""))
            and not NEGATIVE_FEEDBACK.search(str(messages[i].get("text") or ""))
            for i in feedback_indexes
        ):
            status = "confirmed_resolution"
        elif feedback_indexes:
            status = "observed_result"
        elif reply_indexes:
            status = "engineer_answer"
        else:
            status = "unanswered"
        result.append({
            "atomic_qa_id": f"{int(case['id'])}-{position + 1}",
            "question_message_indexes": [question_index],
            "question": str(messages[question_index].get("text") or "").strip(),
            "question_parts": question_parts(str(messages[question_index].get("text") or "")),
            "canonical_question": str(analysis.get("canonical_question") or "").strip() if position == 0 else "",
            "answer_message_indexes": reply_indexes,
            "answer_text": "\n".join(str(messages[i].get("text") or "").strip() for i in reply_indexes),
            "feedback_message_indexes": feedback_indexes,
            "status": status,
            "answer_allowed_after_review": False,
        })
    if not result and case.get("root_question"):
        result.append({
            "atomic_qa_id": f"{int(case['id'])}-1", "question_message_indexes": [],
            "question": str(case["root_question"]).strip(), "question_parts": question_parts(case["root_question"]),
            "canonical_question": str(analysis.get("canonical_question") or "").strip(),
            "answer_message_indexes": [], "answer_text": "", "feedback_message_indexes": [],
            "status": "unanswered", "answer_allowed_after_review": False,
        })
    return result


def model_scope(analysis: dict, case: dict, messages: list[dict]) -> dict[str, list[str]]:
    models = clean_list(analysis.get("models_json"))
    models.extend(clean_list(case.get("models")))
    # Do not turn arbitrary uppercase words into scope; existing extraction is
    # authoritative.  This only deduplicates its two already-known sources.
    models = clean_list(models)
    all_text = "\n".join(str(m.get("text") or "") for m in messages)
    versions = clean_list(VERSION_RE.findall(all_text))
    scope = {field: [] for field in SCOPE_FIELDS}
    scope["models"] = models
    scope["firmware_versions"] = versions
    if models and versions:
        scope["hardware_revisions"] = clean_list(re.findall(r"\b(?:rev(?:ision)?|ревизия)\s*[A-Za-z0-9._-]+", all_text, re.IGNORECASE))
    return scope


def infer_scope_level(scope: dict[str, list[str]], analysis: dict, answer_status: str) -> tuple[str, str]:
    if answer_status == "unanswered" or str(analysis.get("question_quality")) in {"ambiguous", "non_question", "low_value"}:
        return "single_case", "Нет подтверждённого ответа или вопрос требует контекста."
    if scope["models"] and (scope["firmware_versions"] or scope["software_versions"] or scope["operating_modes"]):
        return "model_condition", "Модель и условие должны быть проверены вместе."
    if scope["models"]:
        return "model", "Ответ ограничен явно извлечёнными моделями; расширение до серии/семейства требует проверки."
    products = clean_list(analysis.get("products_json"))
    if products:
        return "brand", "Извлечённый продукт/бренд требует ручной нормализации."
    return "generic", "Модель не извлечена; считать общим знанием только после проверки."


def hierarchy_candidates(scope: dict[str, list[str]]) -> dict[str, Any]:
    series = []
    for model in scope["models"]:
        # Candidate only: DS-N104P -> DS-N, F-VI-3445IPE1 -> F-VI.  Never
        # silently uses this as an answer scope.
        match = re.match(r"^([A-Za-z]+-[A-Za-z]+)", model)
        if match and match.group(1).casefold() not in {x.casefold() for x in series}:
            series.append(match.group(1))
    return {"series_candidates": series, "status": "pending_human_confirmation"}


def make_case(case: dict, analysis: dict | None) -> dict:
    analysis = analysis or {}
    messages = messages_for(case)
    for index, message in enumerate(messages):
        message["message_index"] = index
        message["actor"] = "user" if is_customer_message(message, case, index) else "engineer"
        message["review_role"] = classify_message(message, case, index)
    status, feedback_indexes = feedback_status(messages, case)
    scope = model_scope(analysis, case, messages)
    level, note = infer_scope_level(scope, analysis, status)
    qas = atomic_qa(messages, case, analysis)
    answer_indexes = sorted({i for qa in qas for i in qa["answer_message_indexes"]})
    evidence_indexes = sorted(set(answer_indexes + feedback_indexes))
    feedback_text = "\n".join(
        str(messages[i].get("text") or "").strip() for i in feedback_indexes
    )
    return {
        "support_case_id": int(case["id"]),
        "external_thread_id": str(case.get("external_thread_id") or ""),
        "source_content_hash": str(case.get("content_hash") or analysis.get("source_content_hash") or ""),
        "date_start": case.get("date_start"), "date_end": case.get("date_end"),
        "root_author": case.get("root_author"), "root_question": case.get("root_question"),
        "message_count": len(messages), "messages": messages,
        "message_relations": infer_message_relations({**case, "messages": messages}),
        "analysis": {
            "analysis_id": analysis.get("id"), "prompt_version": analysis.get("prompt_version"),
            "question_quality": analysis.get("question_quality"),
            "canonical_question": analysis.get("canonical_question"),
            "domain": analysis.get("domain"), "knowledge_type": analysis.get("knowledge_type"),
            "knowledge_key": analysis.get("knowledge_key"),
            "family": analysis.get("family"), "action": analysis.get("action"),
            "object_type": analysis.get("object_type"), "context_status": analysis.get("context_status"),
            "secondary_questions": clean_list(analysis.get("secondary_questions_json")),
            "extraction_confidence": analysis.get("extraction_confidence"),
        },
        "atomic_qa": qas,
        "answer_candidate": {
            "text": "\n".join(str(messages[i].get("text") or "").strip() for i in answer_indexes),
            "answer_message_indexes": answer_indexes,
            "feedback_message_indexes": feedback_indexes,
            "evidence_message_indexes": evidence_indexes,
            "confirmation_status": status,
            "customer_feedback_present": bool(feedback_indexes),
            "customer_feedback_was_included": True,
            "customer_feedback_text": feedback_text,
        },
        "scope": scope,
        "scope_level": level,
        "scope_note": note,
        "hierarchy_candidates": hierarchy_candidates(scope),
        "review": {
            "status": "pending", "approved_for_bot": False, "review_note": "",
            "needs_engineer": level in {"single_case", "model_condition"} or status in {"unanswered", "observed_result"},
        },
    }


def load_taxonomy(path: Path) -> dict[int, dict]:
    """Load the existing V2.1 enrichment when available.

    The SQL analysis table predates the intent fields (knowledge_key, family,
    action, and context status).  Merging the checked-in artifact here avoids
    throwing away that work while still allowing the eleven cases omitted by
    the old quality filter to remain in the workbook.
    """
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("intents", []) if isinstance(payload, dict) else []
    result = {}
    for item in items:
        if isinstance(item, dict) and item.get("support_case_id") is not None:
            result[int(item["support_case_id"])] = item
    return result


def merged_analysis(analysis: dict | None, taxonomy: dict | None) -> dict:
    merged = dict(analysis or {})
    for key, value in (taxonomy or {}).items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    # V2.1 calls the model scope ``scope_models`` while SQL calls it
    # ``models_json``.  Keep both available to downstream reviewers.
    if not merged.get("models_json") and merged.get("scope_models"):
        merged["models_json"] = merged["scope_models"]
    return merged


def candidate_from_case(case: dict) -> dict:
    """Return one import-shaped candidate per case.

    One-to-one candidates intentionally avoid automatic grouping.  Reviewers
    can merge reusable cases later, while a case-specific answer can never
    accidentally become a generic bot answer during import.
    """
    analysis = case["analysis"]
    answer = case["answer_candidate"]
    scope_level = {
        "model_condition": "conditional",
        # The database enum intentionally calls this unspecified: a singleton
        # answer is not a claim that the product scope is generic.
        "single_case": "unspecified",
    }.get(case["scope_level"], case["scope_level"])
    answer_text = answer.get("text", "")
    if answer.get("customer_feedback_text"):
        answer_text = (answer_text + "\n\n" if answer_text else "") + (
            "[Подтверждение пользователя — не проверено инженером]: "
            + answer["customer_feedback_text"]
        )
    question_patterns = clean_list([
        case.get("root_question"), analysis.get("canonical_question"),
        *analysis.get("secondary_questions", []),
        *(qa.get("question") for qa in case.get("atomic_qa", [])),
    ])
    claims = []
    if answer.get("text"):
        claims.append({
            "claim": answer["text"], "claim_type": "historical_telegram_answer",
            "evidence": [{
                "source_type": "telegram", "case_id": case["support_case_id"],
                "message_indexes": answer["evidence_message_indexes"],
                "role": "confirmed_resolution" if answer["confirmation_status"] == "confirmed_resolution" else "unconfirmed_claim",
            }],
        })
    if answer.get("customer_feedback_text"):
        claims.append({
            "claim": answer["customer_feedback_text"], "claim_type": "customer_result_feedback",
            "evidence": [{
                "source_type": "telegram", "case_id": case["support_case_id"],
                "message_indexes": answer["feedback_message_indexes"], "role": "confirmed_resolution",
            }],
        })
    role_map = {
        "user_question": "user_report", "engineer_reply": "engineer_instruction",
        "user_report": "user_report", "engineer_hypothesis": "engineer_hypothesis",
        "engineer_instruction": "engineer_instruction", "unconfirmed_claim": "unconfirmed_claim",
        "irrelevant": "irrelevant", "observed_result": "observed_result",
        "confirmed_resolution": "confirmed_resolution",
    }
    telegram_evidence = [{
        "case_id": case["support_case_id"],
        "resolution_confirmed": answer["confirmation_status"] == "confirmed_resolution",
        "message_roles": [{
            "message_index": message.get("message_index"),
            "role": role_map.get(message.get("review_role"), "unconfirmed_claim"),
            "reason": "Deterministic role from author, reply metadata, content and thread position; review required.",
        } for message in case.get("messages", [])],
    }]
    return {
        "candidate_id": f"CASE-{case['support_case_id']:06d}",
        "knowledge_key": analysis.get("knowledge_key") or f"telegram.case.{case['support_case_id']}",
        "title": str(analysis.get("canonical_question") or case.get("root_question") or f"Telegram case #{case['support_case_id']}")[:500],
        "knowledge_type": analysis.get("knowledge_type") or "other",
        "scope": case.get("scope", {}), "scope_level": scope_level,
        "question_patterns": question_patterns, "claims": claims,
        "answer_text": answer_text,
        "answer_status": "pending", "procedure_steps": [], "conditions": [],
        "exceptions": [], "warnings": ["Исторический ответ Telegram; требуется ручная проверка."],
        "confidence": "medium" if answer["confirmation_status"] == "confirmed_resolution" else "low",
        "freshness_sensitive": False, "last_verified_at": None,
        "verification_status": "pending", "review_status": "pending", "review_note": "",
        "production_answer_allowed": False, "frequency": 1,
        "telegram_cases": [case["support_case_id"]], "telegram_evidence": telegram_evidence,
        "message_relations": case.get("message_relations", []),
        "official_sources": [], "conflicts": [],
        "open_questions": [case["scope_note"]],
        "source_case_scope_level": case["scope_level"],
    }


def render_review(cases: list[dict], metadata: dict) -> str:
    counts = Counter(case["scope_level"] for case in cases)
    statuses = Counter(case["answer_candidate"]["confirmation_status"] for case in cases)
    lines = [
        "# Telegram Knowledge Review Workbook", "",
        "> Review-only artifact. Every case is pending and unavailable to the bot until a human approves it.",
        "> Full messages are retained. Customer follow-up messages are explicitly shown as feedback/result evidence.", "",
        "## Summary", "", f"- Cases: {len(cases)}",
        f"- Atomic QA records: {sum(len(case['atomic_qa']) for case in cases)}",
        f"- Scope levels: {dict(sorted(counts.items()))}", f"- Answer statuses: {dict(sorted(statuses.items()))}", "",
    ]
    for case in cases:
        answer = case["answer_candidate"]
        analysis = case["analysis"]
        lines.extend([
            f"## Case #{case['support_case_id']} — {analysis.get('canonical_question') or case.get('root_question') or '—'}", "",
            f"- Review: `{case['review']['status']}`; bot allowed: `false`",
            f"- Scope: `{case['scope_level']}`; models: {', '.join(case['scope']['models']) or '—'}",
            f"- Knowledge key: `{analysis.get('knowledge_key') or '—'}`; quality: `{analysis.get('question_quality') or '—'}`",
            f"- Answer status: `{answer['confirmation_status']}`; customer feedback included: `{str(answer['customer_feedback_was_included']).lower()}`", "",
            "### Atomic QA", "",
        ])
        for qa in case["atomic_qa"]:
            lines.extend([
                f"- **{qa['atomic_qa_id']} / {qa['status']}** Q[{qa['question_message_indexes']}] {qa['question']}",
                f"  A[{qa['answer_message_indexes']}] {qa['answer_text'] or '—'}",
                f"  feedback[{qa['feedback_message_indexes']}]", "",
            ])
        lines.extend(["### Review", "", "[ ] approve   [ ] edit   [ ] reject   [ ] needs_engineer", "", "Review note: ______", ""])
    return "\n".join(lines)


def build(
    cases_sql: Path = DEFAULT_CASES_SQL,
    analysis_sql: Path = DEFAULT_ANALYSIS_SQL,
    taxonomy_path: Path = DEFAULT_TAXONOMY,
) -> dict:
    cases, analyses = load_export(cases_sql, analysis_sql)
    analysis_by_case = choose_analysis(analyses)
    taxonomy = load_taxonomy(taxonomy_path)
    organized = [
        make_case(case, merged_analysis(analysis_by_case.get(int(case["id"])), taxonomy.get(int(case["id"]))))
        for case in cases
    ]
    if len(organized) != 602:
        raise ValueError(f"expected 602 support cases, got {len(organized)}")
    if len(analysis_by_case) != len(organized):
        raise ValueError(f"missing analysis for {len(organized) - len(analysis_by_case)} support cases")
    all_ids = [case["support_case_id"] for case in organized]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate support_case_id in organized artifact")
    candidates = [candidate_from_case(case) for case in organized]
    payload = {
        "schema_version": 1,
        "artifact_type": "telegram_knowledge_review",
        "review_only": True,
        "bot_answer_default": False,
        "source": {
            "cases_sql": str(cases_sql), "analysis_sql": str(analysis_sql),
            "taxonomy_artifact": str(taxonomy_path),
            "analysis_prompt_preferred": ANALYSIS_PROMPT,
            "cases_sha256": hashlib.sha256(cases_sql.read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(analysis_sql.read_bytes()).hexdigest(),
        },
        "summary": {
            "support_cases": len(organized),
            "atomic_qa": sum(len(case["atomic_qa"]) for case in organized),
            "root_author_feedback_cases": sum(case["answer_candidate"]["customer_feedback_present"] for case in organized),
            "confirmed_resolution_cases": sum(case["answer_candidate"]["confirmation_status"] == "confirmed_resolution" for case in organized),
            "unanswered_cases": sum(case["answer_candidate"]["confirmation_status"] == "unanswered" for case in organized),
            "scope_levels": dict(sorted(Counter(case["scope_level"] for case in organized).items())),
            "review_status": {"pending": len(organized), "approved": 0},
        },
        "created_at": utc_now(),
        # Import-shaped one-to-one candidates.  They are deliberately also
        # present beside the lossless case workbook for simple import tools.
        "candidates": candidates,
        "cases": organized,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-sql", type=Path, default=DEFAULT_CASES_SQL)
    parser.add_argument("--analysis-sql", type=Path, default=DEFAULT_ANALYSIS_SQL)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    payload = build(args.cases_sql, args.analysis_sql, args.taxonomy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review.write_text(render_review(payload["cases"], payload["summary"]), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
