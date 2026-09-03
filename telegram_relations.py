"""Small, conservative helpers for Telegram message roles and relations.

This module intentionally does not model a conversation graph.  It only keeps
the few relationships that are useful during review and preserves manual
corrections over importer-generated rows.
"""

from __future__ import annotations

import re


MESSAGE_ROLES = {
    "user_report",
    "engineer_hypothesis",
    "engineer_instruction",
    "observed_result",
    "confirmed_resolution",
    "unconfirmed_claim",
    "irrelevant",
}
RELATION_TYPES = {"answers", "confirm_success", "confirm_failure", "reply_to"}
RELATION_SOURCES = {"manual", "telegram", "inferred"}
SOURCE_PRIORITY = {"inferred": 1, "telegram": 2, "manual": 3}

POSITIVE_FEEDBACK = re.compile(
    r"(?:помогло|помог|заработал[оаи]?|работа(?:ет|ло|ет)|получилось|решено|всё работает|все работает|спасибо.*(?:работ|помог)|worked|works|fixed|solved)",
    re.IGNORECASE,
)
NEGATIVE_FEEDBACK = re.compile(
    r"(?:не помог|не заработ|не работает|не получилось|не реш|ошибк|still|doesn't work|not working)",
    re.IGNORECASE,
)
QUESTION_START = re.compile(
    r"^(?:как|где|что|сколько|какой|какая|какие|можно ли|почему|подскажите|уточните|how|where|what|which|can we|is there)\b",
    re.IGNORECASE,
)
INSTRUCTION = re.compile(
    r"(?:^|\s)(?:проверьте|проверь|укажите|отправьте|отправь|перейдите|перейди|нажмите|нажми|включите|включи|откройте|открой|установите|установи|обновите|обнови|сбросьте|сбрось|замените|замени|check|please|send|open|go to|click|install|update|reset)\b",
    re.IGNORECASE,
)
HYPOTHESIS = re.compile(
    r"(?:возможно|вероятно|скорее всего|похоже|может быть|если я правильно|probably|perhaps|might be|seems)",
    re.IGNORECASE,
)


def message_id(message: dict, index: int) -> str:
    """Return a stable per-thread identity even for incomplete exports."""
    value = message.get("message_id")
    return str(value).strip() if value not in (None, "") else f"index:{index}"


def _root_author(case: dict) -> str:
    return str(case.get("root_author") or "").strip().casefold()


def _author(message: dict) -> str:
    return str(message.get("author") or "").strip().casefold()


def is_customer_message(message: dict, case: dict, index: int) -> bool:
    """Use author as one signal, with first-turn fallback for old exports."""
    root = _root_author(case)
    author = _author(message)
    return bool(root and author == root) or (not root and index == 0)


def is_question_text(text: str) -> bool:
    text = str(text or "").strip()
    return "?" in text or "？" in text or bool(QUESTION_START.search(text))


def classify_message(message: dict, case: dict, index: int, messages: list[dict] | None = None) -> str:
    """Infer a review role using content, position, author and reply metadata.

    The root author is not sufficient to classify a message: a customer can
    report a result or confirm success, and an engineer can ask a question.
    """
    explicit = str(message.get("review_role") or message.get("role") or "").strip()
    if explicit in MESSAGE_ROLES:
        return explicit
    text = str(message.get("text") or message.get("content") or "").strip()
    customer = is_customer_message(message, case, index)
    if customer:
        if index == 0 or is_question_text(text):
            return "user_report"
        if POSITIVE_FEEDBACK.search(text) and not NEGATIVE_FEEDBACK.search(text):
            return "confirmed_resolution"
        return "observed_result"
    if HYPOTHESIS.search(text):
        return "engineer_hypothesis"
    if INSTRUCTION.search(text) or message.get("reply_to_message_id"):
        return "engineer_instruction"
    if not text:
        return "irrelevant"
    return "unconfirmed_claim"


def message_evidence_status(message: dict, role: str | None = None) -> str:
    """Map a reviewed message to the conservative evidence status vocabulary."""
    role = role or str(message.get("effective_role") or message.get("role") or "")
    text = str(message.get("text") or message.get("content") or "")
    if role == "confirmed_resolution" and not NEGATIVE_FEEDBACK.search(text):
        return "confirmed_success"
    if NEGATIVE_FEEDBACK.search(text) or role == "confirm_failure":
        return "confirmed_failure"
    if role in {"engineer_instruction", "engineer_hypothesis", "unconfirmed_claim", "observed_result"}:
        return "supports"
    return "context_only"


def infer_message_relations(case: dict) -> list[dict]:
    """Return native replies and only high-confidence result relations."""
    messages = [item for item in case.get("messages", []) if isinstance(item, dict)]
    identities = {message_id(item, index): index for index, item in enumerate(messages)}
    roles = {
        index: classify_message(item, case, index, messages)
        for index, item in enumerate(messages)
    }
    relations: list[dict] = []

    for index, message in enumerate(messages):
        source_id = message_id(message, index)
        target_raw = message.get("reply_to_message_id")
        if target_raw in (None, ""):
            continue
        target_id = str(target_raw).strip()
        relation_type = "reply_to"
        if target_id in identities and roles[index] == "engineer_instruction":
            target_index = identities[target_id]
            if roles[target_index] == "user_report":
                relation_type = "answers"
        relations.append({
            "source_message_id": source_id,
            "target_message_id": target_id,
            "relation_type": relation_type,
            "source": "telegram",
            "confidence": 1.0,
        })

    for index, message in enumerate(messages):
        role = roles[index]
        if role not in {"confirmed_resolution", "observed_result"}:
            continue
        previous = next(
            (candidate_index for candidate_index in range(index - 1, -1, -1)
             if roles[candidate_index] in {"engineer_instruction", "engineer_hypothesis", "unconfirmed_claim"}),
            None,
        )
        if previous is None:
            continue
        text = str(message.get("text") or "")
        positive = role == "confirmed_resolution"
        if positive or NEGATIVE_FEEDBACK.search(text):
            relations.append({
                "source_message_id": message_id(message, index),
                "target_message_id": message_id(messages[previous], previous),
                "relation_type": "confirm_success" if positive else "confirm_failure",
                "source": "inferred",
                "confidence": 0.92 if positive else 0.88,
            })
    return relations


def preferred_relation_source(current: str | None, incoming: str) -> str:
    """Choose the strongest source without ever replacing a manual row."""
    if current in SOURCE_PRIORITY and SOURCE_PRIORITY[current] >= SOURCE_PRIORITY.get(incoming, 0):
        return current
    return incoming


def merge_relation(existing: dict | None, incoming: dict) -> dict:
    """Pure rebuild rule used by tests and mirrored by the SQL upsert."""
    if not existing:
        return dict(incoming)
    if SOURCE_PRIORITY.get(str(existing.get("source")), 0) >= SOURCE_PRIORITY.get(str(incoming.get("source")), 0):
        return dict(existing)
    return {**existing, **incoming}


def upsert_message_relations(cur, case_id: int, root_author: str | None, messages) -> int:
    """Persist generated relations while preserving manual rows.

    ``cur`` is intentionally duck-typed so the offline role/relation logic
    remains testable without opening a database connection.
    """
    case = {"root_author": root_author, "messages": [item for item in messages or [] if isinstance(item, dict)]}
    count = 0
    for relation in infer_message_relations(case):
        cur.execute(
            """
            INSERT INTO message_relations
              (support_case_id,source_message_id,target_message_id,relation_type,source,confidence)
            VALUES(%s,%s,%s,%s,%s,%s)
            ON CONFLICT(support_case_id,source_message_id,target_message_id,relation_type)
            DO UPDATE SET
              source=CASE
                WHEN message_relations.source='manual' THEN message_relations.source
                WHEN message_relations.source='telegram' AND EXCLUDED.source='inferred' THEN message_relations.source
                ELSE EXCLUDED.source
              END,
              confidence=CASE
                WHEN message_relations.source='manual' THEN message_relations.confidence
                WHEN message_relations.source='telegram' AND EXCLUDED.source='inferred' THEN message_relations.confidence
                ELSE EXCLUDED.confidence
              END,
              updated_at=CURRENT_TIMESTAMP
            """,
            (case_id, relation["source_message_id"], relation["target_message_id"], relation["relation_type"], relation["source"], relation["confidence"]),
        )
        count += 1
    return count
