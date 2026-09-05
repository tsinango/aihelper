"""Small deterministic helpers for long-form V2 Inbox intake.

Bulk intake deliberately has no workflow engine.  It only answers three
questions before the existing learning loop runs: what kind of input arrived,
where are its logical segments, and how complete was processing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


INPUT_MODES = frozenset({
    "control_reply",
    "answer_to_current_question",
    "new_knowledge_payload",
    "bulk_knowledge_payload",
})

_CONTROL_WORDS = frozenset({
    "对", "对的", "正确", "没错", "是的", "确认", "是", "yes", "true", "y",
    "否", "不是", "不对", "不正确", "no", "false", "n", "нет",
    "不知道", "不清楚", "不确定", "не знаю", "неизвестно",
    "跳过", "以后再说", "稍后再说", "先不说", "skip", "later",
})

_NUMBERED_START = re.compile(
    r"(?im)^[ \t]*(?:(?:№\s*\d+|номер\s*\d+)[.)、:：]?|\d{1,3}\s*[.)、:：])[ \t]*"
)
_BULLET_START = re.compile(r"(?m)^[ \t]*(?:[-*•▪●‣])[ \t]+")
_KEY_VALUE_LINE = re.compile(r"(?m)^[ \t]*[^\n:：]{1,80}[：:][^\n]+$")
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])\s*")
_MODEL_TOKEN = re.compile(r"\b[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9./()_-]*\d[A-Za-zА-Яа-я0-9./()_-]*\b")
_INDIVIDUAL_CONFIRMATION_MARKERS = (
    # Explicit limitations are still clear evidence and should be eligible
    # for the batch confirmation. Only boundaries that need a separate scope
    # decision remain individual: version/revision, family-wide claims, and
    # explicit exceptions or unresolved conditions.
    "整个系列", "所有型号", "适用于系列", "hardware revision", "firmware", "revision",
    "除非", "例外", "услови", "кроме", "исключ",
)


def _short_control(content: str) -> bool:
    normalized = re.sub(r"[\s。.!！?？,，:：;；]+", "", str(content or "").casefold())
    return normalized in {word.casefold() for word in _CONTROL_WORDS}


def looks_like_bulk(content: str) -> bool:
    """Return true for unmistakably multi-part input, without an LLM call."""

    text = str(content or "").strip()
    if not text or _short_control(text):
        return False
    numbered = len(_NUMBERED_START.findall(text))
    bullets = len(_BULLET_START.findall(text))
    paragraphs = len([part for part in re.split(r"\n\s*\n", text) if part.strip()])
    key_values = len(_KEY_VALUE_LINE.findall(text))
    model_tokens = {match.group(0).casefold() for match in _MODEL_TOKEN.finditer(text)}
    line_count = len(text.splitlines())
    sentence_count = len(re.findall(r"[。！？!?；;]", text))
    return any((
        numbered >= 2,
        bullets >= 2,
        paragraphs >= 2 and len(text) >= 360,
        key_values >= 3,
        len(model_tokens) >= 3 and line_count >= 3,
        sentence_count >= 3 and len(text) >= 120,
        len(text) >= 700,
    ))


def classify_input_mode(
    content: str,
    *,
    pending_question: str | None = None,
    has_pending: bool = False,
) -> str:
    """Classify the envelope of a turn; semantic intent remains out of scope."""

    text = str(content or "").strip()
    if _short_control(text):
        return "control_reply"
    if looks_like_bulk(text):
        return "bulk_knowledge_payload"
    if has_pending or pending_question:
        return "answer_to_current_question"
    return "new_knowledge_payload"


def _numbered_segments(text: str) -> list[str]:
    matches = list(_NUMBERED_START.finditer(text))
    if len(matches) < 2:
        return []
    result: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        # Keep a heading/preamble attached to the first numbered item.  The
        # raw evidence remains the authoritative full payload, but every
        # source character also needs a deterministic processing owner.
        if index == 0:
            prefix = text[:match.start()].strip()
            if prefix:
                value = f"{prefix}\n{value}" if value else prefix
        if value:
            result.append(value)
    return result


def _line_segments(text: str, marker: re.Pattern[str]) -> list[str]:
    matches = list(marker.finditer(text))
    if len(matches) < 2:
        return []
    result: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        if value:
            result.append(value)
    return result


def _sentence_segments(text: str, *, max_chars: int = 3600) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []
    result: list[str] = []
    current: list[str] = []
    current_size = 0
    for sentence in sentences:
        if current and current_size + len(sentence) + 1 > max_chars:
            result.append(" ".join(current).strip())
            current = []
            current_size = 0
        current.append(sentence)
        current_size += len(sentence) + 1
    if current:
        result.append(" ".join(current).strip())
    return result


def segment_bulk_text(content: str) -> list[dict[str, Any]]:
    """Split a bulk payload deterministically and number every logical segment."""

    text = str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    values = _numbered_segments(text)
    if not values:
        values = _line_segments(text, _BULLET_START)
    if not values:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) >= 2:
            values = paragraphs
    if not values and len(_KEY_VALUE_LINE.findall(text)) >= 3:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if not values:
        values = _sentence_segments(text)
    segments = []
    for index, value in enumerate(values, start=1):
        segments.append({"segment_no": index, "text": value, "status": "pending"})
    return segments


def coverage(
    segments: Iterable[Mapping[str, Any]],
    *,
    facts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build an explicit coverage snapshot; no list slicing or hidden limits."""

    all_segments = list(segments)
    processed: list[int] = []
    failed: list[int] = []
    for segment in all_segments:
        number = int(segment["segment_no"])
        if segment.get("status") == "failed":
            failed.append(number)
        elif segment.get("status") == "processed":
            processed.append(number)
    fact_items = [dict(item) for item in facts]
    return {
        "expected_segments": len(all_segments),
        "processed_segments": len(processed),
        "failed_segments": len(failed),
        "failed_segment_numbers": failed,
        "extracted_facts": fact_items,
    }


def minimum_explicit_claims(source: str) -> int:
    """Estimate a lower bound for atomic claims in one extracted segment.

    This is intentionally conservative and structural.  It is not a second
    extractor: separators used by the source (labeled fields, semicolons,
    lines, or sentence boundaries) only prevent a one-fact response from
    being reported as complete when the segment visibly contains several
    claims.
    """

    text = str(source or "").strip()
    if not text:
        return 0
    labeled_fields = re.findall(
        r"(?:^|[;；\n])\s*[^;；\n:：]{1,80}[：:]\s*[^;；\n]+",
        text,
    )
    separators = re.split(r"[;；\n]+", text)
    structural_parts = [part.strip() for part in separators if part.strip()]
    sentence_count = len(re.findall(r"[。！？!?]", text))
    return max(1, len(labeled_fields), len(structural_parts), sentence_count)


def extraction_coverage_is_complete(
    source: str,
    raw_facts: Iterable[Mapping[str, Any]],
    coverage_data: Mapping[str, Any] | None,
) -> bool:
    """Validate the small extraction coverage contract used for bulk calls.

    A model response is not considered processed merely because it contains a
    fact.  It must explicitly enumerate covered source claims, attach each
    claim to one or more semantic knowledge units (or mark it non-knowledge),
    preserve source excerpts, and say that there are no uncovered claims.  A
    structural lower bound catches the common failure where a multi-field
    segment returns only its first claim and incorrectly claims completion.
    """

    if not isinstance(coverage_data, Mapping):
        return False
    if coverage_data.get("complete") is not True:
        return False
    uncovered = coverage_data.get("uncovered_claims")
    claims = coverage_data.get("claims")
    if not isinstance(uncovered, list) or uncovered:
        return False
    if not isinstance(claims, list) or len(claims) < minimum_explicit_claims(source):
        return False

    facts = [dict(item) for item in raw_facts if isinstance(item, Mapping)]
    source_text = str(source or "")
    excerpts = []
    for fact in facts:
        excerpt = str(fact.get("source_excerpt") or "").strip()
        if not excerpt or excerpt not in source_text:
            return False
        excerpts.append(excerpt)

    seen_fact_indexes: set[int] = set()
    for claim in claims:
        if not isinstance(claim, Mapping):
            return False
        claim_text = str(claim.get("text") or claim.get("source_excerpt") or "").strip()
        indexes = claim.get("knowledge_unit_indexes")
        if indexes is None:
            indexes = claim.get("fact_indexes")
        disposition = str(claim.get("disposition") or "knowledge").strip().casefold()
        if not claim_text or claim_text not in source_text:
            return False
        if indexes == [] and disposition in {"non_knowledge", "marketing", "context"}:
            continue
        if not isinstance(indexes, list) or not indexes:
            return False
        if not all(
            isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(facts)
            for index in indexes
        ):
            return False
        if not any(claim_text in excerpts[index] for index in indexes):
            return False
        seen_fact_indexes.update(indexes)
    if not facts:
        return all(
            isinstance(claim.get("knowledge_unit_indexes"), list)
            and not claim.get("knowledge_unit_indexes")
            and str(claim.get("disposition") or "").strip().casefold()
            in {"non_knowledge", "marketing", "context"}
            for claim in claims
        )
    return seen_fact_indexes == set(range(len(facts))) and len(claims) >= minimum_explicit_claims(source)


def deduplicate_knowledge(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate summary rows by final Knowledge id, then by normalized text."""

    result: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    seen_text: set[str] = set()
    for item in items:
        knowledge_id = item.get("id") or item.get("knowledge_id")
        content = str(item.get("content") or "").strip()
        key = content.casefold()
        if knowledge_id is not None and knowledge_id in seen_ids:
            continue
        if key and key in seen_text:
            continue
        if knowledge_id is not None:
            seen_ids.add(knowledge_id)
        if key:
            seen_text.add(key)
        result.append(dict(item))
    return result


def non_exhaustive_semantics(text: str) -> bool:
    """Recognize list wording that explicitly says the list is not exhaustive."""

    lowered = str(text or "").casefold()
    return any(marker in lowered for marker in ("等", "etc.", "etc", "и т.д", "и т. д", "прочие", "другие", "其他"))


def requires_individual_confirmation(fact: Mapping[str, Any]) -> bool:
    """Keep inference, scope expansion, conditions, exceptions and negatives atomic."""

    if fact.get("derived"):
        return True
    content = str(fact.get("content") or "").casefold()
    return any(marker in content for marker in _INDIVIDUAL_CONFIRMATION_MARKERS)


def parse_batch_confirmation(content: str, total_segments: int) -> set[int] | None:
    """Parse the small, deterministic subset needed for partial batch confirms."""

    text = str(content or "").strip().casefold()
    if not text or total_segments <= 0:
        return None
    if any(token in text for token in ("全部", "所有", "all")):
        return set(range(1, total_segments + 1))
    prefix = re.search(r"确认\s*前\s*(\d+)\s*(?:项|条)?", text)
    if prefix:
        count = min(int(prefix.group(1)), total_segments)
        return set(range(1, count + 1))
    listed = re.search(r"(?:确认|正确|保留)\s*(?:第\s*)?([0-9、,，和及\s-]+)\s*(?:项|条)", text)
    if not listed:
        return None
    values: set[int] = set()
    for token in re.findall(r"\d+", listed.group(1)):
        number = int(token)
        if 1 <= number <= total_segments:
            values.add(number)
    return values or None


# Short aliases keep the small helper pleasant to use from tests and future
# Inbox integrations without introducing another abstraction layer.
detect_input_mode = classify_input_mode
segment_bulk = segment_bulk_text


__all__ = [
    "INPUT_MODES",
    "classify_input_mode",
    "detect_input_mode",
    "coverage",
    "extraction_coverage_is_complete",
    "deduplicate_knowledge",
    "looks_like_bulk",
    "non_exhaustive_semantics",
    "minimum_explicit_claims",
    "requires_individual_confirmation",
    "parse_batch_confirmation",
    "segment_bulk_text",
    "segment_bulk",
]
