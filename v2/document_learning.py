"""Phase 4.2 document-specific learning: complete, reusable Knowledge units.

This pipeline never touches the text-learning splitters
(``segment_bulk_text``, ``_split_obvious_conjoined_unit``,
``_consolidate_related_units``), forced Russian canonicalization, or
per-fact compare.  One learning context -- a whole section, topic, or group
of related slides -- yields several complete units (fact / procedure / rule
/ experience) in a single extraction, each citing the real blocks it came
from.  A procedure keeps its ordered steps, prerequisites, expected result,
warnings, and exceptions together; it is never shredded into atom facts.

Every deterministic check below (block existence, excerpt containment,
identifier fidelity, required structure) guards the pipeline shape only.
Semantic correctness is decided by the engineer at confirm time, and only a
confirm promotes a proposal to answerable Knowledge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from psycopg.types.json import Jsonb

from helpers import identifiers
from llm import parse_json_response
from v2.documents import get_version

log = logging.getLogger("aihelper.v2.document_learning")

V2_DOC_EXTRACT_PROMPT_VERSION = "v2-doc-extract-1"
EXTRACT_MAX_TOKENS = 4000
# One context stays small enough for a bounded model call; oversized
# sections split along block boundaries into numbered parts.
MAX_CONTEXT_CHARS = 8000
MAX_UNIT_CONTENT_CHARS = 12000
UNIT_KINDS = ("fact", "procedure", "rule", "experience")


class DocumentLearnError(ValueError):
    """Invalid extraction output or bad confirm payload (maps to 400/409)."""


class DocumentLearnNotFound(LookupError):
    """No such proposal, version, or job."""


def _text(value: Any, limit: int = MAX_UNIT_CONTENT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise DocumentLearnError(
            f"unit text exceeds {limit} characters; split the context instead of truncating"
        )
    return text


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _clean_applicability(value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DocumentLearnError("applicability must be a JSON object")
    return {str(key): value[key] for key in value}


# -- contexts ---------------------------------------------------------------


def build_learning_contexts(blocks: list[dict]) -> list[dict]:
    """Group text blocks into whole-section / related-slide contexts.

    Image-only and empty blocks carry no text to learn from; they keep their
    4.1 processing state and stay out of extraction contexts.  Tables, notes,
    headings, and paragraphs learn together with their section so steps keep
    their conditions and warnings.
    """

    teachable = [
        block for block in blocks
        if str(block.get("evidence_text") or block.get("text") or "").strip()
        and str(block.get("block_type") or "") != "image"
    ]
    groups: dict[str, dict] = {}
    order: list[str] = []
    for block in blocks:
        if block not in teachable:
            continue
        section = list(block.get("section_path") or [])
        group_title = str(section[0] or "").strip() if section else ""
        if not group_title:
            if block.get("slide_no") is not None:
                group_title = f"Slide {int(block['slide_no'])}"
            elif block.get("page_no") is not None:
                group_title = f"Page {int(block['page_no'])}"
            else:
                group_title = "Untitled"
        if group_title not in groups:
            groups[group_title] = {"title": group_title, "blocks": []}
            order.append(group_title)
        locator = (
            f"slide {int(block['slide_no'])}" if block.get("slide_no") is not None
            else f"page {int(block['page_no'])}" if block.get("page_no") is not None
            else str(block.get("block_key") or "")
        )
        groups[group_title]["blocks"].append({
            "block_id": int(block["id"]),
            "block_key": str(block.get("block_key") or ""),
            "locator": locator,
            "text": str(block.get("evidence_text") or block.get("text") or ""),
        })
    contexts = []
    for title in order:
        pending = list(groups[title]["blocks"])
        part = 0
        while pending:
            part += 1
            chunk: list[dict] = []
            size = 0
            while pending and size + len(pending[0]["text"]) <= MAX_CONTEXT_CHARS:
                item = pending.pop(0)
                chunk.append(item)
                size += len(item["text"])
            if not chunk:
                # One block alone exceeds the budget: learn it solo and flag it.
                item = pending.pop(0)
                chunk.append(item)
            key = title if part == 1 and not pending else f"{title} (part {part})"
            contexts.append({
                "context_key": key,
                "title": title,
                "blocks": chunk,
                "approx_chars": sum(len(item["text"]) for item in chunk),
            })
    return contexts


# -- extraction --------------------------------------------------------------


def build_extract_messages(version: dict, context: dict) -> list[dict]:
    blocks = [
        {"block_id": item["block_id"], "locator": item["locator"], "text": item["text"]}
        for item in context["blocks"]
    ]
    system = (
        "You extract complete, reusable technical knowledge units from one "
        "section of a product manual for an internal support engineer. "
        "Output ONE JSON object only: "
        '{"units": [{"title": "...", "unit_kind": "fact|procedure|rule|experience", '
        '"applicability": {"models": [], "versions": [], "conditions": []}, '
        '"prerequisites": [], "ordered_steps": [], "expected_result": "", '
        '"warnings": [], "exceptions": [], "trigger": "", "result": "", '
        '"observation": "", "content": "...", '
        '"sources": [{"block_id": 0, "excerpt": "..."}]}]}. "'
        "Rules: keep one procedure/rule/experience whole with all its steps, "
        "conditions, and warnings -- never split it into sentence facts. "
        "Every unit needs at least one source citing a supplied block_id with "
        "an excerpt copied verbatim from that block. Never invent model "
        "names, numbers, versions, or steps not stated in the blocks. "
        "Preserve technical identifiers unchanged. Write content in the same "
        "language as the block texts. If a block is unclear, skip it rather "
        "than guessing; an empty units list is acceptable."
    )
    user = {
        "document": {
            "key": str(version.get("document_key") or ""),
            "title": str(version.get("title") or version.get("file_name") or ""),
            "section": str(context.get("title") or ""),
        },
        "blocks": blocks,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def extract_units(llm_service, version: dict, context: dict) -> list[dict]:
    """Run one bounded extraction for a context; shape-checked, not trusted."""

    if llm_service is None:
        raise DocumentLearnError("learning model is not configured")
    messages = build_extract_messages(version, context)
    content = llm_service.extract_structured(
        messages,
        {"type": "object", "properties": {"units": {"type": "array"}}},
        EXTRACT_MAX_TOKENS,
    )
    parsed = parse_json_response(content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("units"), list):
        raise DocumentLearnError("extraction output must be a JSON object with a units list")
    return [unit for unit in parsed["units"] if isinstance(unit, dict)]


# -- deterministic validation -------------------------------------------------


def _allowed_tokens(version: dict, block_texts: dict[int, str]) -> tuple[set[str], set[str]]:
    pool = " ".join([str(version.get("title") or ""), str(version.get("document_key") or ""),
                     *block_texts.values()])
    models = {str(token).upper() for token in identifiers(pool)}
    digits = set(re.findall(r"\d{3,}", pool))
    return models, digits


def validate_units(
    conn, version_id: int, units: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split extraction output into shippable units and hard errors.

    Returns ``(valid, errors)`` where each error names the unit title and
    the failed check.  Errors never auto-repair text; the context stays
    retryable and its blocks stay pending.
    """

    version = get_version(conn, version_id)
    if version is None:
        raise DocumentLearnNotFound(f"V2 document version {int(version_id)} was not found")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT b.id, b.block_key, r.content
            FROM v2_document_blocks b
            LEFT JOIN v2_raw_evidence r ON r.id=b.raw_evidence_id
            WHERE b.version_id=%s
            """,
            (int(version_id),),
        )
        block_map = {
            int(row["id"]): (str(row.get("block_key") or ""), str(row.get("content") or ""))
            for row in cur.fetchall()
        }
    allowed_models, allowed_digits = _allowed_tokens(
        version, {key: text for key, (_, text) in block_map.items()},
    )
    return check_units(version, block_map, allowed_models, allowed_digits, units)


def check_units(version: dict, block_map: dict, allowed_models: set,
                allowed_digits: set, units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pure half of validate_units: no database touched."""
    valid: list[dict] = []
    errors: list[dict] = []
    for position, unit in enumerate(units):
        title = f"unit {position + 1}"
        try:
            title = str(unit.get("title") or "").strip()[:200]
            if not title:
                raise DocumentLearnError("unit title is required")
            kind = str(unit.get("unit_kind") or "").strip()
            if kind not in UNIT_KINDS:
                raise DocumentLearnError(f"unknown unit_kind: {kind!r}")
            content = _text(unit.get("content"))
            if not content:
                raise DocumentLearnError("unit content is required")
            applicability = _clean_applicability(unit.get("applicability"))
            details = unit.get("details") if isinstance(unit.get("details"), dict) else {}
            details = _validate_details(kind, unit, details)
            sources = unit.get("sources")
            if not isinstance(sources, list) or not sources:
                raise DocumentLearnError("at least one block source is required")
            checked_sources = []
            for source in sources:
                if not isinstance(source, dict):
                    raise DocumentLearnError("each source must be an object")
                try:
                    block_id = int(source.get("block_id"))
                except (TypeError, ValueError) as exc:
                    raise DocumentLearnError("source block_id must be an integer") from exc
                if block_id not in block_map:
                    raise DocumentLearnError(f"source block {block_id} is not part of this version")
                excerpt = str(source.get("excerpt") or "").strip()
                if not excerpt:
                    raise DocumentLearnError("source excerpt must not be empty")
                _, block_text = block_map[block_id]
                if _normalized(excerpt) not in _normalized(block_text):
                    raise DocumentLearnError(
                        f"source excerpt is not verbatim from block {block_id}"
                    )
                checked_sources.append({"block_id": block_id, "excerpt": excerpt[:4000]})
            _check_identifiers(title, content, checked_sources, block_map, allowed_models, allowed_digits)
            valid.append({
                "title": title, "unit_kind": kind, "content": content,
                "applicability": applicability, "details": details,
                "sources": checked_sources,
            })
        except DocumentLearnError as exc:
            errors.append({"title": title, "error": str(exc)})
    return valid, errors


def _validate_details(kind: str, unit: dict, details: dict) -> dict:
    """Kind-specific structure floor; unknown keys pass through untouched."""

    result = {str(key): details[key] for key in details}
    strings = ("expected_result", "trigger", "result", "observation")
    lists = ("prerequisites", "ordered_steps", "warnings", "exceptions")
    for key in strings:
        if key in result and not isinstance(result[key], str):
            raise DocumentLearnError(f"details.{key} must be a string")
    for key in lists:
        if key in result:
            if not isinstance(result[key], list) or not all(isinstance(item, str) for item in result[key]):
                raise DocumentLearnError(f"details.{key} must be a list of strings")
    if kind == "procedure":
        steps = list(unit.get("ordered_steps") or result.get("ordered_steps") or [])
        if not [step for step in steps if str(step).strip()]:
            raise DocumentLearnError("procedure units require non-empty ordered_steps")
        result["ordered_steps"] = [str(step) for step in steps]
    if kind == "rule":
        trigger = str(unit.get("trigger") or result.get("trigger") or "").strip()
        outcome = str(unit.get("result") or result.get("result") or "").strip()
        if not trigger or not outcome:
            raise DocumentLearnError("rule units require trigger and result")
        result["trigger"] = trigger
        result["result"] = outcome
    # Top-level convenience fields also land in details so nothing is lost.
    for key in (*strings, *lists):
        if key in unit and key not in result:
            result[key] = unit[key]
    if "title" not in result:
        result["title"] = str(unit.get("title") or "").strip()[:200]
    return result


def _check_identifiers(title, content, sources, block_map, allowed_models, allowed_digits) -> None:
    cited = " ".join(block_map[item["block_id"]][1] for item in sources)
    cited_models = {str(token).upper() for token in identifiers(cited)}
    for token in identifiers(f"{title}\n{content}"):
        if str(token).upper() not in allowed_models and str(token).upper() not in cited_models:
            raise DocumentLearnError(
                f"identifier {token!r} does not occur in the cited blocks or document title"
            )
    cited_digits = set(re.findall(r"\d{3,}", cited))
    for digits in set(re.findall(r"\d{3,}", f"{title}\n{content}")):
        if digits not in allowed_digits and digits not in cited_digits:
            raise DocumentLearnError(
                f"number {digits!r} does not occur in the cited blocks or document title"
            )


# -- proposals -----------------------------------------------------------------


def _proposal_fingerprint(title: str, content: str) -> str:
    return hashlib.sha256(
        f"{_normalized(title)}\n{_normalized(content)}".encode("utf-8")
    ).hexdigest()


def existing_unit_fingerprints(conn, version_id: int) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.details_json, p.fact_text
            FROM v2_learning_proposals p
            WHERE p.origin_document_version_id=%s
              AND p.status IN ('pending_confirmation', 'pending_clarification', 'confirmed')
            """,
            (int(version_id),),
        )
        prints = set()
        for row in cur.fetchall():
            details = row.get("details_json") or {}
            title = details.get("title") if isinstance(details, dict) else ""
            prints.add(_proposal_fingerprint(str(title or ""), str(row.get("fact_text") or "")))
        return prints


def save_unit_proposals(
    conn, version_id: int, context_key: str, units: list[dict],
) -> list[dict]:
    """Persist validated units as pending proposals; deterministic dedup.

    Structured kinds (procedure/rule/experience) are stored with content
    deterministically rendered from their validated details, so the
    readable text and the structure can never disagree downstream.  Plain
    facts keep the extracted prose.
    """

    seen = existing_unit_fingerprints(conn, version_id)
    proposals = []
    with conn.cursor() as cur:
        for unit in units:
            content = unit["content"]
            if unit["unit_kind"] in ("procedure", "rule", "experience"):
                content = render_unit_content(
                    unit["title"], unit["unit_kind"],
                    {**unit.get("details", {}), "applicability": unit["applicability"]},
                )
            fingerprint = _proposal_fingerprint(unit["title"], content)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            details = dict(unit.get("details") or {})
            details["title"] = unit["title"]
            details["sources"] = unit["sources"]
            cur.execute(
                """
                INSERT INTO v2_learning_proposals(
                    thread_id, fact_text, entity_name, proposed_trust,
                    status, comparison_result, comparison_reason,
                    related_knowledge_ids, unit_kind, applicability, revision,
                    details_json, origin_document_version_id
                ) VALUES(NULL, %s, '', 'provisional', 'pending_confirmation',
                         'NEW', 'document_learn', ARRAY[]::BIGINT[], %s, %s, 1, %s, %s)
                RETURNING id, fact_text, entity_name, proposed_trust, status,
                          comparison_result, unit_kind, applicability, revision,
                          details_json, origin_document_version_id,
                          created_at, updated_at
                """,
                (
                    content, unit["unit_kind"], Jsonb(unit["applicability"]),
                    Jsonb(details), int(version_id),
                ),
            )
            proposal = dict(cur.fetchone())
            proposals.append(proposal)
            cur.execute(
                """
                UPDATE v2_document_blocks
                SET processing_state='proposal',
                    state_reason=%s, updated_at=CURRENT_TIMESTAMP
                WHERE version_id=%s AND id = ANY(%s)
                  AND processing_state IN ('pending', 'needs_review')
                """,
                (
                    f"document_learn:{context_key}"[:200], int(version_id),
                    [item["block_id"] for item in unit["sources"]],
                ),
            )
    return proposals


# -- learn jobs ------------------------------------------------------------------


def queue_learn_jobs(conn, version_id: int) -> list[dict]:
    """Create one queued learn job per context; idempotent per context key."""

    from v2.documents import get_blocks

    version = get_version(conn, version_id)
    if version is None:
        raise DocumentLearnNotFound(f"V2 document version {int(version_id)} was not found")
    blocks = get_blocks(conn, version_id)
    rows = [
        {
            "id": int(block["id"]), "block_key": str(block.get("block_key") or ""),
            "section_path": list(block.get("section_path") or []),
            "page_no": block.get("page_no"), "slide_no": block.get("slide_no"),
            "block_type": str(block.get("block_type") or ""),
            "evidence_text": str(block.get("evidence_text") or ""),
        }
        for block in blocks
        if str(block.get("processing_state") or "") in ("pending", "needs_review")
    ]
    contexts = build_learning_contexts(rows)
    jobs = []
    with conn.cursor() as cur:
        for context in contexts:
            key = f"v2doc:learn:{int(version_id)}:{_normalized(context['context_key'])}"
            cur.execute(
                """
                INSERT INTO v2_document_jobs(
                    version_id, stage, checkpoint, idempotency_key, status
                ) VALUES(%s, 'learn', %s, %s, 'queued')
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING id
                """,
                (
                    int(version_id),
                    Jsonb({
                        "context_key": context["context_key"],
                        "block_ids": [item["block_id"] for item in context["blocks"]],
                    }),
                    key[:200],
                ),
            )
            created = cur.fetchone()
            if created:
                jobs.append({"job_id": int(created["id"]), "context_key": context["context_key"]})
        cur.execute(
            """
            UPDATE v2_document_versions
            SET status='learning', updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status IN ('parsed', 'learning_failed')
            """,
            (int(version_id),),
        )
    return jobs


def learn_job_context(conn, job: dict) -> tuple[dict, dict]:
    """Reload the version plus the exact context blocks for one learn job."""

    from v2.documents import get_version

    version = get_version(conn, int(job["version_id"]))
    if version is None:
        raise DocumentLearnNotFound("document version for learn job is gone")
    checkpoint = dict(job.get("checkpoint") or {})
    wanted = [int(item) for item in (checkpoint.get("block_ids") or [])]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT b.id, b.block_key, b.section_path, b.page_no, b.slide_no,
                   b.block_type, r.content AS evidence_text
            FROM v2_document_blocks b
            LEFT JOIN v2_raw_evidence r ON r.id=b.raw_evidence_id
            WHERE b.version_id=%s AND b.id = ANY(%s)
            ORDER BY b.ord, b.id
            """,
            (int(version["id"]), wanted or [-1]),
        )
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["evidence_text"] = str(row.get("evidence_text") or "")
    contexts = build_learning_contexts(rows)
    context = contexts[0] if contexts else {
        "context_key": str(checkpoint.get("context_key") or "empty"),
        "title": "", "blocks": [], "approx_chars": 0,
    }
    return version, context


# -- rendering + confirm ----------------------------------------------------------


def render_unit_content(title: str, unit_kind: str, details: dict) -> str:
    """Deterministically render structured details into readable content."""

    lines = [str(title or "").strip()]
    get = details.get
    applicability = get("applicability") if isinstance(get("applicability"), dict) else None
    if applicability:
        parts = []
        for key in ("models", "versions", "regions", "conditions"):
            values = applicability.get(key)
            if values:
                parts.append(f"{key}: {', '.join(str(item) for item in values)}")
        if parts:
            lines.append("适用范围：" + "；".join(parts))
    for label, key in (("前置条件", "prerequisites"), ("警告", "warnings"), ("例外", "exceptions")):
        values = get(key) or []
        if isinstance(values, list) and [item for item in values if str(item).strip()]:
            lines.append(f"{label}：" + "；".join(str(item).strip() for item in values if str(item).strip()))
    if unit_kind == "procedure":
        steps = [str(item).strip() for item in (get("ordered_steps") or []) if str(item).strip()]
        for index, step in enumerate(steps, start=1):
            lines.append(f"{index}. {step}")
        if get("expected_result"):
            lines.append(f"预期结果：{str(get('expected_result')).strip()}")
    elif unit_kind == "rule":
        if get("trigger"):
            lines.append(f"触发条件：{str(get('trigger')).strip()}")
        if get("result"):
            lines.append(f"结论：{str(get('result')).strip()}")
    elif unit_kind == "experience":
        if get("observation"):
            lines.append(f"现象与结果：{str(get('observation')).strip()}")
    return _text("\n".join(line for line in lines if line))


def get_document_proposal(conn, proposal_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.fact_text, p.entity_name, p.proposed_trust,
                   p.status, p.comparison_result, p.comparison_reason,
                   p.related_knowledge_ids, p.unit_kind, p.applicability,
                   p.revision, p.details_json, p.origin_document_version_id,
                   p.confirmed_knowledge_id, p.created_at, p.updated_at,
                   v.document_key, v.version_label, v.title AS version_title,
                   v.source_authenticity
            FROM v2_learning_proposals p
            JOIN v2_document_versions v ON v.id=p.origin_document_version_id
            WHERE p.id=%s AND p.origin_document_version_id IS NOT NULL
            """,
            (int(proposal_id),),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def list_document_proposals(conn, version_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.fact_text, p.entity_name, p.proposed_trust,
                   p.status, p.comparison_result, p.unit_kind, p.applicability,
                   p.revision, p.details_json, p.origin_document_version_id,
                   p.confirmed_knowledge_id, p.created_at, p.updated_at
            FROM v2_learning_proposals p
            WHERE p.origin_document_version_id=%s
            ORDER BY p.id
            """,
            (int(version_id),),
        )
        return [dict(row) for row in cur.fetchall()]


def confirm_document_proposal(
    conn,
    proposal_id: int,
    *,
    content: str | None = None,
    details: dict | None = None,
    applicability: dict | None = None,
) -> tuple[dict, bool]:
    """Explicitly confirm one document unit; idempotent, no model calls.

    The engineer reviews the whole unit (title/content/structure/sources) in
    the Documents UI and confirms the exact text.  Edited details without
    content re-render content deterministically, so the two never disagree.
    Confirmed units start ``validated`` for their own document version and
    become answerable immediately.
    """

    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM v2_learning_proposals WHERE id=%s FOR UPDATE",
            (int(proposal_id),),
        )
        row = cur.fetchone()
    if not row:
        raise DocumentLearnNotFound(f"V2 document proposal {int(proposal_id)} was not found")
    proposal = dict(row)
    if proposal.get("origin_document_version_id") is None:
        raise DocumentLearnError("only document-learning proposals confirm here")
    if proposal.get("status") == "confirmed":
        knowledge = _confirmed_knowledge(conn, int(proposal["confirmed_knowledge_id"]))
        return knowledge, True
    if proposal.get("status") != "pending_confirmation":
        raise DocumentLearnError("only a pending document proposal can be confirmed")

    stored_details = dict(proposal.get("details_json") or {})
    final_details = dict(stored_details)
    if details is not None:
        if not isinstance(details, dict):
            raise DocumentLearnError("details must be an object")
        final_details.update({str(key): details[key] for key in details})
    final_applicability = (
        _clean_applicability(applicability)
        if applicability is not None
        else dict(proposal.get("applicability") or {})
    )
    title = str(final_details.get("title") or "").strip()[:200] or _text(
        str(proposal.get("fact_text") or "").splitlines()[0] if str(proposal.get("fact_text") or "").strip() else "Document unit",
        200,
    )
    final_details["title"] = title
    if content is not None:
        final_content = _text(content)
        if not final_content:
            raise DocumentLearnError("confirmed content must not be empty")
    elif details is not None:
        final_content = render_unit_content(
            title, str(proposal.get("unit_kind") or "fact"), final_details,
        )
    else:
        final_content = _text(proposal.get("fact_text"))
    sources = list(final_details.get("sources") or [])
    if not sources:
        raise DocumentLearnError("confirmed unit keeps at least one block source")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge(
                title, content, entity_name, trust, active, unit_kind,
                applicability, revision, details_json,
                origin_document_version_id, validation_status
            ) VALUES(%s, %s, '', 'user_confirmed', TRUE, %s, %s, 1, %s, %s, 'validated')
            RETURNING id, title, content, entity_name, trust, active,
                      unit_kind, applicability, revision, details_json,
                      origin_document_version_id, validation_status,
                      created_at, updated_at
            """,
            (
                title, final_content, str(proposal.get("unit_kind") or "fact"),
                Jsonb(final_applicability), Jsonb(final_details),
                int(proposal["origin_document_version_id"]),
            ),
        )
        knowledge = dict(cur.fetchone())
    source_kind = (
        "official_document"
        if str(proposal.get("source_authenticity") or "") in ("official_vendor", "confirmed_copy")
        else "other"
    )
    for source in sources:
        _link_block_source(
            conn, int(knowledge["id"]), int(source["block_id"]),
            source_kind, str(source.get("excerpt") or ""),
        )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET status='confirmed', confirmed_knowledge_id=%s,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (int(knowledge["id"]), int(proposal["id"])),
        )
        cur.execute(
            """
            UPDATE v2_document_blocks
            SET processing_state='knowledge',
                state_reason='', updated_at=CURRENT_TIMESTAMP
            WHERE id = ANY(%s) AND processing_state IN ('proposal', 'pending')
            """,
            ([int(item["block_id"]) for item in sources],),
        )
    from v2.service import _write_knowledge_history

    _write_knowledge_history(conn, int(knowledge["id"]), "confirm", {}, {
        "trust": "user_confirmed", "unit_kind": knowledge.get("unit_kind"),
        "revision": 1, "origin_document_version_id": knowledge.get("origin_document_version_id"),
    })
    return knowledge, False


def _link_block_source(conn, knowledge_id: int, block_id: int, source_kind: str, excerpt: str) -> None:
    """Link a Knowledge unit to the raw evidence behind one document block."""

    with conn.cursor() as cur:
        cur.execute(
            "SELECT raw_evidence_id FROM v2_document_blocks WHERE id=%s",
            (int(block_id),),
        )
        block = cur.fetchone()
    if not block or not block.get("raw_evidence_id"):
        raise DocumentLearnError(f"document block {int(block_id)} has no evidence to cite")
    from v2.learning import _link_source

    _link_source(
        conn, int(knowledge_id), int(block["raw_evidence_id"]),
        source_kind=source_kind, relation="supports",
        source_role="supporting", resolution="accepted",
        excerpt=str(excerpt or "")[:4000],
    )


def _confirmed_knowledge(conn, knowledge_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, entity_name, trust, active,
                   unit_kind, applicability, revision, details_json,
                   origin_document_version_id, validation_status,
                   created_at, updated_at
            FROM v2_knowledge WHERE id=%s
            """,
            (int(knowledge_id),),
        )
        row = cur.fetchone()
    if not row:
        raise DocumentLearnNotFound(f"V2 Knowledge {int(knowledge_id)} was not found")
    return dict(row)


__all__ = [
    "EXTRACT_MAX_TOKENS",
    "UNIT_KINDS",
    "V2_DOC_EXTRACT_PROMPT_VERSION",
    "DocumentLearnError",
    "DocumentLearnNotFound",
    "build_extract_messages",
    "build_learning_contexts",
    "confirm_document_proposal",
    "existing_unit_fingerprints",
    "check_units",
    "extract_units",
    "get_document_proposal",
    "learn_job_context",
    "list_document_proposals",
    "queue_learn_jobs",
    "render_unit_content",
    "save_unit_proposals",
    "validate_units",
]
