"""The small, text-only V2 learning loop.

The database stores raw input, a provisional interpretation and one atomic
proposal at a time.  The only operation that changes a Knowledge row to
``user_confirmed`` is an explicit confirmation reply to the assistant's
recap.  This module is intentionally independent from the V1 review code.
"""

from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from typing import Any

from llm import parse_json_response

from v2.bulk import (
    classify_input_mode,
    deduplicate_knowledge,
    extraction_coverage_is_complete,
    parse_batch_confirmation,
    requires_individual_confirmation,
    segment_bulk_text,
)
from v2.compare import compare_and_ask, safe_question
from v2.retrieval import retrieve_learning_knowledge, store_knowledge_embedding
from v2.service import create_thread, get_thread, json_safe, thread_response
from psycopg.types.json import Jsonb


log = logging.getLogger("aihelper.v2.learning")

TRUST_VALUES = frozenset({
    "official_source", "user_confirmed", "provisional", "conflicted",
})
PROPOSAL_STATUSES = frozenset({
    "pending_confirmation", "pending_clarification", "confirmed", "corrected", "skipped",
    "unknown", "rejected", "superseded",
})

CONFIRM_WORDS = frozenset({
    "对", "对的", "正确", "没错", "是的", "确认", "是", "yes", "true", "y",
})
NEGATIVE_WORDS = frozenset({
    "否", "不是", "不对", "不正确", "no", "false", "n", "нет",
})
UNKNOWN_WORDS = frozenset({"不知道", "不清楚", "不确定", "не знаю", "неизвестно"})
SKIP_WORDS = frozenset({"跳过", "以后再说", "稍后再说", "先不说", "skip", "later"})

_BINARY_QUESTION_MARKERS = (
    "是否", "是不是", "能否", "可否", "有无", "有没有", "对吗", "正确吗",
    "да или нет", "верно ли", "является ли", "ли?",
)

_CONJOINED_PREDICATE_RE = re.compile(
    r"^(?P<left>.+?)\s+(?:and|以及|并且|并)\s+"
    r"(?P<right>(?:supports?|has|have|provides?|uses?|includes?|offers?|features?|is|are|支持|具备|采用|提供|包括).+?)"
    r"[.!。！？?]*$",
    re.IGNORECASE,
)
_SEMANTIC_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "it", "is", "are", "has", "have",
    "and", "or", "to", "of", "for", "with", "while", "can", "may", "will", "at",
    "camera", "cameras", "device", "devices", "model", "产品", "设备", "摄像机",
})

UNDERSTANDING_SYSTEM_PROMPT = """
你是一个产品知识学习助理。你正在和产品专家聊天，不是在维护数据库。
请从用户刚刚提供的原文中先识别 evidence claims，再把语义上属于同一个产品知识的 claims 归并为 knowledge_units，返回严格 JSON：
{
  "claims":[{"id":"c1","text":"原文中的连续明确主张"}],
  "knowledge_units":[{
    "title":"简短标题",
    "canonical_fact":"保守、规范化后的一条可引用知识",
    "entity_name":"明确型号或产品名，没有就空字符串",
    "supporting_claim_ids":["c1"],
    "source_excerpt":"覆盖支持 claims 的连续原文片段",
    "derived":false
  }],
  "coverage":{
    "complete":true,
    "claims":[{"id":"c1","text":"原文中的连续明确主张","knowledge_unit_indexes":[0],"disposition":"knowledge"}],
    "uncovered_claims":[]
  }
}

重要概念：evidence claim 是为了忠实记录原文和 provenance 的支持点；knowledge_unit 才是进入知识库、供用户确认的语义知识。多个 claims 可以支持同一个 knowledge_unit。不要把“一个 claim”或“一个句子”机械当成一个 knowledge_unit。

硬规则：
- 先完整识别原文中的 claims，但允许多个同义复述、同一能力的解释/直接效果、连续描述同一 feature 的 claims 归并为一个 knowledge_unit。
- canonical_fact 必须是保守的规范化表达：可删除营销废话、合并同义内容、明确 subject，但不得加入原文没有的信息，不得扩大产品范围，不得丢失条件、版本、地区、firmware、hardware revision、例外或否定边界。
- 每个被映射到 knowledge_unit 的 claim 的技术意义都必须体现在 canonical_fact 中；不能只保留第一句而静默丢弃后续技术说明。只有确实没有独立可验证产品事实的营销性 claim 才可以标记为 non_knowledge。
- 只有不同型号、不同独立参数、不同条件/版本/地区，或明确矛盾的事实才必须拆成不同 knowledge_units。一个句子可能有多个 units，多句话也可能只有一个 unit。
- 能力与其直接解释或结果可以合并。例如“保持全景同时查看局部细节”与“减少变焦/云台导致的盲区”通常是同一技术能力的一个 knowledge_unit；不要为了句号数量拆分。
- 不要因为 subject 相同或句子中出现“and/以及”就合并独立属性。例如“The camera has 8 MP resolution and supports PoE.”必须输出两个 knowledge_units，分别保留分辨率和供电方式；一个句子不等于一个 unit。
- 反例与正例： “The camera has 8 MP resolution and supports PoE.” → 两个 units；“The camera keeps the overall scene visible while zooming. This reduces blind spots.” → 一个 unit，后一句作为同一能力的直接效果。请优先遵循这个区别。
- canonical_fact 不要逐字复制整段输入，也不要把“本段没有说明某参数”这类元评论写成产品事实。只保留可引用的产品结论，通常比原文更短；但不能为变短而丢失技术意义、条件或否定边界。
- 营销性结论（例如“提升安全性”“优秀用户体验”）如果没有独立可验证的产品事实，可作为 claim 记录，但应将 disposition 设为 non_knowledge，不要单独创建 knowledge_unit。
- source_excerpt 必须是输入中的连续原文片段，并覆盖该 unit 的支持 claims；不要伪造引用。supporting_claim_ids 必须引用 claims 中的 id。
- 不要把单型号推广到系列，不要根据型号命名或行业常识推断范围。
- 手册或文字没有提到某功能，不等于不支持；只有明确写出不支持或用户明确确认，才可表达否定。
- “大概”“可能”“应该”等不确定说法仍然只能作为待确认理解。
- 原文中的“等 / etc. / и т.д.”表示当前列举非穷尽；保留这个含义，不要要求补齐列表。
- 只提取原文明确表达的事实。不要为了“补完整”而询问数据规模、RAG、蒸馏、其他模型或任何原文没有提出的维度。
- 必须处理输入中的每一个逻辑段和每一个编号条目；不要因为事实很多而省略、截断或静默丢弃。可以把同一主题的 claims 合并，但 coverage 必须将每个 claim 映射到一个或多个 knowledge_unit，或明确标记为 non_knowledge。
- 如果输入是编号列表，保留每一项的名称、类型、模型大小/类型、状态和条件（若原文有写）。
- derived 只有在你确实做了原文之外的推断时才为 true；原文的忠实归并或改写必须为 false。
- 不要回答客户问题，不要使用训练知识，不要询问数据库字段或技术实现。
- 只返回 JSON，不要 markdown，不要解释。
""".strip()


def _clean(value: Any, limit: int = 12000) -> str:
    return str(value or "").strip()[:limit]


def classify_reply(content: str) -> str:
    """Classify exact short controls; longer text remains product evidence."""

    normalized = re.sub(r"[\s。.!！?？,，]+", "", _clean(content, 200).casefold())
    if normalized in {item.casefold() for item in CONFIRM_WORDS}:
        return "confirm"
    if normalized in {item.casefold() for item in NEGATIVE_WORDS}:
        return "negative"
    if normalized in {item.casefold() for item in UNKNOWN_WORDS}:
        return "unknown"
    if normalized in {item.casefold() for item in SKIP_WORDS}:
        return "skip"
    return "correction"


def _expected_answer_type(question: str | None) -> str:
    text = str(question or "").strip().casefold()
    return "binary" if any(marker in text for marker in _BINARY_QUESTION_MARKERS) else "free_text"


def _entity_name(text: str) -> str:
    # Keep this deliberately conservative.  An identifier is context, not a
    # license to infer a family or a broader product scope.
    match = re.search(r"(?=[A-Za-z0-9./()\-]*\d)[A-Za-z0-9][A-Za-z0-9./()\-]{2,}", text)
    return match.group(0).upper() if match else ""


def _fallback_facts(content: str) -> list[dict]:
    return [{
        "title": "待确认的产品信息",
        "content": content,
        "entity_name": _entity_name(content),
    }]


def _source_span(source: str, excerpts: list[str]) -> str:
    """Return the smallest source substring covering valid supporting quotes."""

    positions = []
    for excerpt in excerpts:
        value = _clean(excerpt, 12000)
        if not value:
            continue
        start = source.find(value)
        if start < 0:
            return ""
        positions.append((start, start + len(value)))
    if not positions:
        return ""
    return source[min(start for start, _ in positions):max(end for _, end in positions)]


def _structured_knowledge_units(parsed: dict, source: str) -> list[dict] | None:
    """Parse the semantic extraction contract without inventing provenance."""

    raw_units = parsed.get("knowledge_units")
    raw_claims = parsed.get("claims")
    if not isinstance(raw_units, list) or not isinstance(raw_claims, list):
        return None
    known_claim_ids = set()
    for claim in raw_claims:
        if not isinstance(claim, dict):
            return None
        claim_id = _clean(claim.get("id"), 100)
        claim_text = _clean(claim.get("text") or claim.get("source_excerpt"), 12000)
        if not claim_id or not claim_text or claim_text not in source or claim_id in known_claim_ids:
            return None
        known_claim_ids.add(claim_id)
    result = []
    for item in raw_units:
        if not isinstance(item, dict):
            return None
        content = _clean(item.get("canonical_fact") or item.get("content") or item.get("fact"))
        if not content:
            return None
        supplied_excerpt = _clean(item.get("source_excerpt"), 12000)
        if supplied_excerpt and supplied_excerpt in source:
            source_excerpt = supplied_excerpt
        else:
            supporting = item.get("supporting_claims") or item.get("supporting_points") or []
            if isinstance(supporting, str):
                supporting = [supporting]
            if not isinstance(supporting, list):
                return None
            source_excerpt = _source_span(source, [str(value) for value in supporting])
        if not source_excerpt:
            return None
        supporting_claim_ids = item.get("supporting_claim_ids") or []
        if isinstance(supporting_claim_ids, str):
            supporting_claim_ids = [supporting_claim_ids]
        if (
            not isinstance(supporting_claim_ids, list)
            or not supporting_claim_ids
            or not all(isinstance(value, str) and value.strip() for value in supporting_claim_ids)
            or not set(supporting_claim_ids).issubset(known_claim_ids)
        ):
            return None
        result.append({
            "title": _clean(item.get("title"), 500) or content[:120],
            "content": content,
            "entity_name": _clean(item.get("entity_name") or item.get("subject"), 500),
            "derived": bool(item.get("derived") or item.get("inferred")),
            "source_excerpt": source_excerpt,
            "supporting_claim_ids": [value.strip() for value in supporting_claim_ids],
            "supporting_points": [str(value) for value in (item.get("supporting_points") or []) if str(value).strip()],
        })
    return result


def _structured_coverage_matches_claims(parsed: dict) -> bool:
    """Require every declared evidence claim to appear in the coverage map."""

    claims = parsed.get("claims")
    coverage = parsed.get("coverage")
    coverage_claims = coverage.get("claims") if isinstance(coverage, dict) else None
    if not isinstance(claims, list) or not isinstance(coverage_claims, list):
        return False
    claim_ids = {_clean(item.get("id"), 100) for item in claims if isinstance(item, dict)}
    covered_ids = {_clean(item.get("id"), 100) for item in coverage_claims if isinstance(item, dict)}
    return bool(claim_ids) and claim_ids == covered_ids


def _meaningful_tokens(value: str, entity_name: str = "") -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9./()_-]*|[\u4e00-\u9fff]", value)
    }
    entity_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9./()_-]*|[\u4e00-\u9fff]", entity_name)
    }
    return {token for token in tokens if token not in _SEMANTIC_STOPWORDS and token not in entity_tokens}


def _same_entity(left: dict, right: dict) -> bool:
    left_entity = _clean(left.get("entity_name"), 500).casefold()
    right_entity = _clean(right.get("entity_name"), 500).casefold()
    return bool(left_entity and left_entity == right_entity)


def _related_semantic_units(left: dict, right: dict) -> bool:
    """Recognize only high-confidence same-feature continuations.

    This is deliberately lexical and local: it prevents a mechanism sentence
    and its direct effect from becoming two confirmations, while unrelated
    attributes such as resolution and PoE have no meaningful overlap.
    """

    if not _same_entity(left, right):
        return False
    left_tokens = _meaningful_tokens(left.get("content", ""), left.get("entity_name", ""))
    right_tokens = _meaningful_tokens(right.get("content", ""), right.get("entity_name", ""))
    left_numbers = set(re.findall(r"\d+(?:\.\d+)?", left.get("content", "")))
    right_numbers = set(re.findall(r"\d+(?:\.\d+)?", right.get("content", "")))
    if left_numbers and right_numbers and left_numbers.isdisjoint(right_numbers):
        return False
    shared = left_tokens & right_tokens
    smaller = min(len(left_tokens), len(right_tokens))
    return len(shared) >= 2 and smaller > 0 and len(shared) / smaller >= 0.25


def _consolidate_related_units(facts: list[dict]) -> list[dict]:
    """Merge adjacent same-entity units only when their wording overlaps."""

    if len(facts) < 2:
        return facts
    result: list[dict] = []
    for fact in facts:
        if result and _related_semantic_units(result[-1], fact):
            previous = result[-1]
            previous["content"] = f"{previous['content'].rstrip('.。')}。{fact['content'].lstrip()}"
            previous["supporting_claim_ids"] = list(dict.fromkeys(
                [*previous.get("supporting_claim_ids", []), *fact.get("supporting_claim_ids", [])]
            ))
            previous["supporting_points"] = list(dict.fromkeys(
                [*previous.get("supporting_points", []), *fact.get("supporting_points", [])]
            ))
            previous_excerpt = previous.get("source_excerpt", "")
            current_excerpt = fact.get("source_excerpt", "")
            if previous_excerpt and current_excerpt:
                previous["source_excerpt"] = _source_span(
                    previous_excerpt + "\n" + current_excerpt,
                    [previous_excerpt, current_excerpt],
                ) or f"{previous_excerpt}\n{current_excerpt}"
            continue
        result.append(dict(fact))
    return result


def _split_obvious_conjoined_unit(fact: dict) -> list[dict]:
    """Split two explicit predicates in one sentence, without product rules."""

    # The source excerpt may cover several units.  Split only the unit the
    # model actually proposed; otherwise an excerpt shared by two already
    # separate units would be duplicated again.
    source = _clean(fact.get("content"), 12000)
    match = _CONJOINED_PREDICATE_RE.match(source)
    if not match:
        return [fact]
    left = match.group("left").strip()
    right = match.group("right").strip()
    predicate = re.search(
        r"\b(?:has|have|supports?|provides?|uses?|includes?|offers?|features?|is|are)\b|支持|具备|采用|提供|包括",
        left,
        re.IGNORECASE,
    )
    if not predicate:
        return [fact]
    subject = left[:predicate.start()].strip()
    right_content = f"{subject} {right}".strip() if subject else right
    claim_ids = list(fact.get("supporting_claim_ids", []))
    return [
        dict(fact, content=left.rstrip(".。") + ".", supporting_claim_ids=claim_ids),
        dict(fact, content=right_content.rstrip(".。") + ".", supporting_claim_ids=claim_ids),
    ]


def _postprocess_semantic_units(facts: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for fact in facts:
        expanded.extend(_split_obvious_conjoined_unit(fact))
    return _consolidate_related_units(expanded)


def _model_facts(
    content: str,
    llm_service=None,
    context: list[dict] | None = None,
    *,
    max_tokens: int = 1600,
    require_coverage: bool = False,
) -> tuple[list[dict], bool]:
    if llm_service is None:
        return _fallback_facts(content), True
    messages = [{"role": "system", "content": UNDERSTANDING_SYSTEM_PROMPT}]
    if context:
        context_lines = []
        for item in context:
            context_lines.append("待处理事实：" + _clean(item.get("fact_text")))
            if item.get("clarification_question"):
                context_lines.append("AI 已问：" + _clean(item.get("clarification_question")))
        messages.append({
            "role": "user",
            "content": "此前对话上下文：\n" + "\n".join(context_lines),
        })
    messages.append({"role": "user", "content": content})
    try:
        parsed = parse_json_response(llm_service.extract(messages, max_tokens=max_tokens))
        if isinstance(parsed, dict) and "knowledge_units" in parsed:
            facts = _structured_knowledge_units(parsed, content)
            if (
                facts is None
                or not _structured_coverage_matches_claims(parsed)
                or not extraction_coverage_is_complete(
                    content,
                    facts,
                    parsed.get("coverage"),
                )
            ):
                log.warning("V2 semantic extraction contract invalid; keeping a provisional raw interpretation")
                return _fallback_facts(content), True
            return _postprocess_semantic_units(facts), False
        raw_facts = parsed.get("facts") if isinstance(parsed, dict) else None
        if not isinstance(raw_facts, list):
            raw_facts = [parsed] if isinstance(parsed, dict) and parsed.get("content") else []
        facts = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_content = _clean(item.get("content") or item.get("fact") or item.get("text"))
            if not fact_content:
                continue
            facts.append({
                "title": _clean(item.get("title"), 500) or fact_content[:120],
                "content": fact_content,
                "entity_name": _clean(item.get("entity_name") or item.get("entity"), 500),
                "derived": bool(item.get("derived") or item.get("inferred")),
                "source_excerpt": _clean(item.get("source_excerpt"), 12000),
            })
        if facts:
            if require_coverage and not extraction_coverage_is_complete(
                content,
                raw_facts,
                parsed.get("coverage") if isinstance(parsed, dict) else None,
            ):
                log.warning("V2 bulk extraction coverage incomplete; segment remains failed")
                return _fallback_facts(content), True
            # Legacy ``facts`` responses already carry the model's intended
            # unit boundaries.  Only repair a single legacy unit; merging an
            # existing multi-fact response would undo its explicit split.
            return (_postprocess_semantic_units(facts) if len(facts) == 1 else facts), False
    except Exception:
        log.exception("V2 learning understanding failed; keeping a provisional raw interpretation")
    return _fallback_facts(content), True


def _insert_evidence(
    conn,
    content: str,
    thread_id: int,
    *,
    channel: str = "inbox",
    label: str = "Inbox",
    input_mode: str = "new_knowledge_payload",
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_raw_evidence(
                evidence_type, author_role, content, raw_payload,
                source_label, source_locator
            )
            VALUES('user_input', 'product_expert', %s, %s, %s, %s)
            RETURNING id, evidence_type, author_role, content, raw_payload,
                      source_label, source_locator, created_at
            """,
            (
                content,
                Jsonb({"thread_id": thread_id, "channel": channel, "input_mode": input_mode}),
                label,
                f"v2-thread:{thread_id}",
            ),
        )
        return dict(cur.fetchone())


def _insert_message(
    conn,
    thread_id: int,
    role: str,
    message_type: str,
    content: str,
    raw_evidence_id: int | None = None,
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_inbox_messages(
                thread_id, sequence_no, role, message_type, content, raw_evidence_id
            )
            VALUES(
                %s,
                (SELECT COALESCE(MAX(sequence_no), 0) + 1
                 FROM v2_inbox_messages WHERE thread_id=%s),
                %s, %s, %s, %s
            )
            RETURNING id, thread_id, sequence_no, role, message_type, content,
                      raw_evidence_id, created_at
            """,
            (thread_id, thread_id, role, message_type, content, raw_evidence_id),
        )
        row = dict(cur.fetchone())
        cur.execute(
            "UPDATE v2_inbox_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (thread_id,),
        )
    return row


def _ensure_session(
    conn,
    thread_id: int,
    question_budget: int,
    *,
    session_type: str = "active_inbox",
) -> dict:
    budget = max(0, int(question_budget))
    if session_type not in {"passive", "active_inbox", "active_grill"}:
        raise ValueError(f"unknown V2 learning session type: {session_type}")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, session_type, status, question_budget,
                   questions_asked, summary, started_at, completed_at,
                   created_at, updated_at
            FROM v2_learning_sessions
            WHERE thread_id=%s AND status='active'
            ORDER BY id DESC LIMIT 1
            """,
            (thread_id,),
        )
        current = cur.fetchone()
        if current:
            return dict(current)
        cur.execute(
            """
            INSERT INTO v2_learning_sessions(
                thread_id, session_type, status, question_budget
            )
            VALUES(%s, %s, 'active', %s)
            RETURNING id, thread_id, session_type, status, question_budget,
                      questions_asked, summary, started_at, completed_at,
                      created_at, updated_at
            """,
            (thread_id, session_type, budget),
        )
        return dict(cur.fetchone())


def _pending_proposal(conn, thread_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, source_message_id, question_message_id,
                   fact_text, entity_name, proposed_trust, status,
                   confirmed_knowledge_id, resolution_message_id,
                   comparison_result, clarification_question,
                   comparison_reason, related_knowledge_ids,
                   batch_id, segment_no, expected_answer_type, derived, individual_confirmation, paused,
                   created_at, updated_at
            FROM v2_learning_proposals
            WHERE thread_id=%s
              AND status IN ('pending_clarification', 'pending_confirmation')
              AND paused=FALSE
            ORDER BY id LIMIT 1
            """,
            (thread_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _pending_context(conn, thread_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fact_text, clarification_question, comparison_result
            FROM v2_learning_proposals
            WHERE thread_id=%s
              AND status IN ('pending_clarification', 'pending_confirmation')
              AND paused=FALSE
            ORDER BY id
            """,
            (thread_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def _lock_thread(conn, thread_id: int) -> None:
    """Serialize state transitions and message sequence allocation per thread."""

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM v2_inbox_threads WHERE id=%s FOR UPDATE",
            (thread_id,),
        )
        if cur.fetchone() is None:
            raise ValueError(f"V2 thread {thread_id} disappeared during the learning turn")


def _create_knowledge(conn, fact: dict, *, trust: str = "provisional") -> dict:
    if trust not in TRUST_VALUES:
        raise ValueError("unknown V2 trust value")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge(title, content, entity_name, trust, active)
            VALUES(%s, %s, %s, %s, %s)
            RETURNING id, title, content, entity_name, trust, active,
                      created_at, updated_at
            """,
            (
                fact["title"], fact["content"], fact.get("entity_name") or "",
                trust, trust != "conflicted",
            ),
        )
        return dict(cur.fetchone())


def _link_source(
    conn,
    knowledge_id: int,
    evidence_id: int,
    *,
    source_kind: str,
    relation: str = "supports",
    source_role: str = "supporting",
    resolution: str = "accepted",
    excerpt: str = "",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge_sources(
                knowledge_id, raw_evidence_id, source_kind, relation,
                source_role, excerpt, active, resolution
            )
            VALUES(%s, %s, %s, %s, %s, %s, TRUE, %s)
            ON CONFLICT (knowledge_id, raw_evidence_id, relation) DO NOTHING
            """,
            (knowledge_id, evidence_id, source_kind, relation, source_role, excerpt[:4000], resolution),
        )


def _insert_proposal(
    conn,
    thread_id: int,
    message_id: int,
    evidence_id: int,
    fact: dict,
    knowledge_id: int | None,
    *,
    decision: str = "NEW",
    status: str = "pending_confirmation",
    clarification_question: str = "",
    comparison_reason: str = "",
    related_knowledge_ids: list[int] | None = None,
    batch_id: int | None = None,
    segment_no: int | None = None,
    derived: bool = False,
    individual_confirmation: bool = False,
) -> dict:
    if decision not in {"NEW", "CONFIRM", "ENRICH", "CONFLICT", "UNCLEAR"}:
        raise ValueError(f"unknown V2 comparison result: {decision}")
    if status not in {"pending_confirmation", "pending_clarification"}:
        raise ValueError(f"invalid new proposal status: {status}")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_learning_proposals(
                thread_id, source_message_id, fact_text, entity_name,
                proposed_trust, status, confirmed_knowledge_id,
                comparison_result, clarification_question,
                comparison_reason, related_knowledge_ids, batch_id,
                segment_no, expected_answer_type, derived, individual_confirmation
            )
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, thread_id, source_message_id, question_message_id,
                      fact_text, entity_name, proposed_trust, status,
                      confirmed_knowledge_id, resolution_message_id,
                      comparison_result, clarification_question,
                      comparison_reason, related_knowledge_ids,
                      batch_id, segment_no, expected_answer_type, derived, individual_confirmation, paused,
                      created_at, updated_at
            """,
            (
                thread_id,
                message_id,
                fact["content"],
                fact.get("entity_name") or "",
                "conflicted" if decision == "CONFLICT" else "provisional",
                status,
                knowledge_id,
                decision,
                clarification_question,
                comparison_reason,
                list(related_knowledge_ids or []),
                batch_id,
                segment_no,
                _expected_answer_type(clarification_question),
                bool(derived),
                bool(individual_confirmation),
            ),
        )
        proposal = dict(cur.fetchone())
    if knowledge_id is not None:
        _link_source(
            conn,
            knowledge_id,
            evidence_id,
            source_kind="user_input",
            source_role="primary",
            resolution="unresolved",
            excerpt=_clean(fact.get("source_excerpt")) or fact["content"],
        )
    return proposal


def _deduplicate_facts(facts: list[dict]) -> list[dict]:
    """Keep one proposal per normalized fact in a single model response."""

    result = []
    seen = set()
    for fact in facts:
        content = _clean(fact.get("content"))
        entity_name = _clean(fact.get("entity_name"), 500)
        if not content:
            continue
        key = (content.casefold(), entity_name.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "title": _clean(fact.get("title"), 500) or content[:120],
            "content": content,
            "entity_name": entity_name,
            "derived": bool(fact.get("derived")),
            "individual_confirmation": bool(fact.get("individual_confirmation")),
            "source_excerpt": _clean(fact.get("source_excerpt"), 12000),
        })
    return result


def _session_question_count(conn, session_id: int) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT questions_asked, question_budget FROM v2_learning_sessions WHERE id=%s", (session_id,))
        row = cur.fetchone()
    return int(row["questions_asked"]), int(row["question_budget"])


def _ask(conn, proposal: dict, session: dict) -> tuple[dict | None, str | None]:
    if session.get("_unlimited_questions"):
        asked, budget = 0, None
    else:
        asked, budget = _session_question_count(conn, int(session["id"]))
    if budget is not None and asked >= budget:
        return None, None
    if proposal.get("status") == "pending_clarification":
        text = _clean(proposal.get("clarification_question"), 2000)
        if not text:
            text = safe_question({"entity_name": proposal.get("entity_name", "")})
    else:
        fact_text = re.sub(r"[。.!！?？]+$", "", str(proposal["fact_text"] or "").strip())
        text = f"我理解为：{fact_text}。对吗？"
    message_type = (
        "clarification"
        if proposal.get("status") == "pending_clarification"
        else "question"
    )
    # A failed compare or a correction can produce the same safe question as
    # an earlier proposal. Reuse the existing message instead of asking it
    # again and incrementing the question counter.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, sequence_no, role, message_type, content,
                   raw_evidence_id, created_at
            FROM v2_inbox_messages
            WHERE thread_id=%s AND role='assistant'
              AND message_type IN ('question', 'clarification', 'batch_confirmation')
              AND content=%s
            ORDER BY id DESC LIMIT 1
            """,
            (proposal["thread_id"], text),
        )
        previous = cur.fetchone()
    if previous:
        message = dict(previous)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE v2_learning_proposals SET question_message_id=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
                (message["id"], proposal["id"]),
            )
        return message, text
    message = _insert_message(
        conn, int(proposal["thread_id"]), "assistant", message_type, text
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET question_message_id=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (message["id"], proposal["id"]),
        )
        cur.execute(
            """
            UPDATE v2_learning_sessions
            SET questions_asked=questions_asked + 1, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (session["id"],),
        )
    return message, text


def _update_proposal(
    conn,
    proposal_id: int,
    status: str,
    *,
    knowledge_id: int | None = None,
    message_id: int | None = None,
) -> dict:
    if status not in PROPOSAL_STATUSES:
        raise ValueError(f"unknown V2 proposal status: {status}")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET status=%s, confirmed_knowledge_id=COALESCE(%s, confirmed_knowledge_id),
                resolution_message_id=COALESCE(%s, resolution_message_id),
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
              AND status IN ('pending_clarification', 'pending_confirmation')
            RETURNING id, thread_id, source_message_id, question_message_id,
                      fact_text, entity_name, proposed_trust, status,
                      confirmed_knowledge_id, resolution_message_id,
                      comparison_result, clarification_question,
                      comparison_reason, related_knowledge_ids,
                      created_at, updated_at
            """,
            (status, knowledge_id, message_id, proposal_id),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"V2 proposal {proposal_id} is no longer pending")
    return dict(row)


def _organization_review_context(conn, knowledge: dict) -> dict:
    """Add only this Knowledge item's accepted source excerpts.

    Raw evidence can contain several claims and can be linked to several
    Knowledge rows.  Its full content therefore cannot be used as evidence for
    this review.  The Knowledge content and accepted supporting excerpts are
    the complete evidence scope; unresolved, rejected, contextual, and empty
    source links stay out of the organization review.
    """

    context = dict(knowledge)
    knowledge_id = knowledge.get("id")
    if knowledge_id is None:
        return context
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.excerpt
            FROM v2_knowledge_sources s
            WHERE s.knowledge_id=%s
              AND s.active=TRUE
              AND s.relation='supports'
              AND s.resolution='accepted'
              AND char_length(btrim(s.excerpt)) > 0
            ORDER BY s.id
            LIMIT 8
            """,
            (int(knowledge_id),),
        )
        source_excerpts = [str(row["excerpt"] or "") for row in cur.fetchall()]
    if source_excerpts:
        # Keep Knowledge.content as the atomic fact.  organization.py uses
        # these explicitly scoped excerpts for its bounded review without
        # treating the source as a replacement or expansion of the fact.
        context["accepted_source_excerpts"] = source_excerpts
    return context


def _run_local_organization_review(conn, knowledge: dict, *, llm_service=None) -> dict:
    """Best-effort organization after a fact is durably confirmed.

    Organization is deliberately isolated behind a savepoint.  A malformed
    relation, missing additive schema, or any other organization failure must
    not turn a successful Knowledge confirmation into a failed learning turn.
    """

    try:
        from v2.organization import review_local_organization

        transaction = conn.transaction() if callable(getattr(conn, "transaction", None)) else nullcontext()
        with transaction:
            context = _organization_review_context(conn, knowledge)
            return review_local_organization(
                conn,
                context,
                llm_service=llm_service,
            )
    except Exception:
        log.exception(
            "V2 local organization review failed knowledge_id=%s",
            knowledge.get("id"),
        )
        return {
            "action": "UNCLEAR",
            "entity": None,
            "relations": [],
            "reason": "organization review failed; Knowledge was retained",
        }


def _confirm(
    conn,
    proposal: dict,
    evidence: dict,
    message: dict,
    *,
    embedding_client=None,
    llm_service=None,
) -> dict:
    if proposal.get("status") not in (None, "pending_confirmation"):
        raise ValueError("only a pending V2 proposal can be confirmed")
    knowledge_id = proposal.get("confirmed_knowledge_id")
    if not knowledge_id:
        raise ValueError("pending proposal has no provisional Knowledge")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET trust=CASE WHEN trust='official_source'
                           THEN 'official_source'
                           ELSE 'user_confirmed' END,
                active=TRUE, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND active=TRUE
              AND trust IN ('official_source', 'user_confirmed', 'provisional')
            RETURNING id, title, content, entity_name, trust, active,
                      entity_id, created_at, updated_at
            """,
            (knowledge_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError("the Knowledge attached to this proposal is inactive or conflicted")
    knowledge = dict(row)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge_sources s
            SET resolution='accepted'
            FROM v2_inbox_messages m
            WHERE m.id=%s AND s.knowledge_id=%s
              AND s.raw_evidence_id=m.raw_evidence_id
              AND s.active=TRUE AND s.relation='supports'
            """,
            (proposal.get("source_message_id"), knowledge_id),
        )
    _link_source(
        conn,
        int(knowledge_id),
        int(evidence["id"]),
        source_kind="user_confirmation",
        source_role="primary",
        excerpt=evidence["content"],
    )
    _update_proposal(conn, int(proposal["id"]), "confirmed", knowledge_id=int(knowledge_id), message_id=int(message["id"]))
    store_knowledge_embedding(
        conn,
        int(knowledge_id),
        " ".join(filter(None, (knowledge.get("entity_name"), knowledge.get("title"), knowledge.get("content")))),
        embedder=embedding_client,
    )
    _run_local_organization_review(conn, knowledge, llm_service=llm_service)
    return knowledge


def _retire_corrected_knowledge(conn, proposal: dict) -> None:
    """Reject this proposal's source and retire unsupported provisional text."""

    knowledge_id = proposal.get("confirmed_knowledge_id")
    if not knowledge_id:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT raw_evidence_id
            FROM v2_inbox_messages
            WHERE id=%s
            """,
            (proposal.get("source_message_id"),),
        )
        source_message = cur.fetchone()
        if source_message is None or source_message["raw_evidence_id"] is None:
            return
        cur.execute(
            """
            UPDATE v2_knowledge_sources
            SET active=FALSE, resolution='rejected'
            WHERE knowledge_id=%s AND raw_evidence_id=%s AND active=TRUE
            """,
            (knowledge_id, source_message["raw_evidence_id"]),
        )
        cur.execute(
            """
            UPDATE v2_knowledge k
            SET active=FALSE, updated_at=CURRENT_TIMESTAMP
            WHERE k.id=%s AND k.active=TRUE AND k.trust='provisional'
              AND NOT EXISTS (
                SELECT 1 FROM v2_knowledge_sources s
                WHERE s.knowledge_id=k.id AND s.active=TRUE
                  AND s.relation='supports'
              )
            """,
            (knowledge_id,),
        )


def _supersede_proposal_source(conn, proposal: dict) -> None:
    """Close the provisional source behind a clarified conflict without deleting it."""

    knowledge_id = proposal.get("confirmed_knowledge_id")
    if not knowledge_id:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge_sources s
            SET active=FALSE, resolution='superseded'
            FROM v2_inbox_messages m
            WHERE m.id=%s AND s.knowledge_id=%s
              AND s.raw_evidence_id=m.raw_evidence_id
              AND s.active=TRUE AND s.resolution='unresolved'
            """,
            (proposal.get("source_message_id"), knowledge_id),
        )


def _plan_fact(
    conn,
    *,
    thread_id: int,
    message_id: int,
    evidence_id: int,
    fact: dict,
    llm_service,
    embedding_client,
    batch_id: int | None = None,
    segment_no: int | None = None,
) -> dict:
    """Retrieve, compare, and persist one atomic next step."""

    candidates = retrieve_learning_knowledge(
        conn,
        " ".join(filter(None, (fact.get("entity_name"), fact.get("content")))),
        embedder=embedding_client,
        top_k=5,
        same_model_only=True,
    )
    comparison = compare_and_ask(fact, candidates, llm_service)
    decision = comparison["decision"]
    related_ids = (
        [int(comparison["knowledge_id"])]
        if comparison.get("knowledge_id") is not None
        else []
    )

    if decision == "CONFIRM":
        knowledge_id = related_ids[0]
        status = "pending_confirmation"
    elif decision == "NEW" or (decision == "ENRICH" and not comparison.get("question")):
        knowledge = _create_knowledge(conn, fact, trust="provisional")
        knowledge_id = int(knowledge["id"])
        status = "pending_confirmation"
    elif decision == "CONFLICT":
        knowledge = _create_knowledge(conn, fact, trust="conflicted")
        knowledge_id = int(knowledge["id"])
        status = "pending_clarification"
    else:
        # Ambiguous ENRICH and UNCLEAR stay as raw evidence until the expert has supplied
        # the missing scope/version/condition. This prevents provisional text
        # from being merged into trusted Knowledge before confirmation.
        knowledge_id = None
        status = "pending_clarification"

    return _insert_proposal(
        conn,
        thread_id,
        message_id,
        evidence_id,
        fact,
        knowledge_id,
        decision=decision,
        status=status,
        clarification_question=comparison.get("question") or "",
        comparison_reason=comparison.get("reason") or "",
        related_knowledge_ids=related_ids,
        batch_id=batch_id,
        segment_no=segment_no,
        derived=bool(fact.get("derived")),
        individual_confirmation=bool(fact.get("individual_confirmation")),
    )


def _create_batch(conn, thread_id: int, evidence_id: int, raw_source: str, total_segments: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_learning_batches(
                thread_id, raw_evidence_id, raw_source, total_segments
            )
            VALUES(%s, %s, %s, %s)
            RETURNING id, thread_id, raw_evidence_id, raw_source,
                      total_segments, processed_segments, failed_segments,
                      clear_facts, unclear_items, conflicts, status,
                      confirmation_message_id, created_at, updated_at
            """,
            (thread_id, evidence_id, raw_source, total_segments),
        )
        return dict(cur.fetchone())


def _get_batch(conn, batch_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, raw_evidence_id, raw_source,
                   total_segments, processed_segments, failed_segments,
                   clear_facts, unclear_items, conflicts, status,
                   confirmation_message_id, created_at, updated_at
            FROM v2_learning_batches WHERE id=%s
            """,
            (batch_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _update_batch(
    conn,
    batch_id: int,
    *,
    processed_segments: int,
    failed_segments: int,
    clear_facts: list[dict],
    unclear_items: list[dict],
    conflicts: list[dict],
    status: str | None = None,
    confirmation_message_id: int | None = None,
) -> dict:
    if status is None:
        if clear_facts:
            status = "awaiting_confirmation"
        elif unclear_items:
            status = "awaiting_clarification"
        elif failed_segments:
            status = "partial"
        else:
            status = "completed"
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_batches
            SET processed_segments=%s, failed_segments=%s,
                clear_facts=%s, unclear_items=%s, conflicts=%s,
                status=%s, confirmation_message_id=COALESCE(%s, confirmation_message_id),
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            RETURNING id, thread_id, raw_evidence_id, raw_source,
                      total_segments, processed_segments, failed_segments,
                      clear_facts, unclear_items, conflicts, status,
                      confirmation_message_id, created_at, updated_at
            """,
            (
                processed_segments,
                failed_segments,
                Jsonb(clear_facts),
                Jsonb(unclear_items),
                Jsonb(conflicts),
                status,
                confirmation_message_id,
                batch_id,
            ),
        )
        return dict(cur.fetchone())


def _pending_batch(conn, thread_id: int) -> dict | None:
    if not hasattr(conn, "cursor"):
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT b.id, b.thread_id, b.raw_evidence_id, b.raw_source,
                   b.total_segments, b.processed_segments, b.failed_segments,
                   b.clear_facts, b.unclear_items, b.conflicts, b.status,
                   b.confirmation_message_id, b.created_at, b.updated_at
            FROM v2_learning_batches b
            WHERE b.thread_id=%s AND b.status IN (
                'processing', 'awaiting_confirmation', 'awaiting_clarification', 'partial'
            )
              AND EXISTS (
                  SELECT 1 FROM v2_learning_proposals p
                  WHERE p.batch_id=b.id AND p.paused=FALSE
                    AND p.status IN ('pending_confirmation', 'pending_clarification')
              )
            ORDER BY b.id LIMIT 1
            """,
            (thread_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _pause_pending_proposals(conn, thread_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET paused=TRUE, updated_at=CURRENT_TIMESTAMP
            WHERE thread_id=%s AND paused=FALSE
              AND status IN ('pending_clarification', 'pending_confirmation')
            """,
            (thread_id,),
        )


def _resume_paused_proposals(conn, thread_id: int) -> None:
    """Resume an older question only after all newer batch work is settled."""

    if not hasattr(conn, "cursor"):
        return
    if _pending_batch(conn, thread_id):
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET paused=FALSE, updated_at=CURRENT_TIMESTAMP
            WHERE thread_id=%s AND paused=TRUE
              AND status IN ('pending_clarification', 'pending_confirmation')
            """,
            (thread_id,),
        )


def _batch_confirmation_text(conn, batch: dict) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS count
            FROM v2_learning_proposals
            WHERE batch_id=%s AND paused=FALSE AND status='pending_confirmation'
              AND derived=FALSE AND individual_confirmation=FALSE
            """,
            (batch["id"],),
        )
        clear_count = int(cur.fetchone()["count"])
        cur.execute(
            """
            SELECT count(*) AS count
            FROM v2_learning_proposals
            WHERE batch_id=%s AND paused=FALSE
              AND (status='pending_clarification' OR (status='pending_confirmation' AND (derived=TRUE OR individual_confirmation=TRUE)))
            """,
            (batch["id"],),
        )
        unclear_count = int(cur.fetchone()["count"])
    total = int(batch["total_segments"])
    processed = int(batch["processed_segments"])
    failed = int(batch["failed_segments"])
    lines = [f"我收到这批资料，共 {total} 个逻辑条目。", f"- {processed}/{total} 项已处理。"]
    if failed:
        lines.append(f"- {failed} 项解析失败，我没有假装它们已经理解；请补发失败条目。")
    if clear_count:
        lines.extend([
            f"我提取了 {clear_count} 条明确知识。它们都来自你刚才提供的资料，没有额外推断。",
            "是否确认将这些明确事实一起保存？（也可以回复“确认第1、2项”进行部分确认。）",
        ])
    if unclear_count:
        lines.append(f"另有 {unclear_count} 项需要单独确认或澄清，确认明确内容后我会逐项处理。")
    return "\n".join(lines)


def _batch_confirmation_message(conn, batch: dict) -> tuple[dict, str]:
    text = _batch_confirmation_text(conn, batch)
    if batch.get("confirmation_message_id"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, thread_id, sequence_no, role, message_type, content, raw_evidence_id, created_at FROM v2_inbox_messages WHERE id=%s",
                (batch["confirmation_message_id"],),
            )
            previous = cur.fetchone()
        if previous:
            return dict(previous), previous["content"]
    message = _insert_message(conn, int(batch["thread_id"]), "assistant", "batch_confirmation", text)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE v2_learning_batches SET confirmation_message_id=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (message["id"], batch["id"]),
        )
    return message, text


def _batch_detail_message(conn, batch: dict) -> tuple[dict, str]:
    """Render a readable detail view without introducing another UI workflow."""

    lines = ["这批资料的明细："]
    for item in batch.get("clear_facts") or []:
        lines.append(f"- 第{item.get('segment_no')}项：{item.get('content')}")
    for item in batch.get("unclear_items") or []:
        suffix = "（需要澄清）"
        lines.append(f"- 第{item.get('segment_no')}项：{item.get('content')}{suffix}")
    if not (batch.get("clear_facts") or batch.get("unclear_items")):
        lines.append("- 暂无可展示的已提取事实。")
    message = _insert_message(conn, int(batch["thread_id"]), "assistant", "batch_confirmation", "\n".join(lines))
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE v2_learning_batches SET confirmation_message_id=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (message["id"], batch["id"]),
        )
    return message, message["content"]


def _confirm_batch(
    conn,
    batch: dict,
    evidence: dict,
    message: dict,
    *,
    segment_numbers: set[int] | None = None,
    embedding_client=None,
    llm_service=None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, source_message_id, question_message_id,
                   fact_text, entity_name, proposed_trust, status,
                   confirmed_knowledge_id, resolution_message_id,
                   comparison_result, clarification_question,
                   comparison_reason, related_knowledge_ids,
                   batch_id, segment_no, expected_answer_type, derived, individual_confirmation, paused,
                   created_at, updated_at
            FROM v2_learning_proposals
            WHERE batch_id=%s AND paused=FALSE AND status='pending_confirmation'
              AND derived=FALSE AND individual_confirmation=FALSE
            ORDER BY segment_no, id
            """,
            (batch["id"],),
        )
        proposals = [dict(row) for row in cur.fetchall()]
    selected = [
        item for item in proposals
        if segment_numbers is None or int(item.get("segment_no") or 0) in segment_numbers
    ]
    for proposal in selected:
        _confirm(
            conn,
            proposal,
            evidence,
            message,
            embedding_client=embedding_client,
            llm_service=llm_service,
        )
    return len(selected)


def _batch_is_settled(conn, batch_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS count FROM v2_learning_proposals
            WHERE batch_id=%s AND paused=FALSE
              AND status IN ('pending_confirmation', 'pending_clarification')
            """,
            (batch_id,),
        )
        return int(cur.fetchone()["count"]) == 0


def _refresh_batch_state(conn, batch: dict) -> dict:
    """Recompute the small batch status after a confirmation or clarification."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              count(*) FILTER (WHERE status='pending_confirmation' AND derived=FALSE AND individual_confirmation=FALSE AND paused=FALSE) AS clear_count,
              count(*) FILTER (WHERE paused=FALSE AND (status='pending_clarification' OR (status='pending_confirmation' AND (derived=TRUE OR individual_confirmation=TRUE)))) AS unclear_count
            FROM v2_learning_proposals WHERE batch_id=%s
            """,
            (batch["id"],),
        )
        counts = cur.fetchone()
    if int(counts["clear_count"]) > 0:
        status = "awaiting_confirmation"
    elif int(counts["unclear_count"]) > 0:
        status = "awaiting_clarification"
    elif int(batch.get("failed_segments") or 0) > 0:
        status = "partial"
    else:
        status = "completed"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE v2_learning_batches SET status=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (status, batch["id"]),
        )
    return _get_batch(conn, int(batch["id"])) or dict(batch, status=status)


def _learn_bulk_turn(
    conn,
    *,
    clean: str,
    thread_id: int,
    session: dict,
    evidence: dict,
    user_message: dict,
    llm_service,
    embedding_client,
    had_pending: bool,
) -> dict:
    """Process every deterministic segment and leave only unresolved work pending."""

    if had_pending:
        # A new bulk payload is a new evidence unit. Existing questions remain
        # intact and are resumed only after this batch is settled.
        _pause_pending_proposals(conn, thread_id)
    segments = segment_bulk_text(clean)
    batch = _create_batch(conn, thread_id, int(evidence["id"]), clean, len(segments))
    processed_segments = 0
    failed_segments = 0
    clear_facts: list[dict] = []
    unclear_items: list[dict] = []
    conflicts: list[dict] = []
    fallback = False

    for segment in segments:
        segment_number = int(segment["segment_no"])
        try:
            facts, used_fallback = _model_facts(
                segment["text"], llm_service, require_coverage=True
            )
            facts = _deduplicate_facts(facts)
        except Exception:
            log.exception("V2 bulk segment processing failed segment=%s", segment_number)
            facts, used_fallback = [], True
        # A fallback is a raw interpretation, not proof that the segment was
        # understood. It remains in raw evidence and is reported as failed.
        if used_fallback or not facts:
            failed_segments += 1
            fallback = True
            continue
        processed_segments += 1
        for fact in facts:
            fact["individual_confirmation"] = requires_individual_confirmation(fact)
            proposal = _plan_fact(
                conn,
                thread_id=thread_id,
                message_id=int(user_message["id"]),
                evidence_id=int(evidence["id"]),
                fact=fact,
                llm_service=llm_service,
                embedding_client=embedding_client,
                batch_id=int(batch["id"]),
                segment_no=segment_number,
            )
            item = {
                "proposal_id": proposal.get("id"),
                "segment_no": segment_number,
                "content": fact["content"],
                "derived": bool(fact.get("derived")),
                "individual_confirmation": bool(fact.get("individual_confirmation")),
            }
            if (
                proposal.get("status") == "pending_confirmation"
                and not fact.get("derived")
                and not fact.get("individual_confirmation")
            ):
                clear_facts.append(item)
            elif proposal.get("status") == "pending_confirmation":
                unclear_items.append(item | {"question": "这是基于原文之外的推断，需要单独确认。"})
            elif proposal.get("comparison_result") == "CONFLICT":
                conflicts.append(item)
                unclear_items.append(item | {"question": proposal.get("clarification_question", "")})
            else:
                unclear_items.append(item | {"question": proposal.get("clarification_question", "")})

    batch = _update_batch(
        conn,
        int(batch["id"]),
        processed_segments=processed_segments,
        failed_segments=failed_segments,
        clear_facts=clear_facts,
        unclear_items=unclear_items,
        conflicts=conflicts,
    )
    question_message, _, active_proposal = _next_question(conn, thread_id, session)
    if question_message:
        return _result(
            conn,
            thread_id,
            status=_awaiting_status(active_proposal),
            message=question_message,
            proposal=active_proposal,
            batch=batch,
            fallback=fallback,
        )

    if failed_segments:
        response_text = (
            f"这次收到 {len(segments)} 项，其中 {processed_segments} 项已理解，"
            f"{failed_segments} 项解析失败。原始资料已保留，请补发失败条目。"
        )
        response = _insert_message(conn, thread_id, "assistant", "text", response_text)
        return _result(conn, thread_id, status="bulk_partial", message=response, batch=batch, fallback=True)
    _resume_paused_proposals(conn, thread_id)
    response = _insert_message(conn, thread_id, "assistant", "text", "这批资料已处理完成，暂时没有需要确认的内容。")
    return _result(conn, thread_id, status="bulk_completed", message=response, batch=batch)


def _summary(conn, thread_id: int, session: dict) -> tuple[dict | None, str | None]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT k.id, k.title, k.content
            FROM v2_learning_proposals p
            JOIN v2_knowledge k ON k.id=p.confirmed_knowledge_id
            WHERE p.thread_id=%s AND p.status='confirmed'
              AND p.updated_at >= %s
            ORDER BY p.id
            """,
            (thread_id, session["started_at"]),
        )
        facts = deduplicate_knowledge(dict(row) for row in cur.fetchall())
        cur.execute(
            """
            SELECT status, count(*) AS count
            FROM v2_learning_proposals
            WHERE thread_id=%s AND updated_at >= %s
              AND status IN ('corrected', 'unknown', 'skipped')
            GROUP BY status
            """,
            (thread_id, session["started_at"]),
        )
        outcomes = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT COALESCE(sum(total_segments), 0) AS total,
                   COALESCE(sum(processed_segments), 0) AS processed,
                   COALESCE(sum(failed_segments), 0) AS failed
            FROM v2_learning_batches
            WHERE thread_id=%s AND created_at >= %s
            """,
            (thread_id, session["started_at"]),
        )
        batch_coverage = dict(cur.fetchone())
    corrected = outcomes.get("corrected", 0)
    unresolved = outcomes.get("unknown", 0) + outcomes.get("skipped", 0)
    lines = [f"今天我学会了 {len(facts)} 件事："]
    if int(batch_coverage["total"]):
        lines.append(
            f"本次批量资料：{int(batch_coverage['processed'])}/{int(batch_coverage['total'])} 项已处理。"
        )
        if int(batch_coverage["failed"]):
            lines.append(f"还有 {int(batch_coverage['failed'])} 项解析失败，原始内容已保留。")
    lines.extend(f"* {item['content']}" for item in facts)
    if not facts:
        lines.append("* 暂时没有新增已确认知识。")
    if corrected:
        lines.append(f"我修正了 {corrected} 条旧理解。")
    if unresolved:
        lines.append(f"还有 {unresolved} 个问题没有确认。")
    text = "\n".join(lines)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_sessions
            SET status='completed', summary=%s, completed_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (text, session["id"]),
        )
    message = _insert_message(conn, thread_id, "assistant", "summary", text)
    return message, text


def _next_question(conn, thread_id: int, session: dict) -> tuple[dict | None, str | None, dict | None]:
    batch = _pending_batch(conn, thread_id)
    if batch:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) AS count FROM v2_learning_proposals
                WHERE batch_id=%s AND paused=FALSE AND status='pending_confirmation'
                  AND derived=FALSE AND individual_confirmation=FALSE
                """,
                (batch["id"],),
            )
            clear_count = int(cur.fetchone()["count"])
        if clear_count:
            message, _ = _batch_confirmation_message(conn, batch)
            return message, message.get("content"), {
                "batch_id": batch["id"],
                "status": "pending_confirmation",
                "batch": batch,
            }
    proposal = _pending_proposal(conn, thread_id)
    if not proposal:
        return None, None, None
    if proposal.get("question_message_id"):
        with conn.cursor() as cur:
            cur.execute("SELECT id, thread_id, sequence_no, role, message_type, content, raw_evidence_id, created_at FROM v2_inbox_messages WHERE id=%s", (proposal["question_message_id"],))
            message = cur.fetchone()
        return dict(message) if message else None, message["content"] if message else None, proposal
    message, text = _ask(conn, proposal, session)
    return message, text, proposal


def _result(
    conn,
    thread_id: int,
    *,
    status: str,
    message: dict | None = None,
    proposal: dict | None = None,
    batch: dict | None = None,
    fallback: bool = False,
    summary: str | None = None,
) -> dict:
    payload = {
        "thread_id": thread_id,
        "status": status,
        "message": message,
        "proposal": proposal,
        "batch": batch,
        "summary": summary,
        "fallback": fallback,
        **thread_response(conn, thread_id),
    }
    return json_safe(payload)


def _awaiting_status(proposal: dict | None) -> str:
    return (
        "awaiting_clarification"
        if proposal and proposal.get("status") == "pending_clarification"
        else "awaiting_confirmation"
    )


def learn_turn(
    conn,
    content: str,
    *,
    thread_id: int | None = None,
    channel: str = "inbox",
    llm_service=None,
    embedding_client=None,
    question_budget: int = 5,
    persisted_evidence: dict | None = None,
    persisted_user_message: dict | None = None,
) -> dict:
    """Persist one expert turn and advance exactly one atomic confirmation."""

    # Keep the complete raw payload. The HTTP boundary caps a single request,
    # while bulk segmentation ensures each LLM call stays small.
    clean = str(content or "").strip()
    if not clean:
        raise ValueError("content must not be empty")
    transaction = conn.transaction() if callable(getattr(conn, "transaction", None)) else nullcontext()
    with transaction:
        if thread_id is None:
            thread = create_thread(conn, channel=channel, mode="learn")
        else:
            thread = get_thread(conn, int(thread_id))
        current_thread_id = int(thread["id"])
        _lock_thread(conn, current_thread_id)
        requested_session_type = (
            "active_inbox" if channel in {"inbox", "chat"} else "passive"
        )
        session = _ensure_session(
            conn,
            current_thread_id,
            question_budget,
            session_type=requested_session_type,
        )
        # Direct Inbox/Chat work is an explicitly active interaction.  The
        # passive daily budget is for unsolicited questions only.
        session["_unlimited_questions"] = session.get("session_type") in {
            "active_inbox",
            "active_grill",
        }
        pending = _pending_proposal(conn, current_thread_id)
        pending_batch = _pending_batch(conn, current_thread_id)
        pending_status = (
            (pending.get("status") or "pending_confirmation")
            if pending
            else None
        )
        mode = classify_input_mode(
            clean,
            pending_question=pending.get("clarification_question") if pending else None,
            has_pending=bool(pending or pending_batch),
        )
        evidence = persisted_evidence or _insert_evidence(
            conn, clean, current_thread_id, channel=channel, input_mode=mode
        )
        reply_kind = classify_reply(clean) if (pending or pending_batch) else "new"
        message_type = {
            "confirm": "confirmation",
            "negative": "correction",
            "unknown": "unknown",
            "skip": "skip",
            "correction": "correction",
        }.get(reply_kind, "evidence")
        user_message = persisted_user_message or _insert_message(
            conn, current_thread_id, "user", message_type, clean, int(evidence["id"])
        )

        if mode == "bulk_knowledge_payload":
            return _learn_bulk_turn(
                conn,
                clean=clean,
                thread_id=current_thread_id,
                session=session,
                evidence=evidence,
                user_message=user_message,
                llm_service=llm_service,
                embedding_client=embedding_client,
                had_pending=bool(pending or pending_batch),
            )

        # A retry after confirmation (or a bare control word in a new thread)
        # is not product knowledge.  Keep the raw input and explain the state.
        if not pending and not pending_batch and classify_reply(clean) in {"confirm", "negative", "unknown", "skip"}:
            response = _insert_message(
                conn,
                current_thread_id,
                "assistant",
                "text",
                "当前没有待确认的问题；请直接告诉我新的产品信息。",
            )
            return _result(conn, current_thread_id, status="no_pending", message=response)

        batch_selection = (
            parse_batch_confirmation(clean, int(pending_batch["total_segments"]))
            if pending_batch else None
        )
        if pending_batch and clean.casefold() in {"查看明细", "查看详情", "details"}:
            detail_message, _ = _batch_detail_message(conn, pending_batch)
            return _result(
                conn,
                current_thread_id,
                status="awaiting_confirmation",
                message=detail_message,
                proposal={"batch_id": pending_batch["id"], "status": "pending_confirmation", "batch": pending_batch},
                batch=pending_batch,
            )
        if pending_batch and (reply_kind == "confirm" or batch_selection is not None):
            selected = None if reply_kind == "confirm" else batch_selection
            _confirm_batch(
                conn,
                pending_batch,
                evidence,
                user_message,
                segment_numbers=selected,
                embedding_client=embedding_client,
                llm_service=llm_service,
            )
            # A partial confirmation gets a fresh prompt for the remaining
            # clear facts. This is also a repetition guard for the batch UX.
            if selected is not None:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE v2_learning_batches SET confirmation_message_id=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
                        (pending_batch["id"],),
                    )
            batch = _refresh_batch_state(conn, pending_batch)
            if _batch_is_settled(conn, int(batch["id"])):
                _resume_paused_proposals(conn, current_thread_id)
            next_message, _, next_proposal = _next_question(conn, current_thread_id, session)
            if next_message:
                return _result(
                    conn,
                    current_thread_id,
                    status=_awaiting_status(next_proposal),
                    message=next_message,
                    proposal=next_proposal,
                    batch=batch,
                )
            summary_message, summary_text = _summary(conn, current_thread_id, session)
            return _result(
                conn,
                current_thread_id,
                status="confirmed",
                message=summary_message,
                summary=summary_text,
                batch=batch,
            )

        if pending and pending_status == "pending_confirmation" and reply_kind == "confirm":
            _confirm(
                conn,
                pending,
                evidence,
                user_message,
                embedding_client=embedding_client,
                llm_service=llm_service,
            )
            next_message, _, next_proposal = _next_question(conn, current_thread_id, session)
            if next_message:
                return _result(
                    conn,
                    current_thread_id,
                    status=_awaiting_status(next_proposal),
                    message=next_message,
                    proposal=next_proposal,
                )
            summary_message, summary_text = _summary(conn, current_thread_id, session)
            return _result(
                conn,
                current_thread_id,
                status="confirmed",
                message=summary_message,
                summary=summary_text,
            )

        binary_clarification = (
            pending
            and pending_status == "pending_clarification"
            and (pending.get("expected_answer_type") or _expected_answer_type(pending.get("clarification_question"))) == "binary"
            and reply_kind in {"confirm", "negative"}
        )
        if pending and pending_status == "pending_clarification" and reply_kind == "confirm" and not binary_clarification:
            # Clarification questions are deliberately not yes/no questions.  A
            # bare confirmation is not product evidence and must not supersede
            # the unresolved proposal or be extracted as a new fact.
            clarification = _insert_message(
                conn,
                current_thread_id,
                "assistant",
                "clarification",
                _clean(pending.get("clarification_question"), 2000)
                or safe_question({"entity_name": pending.get("entity_name", "")}),
            )
            return _result(
                conn,
                current_thread_id,
                status="awaiting_clarification",
                message=clarification,
                proposal=pending,
            )

        if pending and reply_kind in {"unknown", "skip"}:
            proposal_status = "skipped" if reply_kind == "skip" else "unknown"
            _update_proposal(
                conn,
                int(pending["id"]),
                proposal_status,
                message_id=int(user_message["id"]),
            )
            acknowledgement = (
                "好的，我先记下你现在不知道。以后有新的资料或上下文时再回来。"
                if reply_kind == "unknown" else
                "好的，先跳过这个问题。我不会反复追问。"
            )
            acknowledgement_message = _insert_message(
                conn, current_thread_id, "assistant", "text", acknowledgement
            )
            _refresh_batch_state(conn, pending_batch) if pending_batch else None
            _resume_paused_proposals(conn, current_thread_id)
            next_message, _, next_proposal = _next_question(conn, current_thread_id, session)
            if next_message:
                return _result(
                    conn,
                    current_thread_id,
                    status=_awaiting_status(next_proposal),
                    message=next_message,
                    proposal=next_proposal,
                )
            _, summary_text = _summary(conn, current_thread_id, session)
            return _result(
                conn,
                current_thread_id,
                status=proposal_status,
                message=acknowledgement_message,
                summary=summary_text,
            )

        extraction_input = clean
        if pending and pending_status == "pending_clarification":
            extraction_input = (
                f"需要澄清的原始说法：{pending['fact_text']}\n"
                f"产品专家的补充回答：{clean}"
            )
        facts, fallback = _model_facts(
            extraction_input,
            llm_service,
            _pending_context(conn, current_thread_id) if pending else None,
        )
        facts = _deduplicate_facts(facts)
        if pending:
            if pending_status == "pending_confirmation":
                _retire_corrected_knowledge(conn, pending)
                finished_status = "corrected"
            else:
                _supersede_proposal_source(conn, pending)
                finished_status = "superseded"
            _update_proposal(
                conn,
                int(pending["id"]),
                finished_status,
                message_id=int(user_message["id"]),
            )

        first_proposal = None
        for fact in facts:
            proposal = _plan_fact(
                conn,
                thread_id=current_thread_id,
                message_id=int(user_message["id"]),
                evidence_id=int(evidence["id"]),
                fact=fact,
                llm_service=llm_service,
                embedding_client=embedding_client,
            )
            if first_proposal is None:
                first_proposal = proposal

        if first_proposal is None:
            response = _insert_message(
                conn,
                current_thread_id,
                "assistant",
                "text",
                "我保留了原始输入，但还没有形成可以确认的产品事实。",
            )
            return _result(conn, current_thread_id, status="reused", message=response, fallback=fallback)

        _resume_paused_proposals(conn, current_thread_id)
        question_message, _, active_proposal = _next_question(
            conn, current_thread_id, session
        )
        if not question_message:
            response = _insert_message(
                conn,
                current_thread_id,
                "assistant",
                "text",
                "我先记下这条待确认信息，稍后再问你。",
            )
            return _result(
                conn,
                current_thread_id,
                status="waiting",
                message=response,
                proposal=active_proposal,
                fallback=fallback,
            )
        return _result(
            conn,
            current_thread_id,
            status=_awaiting_status(active_proposal),
            message=question_message,
            proposal=active_proposal,
            fallback=fallback,
        )
