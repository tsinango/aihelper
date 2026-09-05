"""Safe, provider-independent comparison for the V2 learning loop.

This module deliberately does not write to the database and does not change a
Knowledge trust value.  It is the boundary between extracted user evidence
and the learning state machine: the model may classify a new fact against a
small set of retrieved candidates, but Python owns the safety checks.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from llm import LLMService, parse_json_response


log = logging.getLogger("aihelper.v2.compare")


class CompareServiceError(RuntimeError):
    """The comparison judge is unavailable or returned unusable output.

    This is a technical failure, not product ambiguity.  Callers must surface
    it as a retryable processing failure instead of asking the expert a
    business clarification question.
    """


DECISIONS = frozenset({"NEW", "CONFIRM", "ENRICH", "CONFLICT", "UNCLEAR"})
QUESTION_DECISIONS = frozenset({"CONFLICT", "UNCLEAR"})
REQUIRED_CANDIDATE_DECISIONS = frozenset({"CONFIRM", "ENRICH", "CONFLICT"})
ALLOWED_RESULT_KEYS = frozenset({"decision", "knowledge_id", "question", "reason"})

# These terms are implementation vocabulary, not product expertise.  A model
# question containing them is never shown to a product expert.
FORBIDDEN_QUESTION_TERMS = frozenset({
    "database", "db", "sql", "schema", "table", "column", "field",
    "candidate", "knowledge_id", "knowledge key", "knowledge_key",
    "trust", "taxonomy", "ontology", "vector", "embedding", "reranker",
    "api", "json", "provenance", "raw_evidence", "source_id", "id字段",
    "数据库", "数据表", "字段", "候选知识", "知识库", "知识记录",
    "知识id", "知识键",
    "向量", "嵌入", "重排", "技术实现", "代码实现",
    # Open-world prompts are not evidence questions. They ask the expert to
    # complete an unbounded inventory that the source never promised to list.
    "还有其他", "是否还有", "有没有其他", "还有哪些", "还包括哪些", "完整列表", "数据规模", "知识蒸馏", "蒸馏", "rag",
    "other models", "other algorithms", "какие еще", "есть ли еще", "полный список", "другие алгоритмы",
})

PRODUCT_QUESTION_TERMS = frozenset({
    "型号", "系列", "版本", "硬件", "固件", "功能", "支持", "适用",
    "条件", "参数", "结论", "说法", "信息", "产品", "规格", "哪",
    "是否", "什么", "如何", "具体", "哪个", "哪些", "准确",
})

UNCERTAIN_MARKERS = ("大概", "可能", "应该", "也许", "看情况", "probably", "maybe")
VAGUE_VERSION_MARKERS = ("新版", "旧版", "老版", "新版本", "旧版本", "以前")
VAGUE_DIFFERENCE_MARKERS = ("不一样", "有区别", "有差异", "不同")

COMPARE_SYSTEM_PROMPT = """
你是产品知识学习助理，正在向产品专家核实一条新产品事实。
输入的 new_fact 是尚未确认的用户证据；retrieved_knowledge 是系统检索到的候选知识。
只根据这两个输入比较，不使用模型训练知识，不创造产品事实。

只能返回一个严格 JSON 对象，键只能是：
{"decision":"NEW|CONFIRM|ENRICH|CONFLICT|UNCLEAR",
 "knowledge_id":整数或null,
 "question":字符串或null,
 "reason":简短字符串}

判定规则：
- NEW：没有相关候选，或新事实不能安全地与候选合并。
- CONFIRM：新事实与一个候选表达的是同一个产品事实；knowledge_id 必须是输入候选中的 id。
- ENRICH：新事实只补充一个候选的有限细节；knowledge_id 必须是输入候选中的 id。
- CONFLICT：新事实与候选存在矛盾；保留双方，不得选择、覆盖或自动裁决任何一方；knowledge_id 只用于指出相关候选。
- UNCLEAR：型号、系列范围、版本、条件、否定事实或含义不清，无法安全判断。
- NEW 的 knowledge_id 必须为 null。CONFIRM、ENRICH、CONFLICT 必须给出输入候选中的 knowledge_id；UNCLEAR 的 knowledge_id 必须为 null。
- ENRICH 只有在范围、版本或条件仍缺失时才提出问题；如果补充已经原子且范围明确，question 为 null，交给后续复述确认。
- CONFLICT、UNCLEAR 必须提出一个真正需要产品专家回答的问题；一次只问一个问题。
- NEW 和 CONFIRM 的 question 必须为 null；reason 只能说明比较结果，不能添加新的产品事实。
- 手册或文字没有写某功能，不能推断为“不支持”；只有明确否定证据才是否定事实。
- 单个型号的事实不能推广到整个系列；新旧版本、hardware revision、firmware、地区和条件都必须保持原范围。
- 不要询问数据库字段、知识记录、技术实现或审核流程。
- 不要因为资料没有覆盖某个维度就追问其他模型、数据规模、RAG、蒸馏或“完整列表”；“等”只表示当前列举非穷尽。
""".strip()


def _text(value: Any, limit: int = 12000) -> str:
    return str(value or "").strip()[:limit]


def _normalise_fact(fact: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(fact, Mapping):
        raise ValueError("each fact must be an object")
    content = _text(fact.get("content") or fact.get("fact") or fact.get("text"))
    if not content:
        raise ValueError("fact content is required")
    return {
        "title": _text(fact.get("title"), 500) or content[:120],
        "content": content,
        "entity_name": _text(fact.get("entity_name") or fact.get("entity"), 500),
    }


def _normalise_facts(facts: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(facts, Mapping):
        values = [facts]
    elif isinstance(facts, Sequence) and not isinstance(facts, (str, bytes, bytearray)):
        values = list(facts)
    else:
        raise ValueError("facts must be a fact object or a sequence of fact objects")
    if len(values) != 1:
        raise ValueError("compare_and_ask accepts exactly one atomic fact")
    return [_normalise_fact(values[0])]


def _normalise_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("knowledge id must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise ValueError("knowledge id must be an integer")
    if result <= 0:
        raise ValueError("knowledge id must be positive")
    return result


def _normalise_candidates(candidates: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if candidates is None:
        return []
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise ValueError("retrieved Knowledge must be a sequence")
    result = []
    seen: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("each retrieved Knowledge item must be an object")
        knowledge_id = _normalise_id(candidate.get("id"))
        if knowledge_id in seen:
            raise ValueError("retrieved Knowledge contains duplicate ids")
        seen.add(knowledge_id)
        content = _text(candidate.get("content"))
        if not content:
            raise ValueError("retrieved Knowledge content is required")
        # Do not pass arbitrary database columns or a trust value into the
        # judging result.  Trust transitions belong to the caller.
        result.append({
            "id": knowledge_id,
            "title": _text(candidate.get("title"), 500),
            "content": content,
            "entity_name": _text(candidate.get("entity_name"), 500),
        })
    return result


def _candidate_prompt(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "[]"
    return "[\n" + ",\n".join(
        "  " + _json_object(candidate) for candidate in candidates
    ) + "\n]"


def _json_object(value: Mapping[str, Any]) -> str:
    # The values have already been reduced to strings/integers.  Keeping JSON
    # encoding local avoids accepting model-controlled prompt fragments.
    import json

    return json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"))


def _build_messages(fact: dict[str, str], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "new_fact:\n" + _json_object(fact) +
                "\n\nretrieved_knowledge:\n" + _candidate_prompt(candidates)
            ),
        },
    ]


def _safe_label(fact: Mapping[str, str]) -> str:
    """Return a short product label safe to insert into a fallback question."""

    label = _text(fact.get("entity_name"), 80)
    if not label:
        label = "这条产品信息"
    for term in FORBIDDEN_QUESTION_TERMS:
        label = label.replace(term, "")
    label = " ".join(label.split()).strip(" `\"'<>[]{}")
    return label or "这条产品信息"


def safe_question(fact: Mapping[str, str], decision: str = "UNCLEAR") -> str:
    """Make a single product question without exposing implementation details."""

    label = _safe_label(fact)
    if decision == "CONFLICT":
        return f"关于「{label}」这条信息，哪一条产品结论准确？"
    if decision == "ENRICH":
        return f"「{label}」的这项补充信息具体适用于哪个版本或条件？"
    return f"请明确「{label}」这条信息具体适用于哪个型号、版本或条件？"


def intrinsic_clarification_question(fact: Mapping[str, str]) -> str | None:
    """Catch ambiguity in the new statement itself, independent of retrieval."""

    content = _text(fact.get("content"))
    lowered = content.casefold()
    if any(marker in lowered for marker in UNCERTAIN_MARKERS):
        return "这条产品信息目前是不确定说法，你能确认准确结论和适用条件吗？"
    if any(marker in content for marker in VAGUE_VERSION_MARKERS):
        return "这里的新版或旧版具体指哪个硬件 revision、固件版本或产品版本？"
    if any(marker in content for marker in VAGUE_DIFFERENCE_MARKERS):
        return "你说的产品差异具体是什么，并且适用于哪个版本？"
    return None


def _question_is_safe(question: Any) -> bool:
    if not isinstance(question, str):
        return False
    question = question.strip()
    if not question or len(question) > 1200:
        return False
    if question.count("?") + question.count("？") != 1:
        return False
    lowered = question.casefold()
    if any(term.casefold() in lowered for term in FORBIDDEN_QUESTION_TERMS):
        return False
    # Reject placeholders such as "为什么？".  A valid question must carry
    # at least a small amount of product context; this also prevents a model
    # from satisfying the required-question rule mechanically.
    visible = question.replace("?", "").replace("？", "").strip()
    return len(visible) >= 6 and any(term in question for term in PRODUCT_QUESTION_TERMS)


def _fail_closed(fact: dict[str, str], reason: str = "无法安全解析比较结果") -> dict[str, Any]:
    # technical_failure marks a provider/parse/contract failure.  It must not
    # be presented to the expert as product uncertainty; the caller turns it
    # into a retryable job failure.  A genuine semantic UNCLEAR decision from
    # the judge carries no such marker.
    return {
        "decision": "UNCLEAR",
        "knowledge_id": None,
        "question": safe_question(fact, "UNCLEAR"),
        "reason": _text(reason, 500) or "无法安全解析比较结果",
        "technical_failure": True,
    }


def _validate_result(
    raw: Any,
    fact: dict[str, str],
    candidate_ids: set[int],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return _fail_closed(fact, "模型结果不是 JSON 对象")
    if set(raw) - ALLOWED_RESULT_KEYS:
        return _fail_closed(fact, "模型结果包含未允许的字段")

    decision = raw.get("decision")
    if decision not in DECISIONS:
        return _fail_closed(fact, "模型返回了未知判定")

    raw_id = raw.get("knowledge_id")
    knowledge_id: int | None
    if raw_id is None:
        knowledge_id = None
    else:
        try:
            knowledge_id = _normalise_id(raw_id)
        except ValueError:
            return _fail_closed(fact, "模型返回了无效 Knowledge id")
        if knowledge_id not in candidate_ids:
            return _fail_closed(fact, "模型返回的 Knowledge id 不在候选集合中")

    if decision in REQUIRED_CANDIDATE_DECISIONS and knowledge_id is None:
        return _fail_closed(fact, f"{decision} 缺少候选 Knowledge id")
    if decision in {"NEW", "UNCLEAR"} and knowledge_id is not None:
        return _fail_closed(fact, f"{decision} 不允许携带 Knowledge id")

    question = raw.get("question")
    if decision in QUESTION_DECISIONS:
        if not _question_is_safe(question):
            question = safe_question(fact, decision)
    elif decision == "ENRICH" and question is not None:
        if not _question_is_safe(question):
            question = safe_question(fact, decision)
    elif question is not None:
        return _fail_closed(fact, f"{decision} 不应携带问题")

    # Only this allow-listed, normalized shape leaves the module.  In
    # particular, no trust/confidence value from the model can escape here.
    return {
        "decision": decision,
        "knowledge_id": knowledge_id,
        "question": question.strip() if isinstance(question, str) else None,
        "reason": _text(raw.get("reason"), 500),
    }


def compare_and_ask(
    facts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    retrieved_knowledge: Sequence[Mapping[str, Any]] | None,
    llm_service: LLMService,
    *,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """Classify one atomic extracted fact against retrieved V2 Knowledge.

    The plural ``facts`` input is accepted for ergonomic compatibility with
    the extractor, but exactly one fact is required.  This is intentional:
    one comparison produces at most one expert question and keeps atomic
    confirmation intact.  Callers should invoke this once per extracted fact.

    Malformed model output, an unavailable judge, and every unsafe model
    decision return ``UNCLEAR`` with a safe product question.  Provider,
    parsing, and contract failures additionally carry
    ``technical_failure=True`` so callers can distinguish a retryable service
    failure from genuine product ambiguity.  Input shape errors are programmer
    errors and raise ``ValueError`` before an LLM call.
    """

    normalized_facts = _normalise_facts(facts)
    candidates = _normalise_candidates(retrieved_knowledge)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    fact = normalized_facts[0]
    candidate_ids = {int(item["id"]) for item in candidates}
    intrinsic_question = intrinsic_clarification_question(fact)
    # Retrieval returning no related Knowledge is an ordinary, deterministic
    # NEW case.  Do not spend an LLM call asking the model to invent a missing
    # comparison target or an open-world follow-up question.  Intrinsic
    # ambiguity remains a clarification, also without retrieval candidates.
    if not candidates:
        if intrinsic_question:
            return {
                "decision": "UNCLEAR",
                "knowledge_id": None,
                "question": intrinsic_question,
                "reason": "新输入本身包含尚未明确的版本、差异或不确定表述",
            }
        return {
            "decision": "NEW",
            "knowledge_id": None,
            "question": None,
            "reason": "没有相关的既有 Knowledge",
        }
    try:
        response = llm_service.judge(_build_messages(fact, candidates), max_tokens=max_tokens)
        parsed = parse_json_response(response)
    except Exception:
        log.exception("V2 compare judge failed; failing closed")
        return _fail_closed(fact, "比较服务不可用或返回无法解析的结果")
    result = _validate_result(parsed, fact, candidate_ids)
    if intrinsic_question and result["decision"] == "UNCLEAR":
        result["question"] = intrinsic_question
        return result
    if intrinsic_question and (
        result["decision"] in {"NEW", "CONFIRM"}
        or (result["decision"] == "ENRICH" and not result.get("question"))
    ):
        return {
            "decision": "UNCLEAR",
            "knowledge_id": None,
            "question": intrinsic_question,
            "reason": "新输入本身包含尚未明确的版本、差异或不确定表述",
        }
    return result


__all__ = [
    "ALLOWED_RESULT_KEYS",
    "COMPARE_SYSTEM_PROMPT",
    "CompareServiceError",
    "DECISIONS",
    "compare_and_ask",
    "intrinsic_clarification_question",
    "safe_question",
]
