"""Minimal V2 learning retrieval and evidence comparison."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from embeddings import OPENROUTER_EMBEDDING_DIMENSIONS, OPENROUTER_EMBEDDING_MODEL
from llm import parse_json_response

COMPARISON_RESULTS = frozenset({"NEW", "CONFIRM", "ENRICH", "CONFLICT", "UNCLEAR"})
TRUST_ORDER = {"official_source": 4, "user_confirmed": 4, "provisional": 2, "conflicted": 1}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./()_-]*|[\u0400-\u04ff]+|[\u4e00-\u9fff]")
_MODEL_RE = re.compile(r"(?=[A-Za-z0-9./()\-]*\d)[A-Za-z0-9][A-Za-z0-9./()\-]{2,}")

COMPARE_SYSTEM_PROMPT = """
你是 aihelper V2 产品知识比较器。只根据给定的 V2 Knowledge evidence 比较用户新输入，
不使用训练知识、常识或猜测。严格返回 JSON：
{"decision":"NEW|CONFIRM|ENRICH|CONFLICT|UNCLEAR","fact_text":"原子事实","clarifying_question":"最多一个问题，没有则为空","reason":"简短原因","related_knowledge_ids":[1]}
规则：CONFIRM 只能是同一实体、范围、版本和条件下的同一事实；ENRICH 只能补充同一实体的独立细节。
系列泛化、版本/条件扩大、否定事实、低信任来源并入高信任事实，都必须 UNCLEAR 或 CONFLICT。
CONFLICT 只保留双方，不自行选择；UNCLEAR 必须提出一个具体问题。单型号不能推广系列；没有提到功能不能解释为不支持。
related_knowledge_ids 只能使用给定 evidence 的 id，不询问数据库或技术实现。
""".strip()


def _text(value: Any, limit: int = 12000) -> str:
    return str(value or "").strip()[:limit]


def _tokens(value: Any) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(_text(value))}


def _models(value: Any) -> set[str]:
    return {item.casefold() for item in _MODEL_RE.findall(_text(value))}


def _same_model(query: str, row: dict) -> bool:
    entity = _text(row.get("entity_name"), 500).casefold()
    return bool(entity and (entity in query.casefold() or entity in _models(query)))


def _lexical_score(query: str, row: dict) -> float:
    query_tokens = _tokens(query)
    row_tokens = _tokens(" ".join(_text(row.get(key)) for key in ("title", "content", "entity_name")))
    if not query_tokens or not row_tokens:
        return 0.0
    score = len(query_tokens & row_tokens) / math.sqrt(len(query_tokens) * len(row_tokens))
    return min(score + (1.0 if _same_model(query, row) else 0.0), 2.0)


def _vector(value: Any) -> list[float] | None:
    if isinstance(value, str):
        value = [item for item in value.strip().strip("[]").split(",") if item.strip()]
    try:
        result = [float(item) for item in value] if value is not None else []
    except (TypeError, ValueError):
        return None
    return result if result and all(math.isfinite(item) for item in result) else None


def _cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else None


def _rows(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, content, entity_name, trust, active, embedding, embedding_model,
                   created_at, updated_at
            FROM v2_knowledge WHERE active=TRUE ORDER BY id
        """)
        return [dict(row) for row in cur.fetchall()]


def _query_embedding(embedder, query: str) -> list[float] | None:
    if embedder is None:
        return None
    try:
        values = embedder.encode([query], normalize_embeddings=True)
        vector = _vector(values[0]) if values else None
        return vector if vector and len(vector) == OPENROUTER_EMBEDDING_DIMENSIONS else None
    except Exception:
        return None


def retrieve_learning_knowledge(conn, query: str, *, embedder=None, top_k: int = 8,
                                lexical_k: int | None = None, embedding_k: int | None = None) -> list[dict]:
    """Merge same-model-first lexical hits with an exact embedding scan.

    Vectors are compared in Python because the V2 data set is small. No vector
    index is required. Embedding errors or incompatible vectors leave lexical
    retrieval intact.
    """
    query = _text(query)
    if not query:
        return []
    top_k = max(1, min(int(top_k), 100))
    lexical_k = top_k if lexical_k is None else max(1, int(lexical_k))
    embedding_k = top_k if embedding_k is None else max(1, int(embedding_k))
    rows = _rows(conn)
    scored = {}
    for row in rows:
        score = _lexical_score(query, row)
        if score > 0:
            scored[int(row["id"])] = {**row, "lexical_score": score, "embedding_score": None, "retrieval_sources": ["lexical"]}
    lexical = sorted(scored.values(), key=lambda r: (_same_model(query, r), r["lexical_score"], TRUST_ORDER.get(r.get("trust"), 0), -int(r["id"])), reverse=True)
    scored = {int(row["id"]): row for row in lexical[:lexical_k]}
    query_vector = _query_embedding(embedder, query)
    if query_vector is not None:
        embedding = []
        for row in rows:
            if row.get("embedding_model") != OPENROUTER_EMBEDDING_MODEL:
                continue
            score = _cosine(query_vector, _vector(row.get("embedding")) or [])
            if score is not None:
                embedding.append((score, row))
        embedding.sort(key=lambda pair: (pair[0], -int(pair[1]["id"])), reverse=True)
        for score, row in embedding[:embedding_k]:
            item = scored.setdefault(int(row["id"]), {**row, "lexical_score": 0.0, "embedding_score": None, "retrieval_sources": []})
            item["embedding_score"] = score
            if "embedding" not in item["retrieval_sources"]:
                item["retrieval_sources"].append("embedding")
    def rank(row):
        lexical_score = float(row.get("lexical_score") or 0)
        embedding_score = max(0.0, float(row.get("embedding_score") or 0))
        return (_same_model(query, row), lexical_score * .55 + embedding_score * .45, lexical_score, embedding_score, TRUST_ORDER.get(row.get("trust"), 0), -int(row["id"]))
    result = sorted(scored.values(), key=rank, reverse=True)[:top_k]
    for row in result:
        row["combined_score"] = float(float(row.get("lexical_score") or 0) * .55 + max(0.0, float(row.get("embedding_score") or 0)) * .45)
        row.pop("embedding", None)
    return result


def build_compare_messages(query: str, candidates: list[dict]) -> list[dict]:
    evidence = [{key: item.get(key, "") for key in ("id", "title", "content", "entity_name", "trust")} for item in candidates]
    return [
        {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": "用户新输入：\n" + _text(query) + "\n\n相关 Knowledge evidence：\n" + json.dumps(evidence, ensure_ascii=False)},
    ]


def compare_knowledge(query: str, candidates: list[dict], llm_service, *, max_tokens: int = 800) -> dict:
    """Ask OpenRouter for NEW/CONFIRM/ENRICH/CONFLICT/UNCLEAR, fail closed."""
    query = _text(query)
    if not query:
        raise ValueError("query must not be empty")
    candidates = list(candidates or [])
    if not candidates:
        return {"decision": "NEW", "fact_text": query, "clarifying_question": "", "reason": "没有相关 Knowledge。", "related_knowledge_ids": []}
    try:
        parsed = parse_json_response(llm_service.judge(build_compare_messages(query, candidates), max_tokens=max_tokens))
    except Exception:
        parsed = {}
    parsed = parsed if isinstance(parsed, dict) else {}
    decision = _text(parsed.get("decision"), 30).upper()
    if decision not in COMPARISON_RESULTS:
        decision = "UNCLEAR"
    allowed = {int(item["id"]) for item in candidates}
    related = []
    for value in parsed.get("related_knowledge_ids", []) if isinstance(parsed.get("related_knowledge_ids"), list) else []:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value in allowed and value not in related:
            related.append(value)
    if decision in {"CONFIRM", "ENRICH", "CONFLICT"} and not related:
        related = [int(candidates[0]["id"])]
    question = _text(parsed.get("clarifying_question"), 2000)
    if decision == "UNCLEAR" and not question:
        question = "你说的这条信息具体适用于哪个型号、版本或条件？"
    if decision == "CONFLICT" and not question:
        question = "这两条说法存在冲突，哪一条适用于当前产品、版本或条件？"
    return {"decision": decision, "fact_text": _text(parsed.get("fact_text")) or query, "clarifying_question": question, "reason": _text(parsed.get("reason"), 2000), "related_knowledge_ids": related}


retrieve = retrieve_learning_knowledge
compare = compare_knowledge
