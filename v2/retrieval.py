"""Minimal V2 learning retrieval over the small Knowledge collection."""

from __future__ import annotations

import math
import re
from typing import Any

from embeddings import OPENROUTER_EMBEDDING_DIMENSIONS, OPENROUTER_EMBEDDING_MODEL
TRUST_ORDER = {"official_source": 4, "user_confirmed": 4, "provisional": 2, "conflicted": 1}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./()_-]*|[\u0400-\u04ff]+|[\u4e00-\u9fff]")
_MODEL_RE = re.compile(r"(?=[A-Za-z0-9./()\-]*\d)[A-Za-z0-9][A-Za-z0-9./()\-]{2,}")

def _text(value: Any, limit: int = 12000) -> str:
    return str(value or "").strip()[:limit]


def _tokens(value: Any) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(_text(value))}


def _models(value: Any) -> set[str]:
    return {item.casefold() for item in _MODEL_RE.findall(_text(value))}


def _explicit_model_identifiers(value: Any) -> set[str]:
    """Return conservative model-like tokens for comparison isolation."""

    result = set(_models(value))
    for token in _TOKEN_RE.findall(_text(value)):
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./()_-]{2,}", token)
            and "-" in token
            and token == token.upper()
            and any(char.isalpha() for char in token)
        ):
            result.add(token.casefold())
    return result


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


def store_knowledge_embedding(conn, knowledge_id: int, text: str, *, embedder=None) -> bool:
    """Store one compatible vector; embedding failure never blocks learning."""

    embedding = _query_embedding(embedder, _text(text))
    if embedding is None:
        return False
    vector_text = "[" + ",".join(str(float(value)) for value in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET embedding=%s::vector, embedding_model=%s,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (vector_text, OPENROUTER_EMBEDDING_MODEL, int(knowledge_id)),
        )
    return True


def retrieve_learning_knowledge(conn, query: str, *, embedder=None, top_k: int = 8,
                                lexical_k: int | None = None, embedding_k: int | None = None,
                                same_model_only: bool = False) -> list[dict]:
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
    if same_model_only:
        query_models = _explicit_model_identifiers(query)
        if query_models:
            rows = [
                row for row in rows
                if query_models & _explicit_model_identifiers(row.get("entity_name"))
            ]
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


retrieve = retrieve_learning_knowledge
