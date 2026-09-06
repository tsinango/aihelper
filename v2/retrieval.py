"""Minimal V2 learning retrieval over the small Knowledge collection."""

from __future__ import annotations

import math
import re
from typing import Any

from embeddings import OPENROUTER_EMBEDDING_DIMENSIONS, OPENROUTER_EMBEDDING_MODEL
from helpers import identifiers, scope_match
TRUST_ORDER = {"official_source": 4, "user_confirmed": 4, "provisional": 2, "conflicted": 1}
# Trust values that may support an internal answer draft.  This mirrors the
# database CHECK in 013 plus the Phase 3.0 readiness predicate: active,
# trusted, and backed by an accepted supports source.
ANSWER_TRUST_VALUES = ("official_source", "user_confirmed")
# Document versions whose raw text may be quoted by the source fallback.
# Unverified uploads stay human-readable in the UI but never auto-quote.
FALLBACK_AUTHENTICITY = ("official_vendor", "confirmed_copy")
FALLBACK_VERSION_STATUSES = ("parsed", "learning", "complete")
FALLBACK_BLOCK_STATES = ("pending", "evidence_only", "proposal", "knowledge")
# Explicit "check the original" requests, matched case-insensitively.
SOURCE_REQUEST_KEYWORDS = (
    "原文", "核对", "核查", "手册", "表格", "第几页", "哪一页", "幻灯片",
    "说明书", "截图", "界面",
    "original", "source", "manual", "table", "page", "slide",
    "провер", "оригинал", "таблиц", "руководств", "страниц",
)
# High-impact operations where a grounded draft gets one verification read.
HIGH_RISK_KEYWORDS = (
    "恢复出厂", "factory reset", "升级", "firmware", "прошив",
    "密码", "password", "пароль", "反潜回", "联动",
)
MAX_FALLBACK_CHARS = 6000
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./()_-]*|[\u0400-\u04ff]+|[\u4e00-\u9fff]")
_MODEL_RE = re.compile(r"(?=[A-Za-z0-9./()\-]*\d)[A-Za-z0-9][A-Za-z0-9./()\-]{2,}")
_VERSION_RE = re.compile(r"V?\d+(?:[.,]\d+)+|\bBUILD\s*\d+|\b\d{6,8}\b", re.IGNORECASE)
# Normalized (alnum-only, uppercase) tokens that look like product models to
# the identifier regex but are actually codecs, standards, protocols, or power
# ratings.  They must never drive a model-conflict exclusion: e.g. a generic
# "H.265" mention in Knowledge content must not conflict with a question that
# happens to name "H.264".
_SCOPE_STANDARD_TOKENS = frozenset({
    "H264", "H265", "H266", "H264P", "H265P",
    "ONVIF", "POE", "POEP", "WIFI",
    "IP66", "IP67", "IP68", "IK10",
    "RTSP", "RTP", "RTCP", "HTTP", "HTTPS", "TCP", "UDP",
    "X8021", "8021X", "GB28181",
    "12V", "24V", "48V", "220V", "12VDC", "24VAC",
})

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


def _entity_name(row: dict) -> str:
    """Use the current organization name, with a legacy row fallback."""

    return _text(row.get("entity_name") or row.get("legacy_entity_name"), 500)


def _same_model(query: str, row: dict) -> bool:
    entity = _entity_name(row).casefold()
    return bool(entity and (entity in query.casefold() or entity in _models(query)))


def _lexical_score(query: str, row: dict) -> float:
    query_tokens = _tokens(query)
    row_tokens = _tokens(" ".join((_text(row.get("title")), _text(row.get("content")), _entity_name(row))))
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
            SELECT k.id, k.title, k.content, k.entity_id,
                   COALESCE(e.name, k.entity_name) AS entity_name,
                   k.entity_name AS legacy_entity_name,
                   k.trust, k.active, k.embedding, k.embedding_model,
                   k.created_at, k.updated_at
            FROM (
              SELECT id, title, content, entity_id, entity_name, trust, active,
                     embedding, embedding_model, created_at, updated_at
              FROM v2_knowledge
              WHERE active=TRUE
            ) k
            LEFT JOIN v2_entities e ON e.id=k.entity_id
            ORDER BY k.id
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


def _merge_embedding_hits(scored: dict, rows: list[dict], query_vector: list[float] | None, embedding_k: int) -> None:
    """Merge an exact embedding scan into lexical hits without touching order."""

    if query_vector is None:
        return
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


def _scope_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _scope_models(value: Any) -> list[str]:
    """Model-like identifiers minus standards, bare numbers, and versions.

    V2 Knowledge has no structured scope columns; model scope is read from
    title/content/entity text.  Bare numbers (``30``), version-shaped tokens
    (``5.6.11``), and codec/standard tokens (``H.265``, ``ONVIF``) are not
    product models and must not participate in conflict checks.
    """

    models = []
    for token in identifiers(value):
        key = _scope_key(token)
        if not key or key.isdigit():
            continue
        if _VERSION_RE.fullmatch(token):
            continue
        if key in _SCOPE_STANDARD_TOKENS:
            continue
        models.append(token.upper())
    return sorted(set(models), key=lambda item: (-len(item), item))


def _version_tokens(value: Any) -> list[str]:
    """Firmware/build/version tokens such as ``5.6.11`` or ``build 241112``."""

    tokens = set()
    for match in _VERSION_RE.findall(_text(value)):
        key = re.sub(r"[^A-Z0-9]", "", str(match).upper())
        if key:
            tokens.add(key)
            # "build 241112" and a bare "241112" denote the same build.
            digits = re.sub(r"[^0-9]", "", key)
            if digits and digits != key:
                tokens.add(digits)
    return sorted(tokens)


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
                if query_models & _explicit_model_identifiers(_entity_name(row))
            ]
    scored = {}
    for row in rows:
        score = _lexical_score(query, row)
        if score > 0:
            scored[int(row["id"])] = {**row, "lexical_score": score, "embedding_score": None, "retrieval_sources": ["lexical"]}
    lexical = sorted(scored.values(), key=lambda r: (_same_model(query, r), r["lexical_score"], TRUST_ORDER.get(r.get("trust"), 0), -int(r["id"])), reverse=True)
    scored = {int(row["id"]): row for row in lexical[:lexical_k]}
    query_vector = _query_embedding(embedder, query)
    _merge_embedding_hits(scored, rows, query_vector, embedding_k)
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


def _answer_eligible_rows(conn) -> list[dict]:
    """Active, trusted Knowledge backed by an accepted supports source.

    This is the same predicate the Phase 3.0 readiness check uses.  The
    supporting raw evidence itself must still be active: superseded or
    redacted evidence cannot ground an answer even when the source link was
    accepted earlier.  Rows with an accepted contradicting source link are
    additionally excluded: a known conflict must never silently support an
    answer draft.  Document-learned units additionally need an explicit
    validation for their own version; pre-document rows (NULL origin) keep
    the old trust/source gate.
    """

    with conn.cursor() as cur:
        cur.execute("""
            SELECT k.id, k.title, k.content, k.entity_id,
                   COALESCE(e.name, k.entity_name) AS entity_name,
                   k.entity_name AS legacy_entity_name,
                   k.trust, k.active, k.embedding, k.embedding_model,
                   k.unit_kind, k.applicability, k.revision,
                   k.origin_document_version_id, k.validation_status,
                   k.created_at, k.updated_at
            FROM v2_knowledge k
            LEFT JOIN v2_entities e ON e.id=k.entity_id
            WHERE k.active=TRUE
              AND k.trust IN ('official_source', 'user_confirmed')
              AND (k.origin_document_version_id IS NULL
                   OR k.validation_status='validated')
              AND EXISTS (
                    SELECT 1 FROM v2_knowledge_sources s
                    JOIN v2_raw_evidence r ON r.id=s.raw_evidence_id
                    WHERE s.knowledge_id=k.id
                      AND s.active=TRUE
                      AND s.relation='supports'
                      AND s.resolution='accepted'
                      AND r.evidence_status='active'
                  )
              AND NOT EXISTS (
                    SELECT 1 FROM v2_knowledge_sources c
                    WHERE c.knowledge_id=k.id
                      AND c.active=TRUE
                      AND c.relation='contradicts'
                      AND c.resolution='accepted'
                  )
            ORDER BY k.id
        """)
        return [dict(row) for row in cur.fetchall()]


def _answer_sources(conn, knowledge_ids: list[int]) -> dict[int, list[dict]]:
    """Accepted supports sources with raw-evidence locators, keyed by Knowledge.

    Only sources whose raw evidence is still active are returned: excerpts of
    superseded or redacted evidence must never surface in an answer snapshot.
    """

    result: dict[int, list[dict]] = {}
    if not knowledge_ids:
        return result
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.id AS source_id, s.knowledge_id, s.source_kind,
                   s.source_role, s.excerpt, s.relation, s.resolution,
                   r.id AS raw_evidence_id, r.evidence_type,
                   r.source_label, r.source_locator, r.evidence_status
            FROM v2_knowledge_sources s
            JOIN v2_raw_evidence r ON r.id=s.raw_evidence_id
            WHERE s.knowledge_id = ANY(%s)
              AND s.active=TRUE
              AND s.relation='supports'
              AND s.resolution='accepted'
              AND r.evidence_status='active'
            ORDER BY s.knowledge_id, s.id
            """,
            (list({int(item) for item in knowledge_ids}),),
        )
        for row in cur.fetchall():
            item = dict(row)
            try:
                key = int(item.get("knowledge_id"))
            except (TypeError, ValueError):
                continue
            result.setdefault(key, []).append(item)
    return result


def _row_scope(row: dict) -> tuple[list[str], list[str]]:
    text = " ".join((_text(row.get("title")), _text(row.get("content")), _entity_name(row)))
    return _scope_models(text), _version_tokens(text)


def _conflict_reason(query_models: list[str], query_versions: list[str], row: dict) -> str | None:
    """Deterministic exclusion reason, or None when the row stays eligible."""

    row_models, row_versions = _row_scope(row)
    if query_models and row_models and scope_match(query_models, row_models) == "conflict":
        return "model_conflict"
    if query_versions and row_versions and not (set(query_versions) & set(row_versions)):
        return "version_conflict"
    return None


def retrieve_for_answer(conn, question: str, *, embedder=None, top_k: int = 5,
                         lexical_k: int | None = None, embedding_k: int | None = None) -> dict:
    """Independent answer retrieval with eligibility filtering before ranking.

    Unlike ``retrieve_learning_knowledge`` (which surfaces every active row so
    a judge can compare and clarify), this entry point only ranks rows that
    may support an answer: active, trusted (official_source/user_confirmed),
    backed by an accepted supports source, and free of known model/version
    conflicts with the question.  Rows filtered out for model/version reasons
    are reported in ``topical_excluded`` so the answer service can distinguish
    "nothing covers this device" (unsupported) from "the model or version is
    missing/ambiguous" (needs_clarification).  Embedding failures leave lexical
    retrieval intact.
    """

    question = _text(question)
    top_k = max(1, min(int(top_k), 20))
    lexical_k = top_k if lexical_k is None else max(1, int(lexical_k))
    embedding_k = top_k if embedding_k is None else max(1, int(embedding_k))
    query_models = _scope_models(question)
    query_versions = _version_tokens(question)
    diagnostics: dict[str, Any] = {
        "query_models": query_models,
        "query_versions": query_versions,
        "eligible_ids": [],
        "topical_excluded": [],
        "topical_scopes": [],
        "lexical_only": True,
    }
    if not question:
        return {"candidates": [], "diagnostics": diagnostics}
    rows = _answer_eligible_rows(conn)
    # Defense in depth: the SQL gate above is authoritative (covered by the
    # PostgreSQL integration tests), but never rank a row that is not active
    # and trusted even if a future query edit lets one through.
    rows = [
        row for row in rows
        if row.get("active") and str(row.get("trust") or "") in ANSWER_TRUST_VALUES
    ]
    diagnostics["eligible_ids"] = [int(row["id"]) for row in rows]
    topical: list[dict] = []
    for row in rows:
        score = _lexical_score(question, row)
        if score <= 0:
            continue
        row_models, row_versions = _row_scope(row)
        reason = _conflict_reason(query_models, query_versions, row)
        if reason:
            diagnostics["topical_excluded"].append({
                "knowledge_id": int(row["id"]),
                "reason": reason,
                "scope_models": row_models,
                "scope_versions": row_versions,
            })
            continue
        topical.append({**row, "lexical_score": score, "embedding_score": None, "retrieval_sources": ["lexical"]})
    # Distinct model scopes among lexically topical rows (before exclusion):
    # the answer service uses this to detect a missing model that would
    # change the conclusion, without generalizing any single scope.
    seen_scopes: list[list[str]] = []
    for row in rows:
        if _lexical_score(question, row) <= 0:
            continue
        row_models, _row_versions = _row_scope(row)
        if row_models and row_models not in seen_scopes:
            seen_scopes.append(row_models)
    diagnostics["topical_scopes"] = seen_scopes
    topical.sort(key=lambda r: (_same_model(question, r), r["lexical_score"], -int(r["id"])), reverse=True)
    scored = {int(row["id"]): row for row in topical[:lexical_k]}
    query_vector = _query_embedding(embedder, question)
    diagnostics["lexical_only"] = query_vector is None
    _merge_embedding_hits(scored, rows, query_vector, embedding_k)
    # Keep embedding-only additions eligible too: they passed the SQL gate and
    # carry no lexical conflict signal to check beyond scope, which the merge
    # pool already satisfied by construction.  Re-check scope defensively.
    finalists = []
    for row in scored.values():
        if int(row["id"]) not in {int(item["id"]) for item in topical}:
            if _conflict_reason(query_models, query_versions, row):
                continue
        finalists.append(row)
    def rank(row):
        lexical_score = float(row.get("lexical_score") or 0)
        embedding_score = max(0.0, float(row.get("embedding_score") or 0))
        return (_same_model(question, row), lexical_score * .55 + embedding_score * .45, lexical_score, embedding_score, -int(row["id"]))
    candidates = sorted(finalists, key=rank, reverse=True)[:top_k]
    sources = _answer_sources(conn, [int(row["id"]) for row in candidates])
    for row in candidates:
        row["combined_score"] = float(float(row.get("lexical_score") or 0) * .55 + max(0.0, float(row.get("embedding_score") or 0)) * .45)
        row["sources"] = sources.get(int(row["id"]), [])
        row.pop("embedding", None)
    return {"candidates": candidates, "diagnostics": diagnostics}


def explicit_source_request(question: str) -> bool:
    """Deterministic trigger: the engineer asks to check the original."""

    folded = _text(question).casefold()
    return any(keyword in folded for keyword in SOURCE_REQUEST_KEYWORDS)


def high_risk_operation(question: str) -> bool:
    """Deterministic trigger: high-impact operations get a verification read."""

    folded = _text(question).casefold()
    return any(keyword in folded for keyword in HIGH_RISK_KEYWORDS)


def _version_applicability_satisfied(version: dict, query_models: list[str],
                                     query_versions: list[str]) -> bool:
    """Exclude versions whose stated scope contradicts the question.

    Unstated scope on either side never excludes: unknown is not a mismatch.
    """

    applicability = version.get("applicability") or {}
    if not isinstance(applicability, dict):
        return True
    for key, query_values in (("models", query_models), ("versions", query_versions)):
        stated = applicability.get(key) or []
        if not stated or not query_values:
            continue
        stated_keys = {_scope_key(str(item)) for item in stated if str(item).strip()}
        query_keys = {_scope_key(str(item)) for item in query_values if str(item).strip()}
        stated_keys.discard("")
        query_keys.discard("")
        if stated_keys and query_keys and stated_keys.isdisjoint(query_keys):
            return False
    return True


def retrieve_document_evidence(conn, question: str, *, top_k: int = 3,
                                document_version_id: int | None = None) -> dict:
    """Lexical fallback over qualified raw document blocks.

    Only raw block text is quoted -- never unreviewed LLM summaries.  A
    version qualifies with confirmed authenticity, a non-failed parse, and
    a scope that does not contradict the question.  Blocks under human
    review or failed states never auto-quote.  Whole blocks only: callers
    must refuse to silently truncate a block that does not fit.
    """

    question = _text(question)
    top_k = max(1, min(int(top_k), 5))
    diagnostics: dict[str, Any] = {
        "query_models": [], "query_versions": [],
        "scanned_version_ids": [], "excluded_versions": [],
        "candidate_block_ids": [],
    }
    if not question:
        return {"evidence": [], "diagnostics": diagnostics}
    query_models = _scope_models(question)
    query_versions = _version_tokens(question)
    diagnostics["query_models"] = query_models
    diagnostics["query_versions"] = query_versions
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id, v.document_key, v.version_label, v.title,
                   v.source_authenticity, v.status, v.applicability,
                   b.id AS block_id, b.block_key, b.page_no, b.slide_no,
                   b.block_type, b.section_path, b.processing_state,
                   r.content AS block_text
            FROM v2_document_blocks b
            JOIN v2_document_versions v ON v.id=b.version_id
            JOIN v2_raw_evidence r ON r.id=b.raw_evidence_id
            WHERE v.source_authenticity = ANY(%s)
              AND v.status = ANY(%s)
              AND b.processing_state = ANY(%s)
              AND r.content <> ''
            """,
            (list(FALLBACK_AUTHENTICITY), list(FALLBACK_VERSION_STATUSES),
             list(FALLBACK_BLOCK_STATES)),
        )
        rows = [dict(row) for row in cur.fetchall()]
    if document_version_id is not None:
        rows = [row for row in rows if int(row["id"]) == int(document_version_id)]
    versions: dict[int, dict] = {}
    for row in rows:
        versions.setdefault(int(row["id"]), {
            "id": int(row["id"]), "document_key": str(row.get("document_key") or ""),
            "version_label": str(row.get("version_label") or ""),
            "title": str(row.get("title") or ""),
            "source_authenticity": str(row.get("source_authenticity") or ""),
            "status": str(row.get("status") or ""),
            "applicability": row.get("applicability") or {},
        })
    diagnostics["scanned_version_ids"] = sorted(versions)
    scored = []
    for row in rows:
        version = versions[int(row["id"])]
        if not _version_applicability_satisfied(version, query_models, query_versions):
            continue
        section = " ".join(str(part) for part in (row.get("section_path") or []))
        score = _lexical_score(question, {
            "title": section, "content": str(row.get("block_text") or ""),
            "entity_name": "",
        })
        if score <= 0:
            continue
        if row.get("slide_no") is not None:
            locator = f"slide {int(row['slide_no'])}"
        elif row.get("page_no") is not None:
            locator = f"page {int(row['page_no'])}"
        else:
            locator = str(row.get("block_key") or "")
        scored.append((score, {
            "version_id": int(row["id"]),
            "document_key": version["document_key"],
            "version_label": version["version_label"],
            "source_authenticity": version["source_authenticity"],
            "block_id": int(row["block_id"]),
            "block_key": str(row.get("block_key") or ""),
            "locator": locator,
            "section_path": [str(part) for part in (row.get("section_path") or [])],
            "block_type": str(row.get("block_type") or ""),
            "text": str(row.get("block_text") or ""),
            "lexical_score": float(score),
        }))
    scored.sort(key=lambda pair: (pair[0], -pair[1]["block_id"]), reverse=True)
    evidence = [item for _, item in scored[:top_k]]
    diagnostics["candidate_block_ids"] = [item["block_id"] for item in evidence]
    diagnostics["excluded_versions"] = sorted(
        set(versions) - {item["version_id"] for _, item in scored}
    )
    return {"evidence": evidence, "diagnostics": diagnostics}
