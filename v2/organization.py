"""Small, evidence-backed organization helpers for V2.

The organization layer is intentionally separate from facts and the learning
state machine.  It only maintains exact-deduplicated entities and local
``belongs_to`` relations.  Callers own transactions; no helper commits.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from llm import parse_json_response


log = logging.getLogger("aihelper.v2.organization")


ENTITY_TYPES = frozenset({
    "brand", "category", "product_line", "series", "model", "concept",
})
RELATION_TYPES = frozenset({"belongs_to"})
TRUSTED_KNOWLEDGE = frozenset({"official_source", "user_confirmed"})
PROVENANCE_KINDS = frozenset({
    "official_source", "user_confirmed", "explicit_user_confirmation",
})
REVIEW_ACTIONS = frozenset({
    "NO_CHANGE", "CREATE_ENTITY", "CREATE_RELATION", "MOVE_RELATION", "UNCLEAR",
})
ORGANIZATION_PROPOSAL_KEYS = frozenset({
    "action", "subject_entity", "target_parent", "new_entity", "entity_type",
    "relation_type", "reason", "evidence_quote", "confidence",
})


class OrganizationError(ValueError):
    """Base error for invalid organization data or unsafe structural changes."""


class ProvenanceError(OrganizationError):
    """Raised when a relation lacks acceptable evidence."""


class CycleError(OrganizationError):
    """Raised when a new parent relation would create a cycle."""


def normalize_entity_name(name: str) -> str:
    """Return a stable exact-dedup key, without inventing aliases.

    Whitespace is ignored because formatting around model separators is not a
    meaningful identity distinction.  Other characters remain meaningful.
    """

    if not isinstance(name, str):
        raise TypeError("entity name must be text")
    value = unicodedata.normalize("NFKC", name).strip().casefold()
    value = value.replace("–", "-").replace("—", "-").replace("／", "/")
    return re.sub(r"\s+", "", value)


def _dict(row: Any) -> dict:
    return dict(row) if row is not None else {}


def _validate_entity_type(entity_type: str) -> str:
    value = str(entity_type or "").strip()
    if value not in ENTITY_TYPES:
        raise OrganizationError(f"unknown entity type: {value}")
    return value


def _validate_relation_type(relation_type: str) -> str:
    value = str(relation_type or "").strip()
    if value not in RELATION_TYPES:
        raise OrganizationError(f"unknown relation type: {value}")
    return value


def lookup_entity(conn, name: str, *, active_only: bool = False) -> dict | None:
    """Find an entity by the exact application normalization key."""

    normalized = normalize_entity_name(name)
    if not normalized:
        return None
    where_active = " AND active=TRUE" if active_only else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, normalized_name, entity_type, active,
                   created_at, updated_at
            FROM v2_entities
            WHERE normalized_name=%s{where_active}
            LIMIT 1
            """,
            (normalized,),
        )
        row = cur.fetchone()
    return _dict(row) or None


def get_or_create_entity(
    conn,
    name: str,
    *,
    entity_type: str = "concept",
    active: bool = True,
) -> dict:
    """Get an exact entity or create it; the caller owns the transaction."""

    display_name = str(name or "").strip()
    normalized = normalize_entity_name(display_name)
    if not normalized:
        raise OrganizationError("entity name cannot be empty")
    entity_type = _validate_entity_type(entity_type)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_entities(name, normalized_name, entity_type, active)
            VALUES(%s, %s, %s, %s)
            ON CONFLICT (normalized_name) DO UPDATE
              SET updated_at=v2_entities.updated_at
            RETURNING id, name, normalized_name, entity_type, active,
                      created_at, updated_at
            """,
            (display_name, normalized, entity_type, bool(active)),
        )
        row = cur.fetchone()
    if row is None:
        raise OrganizationError("entity insert did not return a row")
    return _dict(row)


def get_entity_by_id(conn, entity_id: int, *, active_only: bool = False) -> dict | None:
    where_active = " AND active=TRUE" if active_only else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, normalized_name, entity_type, active,
                   created_at, updated_at
            FROM v2_entities
            WHERE id=%s{where_active}
            """,
            (int(entity_id),),
        )
        row = cur.fetchone()
    return _dict(row) or None


def _entity_ref(conn, value: int | str | dict, *, entity_type: str = "concept") -> dict:
    if isinstance(value, dict):
        if value.get("id") is not None:
            entity = get_entity_by_id(conn, int(value["id"]))
            if entity is None:
                raise OrganizationError(f"entity {value['id']} was not found")
            return entity
        entity_type = value.get("entity_type", entity_type)
        display_name = str(value.get("name", ""))
        requested_type = _validate_entity_type(entity_type)
        existing = lookup_entity(conn, display_name)
        if existing is not None:
            if existing.get("entity_type") != requested_type and requested_type != "concept":
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE v2_entities SET entity_type=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
                        (requested_type, int(existing["id"])),
                    )
                existing["entity_type"] = requested_type
            return existing
        return get_or_create_entity(conn, display_name, entity_type=requested_type)
    if isinstance(value, int):
        entity = get_entity_by_id(conn, value)
        if entity is None:
            raise OrganizationError(f"entity {value} was not found")
        return entity
    existing = lookup_entity(conn, str(value))
    if existing:
        return existing
    return get_or_create_entity(conn, str(value), entity_type=entity_type)


def _knowledge_provenance(conn, source_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, trust, active
            FROM v2_knowledge
            WHERE id=%s
            """,
            (int(source_id),),
        )
        row = cur.fetchone()
    if not row:
        raise ProvenanceError(f"source Knowledge {source_id} was not found")
    result = _dict(row)
    if result.get("trust") not in TRUSTED_KNOWLEDGE:
        raise ProvenanceError(
            "formal organization relations require official_source or user_confirmed Knowledge"
        )
    if result.get("active") is False:
        raise ProvenanceError("inactive Knowledge cannot be relation provenance")
    return result


def _validate_provenance(
    conn,
    *,
    source_id: int | None,
    provenance: str | None,
    provenance_kind: str | None,
) -> tuple[int | None, str, str]:
    kind = str(provenance_kind or "").strip()
    explanation = str(provenance or "").strip()
    if kind not in PROVENANCE_KINDS:
        raise ProvenanceError(f"unknown provenance kind: {kind}")
    if not explanation:
        raise ProvenanceError("formal organization relations require provenance text")
    if source_id is not None:
        source = _knowledge_provenance(conn, int(source_id))
        if kind == "explicit_user_confirmation" and source.get("trust") != "user_confirmed":
            raise ProvenanceError("explicit_user_confirmation requires user_confirmed Knowledge")
        if kind == "official_source" and source.get("trust") != "official_source":
            raise ProvenanceError("official_source provenance requires official_source Knowledge")
        if kind == "user_confirmed" and source.get("trust") not in TRUSTED_KNOWLEDGE:
            raise ProvenanceError("user_confirmed provenance requires trusted Knowledge")
    elif kind != "explicit_user_confirmation":
        raise ProvenanceError("a relation without source_id must be an explicit user confirmation")
    return (int(source_id) if source_id is not None else None, explanation, kind)


def get_relation(
    conn,
    parent_entity_id: int,
    child_entity_id: int,
    *,
    relation_type: str = "belongs_to",
    active_only: bool = True,
) -> dict | None:
    relation_type = _validate_relation_type(relation_type)
    active_clause = " AND active=TRUE" if active_only else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, parent_entity_id, child_entity_id, relation_type,
                   source_id, provenance, provenance_kind, active,
                   created_at, updated_at
            FROM v2_entity_relations
            WHERE parent_entity_id=%s AND child_entity_id=%s
              AND relation_type=%s{active_clause}
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(parent_entity_id), int(child_entity_id), relation_type),
        )
        row = cur.fetchone()
    return _dict(row) or None


def lookup_relation(
    conn,
    parent_entity_id: int,
    child_entity_id: int,
    *,
    relation_type: str = "belongs_to",
    active_only: bool = True,
) -> dict | None:
    """Named lookup alias kept beside ``get_relation`` for callers reading SQL."""

    return get_relation(
        conn,
        parent_entity_id,
        child_entity_id,
        relation_type=relation_type,
        active_only=active_only,
    )


def create_relation(
    conn,
    parent_entity_id: int,
    child_entity_id: int,
    *,
    source_id: int | None,
    provenance: str,
    provenance_kind: str,
    relation_type: str = "belongs_to",
) -> dict:
    """Create an active relation after trust, provenance, and cycle checks."""

    relation_type = _validate_relation_type(relation_type)
    parent_id, child_id = int(parent_entity_id), int(child_entity_id)
    if parent_id == child_id:
        raise CycleError("an entity cannot belong to itself")
    if get_entity_by_id(conn, parent_id, active_only=True) is None:
        raise OrganizationError(f"active parent entity {parent_id} was not found")
    if get_entity_by_id(conn, child_id, active_only=True) is None:
        raise OrganizationError(f"active child entity {child_id} was not found")
    source_id, provenance, provenance_kind = _validate_provenance(
        conn,
        source_id=source_id,
        provenance=provenance,
        provenance_kind=provenance_kind,
    )
    if would_create_cycle(conn, parent_id, child_id, relation_type=relation_type):
        raise CycleError("relation would create an organization cycle")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_entity_relations(
                parent_entity_id, child_entity_id, relation_type,
                source_id, provenance, provenance_kind, active
            )
            VALUES(%s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (parent_entity_id, child_entity_id, relation_type)
              WHERE active=TRUE DO UPDATE
                SET updated_at=v2_entity_relations.updated_at
            RETURNING id, parent_entity_id, child_entity_id, relation_type,
                      source_id, provenance, provenance_kind, active,
                      created_at, updated_at
            """,
            (parent_id, child_id, relation_type, source_id, provenance, provenance_kind),
        )
        row = cur.fetchone()
    if row is None:
        raise OrganizationError("relation insert did not return a row")
    return _dict(row)


def would_create_cycle(
    conn,
    parent_entity_id: int,
    child_entity_id: int,
    *,
    relation_type: str = "belongs_to",
) -> bool:
    """Check whether parent -> child closes an existing parent chain."""

    relation_type = _validate_relation_type(relation_type)
    parent_id, child_id = int(parent_entity_id), int(child_entity_id)
    if parent_id == child_id:
        return True
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE ancestors(entity_id, path) AS (
              SELECT %s::BIGINT, ARRAY[%s::BIGINT]
              UNION ALL
              SELECT r.parent_entity_id, a.path || r.parent_entity_id
              FROM ancestors a
              JOIN v2_entity_relations r
                ON r.child_entity_id=a.entity_id
               AND r.relation_type=%s AND r.active=TRUE
              WHERE NOT r.parent_entity_id = ANY(a.path)
            )
            SELECT EXISTS(
              SELECT 1 FROM ancestors WHERE entity_id=%s
            ) AS would_cycle
            """,
            (parent_id, parent_id, relation_type, child_id),
        )
        row = cur.fetchone()
    return bool(_dict(row).get("would_cycle"))


def move_relation(
    conn,
    child_entity_id: int,
    new_parent_entity_id: int,
    *,
    source_id: int | None,
    provenance: str,
    provenance_kind: str,
    relation_type: str = "belongs_to",
) -> dict:
    """Retire the current parent relation and add a new auditable one."""

    relation_type = _validate_relation_type(relation_type)
    child_id, new_parent_id = int(child_entity_id), int(new_parent_entity_id)
    if get_entity_by_id(conn, new_parent_id, active_only=True) is None:
        raise OrganizationError(f"active parent entity {new_parent_id} was not found")
    if get_entity_by_id(conn, child_id, active_only=True) is None:
        raise OrganizationError(f"active child entity {child_id} was not found")
    if would_create_cycle(conn, new_parent_id, child_id, relation_type=relation_type):
        raise CycleError("relation move would create an organization cycle")
    source_id, provenance, provenance_kind = _validate_provenance(
        conn,
        source_id=source_id,
        provenance=provenance,
        provenance_kind=provenance_kind,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, parent_entity_id, child_entity_id, relation_type,
                   source_id, provenance, provenance_kind, active,
                   created_at, updated_at
            FROM v2_entity_relations
            WHERE child_entity_id=%s AND relation_type=%s AND active=TRUE
            ORDER BY id DESC LIMIT 1
            """,
            (child_id, relation_type),
        )
        current = _dict(cur.fetchone()) or None
    if current and int(current["parent_entity_id"]) == new_parent_id:
        return current
    if current:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE v2_entity_relations
                SET active=FALSE, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s AND active=TRUE
                """,
                (int(current["id"]),),
            )
    return create_relation(
        conn,
        new_parent_id,
        child_id,
        source_id=source_id,
        provenance=provenance,
        provenance_kind=provenance_kind,
        relation_type=relation_type,
    )


def list_local_context(
    conn,
    entity_id: int,
    *,
    max_ancestor_depth: int = 5,
    max_siblings: int = 8,
    max_children: int = 8,
) -> dict:
    """Return only one entity's immediate neighborhood, never the full tree."""

    entity = get_entity_by_id(conn, entity_id)
    if entity is None:
        return {
            "entity": None, "current_parent": None, "siblings": [],
            "nearby_ancestors": [], "children": [],
        }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.name, p.normalized_name, p.entity_type, p.active,
                   p.created_at, p.updated_at
            FROM v2_entity_relations r
            JOIN v2_entities p ON p.id=r.parent_entity_id AND p.active=TRUE
            WHERE r.child_entity_id=%s AND r.relation_type='belongs_to'
              AND r.active=TRUE
            ORDER BY r.id DESC LIMIT 1
            """,
            (int(entity_id),),
        )
        parent = _dict(cur.fetchone()) or None
        parent_id = parent["id"] if parent else None

        if parent_id is None:
            siblings = []
        else:
            cur.execute(
                """
                SELECT e.id, e.name, e.normalized_name, e.entity_type, e.active,
                       e.created_at, e.updated_at
                FROM v2_entity_relations r
                JOIN v2_entities e ON e.id=r.child_entity_id AND e.active=TRUE
                WHERE r.parent_entity_id=%s AND r.relation_type='belongs_to'
                  AND r.active=TRUE AND e.id<>%s
                ORDER BY e.name, e.id
                LIMIT %s
                """,
                (int(parent_id), int(entity_id), max(1, int(max_siblings))),
            )
            siblings = [_dict(row) for row in cur.fetchall()]

        cur.execute(
            """
            WITH RECURSIVE ancestors AS (
              SELECT p.id, p.name, p.normalized_name, p.entity_type, p.active,
                     p.created_at, p.updated_at, 1 AS depth, ARRAY[p.id] AS path
              FROM v2_entity_relations r
              JOIN v2_entities p ON p.id=r.parent_entity_id AND p.active=TRUE
              WHERE r.child_entity_id=%s AND r.relation_type='belongs_to'
                AND r.active=TRUE
              UNION ALL
              SELECT p.id, p.name, p.normalized_name, p.entity_type, p.active,
                     p.created_at, p.updated_at, a.depth + 1, a.path || p.id
              FROM ancestors a
              JOIN v2_entity_relations r
                ON r.child_entity_id=a.id AND r.relation_type='belongs_to'
               AND r.active=TRUE
              JOIN v2_entities p ON p.id=r.parent_entity_id AND p.active=TRUE
              WHERE a.depth < %s AND NOT p.id = ANY(a.path)
            )
            SELECT id, name, normalized_name, entity_type, active,
                   created_at, updated_at, depth
            FROM ancestors ORDER BY depth
            """,
            (int(entity_id), max(1, int(max_ancestor_depth))),
        )
        ancestors = [_dict(row) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT e.id, e.name, e.normalized_name, e.entity_type, e.active,
                   e.created_at, e.updated_at
            FROM v2_entity_relations r
            JOIN v2_entities e ON e.id=r.child_entity_id AND e.active=TRUE
            WHERE r.parent_entity_id=%s AND r.relation_type='belongs_to'
              AND r.active=TRUE
            ORDER BY e.name, e.id
            LIMIT %s
            """,
            (int(entity_id), max(1, int(max_children))),
        )
        children = [_dict(row) for row in cur.fetchall()]
    return {
        "entity": entity,
        "current_parent": parent,
        "siblings": siblings,
        "nearby_ancestors": ancestors,
        "children": children,
    }


def link_knowledge_to_entity(conn, knowledge_id: int, entity_id: int) -> None:
    """Add the nullable organization link without changing the fact itself."""

    if get_entity_by_id(conn, entity_id) is None:
        raise OrganizationError(f"entity {entity_id} was not found")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET entity_id=%s, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND (entity_id IS NULL OR entity_id=%s)
            """,
            (int(entity_id), int(knowledge_id), int(entity_id)),
        )


def backfill_knowledge_entity_links(conn, *, limit: int = 100) -> int:
    """Create/link only explicit ``entity_name`` values; never infer parents."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, entity_name
            FROM v2_knowledge
            WHERE entity_id IS NULL AND char_length(btrim(entity_name)) > 0
            ORDER BY id LIMIT %s
            """,
            (max(1, int(limit)),),
        )
        candidates = [_dict(row) for row in cur.fetchall()]
    linked = 0
    for candidate in candidates:
        entity = lookup_entity(conn, candidate.get("entity_name", ""))
        if entity is None:
            name = str(candidate.get("entity_name") or "").strip()
            entity_type = "model" if re.search(r"\d", name) else "concept"
            entity = get_or_create_entity(conn, name, entity_type=entity_type)
        link_knowledge_to_entity(conn, int(candidate["id"]), int(entity["id"]))
        linked += 1
    return linked


def _chain_items(chain: Any) -> list[dict]:
    if isinstance(chain, dict):
        chain = chain.get("entities") or chain.get("chain") or []
    if not isinstance(chain, (list, tuple)):
        return []
    result = []
    for item in chain:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            item_type = item.get("entity_type", "concept")
        else:
            name, item_type = str(item).strip(), "concept"
        if not name:
            return []
        result.append({"name": name, "entity_type": _validate_entity_type(item_type)})
    return result


def _entity_type_for_chain_position(name: str, position: int, length: int, text: str = "") -> str:
    """Classify a plainly stated chain without using product-specific names."""

    if re.search(r"(?:系列|series)$", name, re.IGNORECASE):
        return "series"
    if re.search(r"(?:产品线|product\s+line)$", name, re.IGNORECASE):
        return "product_line"
    if position == 0 and length > 1:
        # Only owner/role grammar gives the outer node a brand hint.  A bare
        # ``X belongs to Y`` does not tell us whether Y is a brand or a
        # category, so leave that distinction unforced.
        return "brand" if re.search(r"的|中的|['’]s", text, re.IGNORECASE) else "category"
    if position == length - 1:
        return "model" if re.search(r"\d", name) else "concept"
    return "category"


def _clean_organization_label(value: str) -> str:
    value = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", value)
    value = re.sub(r"([\u4e00-\u9fff])([A-Za-z])", r"\1 \2", value)
    value = re.sub(r"^[\s:：\-]+|[\s。；，,;]+$", "", value.strip())
    if value.endswith("产品") and not value.endswith("产品线"):
        value = value[:-2].rstrip()
    return value


def _parse_parent_phrase(phrase: str) -> list[str]:
    """Parse only explicit parent wording from one relationship clause.

    The parser deliberately needs a marker such as ``的``, ``'s`` or ``中
    的``.  A bare co-occurrence of two product-looking names is not enough to
    create a relation.
    """

    phrase = _clean_organization_label(phrase)
    phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase, flags=re.IGNORECASE)
    if not phrase:
        return []

    if "中的" in phrase:
        outer, leaf = phrase.split("中的", 1)
        outer = re.sub(r"^[\s:：\-]+|[\s。；，,;]+$", "", outer.strip())
        leaf = _clean_organization_label(leaf)
        if not outer or not leaf:
            return []
        # ``Brand category 中的 leaf`` explicitly names all three levels.
        match = re.match(r"^([A-Za-z][A-Za-z0-9._-]*)(.+)$", outer)
        if match:
            middle = re.sub(r"^[\s:：\-]+|[\s。；，,;]+$", "", match.group(2).strip())
            return [match.group(1), middle, leaf] if middle else []
        return [outer, leaf]

    if "的" in phrase:
        outer, leaf = phrase.split("的", 1)
        outer = _clean_organization_label(outer)
        leaf = _clean_organization_label(leaf)
        return [outer, leaf] if outer and leaf else []

    possessive = re.match(r"^(.+?)['’]s(.+)$", phrase, flags=re.IGNORECASE)
    if possessive:
        outer = _clean_organization_label(possessive.group(1))
        leaf = _clean_organization_label(possessive.group(2))
        return [outer, leaf] if outer and leaf else []

    if re.search(r"(?:系列|series)$", phrase, re.IGNORECASE):
        return [_clean_organization_label(phrase)]

    # English/Chinese mixed product wording may omit 的 but still explicitly
    # put a named brand before the category, e.g. ``Brand 门禁控制器``.
    leading_brand = re.match(r"^([A-Za-z][A-Za-z0-9._-]*)\s+([\u4e00-\u9fff].+)$", phrase)
    if leading_brand:
        outer = leading_brand.group(1)
        leaf = _clean_organization_label(leading_brand.group(2))
        if leaf and not re.search(r"\b(?:is|an|a|the)\b", leaf, re.IGNORECASE):
            return [outer, leaf]

    return []


def _explicit_parent_chain(subject: str, text: str) -> list[str]:
    """Return a chain only when the source explicitly states its edges."""

    display_text = " ".join(text.splitlines())
    compact = re.sub(r"\s+", "", display_text)
    compact_subject = re.sub(r"\s+", "", subject)
    if not compact_subject or compact_subject.casefold() not in compact.casefold():
        return []

    relation = r"(?P<verb>is\s*part\s*of|belongs\s*to|是|为|属于|归属于|归到|is)"
    escaped_subject = re.escape(subject.strip())
    direct = re.search(
        escaped_subject + r"\s*" + relation + r"\s*(?P<parent>[^，。；,.;]+)",
        display_text,
        re.IGNORECASE,
    )
    if not direct:
        direct = re.search(
            re.escape(compact_subject) + relation + r"(?P<parent>[^，。；,.;]+)",
            compact,
            re.IGNORECASE,
        )
    if not direct:
        return []
    direct_parents = _parse_parent_phrase(direct.group("parent"))
    verb_key = re.sub(r"\s+", "", direct.group("verb")).casefold()
    if not direct_parents and verb_key in {"属于", "归属于", "归到", "belongsto", "ispartof"}:
        bare_parent = _clean_organization_label(direct.group("parent"))
        if bare_parent:
            direct_parents = [bare_parent]
    if not direct_parents:
        return []

    # A direct series statement may be followed by another explicit clause
    # that places that exact series inside additional named levels.  Reuse
    # those words only; never manufacture a broader hierarchy.
    if len(direct_parents) == 1:
        series_name = direct_parents[0]
        series_relation = re.search(
            re.escape(series_name) + r"\s*" + relation
            + r"\s*(?P<parent>[^，。；,.;]+)",
            display_text,
            re.IGNORECASE,
        )
        if not series_relation:
            series_relation = re.search(
                re.escape(re.sub(r"\s+", "", series_name)) + relation
                + r"(?P<parent>[^，。；,.;]+)",
                compact,
                re.IGNORECASE,
            )
        if series_relation:
            outer_parents = _parse_parent_phrase(series_relation.group("parent"))
            if outer_parents:
                return outer_parents + [series_name, subject.strip()]
        return [series_name, subject.strip()]

    return direct_parents + [subject.strip()]


def extract_explicit_chain(knowledge: dict) -> list[dict]:
    """Read only a structured or plainly stated, evidence-backed chain.

    Ordinary fact prose and model-like names are intentionally ignored.  The
    natural-language shortcut recognizes relationship grammar, not product
    names.  It never expands a series or invents a category from an identifier.
    """

    if not isinstance(knowledge, dict):
        return []
    structured = _chain_items(
        knowledge.get("explicit_chain") or knowledge.get("organization_chain")
    )
    if structured:
        return structured

    subject = str(knowledge.get("entity_name") or "").strip()
    if not subject:
        return []
    text = "\n".join(str(knowledge.get(key) or "") for key in ("title", "content"))
    names = _explicit_parent_chain(subject, text)
    if not names:
        return []
    return [
        {"name": name, "entity_type": _entity_type_for_chain_position(name, index, len(names), text)}
        for index, name in enumerate(names)
    ]


def local_organization_review(
    conn,
    knowledge: dict,
    *,
    explicit_chain: Any = None,
    proposed_parent: int | str | dict | None = None,
    parent_entity_type: str = "concept",
    provenance: str | None = None,
    provenance_kind: str | None = None,
    source_id: int | None = None,
) -> dict:
    """Apply one conservative local review.

    ``explicit_chain`` is deliberately structured evidence in parent-to-child
    order.  Without it, this function does not turn names or prose into a
    hierarchy.  A provisional/conflicted Knowledge item therefore returns
    ``NO_CHANGE`` and cannot alter the organization layer.
    """

    if not isinstance(knowledge, dict):
        raise TypeError("knowledge must be a mapping")
    action = "NO_CHANGE"
    result: dict = {"action": action, "entity": None, "relations": []}
    trust = knowledge.get("trust")
    if trust not in TRUSTED_KNOWLEDGE:
        result["reason"] = "knowledge is not trusted structural evidence"
        return result

    source_id = int(source_id if source_id is not None else knowledge.get("id")) if (source_id is not None or knowledge.get("id") is not None) else None
    if source_id is None:
        result["action"] = "UNCLEAR"
        result["reason"] = "trusted organization evidence needs a Knowledge source id"
        return result
    if provenance is None:
        provenance = f"Knowledge #{source_id} explicitly records this organization"
    if provenance_kind is None:
        provenance_kind = trust

    subject_value = knowledge.get("entity_id") or knowledge.get("entity_name")
    if not subject_value:
        result["action"] = "UNCLEAR"
        result["reason"] = "the Knowledge item has no explicit subject entity"
        return result
    if isinstance(subject_value, int):
        subject_existing = get_entity_by_id(conn, subject_value)
    else:
        subject_existing = lookup_entity(conn, str(subject_value))
    subject = _entity_ref(conn, subject_value, entity_type="model")
    subject_created = subject_existing is None
    result["entity"] = subject
    if knowledge.get("id") is not None:
        link_knowledge_to_entity(conn, int(knowledge["id"]), int(subject["id"]))

    chain = _chain_items(explicit_chain)
    if chain:
        subject_positions = [
            index for index, item in enumerate(chain)
            if normalize_entity_name(item["name"]) == subject["normalized_name"]
        ]
        if not subject_positions:
            return {"action": "UNCLEAR", "entity": subject, "relations": [], "reason": "explicit chain subject does not match Knowledge"}
        if subject_positions != [len(chain) - 1]:
            return {"action": "UNCLEAR", "entity": subject, "relations": [], "reason": "the reviewed entity must be the leaf of the explicit chain"}
        entities = [
            _entity_ref(conn, item["name"], entity_type=item["entity_type"])
            for item in chain
        ]
        if not any(int(entity["id"]) == int(subject["id"]) for entity in entities):
            return {"action": "UNCLEAR", "entity": subject, "relations": [], "reason": "explicit chain subject does not match existing entity"}
        # If a new explicit layer is inserted between the current parent and
        # the subject (for example, a newly evidenced series), retain the
        # existing parent above that layer.  This is a local tree insertion,
        # not a guessed relation: the old parent is already confirmed and the
        # new layer is present in the current evidence.
        first_parent = entities[-2]
        current_subject_context = list_local_context(conn, int(subject["id"]))
        old_subject_parent = (current_subject_context.get("current_parent") or {}).get("id")
        chain_ids = {int(entity["id"]) for entity in entities}
        if old_subject_parent is not None and int(old_subject_parent) not in chain_ids:
            first_parent_context = list_local_context(conn, int(first_parent["id"]))
            first_parent_current = (first_parent_context.get("current_parent") or {}).get("id")
            if first_parent_current is not None and int(first_parent_current) != int(old_subject_parent):
                return {
                    "action": "UNCLEAR",
                    "entity": subject,
                    "relations": [],
                    "reason": "explicit layer conflicts with the existing local parent structure",
                }
            if first_parent_current is None and int(first_parent["id"]) != int(old_subject_parent):
                bridge = create_relation(
                    conn,
                    int(old_subject_parent),
                    int(first_parent["id"]),
                    source_id=source_id,
                    provenance=provenance + "; retained the existing local parent above the explicit layer",
                    provenance_kind=provenance_kind,
                )
                result["action"] = "CREATE_RELATION"
                result["relations"].append(bridge)
        for parent, child in zip(entities, entities[1:]):
            current = get_relation(conn, int(parent["id"]), int(child["id"]))
            if current:
                continue
            old_parent = None
            context = list_local_context(conn, int(child["id"]))
            if context.get("current_parent"):
                old_parent = int(context["current_parent"]["id"])
            if old_parent is not None and old_parent != int(parent["id"]):
                relation = move_relation(
                    conn, int(child["id"]), int(parent["id"]),
                    source_id=source_id, provenance=provenance,
                    provenance_kind=provenance_kind,
                )
                result["action"] = "MOVE_RELATION"
            else:
                relation = create_relation(
                    conn, int(parent["id"]), int(child["id"]),
                    source_id=source_id, provenance=provenance,
                    provenance_kind=provenance_kind,
                )
                if result["action"] == "NO_CHANGE":
                    result["action"] = "CREATE_RELATION"
            result["relations"].append(relation)
        return result

    if proposed_parent is None:
        if subject_created:
            result["action"] = "CREATE_ENTITY"
        return result
    parent = _entity_ref(conn, proposed_parent, entity_type=parent_entity_type)
    relation = get_relation(conn, int(parent["id"]), int(subject["id"]))
    if relation:
        return result
    context = list_local_context(conn, int(subject["id"]))
    if context.get("current_parent"):
        relation = move_relation(
            conn, int(subject["id"]), int(parent["id"]),
            source_id=source_id, provenance=provenance,
            provenance_kind=provenance_kind,
        )
        result["action"] = "MOVE_RELATION"
    else:
        relation = create_relation(
            conn, int(parent["id"]), int(subject["id"]),
            source_id=source_id, provenance=provenance,
            provenance_kind=provenance_kind,
        )
        result["action"] = "CREATE_RELATION"
    result["relations"].append(relation)
    return result


ORGANIZATION_SYSTEM_PROMPT = """
你是产品知识助理，只负责一次很小范围的组织位置建议。你收到的输入只有：
当前已确认 Knowledge、当前实体、当前 parent/少量 ancestors、少量 siblings 和少量候选实体。
不要使用训练知识，不要补全行业分类，不要根据型号或名称猜 parent，不要创建未出现在证据中的中间层。

只返回严格 JSON 对象，键只能是：
{"action":"NO_CHANGE|CREATE_ENTITY|CREATE_RELATION|MOVE_RELATION|UNCLEAR",
 "subject_entity":"当前实体名称",
 "target_parent":"parent 名称或 null",
 "new_entity":null,
 "entity_type":"brand|category|product_line|series|model|concept 或 null",
 "relation_type":"belongs_to",
 "reason":"简短原因",
 "evidence_quote":"来自当前 Knowledge 的连续原文，或空字符串",
 "confidence":"explicit|ambiguous"}

只有当前 Knowledge 明确表达 child 属于 parent 时才建议 CREATE_RELATION 或 MOVE_RELATION。
关系类型永远只能是 belongs_to。只有证据中明确出现的实体才能放入 new_entity；不要发明中间层或把单个型号推广到系列。
provisional 或 conflicted 不会进入此调用；即使看到不确定措辞也只能返回 UNCLEAR。
大多数情况返回 NO_CHANGE。LLM 只提出建议，不能返回 SQL，也不能执行数据库操作。
""".strip()


class OrganizationProposalError(OrganizationError):
    """Raised when an LLM organization proposal is not safe to consider."""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _entity_snapshot(entity: dict | None) -> dict | None:
    if not entity:
        return None
    return {
        "id": int(entity["id"]),
        "name": str(entity.get("name") or ""),
        "entity_type": str(entity.get("entity_type") or "concept"),
        "active": bool(entity.get("active", True)),
    }


def _relation_snapshot(relation: dict | None) -> dict | None:
    if not relation:
        return None
    return {
        "id": int(relation["id"]),
        "parent_entity_id": int(relation["parent_entity_id"]),
        "child_entity_id": int(relation["child_entity_id"]),
        "relation_type": str(relation.get("relation_type") or "belongs_to"),
        "source_id": relation.get("source_id"),
        "provenance": str(relation.get("provenance") or "")[:1000],
        "provenance_kind": str(relation.get("provenance_kind") or ""),
        "active": bool(relation.get("active", True)),
    }


def _search_terms_for_context(knowledge: Mapping[str, Any]) -> list[str]:
    text = " ".join(str(knowledge.get(key) or "") for key in ("entity_name", "title", "content"))
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        key = normalize_entity_name(value) if value else ""
        if key and len(key) >= 2 and key not in seen:
            seen.add(key)
            terms.append(value)

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/()\-]*|[\u4e00-\u9fff]{2,}", text):
        add(token)
        if re.search(r"[\u4e00-\u9fff]", token):
            add(token.lstrip("的其是为属于归到和与").rstrip("产品"))
    return terms[:8]


def _lexical_context_candidates(
    conn,
    knowledge: Mapping[str, Any],
    *,
    exclude_entity_id: int | None,
    limit: int = 8,
) -> list[dict]:
    """Find a few name matches; this query is never a full entity dump."""

    terms = _search_terms_for_context(knowledge)
    if not terms:
        return []
    clauses: list[str] = []
    params: list[str | int] = []
    for term in terms:
        pattern = f"%{term}%"
        clauses.append("(normalized_name ILIKE %s OR name ILIKE %s)")
        params.extend((pattern, pattern))
    sql = f"""
        SELECT id, name, normalized_name, entity_type, active,
               created_at, updated_at
        FROM v2_entities
        WHERE active=TRUE AND ({' OR '.join(clauses)})
        ORDER BY lower(name), id
        LIMIT %s
    """
    params.append(max(1, int(limit)))
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = [_dict(row) for row in cur.fetchall()]
    except Exception:
        # Candidate retrieval is an optimization.  A missing optional query
        # must not turn a durable Knowledge confirmation into a failure.
        log.debug("organization local candidate lookup failed", exc_info=True)
        return []
    return [
        _entity_snapshot(row)
        for row in rows
        if exclude_entity_id is None or int(row["id"]) != int(exclude_entity_id)
    ][:max(1, int(limit))]


def build_local_organization_context(
    conn,
    knowledge: Mapping[str, Any],
    *,
    max_siblings: int = 8,
    max_candidates: int = 8,
) -> dict:
    """Build the bounded neighborhood sent to the organization model."""

    entity_id = knowledge.get("entity_id")
    entity = get_entity_by_id(conn, int(entity_id)) if entity_id is not None else None
    if entity is None and knowledge.get("entity_name"):
        entity = lookup_entity(conn, str(knowledge["entity_name"]))
    local = list_local_context(conn, int(entity["id"])) if entity else {
        "entity": None, "current_parent": None, "siblings": [],
        "nearby_ancestors": [], "children": [],
    }
    parent_relation = None
    parent = local.get("current_parent")
    if parent:
        try:
            parent_relation = get_relation(conn, int(parent["id"]), int(entity["id"]))
        except Exception:
            log.debug("organization parent provenance lookup failed", exc_info=True)
    current_parent = None
    if parent:
        current_parent = {
            "entity": _entity_snapshot(parent),
            "relation": _relation_snapshot(parent_relation),
        }
    nearby_relations: list[dict] = []
    local_ids = {
        int(item["id"])
        for item in ([entity] if entity else [])
        + ([parent] if parent else [])
        + list(local.get("siblings") or [])
        + list(local.get("nearby_ancestors") or [])[:2]
    }
    if local_ids:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, parent_entity_id, child_entity_id, relation_type,
                           source_id, provenance, provenance_kind, active,
                           created_at, updated_at
                    FROM v2_entity_relations
                    WHERE active=TRUE
                      AND (parent_entity_id = ANY(%s) OR child_entity_id = ANY(%s))
                    ORDER BY id DESC LIMIT 12
                    """,
                    (list(local_ids), list(local_ids)),
                )
                nearby_relations = [_dict(row) for row in cur.fetchall()]
        except Exception:
            log.debug("organization local provenance lookup failed", exc_info=True)
    local_names = {
        int(item["id"]): str(item.get("name") or "")
        for item in ([entity] if entity else [])
        + ([parent] if parent else [])
        + list(local.get("siblings") or [])
        + list(local.get("nearby_ancestors") or [])
    }
    nearby_relations = [
        dict(
            _relation_snapshot(relation) or {},
            parent_name=local_names.get(int(relation["parent_entity_id"]), ""),
            child_name=local_names.get(int(relation["child_entity_id"]), ""),
        )
        for relation in nearby_relations
    ]
    current_knowledge = {
        key: knowledge.get(key)
        for key in ("id", "title", "content", "entity_name", "trust")
        if key in knowledge
    }
    return {
        "current_knowledge": current_knowledge,
        "current_entity": _entity_snapshot(entity),
        "current_parent": current_parent,
        "nearby_ancestors": [
            _entity_snapshot(item) for item in (local.get("nearby_ancestors") or [])[:2]
        ],
        "siblings": [
            _entity_snapshot(item) for item in (local.get("siblings") or [])[:max(1, int(max_siblings))]
        ],
        "candidate_entities": _lexical_context_candidates(
            conn,
            knowledge,
            exclude_entity_id=int(entity["id"]) if entity else None,
            limit=max_candidates,
        ),
        "nearby_relation_provenance": nearby_relations,
    }


def validate_organization_proposal(raw: Any) -> dict:
    """Validate only the model's JSON shape; DB/evidence checks happen later."""

    if not isinstance(raw, Mapping):
        raise OrganizationProposalError("organization proposal must be a JSON object")
    unknown = set(raw) - set(ORGANIZATION_PROPOSAL_KEYS)
    if unknown:
        raise OrganizationProposalError(f"organization proposal has unknown keys: {sorted(unknown)}")
    action = str(raw.get("action") or "").strip()
    if action not in REVIEW_ACTIONS:
        raise OrganizationProposalError(f"unknown organization action: {action}")
    subject = str(raw.get("subject_entity") or "").strip()
    target = raw.get("target_parent")
    target = str(target).strip() if target is not None else None
    if target == "":
        target = None
    reason = str(raw.get("reason") or "").strip()[:1000]
    evidence_quote = str(raw.get("evidence_quote") or "").strip()[:4000]
    confidence = str(raw.get("confidence") or "").strip()
    if confidence not in {"explicit", "ambiguous"}:
        raise OrganizationProposalError("organization proposal confidence is invalid")
    relation_type = str(raw.get("relation_type") or "belongs_to").strip()
    _validate_relation_type(relation_type)
    entity_type = raw.get("entity_type")
    if entity_type is not None:
        entity_type = _validate_entity_type(str(entity_type))
    new_entity = raw.get("new_entity")
    if isinstance(new_entity, str):
        new_entity = {"name": new_entity}
    if new_entity is not None:
        if not isinstance(new_entity, Mapping):
            raise OrganizationProposalError("new_entity must be an object or null")
        new_name = str(new_entity.get("name") or "").strip()
        if not new_name:
            raise OrganizationProposalError("new_entity.name is required")
        new_type = new_entity.get("entity_type", entity_type or "concept")
        new_entity = {
            "name": new_name,
            "entity_type": _validate_entity_type(str(new_type)),
        }
    if action in {"CREATE_RELATION", "MOVE_RELATION"} and (not subject or not target):
        raise OrganizationProposalError("relation proposal requires subject_entity and target_parent")
    if action == "CREATE_ENTITY" and not subject and not new_entity:
        raise OrganizationProposalError("entity proposal requires a subject or new_entity")
    if action not in {"NO_CHANGE", "UNCLEAR"}:
        if not evidence_quote:
            raise OrganizationProposalError("structural proposal requires evidence_quote")
        if confidence != "explicit":
            raise OrganizationProposalError("ambiguous evidence cannot change organization")
        if not reason:
            raise OrganizationProposalError("structural proposal requires a reason")
    return {
        "action": action,
        "subject_entity": subject,
        "target_parent": target,
        "new_entity": new_entity,
        "entity_type": entity_type,
        "relation_type": relation_type,
        "reason": reason,
        "evidence_quote": evidence_quote,
        "confidence": confidence,
    }


def propose_local_organization(
    knowledge: Mapping[str, Any],
    local_context: Mapping[str, Any],
    llm_service,
) -> dict:
    """Ask the sole LLM provider for a bounded, non-executing proposal."""

    if llm_service is None:
        raise OrganizationProposalError("organization review has no LLM service")
    messages = [
        {"role": "system", "content": ORGANIZATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "local_context:\n" + _json_text(dict(local_context)),
        },
    ]
    complete_json = getattr(llm_service, "complete_json", None)
    if callable(complete_json):
        response = complete_json(messages, max_tokens=900)
    else:
        complete = getattr(llm_service, "complete", None) or getattr(llm_service, "extract", None)
        if not callable(complete):
            raise OrganizationProposalError("LLM service cannot complete an organization proposal")
        response = complete(messages, max_tokens=900)
    parsed = parse_json_response(response)
    return validate_organization_proposal(parsed)


def _contains_evidence(text: str, quote: str) -> bool:
    source = normalize_entity_name(str(text or ""))
    excerpt = normalize_entity_name(str(quote or ""))
    return bool(excerpt and excerpt in source)


def _proposal_name_in_evidence(name: str, evidence_quote: str) -> bool:
    return bool(name and normalize_entity_name(name) in normalize_entity_name(evidence_quote))


def _evidence_has_explicit_parent_marker(
    subject_name: str,
    parent_name: str,
    evidence_quote: str,
) -> bool:
    """Reject co-occurrence-only proposals, including naming-based guesses."""

    compact = normalize_entity_name(evidence_quote)
    subject = normalize_entity_name(subject_name)
    parent = normalize_entity_name(parent_name)
    subject_at = compact.find(subject)
    parent_at = compact.find(parent, subject_at + len(subject)) if subject_at >= 0 else -1
    if subject_at < 0 or parent_at < 0:
        return False
    between = compact[subject_at + len(subject):parent_at]
    if any(mark in between for mark in ("。", ".", "!", "！", "?", "？", ";", "；")):
        return False
    return bool(re.search(
        r"属于|归属于|归到|是|为|belongs?to|partof|memberof|under|beneath|"
        r"categorized|classified|归类",
        between,
        re.IGNORECASE,
    ))


def _validate_proposal_against_context(
    conn,
    knowledge: Mapping[str, Any],
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> tuple[dict, dict, dict | None]:
    """Apply Python-owned scope/evidence/parent checks before any mutation."""

    action = proposal["action"]
    if action in {"NO_CHANGE", "UNCLEAR"}:
        return dict(proposal), {}, None
    source_text = "\n".join(
        str(value or "")
        for value in (
            knowledge.get("content"),
            knowledge.get("source_content"),
            knowledge.get("source_text"),
            (context.get("current_knowledge") or {}).get("content"),
        )
        if value
    )
    if not _contains_evidence(source_text, proposal["evidence_quote"]):
        raise OrganizationProposalError("evidence_quote is not in the confirmed Knowledge content")
    subject = context.get("current_entity") or {}
    if not subject or normalize_entity_name(proposal["subject_entity"]) != normalize_entity_name(subject.get("name")):
        raise OrganizationProposalError("proposal subject is outside the current entity scope")
    if not _proposal_name_in_evidence(proposal["subject_entity"], proposal["evidence_quote"]):
        raise OrganizationProposalError("proposal subject is not present in evidence_quote")
    target_entity = lookup_entity(conn, proposal["target_parent"]) if proposal.get("target_parent") else None
    new_entity = proposal.get("new_entity")
    if new_entity:
        if not _proposal_name_in_evidence(new_entity["name"], proposal["evidence_quote"]):
            raise OrganizationProposalError("new entity is not present in evidence_quote")
        if proposal.get("target_parent") and normalize_entity_name(new_entity["name"]) != normalize_entity_name(proposal["target_parent"]):
            raise OrganizationProposalError("new_entity must be the proposed target parent")
    if action in {"CREATE_RELATION", "MOVE_RELATION"}:
        if not _proposal_name_in_evidence(proposal["target_parent"], proposal["evidence_quote"]):
            raise OrganizationProposalError("target parent is not present in evidence_quote")
        if not _evidence_has_explicit_parent_marker(
            proposal["subject_entity"],
            proposal["target_parent"],
            proposal["evidence_quote"],
        ):
            raise OrganizationProposalError("evidence_quote does not explicitly describe a parent relation")
        if target_entity is None and new_entity is None:
            raise OrganizationProposalError("unknown target parent requires an explicitly evidenced new_entity")
        if target_entity is not None and not target_entity.get("active", True):
            raise OrganizationProposalError("target parent is inactive")
        current_parent = (context.get("current_parent") or {}).get("entity") or {}
        same_parent = current_parent and normalize_entity_name(current_parent.get("name")) == normalize_entity_name(proposal["target_parent"])
        if action == "MOVE_RELATION" and not current_parent:
            raise OrganizationProposalError("MOVE_RELATION requires an existing active parent")
        if action == "CREATE_RELATION" and current_parent and not same_parent:
            raise OrganizationProposalError("an existing parent requires MOVE_RELATION")
        if action == "MOVE_RELATION" and same_parent:
            raise OrganizationProposalError("MOVE_RELATION target is already the active parent")
    if action == "CREATE_ENTITY" and new_entity and not _proposal_name_in_evidence(new_entity["name"], proposal["evidence_quote"]):
        raise OrganizationProposalError("created entity is not present in evidence_quote")
    return dict(proposal), dict(subject), target_entity


def _apply_organization_proposal(
    conn,
    knowledge: Mapping[str, Any],
    proposal: Mapping[str, Any],
    subject: dict,
    target_entity: dict | None,
) -> dict:
    source_id = int(knowledge["id"])
    provenance = (
        f"Knowledge #{source_id}: {proposal.get('reason') or 'explicit local organization'}; "
        f"evidence: {proposal.get('evidence_quote') or ''}"
    )
    if proposal["action"] == "CREATE_ENTITY":
        entity_ref = proposal.get("new_entity") or {
            "name": subject["name"],
            "entity_type": proposal.get("entity_type") or subject.get("entity_type") or "concept",
        }
        entity = _entity_ref(conn, entity_ref)
        if int(entity["id"]) == int(subject["id"]):
            link_knowledge_to_entity(conn, source_id, int(entity["id"]))
        return {"action": "CREATE_ENTITY", "entity": entity, "relations": []}
    if proposal["action"] in {"NO_CHANGE", "UNCLEAR"}:
        return {"action": proposal["action"], "entity": subject, "relations": [], "reason": proposal.get("reason", "")}
    parent_ref = target_entity or proposal.get("new_entity")
    if not parent_ref:
        raise OrganizationProposalError("validated relation has no parent entity")
    return local_organization_review(
        conn,
        dict(knowledge, entity_id=int(subject["id"])),
        proposed_parent=parent_ref,
        parent_entity_type=str((parent_ref.get("entity_type") if isinstance(parent_ref, Mapping) else "concept") or "concept"),
        provenance=provenance,
        provenance_kind=str(knowledge["trust"]),
        source_id=source_id,
    )


def review_local_organization(
    conn,
    knowledge: Mapping[str, Any],
    *,
    llm_service=None,
) -> dict:
    """Run one bounded review and keep all LLM output outside the DB layer."""

    if not isinstance(knowledge, Mapping):
        raise TypeError("knowledge must be a mapping")
    if knowledge.get("trust") not in TRUSTED_KNOWLEDGE:
        return {"action": "NO_CHANGE", "entity": None, "relations": [], "reason": "knowledge is not trusted structural evidence"}
    explicit_chain = extract_explicit_chain(dict(knowledge))
    if explicit_chain:
        return local_organization_review(conn, dict(knowledge), explicit_chain=explicit_chain)
    # Linking a confirmed fact to its exact subject is deterministic and is
    # useful even when the optional organization model is unavailable.
    try:
        baseline = local_organization_review(conn, dict(knowledge))
    except Exception:
        log.exception("organization_review_failed knowledge_id=%s", knowledge.get("id"))
        return {
            "action": "UNCLEAR",
            "entity": None,
            "relations": [],
            "reason": "organization review failed; Knowledge was retained",
        }
    if llm_service is None:
        return baseline
    try:
        context = build_local_organization_context(conn, knowledge)
        proposal = propose_local_organization(knowledge, context, llm_service)
        checked, subject, target_entity = _validate_proposal_against_context(
            conn, knowledge, context, proposal,
        )
        if checked["action"] in {"NO_CHANGE", "UNCLEAR"}:
            if checked["action"] == "NO_CHANGE":
                return baseline
            return {
                "action": "UNCLEAR",
                "entity": baseline.get("entity"),
                "relations": [],
                "reason": checked.get("reason") or "organization structure is unclear",
            }
        return _apply_organization_proposal(conn, knowledge, checked, subject, target_entity)
    except Exception:
        log.exception("organization_review_failed knowledge_id=%s", knowledge.get("id"))
        return {
            "action": baseline.get("action", "NO_CHANGE"),
            "entity": baseline.get("entity"),
            "relations": [],
            "reason": "organization review failed; Knowledge was retained",
        }


__all__ = [
    "CycleError", "OrganizationError", "ProvenanceError", "backfill_knowledge_entity_links",
    "create_relation", "get_entity_by_id", "get_or_create_entity", "get_relation",
    "extract_explicit_chain", "link_knowledge_to_entity", "list_local_context",
    "local_organization_review", "review_local_organization", "propose_local_organization",
    "build_local_organization_context", "validate_organization_proposal",
    "OrganizationProposalError",
    "lookup_entity", "lookup_relation", "move_relation", "normalize_entity_name",
    "would_create_cycle",
]
