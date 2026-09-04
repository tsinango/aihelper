"""Small, evidence-backed organization helpers for V2.

The organization layer is intentionally separate from facts and the learning
state machine.  It only maintains exact-deduplicated entities and local
``belongs_to`` relations.  Callers own transactions; no helper commits.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


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


def list_local_context(conn, entity_id: int, *, max_ancestor_depth: int = 5) -> dict:
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
                """,
                (int(parent_id), int(entity_id)),
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
            """,
            (int(entity_id),),
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


def extract_explicit_chain(knowledge: dict) -> list[dict]:
    """Read only an explicitly structured or plainly stated chain.

    Ordinary fact prose and model-like names are intentionally ignored.  A
    small set of unambiguous relationship sentences is also accepted so the
    normal Inbox confirmation path can organize facts without adding a second
    schema.  The parser never expands a series from a model name alone.
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
    text = "\n".join(
        str(knowledge.get(key) or "")
        for key in ("title", "content")
    )
    compact = " ".join(re.sub(r"\s+", "", segment) for segment in text.splitlines())
    compact_subject = re.sub(r"\s+", "", subject)
    if not compact_subject or compact_subject.casefold() not in compact.casefold():
        return []

    model_candidates = [
        candidate
        for segment in text.splitlines()
        for candidate in re.findall(
            r"[A-Za-z][A-Za-z0-9]*(?:[-./()][A-Za-z0-9]+)+",
            re.sub(r"\s+", "", segment),
        )
        if re.search(r"\d", candidate)
    ]
    model_name = model_candidates[0] if model_candidates else subject
    relation_subject = compact_subject if re.search(r"\d", compact_subject) else model_name

    # A series is accepted only when the source explicitly relates this exact
    # model to the named series.  This intentionally rejects name-prefix
    # generalization such as treating every ``F-NR`` model as an NVR.
    series_match = re.search(
        r"(?<![A-Za-z0-9])" + re.escape(relation_subject)
        + r".{0,100}?(?:属于|是|为|belongs?to|is).{0,70}?([A-Za-z0-9][A-Za-z0-9./()\-]*)系列",
        compact,
        re.IGNORECASE,
    )
    series_name = f"{series_match.group(1)} 系列" if series_match else ""
    series_base = re.sub(r"\s*系列$", "", series_name)

    # These patterns require an explicit relationship involving the subject
    # or its explicitly named series.  Merely co-occurring product names do
    # not create structure.
    subject_to_iflow_nvr = re.search(
        r"(?<![A-Za-z0-9])" + re.escape(relation_subject)
        + r".{0,100}(?:是|为|属于|is|belongs?to).{0,120}iflow.{0,100}nvr",
        compact,
        re.IGNORECASE,
    )
    series_to_iflow_nvr = bool(series_name) and re.search(
        r"(?<![A-Za-z0-9])" + re.escape(series_base)
        + r"系列.{0,100}(?:是|为|属于|is|belongs?to).{0,120}iflow.{0,100}nvr",
        compact,
        re.IGNORECASE,
    )
    if not (subject_to_iflow_nvr or series_to_iflow_nvr):
        return ([
            {"name": series_name, "entity_type": "series"},
            {"name": model_name if model_candidates else subject, "entity_type": "model"},
        ] if series_name else [])

    chain = [{"name": "iFlow", "entity_type": "brand"}]
    if re.search(r"后端产品|backend\s*products?", compact, re.IGNORECASE):
        chain.append({"name": "后端产品", "entity_type": "category"})
    chain.append({"name": "NVR", "entity_type": "category"})
    if series_name:
        chain.append({"name": series_name, "entity_type": "series"})
    chain.append({
        "name": model_name if model_candidates else subject,
        "entity_type": "model",
    })
    return chain


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
        entities = [
            _entity_ref(conn, item["name"], entity_type=item["entity_type"])
            for item in chain
        ]
        if not any(int(entity["id"]) == int(subject["id"]) for entity in entities):
            return {"action": "UNCLEAR", "entity": subject, "relations": [], "reason": "explicit chain subject does not match existing entity"}
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


__all__ = [
    "CycleError", "OrganizationError", "ProvenanceError", "backfill_knowledge_entity_links",
    "create_relation", "get_entity_by_id", "get_or_create_entity", "get_relation",
    "extract_explicit_chain", "link_knowledge_to_entity", "list_local_context",
    "local_organization_review",
    "lookup_entity", "lookup_relation", "move_relation", "normalize_entity_name",
    "would_create_cycle",
]
