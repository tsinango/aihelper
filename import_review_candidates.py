#!/usr/bin/env python3
"""Import the review-only pilot artifact into PostgreSQL once.

The JSON artifact is an import source, not a runtime data store. Existing
candidate rows are never overwritten, so a rerun cannot erase human review.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from build_verified_knowledge_pilot import load_env_file
from telegram_relations import upsert_message_relations


DEFAULT_INPUT = Path("/opt/aihelper/data/telegram_knowledge_review.json")

ROLE_VALUES = {
    "user_report", "engineer_hypothesis", "engineer_instruction",
    "observed_result", "confirmed_resolution", "unconfirmed_claim", "irrelevant",
}
EVIDENCE_VALUES = {"supports", "partial", "irrelevant", "conflict", "unreviewed"}


def jsonb(value):
    return Jsonb(value)


def load_source(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("artifact_type") == "telegram_knowledge_review":
        candidates = source.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 602:
            raise ValueError("Telegram review artifact must contain 602 candidates")
        return source
    if source.get("provider") != "openrouter":
        raise ValueError("pilot artifact is not an OpenRouter artifact")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("pilot artifact has no candidates")
    return source


def case_role_map(candidate: dict) -> dict[int, dict[int, dict]]:
    result = {}
    for item in candidate.get("telegram_evidence", []):
        if not isinstance(item, dict):
            continue
        try:
            case_id = int(item["case_id"])
        except (KeyError, TypeError, ValueError):
            continue
        result[case_id] = {}
        for role in item.get("message_roles", []):
            if not isinstance(role, dict):
                continue
            try:
                message_index = int(role.get("message_index", -1))
            except (TypeError, ValueError):
                message_index = -1
            value = str(role.get("role", "unconfirmed_claim"))
            if value not in ROLE_VALUES:
                value = "unconfirmed_claim"
            result[case_id][message_index] = {
                "role": value,
                "reason": str(role.get("reason", ""))[:400],
            }
    return result


def cited_evidence_ids(candidate: dict) -> set[str]:
    result = set()
    for claim in candidate.get("claims", []):
        if not isinstance(claim, dict):
            continue
        for evidence in claim.get("evidence", []):
            if isinstance(evidence, dict) and evidence.get("evidence_id"):
                result.add(str(evidence["evidence_id"]))
    return result


def import_candidate(conn, candidate: dict, case_rows: dict[int, dict]) -> bool:
    candidate_id = str(candidate["candidate_id"])
    candidate_cases = []
    for value in candidate.get("telegram_cases", []):
        try:
            case_id = int(value)
        except (TypeError, ValueError):
            continue
        if case_id in case_rows and case_id not in candidate_cases:
            candidate_cases.append(case_id)
    ai_payload = dict(candidate)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO verified_knowledge_candidates
              (candidate_id, knowledge_key, title, knowledge_type, scope,
               question_patterns, claims, procedure_steps, conditions, exceptions,
               warnings, confidence, freshness_sensitive, last_verified_at,
               verification_status, review_status, review_note, publication_status,
               production_answer_allowed, frequency, ai_payload, human_overrides,
               effective_payload, answer_text, answer_status, scope_level)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    'pending','pending',%s,'draft',FALSE,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (candidate_id) DO UPDATE SET
              knowledge_key=EXCLUDED.knowledge_key,
              title=EXCLUDED.title,
              knowledge_type=EXCLUDED.knowledge_type,
              scope=EXCLUDED.scope,
              question_patterns=EXCLUDED.question_patterns,
              claims=EXCLUDED.claims,
              procedure_steps=EXCLUDED.procedure_steps,
              conditions=EXCLUDED.conditions,
              exceptions=EXCLUDED.exceptions,
              warnings=EXCLUDED.warnings,
              confidence=EXCLUDED.confidence,
              freshness_sensitive=EXCLUDED.freshness_sensitive,
              last_verified_at=EXCLUDED.last_verified_at,
              review_note=EXCLUDED.review_note,
              frequency=EXCLUDED.frequency,
              ai_payload=EXCLUDED.ai_payload,
              effective_payload=EXCLUDED.effective_payload,
              answer_text=EXCLUDED.answer_text,
              answer_status=EXCLUDED.answer_status,
              scope_level=EXCLUDED.scope_level,
              updated_at=CURRENT_TIMESTAMP
            WHERE verified_knowledge_candidates.candidate_id LIKE 'CASE-%%'
              AND verified_knowledge_candidates.review_status='pending'
              AND verified_knowledge_candidates.verification_status='pending'
              AND verified_knowledge_candidates.publication_status='draft'
              AND verified_knowledge_candidates.production_answer_allowed=FALSE
            RETURNING id
            """,
            (
                candidate_id,
                str(candidate.get("knowledge_key", "")),
                str(candidate.get("title", "")),
                str(candidate.get("knowledge_type", "other")),
                jsonb(candidate.get("scope", {})),
                jsonb(candidate.get("question_patterns", [])),
                jsonb(candidate.get("claims", [])),
                jsonb(candidate.get("procedure_steps", [])),
                jsonb(candidate.get("conditions", [])),
                jsonb(candidate.get("exceptions", [])),
                jsonb(candidate.get("warnings", [])),
                str(candidate.get("confidence", "low")),
                bool(candidate.get("freshness_sensitive", False)),
                candidate.get("last_verified_at"),
                str(candidate.get("review_note", "")),
                len(candidate_cases),
                jsonb(ai_payload),
                jsonb({}),
                jsonb(ai_payload),
                str(candidate.get("answer_text", "")),
                str(candidate.get("answer_status", "pending")),
                ("unspecified" if str(candidate.get("scope_level", "unspecified")) == "single_case" else str(candidate.get("scope_level", "unspecified"))),
            ),
        )
        inserted = cur.fetchone() is not None
        cur.execute("SELECT id FROM verified_knowledge_candidates WHERE candidate_id=%s", (candidate_id,))
        candidate_pk = cur.fetchone()["id"]

        for position, case_id in enumerate(candidate_cases):
            cur.execute(
                """
                INSERT INTO verified_knowledge_candidate_cases(candidate_id, support_case_id, case_position)
                VALUES(%s,%s,%s) ON CONFLICT(candidate_id, support_case_id) DO NOTHING
                """,
                (candidate_id, case_id, position),
            )

        role_map = case_role_map(candidate)
        for case_id in candidate_cases:
            messages = case_rows[case_id].get("messages") or []
            upsert_message_relations(cur, case_id, case_rows[case_id].get("root_author"), messages)
            for message_index, message in enumerate(messages):
                role = role_map.get(case_id, {}).get(message_index, {})
                ai_role = role.get("role", "unconfirmed_claim")
                cur.execute(
                    """
                    INSERT INTO verified_knowledge_candidate_message_roles
                      (candidate_id, support_case_id, message_index, ai_role,
                       effective_role, ai_reason)
                    VALUES(%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(candidate_id, support_case_id, message_index) DO UPDATE SET
                      ai_role=EXCLUDED.ai_role,
                      effective_role=CASE
                        WHEN verified_knowledge_candidate_message_roles.human_role IS NULL
                        THEN EXCLUDED.ai_role
                        ELSE verified_knowledge_candidate_message_roles.effective_role
                      END,
                      ai_reason=EXCLUDED.ai_reason,
                      updated_at=CURRENT_TIMESTAMP
                    """,
                    (candidate_id, case_id, message_index, ai_role, ai_role, role.get("reason", "")),
                )

        cited_ids = cited_evidence_ids(candidate)
        for item in candidate.get("official_sources", []):
            if not isinstance(item, dict) or not item.get("evidence_id"):
                continue
            evidence_id = str(item["evidence_id"])
            document_id = item.get("document_id")
            chunk_id = item.get("chunk_id")
            excerpt = ""
            if document_id is not None and chunk_id is not None:
                cur.execute("SELECT content FROM document_chunks WHERE id=%s AND document_id=%s", (chunk_id, document_id))
                chunk = cur.fetchone()
                excerpt = str(chunk["content"] if chunk else "")[:2400]
            ai_relation = "supports" if evidence_id in cited_ids else "unreviewed"
            cur.execute(
                """
                INSERT INTO verified_knowledge_candidate_evidence
                  (candidate_id, evidence_id, source_type, document_id,
                   document_title, page, chunk_id, excerpt,
                   ai_evidence_relation, effective_evidence_relation)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(candidate_id, evidence_id) DO NOTHING
                """,
                (
                    candidate_id, evidence_id, str(item.get("source_type", "official_document")),
                    document_id, str(item.get("document_title", "")), item.get("page"), chunk_id,
                    excerpt, ai_relation, ai_relation,
                ),
            )

        for position, claim in enumerate(candidate.get("claims", [])):
            if not isinstance(claim, dict):
                continue
            cur.execute(
                """
                INSERT INTO verified_knowledge_claims
                  (candidate_id, claim_position, ai_claim, effective_claim)
                VALUES(%s,%s,%s,%s)
                ON CONFLICT(candidate_id, claim_position) DO NOTHING
                """,
                (candidate_id, position, jsonb(claim), jsonb(claim)),
            )
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("/etc/ai-sales-engineer.env"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    load_env_file(args.env_file)
    source = load_source(args.input)
    candidate_ids = [str(candidate["candidate_id"]) for candidate in source["candidates"]]
    case_ids = sorted({int(case_id) for candidate in source["candidates"] for case_id in candidate.get("telegram_cases", [])})
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, root_author, messages FROM support_cases WHERE id=ANY(%s)", (case_ids,))
            case_rows = {int(row["id"]): dict(row) for row in cur.fetchall()}
        inserted = sum(import_candidate(conn, candidate, case_rows) for candidate in source["candidates"])
        conn.commit()
    print(json.dumps({
        "artifact_type": source.get("artifact_type"),
        "candidates_seen": len(candidate_ids),
        "candidates_inserted_or_refreshed": inserted,
        "candidate_ids": candidate_ids,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
