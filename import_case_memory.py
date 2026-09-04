#!/usr/bin/env python3
"""Materialize every extracted Telegram case as traceable AI-derived memory.

The importer is deliberately idempotent.  It does not promote a case to
Verified Knowledge; a reviewer still has to approve a candidate separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from embeddings import OPENROUTER_EMBEDDING_MODEL, OpenRouterEmbeddingClient, read_openrouter_token
from telegram_relations import classify_message, message_id, upsert_message_relations


def as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def message_texts(messages, root_author: str | None) -> tuple[str, str]:
    """Return the first question and all later technical context.

    The old importer discarded later messages from ``root_author``.  A memory
    item is recall evidence, so it must retain customer follow-ups and results
    rather than treating author identity as an answer boundary.
    """
    question = ""
    context = []
    message_list = [item for item in as_list(messages) if isinstance(item, dict)]
    case = {"root_author": root_author, "messages": message_list}
    for index, message in enumerate(message_list):
        if not isinstance(message, dict):
            continue
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        if not question:
            question = text
            continue
        role = classify_message(message, case, index, message_list)
        context.append(f"[{role}] {text}")
    return question, "\n".join(context).strip()


def message_evidence(case: dict) -> list[dict]:
    messages = [item for item in as_list(case.get("messages")) if isinstance(item, dict)]
    evidence = []
    for index, message in enumerate(messages):
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        role = classify_message(message, case, index, messages)
        if role == "confirmed_resolution":
            status = "confirmed_success"
        elif role == "observed_result" and "не" in text.casefold():
            status = "confirmed_failure"
        elif role in {"engineer_instruction", "unconfirmed_claim"}:
            status = "supports"
        else:
            status = "context_only"
        evidence.append({
            "source_type": "telegram_message",
            "support_case_id": int(case["id"]),
            "message_id": message_id(message, index),
            "message_index": index,
            "excerpt": text[:4000],
            "evidence_role": role,
            "evidence_status": status,
        })
    return evidence


def build_memory(case: dict, analysis: dict) -> dict:
    models = as_list(analysis.get("models_json"))
    canonical = str(analysis.get("canonical_question") or case.get("root_question") or "").strip()
    root_question, answer_text = message_texts(case.get("messages"), case.get("root_author"))
    question_patterns = [value for value in (canonical, root_question) if value]
    context_required = str(analysis.get("context_status") or "") == "context_required"
    quality = str(analysis.get("question_quality") or "good")
    if quality in {"non_question", "low_value"}:
        status = "rejected"
        answer_allowed = False
    elif bool(case.get("production_answer_allowed")):
        status = "verified"
        answer_allowed = True
    elif context_required:
        status = "needs_context"
        answer_allowed = True
    else:
        status = "ai_derived"
        answer_allowed = True
    # Historical Telegram memory is recall/candidate evidence only.  It must
    # never become authoritative customer-facing knowledge merely because it
    # has an answer-shaped text.
    answer_allowed = status == "verified" and bool(answer_text)

    claims = []
    if answer_text:
        claims.append({
            "claim": answer_text,
            "claim_type": "historical_case_answer",
            "evidence": [{"source_type": "telegram_case", "support_case_id": case["id"], "status": "historical"}],
        })
    all_messages = "\n".join(str(item.get("text") or "").strip() for item in as_list(case.get("messages")) if isinstance(item, dict))
    searchable = " ".join(str(value).strip() for value in (
        canonical, root_question, all_messages, answer_text, analysis.get("knowledge_key"),
        analysis.get("knowledge_type"), *models,
        *as_list(analysis.get("features_json")),
        *as_list(analysis.get("symptoms_json")),
    ) if str(value or "").strip())
    evidence = message_evidence(case)
    if not evidence:
        evidence = [{
            "source_type": "telegram_case", "support_case_id": case["id"],
            "excerpt": answer_text[:4000], "relation": "question_only",
            "evidence_role": "context_only", "evidence_status": "context_only",
        }]
    return {
        "support_case_id": int(case["id"]),
        "knowledge_key": str(analysis.get("knowledge_key") or "other.other"),
        "canonical_question": canonical,
        "knowledge_type": str(analysis.get("knowledge_type") or "other"),
        "scope": {"models": [str(model).strip() for model in models if str(model).strip()]},
        "question_patterns": question_patterns,
        "answer_text": answer_text,
        "claims": claims,
        "procedure_steps": [answer_text] if answer_text else [],
        "conditions": [],
        "exceptions": [],
        "warnings": ["Источник — исторический ответ Telegram; требуется ручная проверка."] if status != "verified" else [],
        "evidence": evidence,
        "source_status": status,
        "answer_allowed": answer_allowed,
        "requires_context": context_required,
        "source_confidence": max(0.0, min(1.0, float(analysis.get("extraction_confidence") or 0))),
        "searchable_text": searchable,
        "source_content_hash": str(analysis.get("source_content_hash") or case.get("content_hash") or ""),
        "last_verified_at": None,
    }


def load_rows(cur) -> list[dict]:
    cur.execute(
        """
        SELECT DISTINCT ON (sc.id)
          sc.id,sc.root_author,sc.root_question,sc.messages,
          sc.production_answer_allowed,sc.content_hash,
          a.canonical_question,a.knowledge_type,a.models_json,a.features_json,
          a.symptoms_json,a.extraction_confidence,a.source_content_hash,
          a.prompt_version
        FROM support_cases sc
        JOIN support_case_analysis a ON a.support_case_id=sc.id
        ORDER BY sc.id,
          CASE WHEN a.prompt_version LIKE '%V2_1%' THEN 0 ELSE 1 END,
          a.id DESC
        """
    )
    return [dict(row) for row in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("/etc/aihelper.env"))
    parser.add_argument(
        "--artifact", type=Path,
        default=Path(__file__).with_name("data") / "knowledge_intents_v1_1_openrouter.json",
        help="optional OpenRouter-only V1.1 artifact containing deterministic knowledge keys",
    )
    parser.add_argument("--embed", action="store_true", help="embed inserted rows with OpenRouter Nemotron")
    args = parser.parse_args()
    if args.env_file and args.env_file.is_file():
        for line in args.env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    embedder = None
    model_name = OPENROUTER_EMBEDDING_MODEL
    if args.embed:
        token = read_openrouter_token(os.getenv("OPENROUTER_TOKEN_FILE", str(Path(__file__).with_name("openrouter"))))
        if not token:
            raise SystemExit("OpenRouter token is required when --embed is used")
        embedder = OpenRouterEmbeddingClient(
            token,
            timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
        )

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            rows = load_rows(cur)
            artifact = {}
            if args.artifact.is_file():
                payload = json.loads(args.artifact.read_text(encoding="utf-8"))
                artifact = {
                    int(item["support_case_id"]): item
                    for item in payload.get("intents", [])
                    if isinstance(item, dict) and item.get("support_case_id") is not None
                }
            memories = []
            for row in rows:
                case = dict(row)
                upsert_message_relations(cur, int(row["id"]), row.get("root_author"), row.get("messages"))
                analysis = dict(row)
                analysis.update(artifact.get(int(row["id"]), {}))
                analysis["models_json"] = analysis.get("models_json", analysis.get("scope_models", []))
                analysis["source_content_hash"] = analysis.get("source_content_hash") or row.get("source_content_hash")
                memory = build_memory(case, analysis)
                memories.append(memory)
                cur.execute(
                    """
                    INSERT INTO case_knowledge_memory
                      (support_case_id,knowledge_key,canonical_question,knowledge_type,scope,
                       question_patterns,answer_text,claims,procedure_steps,conditions,exceptions,
                       warnings,evidence,source_status,answer_allowed,requires_context,
                       source_confidence,searchable_text,source_content_hash,last_verified_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (support_case_id) DO UPDATE SET
                      knowledge_key=EXCLUDED.knowledge_key,
                      canonical_question=EXCLUDED.canonical_question,
                      knowledge_type=EXCLUDED.knowledge_type,
                      scope=EXCLUDED.scope,
                      question_patterns=EXCLUDED.question_patterns,
                      answer_text=EXCLUDED.answer_text,
                      claims=EXCLUDED.claims,
                      procedure_steps=EXCLUDED.procedure_steps,
                      conditions=EXCLUDED.conditions,
                      exceptions=EXCLUDED.exceptions,
                      warnings=EXCLUDED.warnings,
                      evidence=EXCLUDED.evidence,
                      source_status=CASE WHEN case_knowledge_memory.source_status='verified' THEN 'verified' ELSE EXCLUDED.source_status END,
                      answer_allowed=CASE
                        WHEN case_knowledge_memory.source_status IN ('verified','rejected')
                        THEN case_knowledge_memory.answer_allowed
                        ELSE EXCLUDED.answer_allowed
                      END,
                      requires_context=EXCLUDED.requires_context,
                      source_confidence=EXCLUDED.source_confidence,
                      searchable_text=EXCLUDED.searchable_text,
                      source_content_hash=EXCLUDED.source_content_hash,
                      updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        memory["support_case_id"], memory["knowledge_key"], memory["canonical_question"],
                        memory["knowledge_type"], Jsonb(memory["scope"]), Jsonb(memory["question_patterns"]),
                        memory["answer_text"], Jsonb(memory["claims"]), Jsonb(memory["procedure_steps"]),
                        Jsonb(memory["conditions"]), Jsonb(memory["exceptions"]), Jsonb(memory["warnings"]),
                        Jsonb(memory["evidence"]), memory["source_status"], memory["answer_allowed"],
                        memory["requires_context"], memory["source_confidence"], memory["searchable_text"],
                        memory["source_content_hash"], memory["last_verified_at"],
                    ),
                )
            cur.execute(
                """
                WITH links AS (
                  SELECT DISTINCT ON (cc.support_case_id)
                    cc.support_case_id,vc.candidate_id,vc.review_status,
                    vc.production_answer_allowed
                  FROM verified_knowledge_candidate_cases cc
                  JOIN verified_knowledge_candidates vc ON vc.candidate_id=cc.candidate_id
                  ORDER BY cc.support_case_id,vc.updated_at DESC,vc.id DESC
                )
                UPDATE case_knowledge_memory m
                SET source_candidate_id=vc.candidate_id,
                    source_status=CASE
                      WHEN vc.review_status='approved' THEN 'verified'
                      WHEN vc.review_status IN ('rejected','duplicate') THEN 'rejected'
                      ELSE m.source_status
                    END,
                    answer_allowed=CASE
                      WHEN vc.review_status='approved' THEN vc.production_answer_allowed
                      WHEN vc.review_status IN ('rejected','duplicate') THEN FALSE
                      ELSE m.answer_allowed
                    END,
                    updated_at=CURRENT_TIMESTAMP
                FROM links vc
                WHERE vc.support_case_id=m.support_case_id
                """
            )
            if embedder is not None:
                for start in range(0, len(memories), 8):
                    batch = memories[start:start + 8]
                    vectors = embedder.encode(
                        [item["searchable_text"] for item in batch],
                        batch_size=8, normalize_embeddings=True,
                    )
                    for item, embedding in zip(batch, vectors, strict=True):
                        cur.execute(
                            "UPDATE case_knowledge_memory SET embedding=%s::vector,embedding_model=%s,updated_at=CURRENT_TIMESTAMP WHERE support_case_id=%s",
                            (str(embedding), model_name, item["support_case_id"]),
                        )
        print(json.dumps({"processed_cases": len(rows), "embedded": bool(embedder)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
