"""Phase 3.2 correction -> Experience -> retest loop.

One ``v2_answer_feedback`` row records an engineer correction against one
answer run and doubles as the lightweight unresolved-gap queue.  Experience
stays inside ``v2_knowledge`` (``unit_kind='experience'``); no second
experience store exists.

Kind semantics (fixed at submit time):

- ``reply_only``: the edited reply is stored for this run only.  It never
  creates Knowledge, a proposal, or a source link.
- ``save_experience``: submit stores the correction as raw evidence plus a
  provisional Knowledge row and a ``pending_confirmation`` proposal (model
  output is never trusted early).  An explicit, idempotent confirm flips the
  Knowledge to ``user_confirmed`` in the same transaction.  Updating known
  Knowledge requires the target id plus its expected revision; a stale
  revision is a 409, never an auto-merge.
- ``missing_information`` / ``retrieval_failure`` / ``generation_failure``:
  lightweight gap records for the unresolved queue; they never write
  Knowledge on their own.
- ``field_result_success`` / ``field_result_failure``: field outcomes
  attached to a run; ``field_result`` records success/failure, unknown stays
  NULL and is never auto-filled.

The confirm path is pure database work: no LLM, embedding, or organization
call, so an engineer confirmation completes even when model services are
down.  Embeddings are left NULL for the asynchronous backfill; lexical
retrieval keeps the confirmed Experience answerable immediately.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from v2.learning import _insert_evidence, _link_source

FEEDBACK_KINDS = (
    "reply_only",
    "save_experience",
    "missing_information",
    "retrieval_failure",
    "generation_failure",
    "field_result_success",
    "field_result_failure",
)
FEEDBACK_STATUSES = ("open", "confirmed", "closed")
# Phase 5.3 failure taxonomy.  A failure is never auto-converted into new
# Knowledge: each category names its default corrective action instead.
FAILURE_CATEGORIES = (
    "missing_source",
    "knowledge_missing",
    "retrieval_failure",
    "generation_failure",
    "applicability_version_failure",
    "service_failure",
)
FAILURE_ACTIONS = {
    "missing_source": "补资料或形成待确认 Experience",
    "knowledge_missing": "修改/重提炼具体单元，回跑受影响问题",
    "retrieval_failure": "检查型号/别名、范围过滤和排序，保留诊断样本",
    "generation_failure": "修改生成约束/完整流程呈现并复测；不重复新增同样知识",
    "applicability_version_failure": "修适用性和版本资格，不提高相似度阈值掩盖问题",
    "service_failure": "延后重试/提示服务状态，不制造产品知识提问",
}
# Real questions proving qualified evidence was missed before any retrieval
# improvement is allowed (astra gate).
RETRIEVAL_GATE_NEEDED = 10
UNIT_KINDS = ("fact", "procedure", "rule", "experience")
FIELD_RESULTS = ("success", "failure")
VERDICTS = ("pass", "fail")
# Corrections are stored verbatim; overlong input is rejected, never cut.
MAX_CORRECTION_CHARS = 12000


class FeedbackNotFound(LookupError):
    """No such feedback row or answer run."""


class FeedbackConflict(ValueError):
    """The requested transition is not allowed (maps to HTTP 409)."""


class StaleRevision(FeedbackConflict):
    """The target Knowledge changed since the engineer saw it (maps to 409)."""


def _text(value: Any, limit: int = MAX_CORRECTION_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise FeedbackConflict(
            f"correction text exceeds {limit} characters; shorten it instead of truncating"
        )
    return text


def _clean_applicability(value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FeedbackConflict("applicability must be a JSON object")
    return {str(key): value[key] for key in value}


def _title_for(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            return clean[:120]
    return text[:120] or "Engineer correction"


_FEEDBACK_COLUMNS = (
    "f.id, f.answer_run_id, f.idempotency_key, f.feedback_kind, "
    "f.correction_text, f.applicability, f.unit_kind, "
    "f.target_knowledge_id, f.expected_revision, f.raw_evidence_id, "
    "f.proposal_id, f.knowledge_id, f.status, f.field_result, "
    "f.expected_knowledge_ids, f.reviewer_label, f.created_at, f.updated_at"
)


def _clean_expected_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise FeedbackConflict("expected_knowledge_ids must be a list of integers")
    cleaned = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise FeedbackConflict("expected_knowledge_ids must be a list of integers") from exc
        if number <= 0:
            raise FeedbackConflict("expected_knowledge_ids must be positive integers")
        cleaned.append(number)
    return sorted(set(cleaned))


def _feedback_to_dict(row: dict) -> dict:
    return dict(row)


def get_feedback(conn, feedback_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_FEEDBACK_COLUMNS}, r.question AS run_question,
                   r.answer_status AS run_answer_status
            FROM v2_answer_feedback f
            JOIN v2_answer_runs r ON r.id=f.answer_run_id
            WHERE f.id=%s
            """,
            (int(feedback_id),),
        )
        row = cur.fetchone()
    return _feedback_to_dict(row) if row else None


def list_feedback_for_run(conn, run_id: int) -> list[dict]:
    """Every correction ever filed against one run, newest first."""

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_FEEDBACK_COLUMNS}
            FROM v2_answer_feedback f
            WHERE f.answer_run_id=%s
            ORDER BY f.id DESC
            """,
            (int(run_id),),
        )
        return [_feedback_to_dict(row) for row in cur.fetchall()]


def list_unresolved_feedback(conn, limit: int = 50) -> list[dict]:
    """Open gap queue for the Inbox filter; business gaps only, no tickets."""

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_FEEDBACK_COLUMNS}, r.question AS run_question,
                   r.answer_status AS run_answer_status
            FROM v2_answer_feedback f
            JOIN v2_answer_runs r ON r.id=f.answer_run_id
            WHERE f.status='open'
            ORDER BY f.id DESC
            LIMIT %s
            """,
            (max(1, min(int(limit), 200)),),
        )
        return [_feedback_to_dict(row) for row in cur.fetchall()]


def count_unresolved_feedback(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS count FROM v2_answer_feedback WHERE status='open'"
        )
        row = cur.fetchone()
    return int(row["count"]) if row else 0


def _get_run(conn, run_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, question, context_json FROM v2_answer_runs WHERE id=%s",
            (int(run_id),),
        )
        row = cur.fetchone()
    if not row:
        raise FeedbackNotFound(f"V2 answer run {int(run_id)} was not found")
    return dict(row)


def _get_target(conn, knowledge_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, trust, active, revision
            FROM v2_knowledge WHERE id=%s
            """,
            (int(knowledge_id),),
        )
        row = cur.fetchone()
    if not row:
        raise FeedbackNotFound(f"V2 Knowledge {int(knowledge_id)} was not found")
    target = dict(row)
    if not target.get("active"):
        raise FeedbackConflict("target Knowledge is inactive; restore it first")
    return target


def create_feedback(
    conn,
    *,
    answer_run_id: int,
    idempotency_key: str | None,
    feedback_kind: str,
    correction_text: str = "",
    applicability: Any = None,
    unit_kind: str = "experience",
    target_knowledge_id: int | None = None,
    expected_revision: int | None = None,
    field_result: str | None = None,
    reviewer_label: str = "",
    expected_knowledge_ids: list[int] | None = None,
) -> tuple[dict, bool]:
    """File one correction; same idempotency key returns the stored row."""

    kind = str(feedback_kind or "").strip()
    if kind not in FEEDBACK_KINDS:
        raise FeedbackConflict(f"unknown feedback kind: {kind!r}")
    if unit_kind not in UNIT_KINDS:
        raise FeedbackConflict(f"unknown unit kind: {unit_kind!r}")
    if field_result is not None and field_result not in FIELD_RESULTS:
        raise FeedbackConflict(f"unknown field result: {field_result!r}")
    expected_ids = _clean_expected_ids(expected_knowledge_ids)
    if expected_ids and kind != "retrieval_failure":
        raise FeedbackConflict("expected_knowledge_ids only apply to retrieval_failure")
    text = _text(correction_text)
    if kind in ("reply_only", "save_experience") and not text:
        raise FeedbackConflict(f"{kind} requires a correction text")
    clean_applicability = _clean_applicability(applicability)
    key = str(idempotency_key or "").strip()[:200] or str(uuid.uuid4())
    run = _get_run(conn, answer_run_id)

    target = None
    if target_knowledge_id is not None:
        if kind != "save_experience":
            raise FeedbackConflict("only save_experience takes an update target")
        if expected_revision is None:
            raise FeedbackConflict(
                "updating known Knowledge requires its expected revision"
            )
        target = _get_target(conn, target_knowledge_id)
        expected_revision = int(expected_revision)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM v2_answer_feedback WHERE idempotency_key=%s",
            (key,),
        )
        existing = cur.fetchone()
    if existing:
        row = get_feedback(conn, int(existing["id"]))
        assert row is not None
        return row, True

    evidence = _insert_evidence(
        conn,
        text or f"{kind} reported against answer run {int(run['id'])}",
        0,
        channel="answer_feedback",
        label="Answer feedback",
        input_mode="answer_correction",
    )
    # _insert_evidence always records thread 0 in raw_payload; point the
    # payload at the corrected run instead (feedback has no Inbox thread).
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_raw_evidence
            SET raw_payload=%s, source_locator=%s
            WHERE id=%s
            """,
            (
                Jsonb({
                    "answer_run_id": int(run["id"]),
                    "feedback_kind": kind,
                    "channel": "answer_feedback",
                }),
                f"v2-answer-run:{int(run['id'])}",
                int(evidence["id"]),
            ),
        )

    status = "closed" if kind == "reply_only" else "open"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_answer_feedback(
                answer_run_id, idempotency_key, feedback_kind, correction_text,
                applicability, unit_kind, target_knowledge_id,
                expected_revision, raw_evidence_id, status, field_result,
                expected_knowledge_ids, reviewer_label
            ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                int(run["id"]), key, kind, text, Jsonb(clean_applicability),
                unit_kind,
                int(target["id"]) if target else None,
                expected_revision,
                int(evidence["id"]), status, field_result,
                expected_ids,
                str(reviewer_label or "").strip()[:200],
            ),
        )
        feedback_id = int(cur.fetchone()["id"])

    if kind == "save_experience":
        if target is None:
            knowledge = _insert_provisional_experience(
                conn, text, clean_applicability, unit_kind,
            )
            knowledge_id: int | None = int(knowledge["id"])
            comparison = "NEW"
            related: list[int] = []
        else:
            knowledge_id = None
            comparison = "ENRICH"
            related = [int(target["id"])]
        _link_source(
            conn,
            int(target["id"]) if target else int(knowledge["id"]),
            int(evidence["id"]),
            source_kind="user_input",
            relation="supports",
            source_role="supporting",
            resolution="unresolved",
            excerpt=text[:4000],
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO v2_learning_proposals(
                    thread_id, fact_text, entity_name, proposed_trust,
                    status, comparison_result, comparison_reason,
                    related_knowledge_ids, unit_kind, applicability, revision
                ) VALUES(NULL, %s, '', 'provisional', 'pending_confirmation',
                         %s, 'answer_feedback', %s, %s, %s, 1)
                RETURNING id
                """,
                (text, comparison, related, unit_kind, Jsonb(clean_applicability)),
            )
            proposal_id = int(cur.fetchone()["id"])
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE v2_answer_feedback
                SET proposal_id=%s, knowledge_id=%s, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s
                """,
                (proposal_id, knowledge_id, feedback_id),
            )
    row = get_feedback(conn, feedback_id)
    assert row is not None
    return row, False


def _insert_provisional_experience(
    conn, text: str, applicability: dict, unit_kind: str,
) -> dict:
    """Store the submitted correction as provisional; never trusted early."""

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge(
                title, content, entity_name, trust, active,
                unit_kind, applicability, revision
            ) VALUES(%s, %s, '', 'provisional', TRUE, %s, %s, 1)
            RETURNING id, title, content, entity_name, trust, active,
                      unit_kind, applicability, revision,
                      created_at, updated_at
            """,
            (_title_for(text), text, unit_kind, Jsonb(applicability)),
        )
        return dict(cur.fetchone())


def _knowledge_before(conn, knowledge_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, entity_name, trust, active,
                   unit_kind, applicability, revision
            FROM v2_knowledge WHERE id=%s
            """,
            (int(knowledge_id),),
        )
        row = cur.fetchone()
    if not row:
        raise FeedbackNotFound(f"V2 Knowledge {int(knowledge_id)} was not found")
    return dict(row)


def _write_confirm_history(conn, knowledge_id: int, before: dict, after: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO v2_knowledge_history(
                knowledge_id, action, before_json, after_json
            ) VALUES(%s, 'confirm', %s, %s)
            """,
            (int(knowledge_id), Jsonb(before), Jsonb(after)),
        )


def confirm_feedback(
    conn,
    feedback_id: int,
    *,
    confirmed_text: str | None = None,
    applicability: Any = None,
    reviewer_label: str = "",
) -> tuple[dict, bool]:
    """Explicitly confirm one Experience; idempotent, no model calls.

    The engineer confirms the exact text shown to them.  New Experience flips
    the provisional row to ``user_confirmed``; an update requires the stored
    target id plus a matching revision, otherwise 409.  Embedding recompute
    is left to the asynchronous backfill.
    """

    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM v2_answer_feedback WHERE id=%s FOR UPDATE",
            (int(feedback_id),),
        )
        row = cur.fetchone()
    if not row:
        raise FeedbackNotFound(f"V2 answer feedback {int(feedback_id)} was not found")
    feedback = dict(row)
    if feedback.get("status") == "confirmed":
        knowledge = _knowledge_before(conn, int(feedback["knowledge_id"]))
        return knowledge, True
    if feedback.get("status") != "open":
        raise FeedbackConflict("only an open save_experience feedback can be confirmed")
    if feedback.get("feedback_kind") != "save_experience":
        raise FeedbackConflict("only save_experience feedback carries confirmable Experience")
    if not feedback.get("proposal_id"):
        raise FeedbackConflict("save_experience feedback has no proposal to confirm")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM v2_learning_proposals WHERE id=%s FOR UPDATE",
            (int(feedback["proposal_id"]),),
        )
        proposal_row = cur.fetchone()
    if not proposal_row:
        raise FeedbackConflict("the Experience proposal is gone")
    proposal = dict(proposal_row)
    if proposal.get("status") != "pending_confirmation":
        raise FeedbackConflict("the Experience proposal is no longer pending")

    final_text = _text(confirmed_text) if confirmed_text is not None else _text(proposal.get("fact_text") or "")
    if not final_text:
        raise FeedbackConflict("confirmed Experience text must not be empty")
    clean_applicability = (
        _clean_applicability(applicability)
        if applicability is not None
        else dict(feedback.get("applicability") or {})
    )
    label = str(reviewer_label or feedback.get("reviewer_label") or "").strip()[:200]

    target_id = feedback.get("target_knowledge_id")
    if target_id is not None:
        knowledge = _confirm_update(
            conn, feedback, proposal, int(target_id),
            int(feedback["expected_revision"]),
            final_text, clean_applicability,
        )
    else:
        if not feedback.get("knowledge_id"):
            raise FeedbackConflict("save_experience feedback has no provisional Knowledge")
        knowledge = _confirm_new(
            conn, feedback, proposal, int(feedback["knowledge_id"]),
            final_text, clean_applicability,
        )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_answer_feedback
            SET status='confirmed', knowledge_id=%s, reviewer_label=%s,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (int(knowledge["id"]), label, int(feedback["id"])),
        )
    return knowledge, False


def _accept_feedback_source(conn, knowledge_id: int, evidence_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge_sources
            SET resolution='accepted'
            WHERE knowledge_id=%s AND raw_evidence_id=%s
              AND active=TRUE AND relation='supports'
            """,
            (int(knowledge_id), int(evidence_id)),
        )


def _confirm_new(conn, feedback: dict, proposal: dict, knowledge_id: int,
                 final_text: str, applicability: dict) -> dict:
    before = _knowledge_before(conn, knowledge_id)
    if not before.get("active") or before.get("trust") != "provisional":
        raise FeedbackConflict("the provisional Experience is no longer confirmable")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET trust='user_confirmed', content=%s, applicability=%s,
                revision=revision+1, active=TRUE, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND active=TRUE AND trust='provisional'
            RETURNING id, title, content, entity_name, trust, active,
                      unit_kind, applicability, revision,
                      created_at, updated_at
            """,
            (final_text, Jsonb(applicability), knowledge_id),
        )
        updated = cur.fetchone()
    if not updated:
        raise FeedbackConflict("the provisional Experience changed under confirm")
    knowledge = dict(updated)
    _accept_feedback_source(conn, knowledge_id, int(feedback["raw_evidence_id"]))
    _finish_proposal(conn, int(proposal["id"]), knowledge_id)
    _write_confirm_history(conn, knowledge_id, _history_view(before), _history_view(knowledge))
    return knowledge


def _confirm_update(conn, feedback: dict, proposal: dict, target_id: int,
                    expected_revision: int, final_text: str,
                    applicability: dict) -> dict:
    before = _knowledge_before(conn, target_id)
    if not before.get("active"):
        raise FeedbackConflict("target Knowledge is inactive; restore it first")
    if int(before.get("revision") or 0) != int(expected_revision):
        raise StaleRevision(
            f"target Knowledge is at revision {before.get('revision')}, "
            f"not {int(expected_revision)}; re-read it before confirming"
        )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_knowledge
            SET content=%s, applicability=%s,
                trust=CASE WHEN trust='provisional'
                           THEN 'user_confirmed' ELSE trust END,
                revision=revision+1, active=TRUE, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND active=TRUE AND revision=%s
            RETURNING id, title, content, entity_name, trust, active,
                      unit_kind, applicability, revision,
                      created_at, updated_at
            """,
            (final_text, Jsonb(applicability), target_id, int(expected_revision)),
        )
        updated = cur.fetchone()
    if not updated:
        # Lost a race with another writer after the revision check.
        raise StaleRevision("target Knowledge changed under confirm; re-read it")
    knowledge = dict(updated)
    _accept_feedback_source(conn, target_id, int(feedback["raw_evidence_id"]))
    _finish_proposal(conn, int(proposal["id"]), target_id)
    _write_confirm_history(conn, target_id, _history_view(before), _history_view(knowledge))
    return knowledge


def _history_view(knowledge: dict) -> dict:
    return {
        "trust": str(knowledge.get("trust") or ""),
        "unit_kind": str(knowledge.get("unit_kind") or ""),
        "applicability": knowledge.get("applicability") or {},
        "revision": int(knowledge.get("revision") or 0),
        "content": str(knowledge.get("content") or ""),
    }


def _finish_proposal(conn, proposal_id: int, knowledge_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_learning_proposals
            SET status='confirmed', confirmed_knowledge_id=%s,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            """,
            (int(knowledge_id), int(proposal_id)),
        )


def close_feedback(conn, feedback_id: int) -> dict:
    """Close an open gap record without creating Knowledge."""

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_answer_feedback
            SET status='closed', updated_at=CURRENT_TIMESTAMP
            WHERE id=%s AND status='open'
            RETURNING id
            """,
            (int(feedback_id),),
        )
        updated = cur.fetchone()
    if not updated:
        row = get_feedback(conn, feedback_id)
        if row is None:
            raise FeedbackNotFound(f"V2 answer feedback {int(feedback_id)} was not found")
        return row
    row = get_feedback(conn, int(updated["id"]))
    assert row is not None
    return row


def set_answer_verdict(
    conn, run_id: int, *, verdict: str,
    reason: str = "", reviewer_label: str = "",
) -> dict:
    """Record a human retest judgement; never written by the model."""

    clean = str(verdict or "").strip()
    if clean not in VERDICTS:
        raise FeedbackConflict(f"verdict must be one of {VERDICTS}")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_answer_runs
            SET reviewer_verdict=%s, reviewer_reason=%s, reviewer_label=%s,
                reviewed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
            WHERE id=%s
            RETURNING id
            """,
            (
                clean, str(reason or "").strip()[:2000],
                str(reviewer_label or "").strip()[:200], int(run_id),
            ),
        )
        updated = cur.fetchone()
    if not updated:
        raise FeedbackNotFound(f"V2 answer run {int(run_id)} was not found")
    from v2.answering import get_answer_run

    row = get_answer_run(conn, int(updated["id"]))
    assert row is not None
    return row


def retest_feedback(
    feedback_id: int,
    *,
    db_factory,
    llm_service=None,
    embedding_client=None,
    idempotency_key: str | None = None,
    top_k: int = 5,
) -> dict:
    """Answer the original question again from current Knowledge.

    Creates a brand-new run linked via ``retest_of``/``feedback_id``; the old
    run is never overwritten and the correction text is never injected into
    the model input.
    """

    from v2.answering import answer_question

    with db_factory() as conn:
        feedback = get_feedback(conn, feedback_id)
        if feedback is None:
            raise FeedbackNotFound(f"V2 answer feedback {int(feedback_id)} was not found")
        run_id = int(feedback["answer_run_id"])
        with conn.cursor() as cur:
            cur.execute(
                "SELECT question, context_json FROM v2_answer_runs WHERE id=%s",
                (run_id,),
            )
            original = cur.fetchone()
        if not original:
            raise FeedbackNotFound(f"V2 answer run {run_id} was not found")
        original = dict(original)
    context = dict(original.get("context_json") or {})
    context["retest_of"] = run_id
    context["feedback_id"] = int(feedback["id"])
    return answer_question(
        str(original.get("question") or ""),
        context=context,
        idempotency_key=(str(idempotency_key or "").strip() or str(uuid.uuid4())),
        db_factory=db_factory,
        llm_service=llm_service,
        embedding_client=embedding_client,
        top_k=top_k,
        retest_of=run_id,
        feedback_id=int(feedback["id"]),
    )


def classify_run_failure(run: dict, feedback: dict | None = None) -> tuple[str, str]:
    """Map one failed run to a Phase 5.3 category plus its default action.

    Engineer-filed feedback kinds outrank heuristics.  The mapping never
    creates Knowledge and never changes retrieval by itself.
    """

    feedback_kind = str((feedback or {}).get("feedback_kind") or "")
    if feedback_kind == "retrieval_failure":
        category = "retrieval_failure"
    elif feedback_kind == "generation_failure":
        category = "generation_failure"
    elif feedback_kind == "missing_information":
        category = "knowledge_missing"
    elif feedback_kind == "field_result_failure":
        category = "applicability_version_failure"
    elif str(run.get("execution_status") or "") == "failed" or str(
        run.get("answer_status") or ""
    ) == "service_error":
        category = "service_failure"
    elif str(run.get("answer_status") or "") == "needs_clarification" or str(
        run.get("reason_code") or ""
    ) in ("model_not_covered", "missing_model", "missing_version"):
        category = "applicability_version_failure"
    else:
        trace = run.get("retrieval_trace") or {}
        candidates = list(trace.get("candidate_knowledge_ids") or [])
        eligible = list(trace.get("eligible_knowledge_ids") or [])
        document = trace.get("document") or {}
        if candidates:
            category = "generation_failure"
        elif eligible:
            category = "knowledge_missing"
        elif list(document.get("candidate_block_ids") or []):
            category = "knowledge_missing"
        else:
            category = "missing_source"
    return category, FAILURE_ACTIONS[category]


def list_failures(conn, *, days: int = 7, limit: int = 50) -> list[dict]:
    """Recent unjudged failures with categories for the lightweight view."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.question, r.execution_status, r.answer_status,
                   r.reason_code, r.reviewer_verdict, r.created_at,
                   r.retrieval_trace, r.retest_of, r.feedback_id
            FROM v2_answer_runs r
            WHERE r.created_at >= CURRENT_TIMESTAMP - make_interval(days => %s)
              AND (r.answer_status IN ('unsupported', 'service_error')
                   OR r.execution_status = 'failed')
              AND (r.reviewer_verdict IS NULL OR r.reviewer_verdict = 'fail')
            ORDER BY r.id DESC
            LIMIT %s
            """,
            (max(1, min(int(days), 90)), max(1, min(int(limit), 200))),
        )
        runs = [dict(row) for row in cur.fetchall()]
    items = []
    for run in runs:
        feedback = None
        if run.get("feedback_id"):
            feedback = get_feedback(conn, int(run["feedback_id"]))
        if feedback is None:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {_FEEDBACK_COLUMNS}
                    FROM v2_answer_feedback f
                    WHERE f.answer_run_id=%s AND f.status='open'
                    ORDER BY f.id DESC LIMIT 1
                    """,
                    (int(run["id"]),),
                )
                row = cur.fetchone()
                feedback = _feedback_to_dict(row) if row else None
        category, action = classify_run_failure(run, feedback)
        trace = run.get("retrieval_trace") or {}
        items.append({
            "run_id": int(run["id"]),
            "question": str(run.get("question") or "")[:300],
            "answer_status": str(run.get("answer_status") or ""),
            "reason_code": str(run.get("reason_code") or ""),
            "reviewer_verdict": run.get("reviewer_verdict"),
            "category": category,
            "default_action": action,
            "feedback_id": (int(feedback["id"]) if feedback else None),
            "feedback_kind": str((feedback or {}).get("feedback_kind") or ""),
            "eligible_count": len(list(trace.get("eligible_knowledge_ids") or [])),
            "candidate_count": len(list(trace.get("candidate_knowledge_ids") or [])),
            "created_at": run.get("created_at"),
        })
    return items


def retrieval_gate_progress(conn) -> dict:
    """How far the retrieval-improvement gate has progressed (X/10).

    Only retrieval_failure feedback whose expected Knowledge is currently
    answer-eligible counts: the evidence must provably be in the library.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.expected_knowledge_ids
            FROM v2_answer_feedback f
            WHERE f.feedback_kind='retrieval_failure'
              AND f.status IN ('open', 'confirmed')
              AND coalesce(array_length(f.expected_knowledge_ids, 1), 0) > 0
            ORDER BY f.id
            """,
        )
        rows = [dict(row) for row in cur.fetchall()]
    qualifying = []
    for row in rows:
        expected = [int(item) for item in (row.get("expected_knowledge_ids") or [])]
        if not expected:
            continue
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT k.id
                FROM v2_knowledge k
                WHERE k.id = ANY(%s)
                  AND k.active=TRUE
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
                """,
                (expected,),
            )
            eligible = {int(item["id"]) for item in cur.fetchall()}
        if eligible:
            qualifying.append({"feedback_id": int(row["id"]),
                               "eligible_expected_ids": sorted(eligible)})
    return {
        "qualifying_cases": len(qualifying),
        "needed": RETRIEVAL_GATE_NEEDED,
        "gate_open": len(qualifying) >= RETRIEVAL_GATE_NEEDED,
        "cases": qualifying,
    }


__all__ = [
    "FAILURE_ACTIONS",
    "FAILURE_CATEGORIES",
    "FEEDBACK_KINDS",
    "FEEDBACK_STATUSES",
    "FIELD_RESULTS",
    "RETRIEVAL_GATE_NEEDED",
    "UNIT_KINDS",
    "VERDICTS",
    "FeedbackConflict",
    "FeedbackNotFound",
    "StaleRevision",
    "classify_run_failure",
    "close_feedback",
    "confirm_feedback",
    "count_unresolved_feedback",
    "create_feedback",
    "get_feedback",
    "list_failures",
    "list_feedback_for_run",
    "list_unresolved_feedback",
    "retest_feedback",
    "retrieval_gate_progress",
    "set_answer_verdict",
]
