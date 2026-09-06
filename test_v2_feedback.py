"""Unit tests for the Phase 3.2 correction -> Experience -> retest loop.

A fake in-memory store emulates the feedback, evidence, Knowledge,
proposal, source, history, and run tables so submit/confirm/close/verdict
semantics, idempotency, and revision guards are exercised without
PostgreSQL.  Real SQL persistence is covered by the PostgreSQL integration
tests.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from v2.feedback import (
    FeedbackConflict,
    FeedbackNotFound,
    StaleRevision,
    close_feedback,
    confirm_feedback,
    count_unresolved_feedback,
    create_feedback,
    get_feedback,
    list_feedback_for_run,
    list_unresolved_feedback,
    set_answer_verdict,
)


def _unwrap(value):
    return getattr(value, "obj", value)


def _now():
    return datetime.now(timezone.utc)


class FakeCursor:
    def __init__(self, state):
        self.state = state
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    # -- helpers ---------------------------------------------------------
    def _feedback_view(self, item):
        run = self.state["runs"].get(int(item["answer_run_id"]), {})
        return {
            **dict(item),
            "run_question": run.get("question", ""),
            "run_answer_status": run.get("answer_status", ""),
        }

    # -- dispatch ----------------------------------------------------------
    def execute(self, query, params=()):
        text = " ".join(str(query).split())
        self._result = []
        state = self.state
        if text.startswith("SELECT id, question, context_json FROM v2_answer_runs"):
            row = state["runs"].get(int(params[0]))
            self._result = [dict(row)] if row else []
        elif text.startswith("SELECT id FROM v2_answer_feedback WHERE idempotency_key="):
            for item in state["feedback"].values():
                if item["idempotency_key"] == params[0]:
                    self._result = [{"id": item["id"]}]
        elif text.startswith("INSERT INTO v2_raw_evidence("):
            evidence_id = state["next_id"]
            state["next_id"] += 1
            state["evidence"][evidence_id] = {
                "id": evidence_id, "evidence_type": "user_input",
                "author_role": "product_expert", "content": params[0],
                "raw_payload": _unwrap(params[1]), "source_label": params[2],
                "source_locator": params[3], "created_at": _now(),
            }
            self._result = [dict(state["evidence"][evidence_id])]
        elif text.startswith("UPDATE v2_raw_evidence"):
            item = state["evidence"][int(params[2])]
            item["raw_payload"] = _unwrap(params[0])
            item["source_locator"] = params[1]
        elif text.startswith("INSERT INTO v2_answer_feedback("):
            feedback_id = state["next_id"]
            state["next_id"] += 1
            state["feedback"][feedback_id] = {
                "id": feedback_id, "answer_run_id": int(params[0]),
                "idempotency_key": params[1], "feedback_kind": params[2],
                "correction_text": params[3], "applicability": _unwrap(params[4]),
                "unit_kind": params[5], "target_knowledge_id": params[6],
                "expected_revision": params[7], "raw_evidence_id": int(params[8]),
                "proposal_id": None, "knowledge_id": None, "status": params[9],
                "field_result": params[10], "expected_knowledge_ids": list(params[11]),
                "reviewer_label": params[12],
                "created_at": _now(), "updated_at": _now(),
            }
            self._result = [{"id": feedback_id}]
        elif text.startswith("INSERT INTO v2_knowledge("):
            knowledge_id = state["next_id"]
            state["next_id"] += 1
            state["knowledge"][knowledge_id] = {
                "id": knowledge_id, "title": params[0], "content": params[1],
                "entity_name": "", "trust": "provisional", "active": True,
                "unit_kind": params[2], "applicability": _unwrap(params[3]),
                "revision": 1, "created_at": _now(), "updated_at": _now(),
            }
            self._result = [dict(state["knowledge"][knowledge_id])]
        elif text.startswith("INSERT INTO v2_knowledge_sources("):
            state["sources"].append({
                "knowledge_id": int(params[0]), "raw_evidence_id": int(params[1]),
                "source_kind": params[2], "relation": params[3],
                "source_role": params[4], "excerpt": params[5],
                "resolution": params[6], "active": True,
            })
        elif text.startswith("INSERT INTO v2_learning_proposals("):
            proposal_id = state["next_id"]
            state["next_id"] += 1
            state["proposals"][proposal_id] = {
                "id": proposal_id, "thread_id": None, "fact_text": params[0],
                "entity_name": "", "proposed_trust": "provisional",
                "status": "pending_confirmation", "comparison_result": params[1],
                "comparison_reason": "answer_feedback",
                "related_knowledge_ids": list(params[2]),
                "unit_kind": params[3], "applicability": _unwrap(params[4]),
                "revision": 1, "confirmed_knowledge_id": None,
            }
            self._result = [{"id": proposal_id}]
        elif text.startswith("UPDATE v2_answer_feedback SET proposal_id="):
            item = state["feedback"][int(params[2])]
            item["proposal_id"] = int(params[0])
            item["knowledge_id"] = int(params[1]) if params[1] is not None else None
            item["updated_at"] = _now()
        elif text.startswith("SELECT * FROM v2_answer_feedback WHERE id="):
            item = state["feedback"].get(int(params[0]))
            self._result = [dict(item)] if item else []
        elif text.startswith("SELECT * FROM v2_learning_proposals WHERE id="):
            item = state["proposals"].get(int(params[0]))
            self._result = [dict(item)] if item else []
        elif text.startswith("SELECT id, title, content, entity_name, trust, active"):
            item = state["knowledge"].get(int(params[0]))
            self._result = [dict(item)] if item else []
        elif text.startswith("UPDATE v2_knowledge SET trust='user_confirmed'"):
            item = state["knowledge"][int(params[2])]
            assert item["active"] and item["trust"] == "provisional"
            item["content"] = params[0]
            item["applicability"] = _unwrap(params[1])
            item["trust"] = "user_confirmed"
            item["revision"] += 1
            item["updated_at"] = _now()
            self._result = [dict(item)]
        elif text.startswith("UPDATE v2_knowledge SET content="):
            item = state["knowledge"][int(params[2])]
            if int(item["revision"]) != int(params[3]):
                self._result = []
            else:
                item["content"] = params[0]
                item["applicability"] = _unwrap(params[1])
                if item["trust"] == "provisional":
                    item["trust"] = "user_confirmed"
                item["revision"] += 1
                item["updated_at"] = _now()
                self._result = [dict(item)]
        elif text.startswith("UPDATE v2_knowledge_sources SET resolution='accepted'"):
            for source in state["sources"]:
                if (int(source["knowledge_id"]) == int(params[0])
                        and int(source["raw_evidence_id"]) == int(params[1])):
                    source["resolution"] = "accepted"
        elif text.startswith("UPDATE v2_learning_proposals SET status='confirmed'"):
            item = state["proposals"][int(params[1])]
            item["status"] = "confirmed"
            item["confirmed_knowledge_id"] = int(params[0])
        elif text.startswith("INSERT INTO v2_knowledge_history("):
            state["history"].append({
                "knowledge_id": int(params[0]), "action": "confirm",
                "before_json": _unwrap(params[1]), "after_json": _unwrap(params[2]),
            })
        elif text.startswith("UPDATE v2_answer_feedback SET status='confirmed'"):
            item = state["feedback"][int(params[2])]
            item["status"] = "confirmed"
            item["knowledge_id"] = int(params[0])
            item["reviewer_label"] = params[1]
            item["updated_at"] = _now()
        elif text.startswith("UPDATE v2_answer_feedback SET status='closed'"):
            item = state["feedback"].get(int(params[0]))
            if item and item["status"] == "open":
                item["status"] = "closed"
                item["updated_at"] = _now()
                self._result = [{"id": item["id"]}]
            else:
                self._result = []
        elif text.startswith("UPDATE v2_answer_runs SET reviewer_verdict="):
            row = state["runs"].get(int(params[3]))
            if row:
                row["reviewer_verdict"] = params[0]
                row["reviewer_reason"] = params[1]
                row["reviewer_label"] = params[2]
                row["reviewed_at"] = _now()
                self._result = [{"id": row["id"]}]
            else:
                self._result = []
        elif "FROM v2_answer_feedback f" in text:
            items = list(state["feedback"].values())
            if "WHERE f.answer_run_id=" in text:
                items = [i for i in items if int(i["answer_run_id"]) == int(params[0])]
            elif "WHERE f.status='open'" in text:
                items = [i for i in items if i["status"] == "open"]
            elif "WHERE f.id=" in text:
                item = state["feedback"].get(int(params[0]))
                items = [item] if item else []
            limit = params[-1] if "LIMIT %s" in text else None
            views = [self._feedback_view(i) for i in sorted(items, key=lambda i: -i["id"])]
            self._result = views[: int(limit)] if limit else views
        elif text.startswith("SELECT count(*) AS count FROM v2_answer_feedback"):
            self._result = [{"count": sum(1 for i in state["feedback"].values() if i["status"] == "open")}]
        elif "FROM v2_answer_runs" in text and "WHERE id=" in text:
            row = state["runs"].get(int(params[0]))
            self._result = [dict(row)] if row else []
        elif text.startswith("SELECT id, trust, active, revision FROM v2_knowledge"):
            item = state["knowledge"].get(int(params[0]))
            self._result = [dict(item)] if item else []
        else:  # pragma: no cover - every query above is exercised below
            raise AssertionError(f"unhandled query: {text[:100]}")


class FakeConn:
    def __init__(self, state):
        self.state = state

    def cursor(self):
        return FakeCursor(self.state)


def _run(run_id=7, question="Где включается ONVIF?"):
    return {
        "id": run_id, "question": question, "context_json": {},
        "answer_status": "unsupported",
    }


class FeedbackValidationTest(unittest.TestCase):
    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(FeedbackConflict):
            create_feedback(FakeConn({}), answer_run_id=1,
                            idempotency_key="k", feedback_kind="nope")

    def test_correction_kinds_require_text(self):
        for kind in ("reply_only", "save_experience"):
            with self.assertRaises(FeedbackConflict):
                create_feedback(FakeConn({}), answer_run_id=1,
                                idempotency_key="k", feedback_kind=kind,
                                correction_text="  ")

    def test_overlong_correction_is_rejected_not_truncated(self):
        with self.assertRaises(FeedbackConflict):
            create_feedback(FakeConn({}), answer_run_id=1,
                            idempotency_key="k", feedback_kind="reply_only",
                            correction_text="x" * 12001)

    def test_applicability_must_be_an_object(self):
        with self.assertRaises(FeedbackConflict):
            create_feedback(FakeConn({}), answer_run_id=1,
                            idempotency_key="k", feedback_kind="reply_only",
                            correction_text="ok", applicability=["DS-2CD"])

    def test_update_target_requires_save_experience_and_revision(self):
        state = {"runs": {1: _run(1)}, "feedback": {}, "evidence": {},
                 "knowledge": {}, "proposals": {}, "sources": [],
                 "history": [], "next_id": 100}
        with self.assertRaises(FeedbackConflict):
            create_feedback(FakeConn(state), answer_run_id=1,
                            idempotency_key="k", feedback_kind="reply_only",
                            correction_text="ok", target_knowledge_id=5,
                            expected_revision=1)
        with self.assertRaises(FeedbackConflict):
            create_feedback(FakeConn(state), answer_run_id=1,
                            idempotency_key="k", feedback_kind="save_experience",
                            correction_text="ok", target_knowledge_id=5)

    def test_missing_run_is_not_found(self):
        state = {"runs": {}, "feedback": {}, "evidence": {},
                 "knowledge": {}, "proposals": {}, "sources": [],
                 "history": [], "next_id": 100}
        with self.assertRaises(FeedbackNotFound):
            create_feedback(FakeConn(state), answer_run_id=9,
                            idempotency_key="k", feedback_kind="reply_only",
                            correction_text="ok")


class ReplyOnlyTest(unittest.TestCase):
    def setUp(self):
        self.state = {"runs": {7: _run()}, "feedback": {}, "evidence": {},
                      "knowledge": {}, "proposals": {}, "sources": [],
                      "history": [], "next_id": 100}
        self.conn = FakeConn(self.state)

    def test_reply_only_never_touches_knowledge(self):
        item, duplicate = create_feedback(
            self.conn, answer_run_id=7, idempotency_key="once",
            feedback_kind="reply_only", correction_text="Скажите так.",
        )
        self.assertFalse(duplicate)
        self.assertEqual(item["status"], "closed")
        self.assertIsNone(item["proposal_id"])
        self.assertIsNone(item["knowledge_id"])
        self.assertEqual(self.state["knowledge"], {})
        self.assertEqual(self.state["proposals"], {})
        self.assertEqual(self.state["sources"], [])
        self.assertEqual(len(self.state["evidence"]), 1)

    def test_confirm_reply_only_is_rejected(self):
        item, _ = create_feedback(
            self.conn, answer_run_id=7, idempotency_key="once",
            feedback_kind="reply_only", correction_text="Скажите так.",
        )
        with self.assertRaises(FeedbackConflict):
            confirm_feedback(self.conn, int(item["id"]))


class SaveExperienceTest(unittest.TestCase):
    def setUp(self):
        self.state = {"runs": {7: _run()}, "feedback": {}, "evidence": {},
                      "knowledge": {}, "proposals": {}, "sources": [],
                      "history": [], "next_id": 100}
        self.conn = FakeConn(self.state)

    def _submit(self, key="exp-1", **kwargs):
        params = {"answer_run_id": 7, "idempotency_key": key,
                  "feedback_kind": "save_experience",
                  "correction_text": "ONVIF включается в Network → Advanced.",
                  "applicability": {"models": ["DS-2CD2387G2P-LSU/SL"]}}
        params.update(kwargs)
        return create_feedback(self.conn, **params)

    def test_submit_stages_provisional_knowledge_and_pending_proposal(self):
        item, duplicate = self._submit()
        self.assertFalse(duplicate)
        self.assertEqual(item["status"], "open")
        knowledge = self.state["knowledge"][int(item["knowledge_id"])]
        self.assertEqual(knowledge["trust"], "provisional")
        self.assertEqual(knowledge["unit_kind"], "experience")
        self.assertEqual(knowledge["applicability"], {"models": ["DS-2CD2387G2P-LSU/SL"]})
        proposal = self.state["proposals"][int(item["proposal_id"])]
        self.assertEqual(proposal["status"], "pending_confirmation")
        self.assertEqual(proposal["comparison_result"], "NEW")
        (source,) = self.state["sources"]
        self.assertEqual(source["resolution"], "unresolved")

    def test_duplicate_key_returns_stored_row_without_new_knowledge(self):
        first, _ = self._submit()
        second, duplicate = self._submit()
        self.assertTrue(duplicate)
        self.assertEqual(int(first["id"]), int(second["id"]))
        self.assertEqual(len(self.state["knowledge"]), 1)

    def test_confirm_flips_to_user_confirmed_and_accepts_source(self):
        item, _ = self._submit()
        knowledge, duplicate = confirm_feedback(
            self.conn, int(item["id"]), reviewer_label="op",
        )
        self.assertFalse(duplicate)
        self.assertEqual(knowledge["trust"], "user_confirmed")
        self.assertEqual(knowledge["revision"], 2)
        (source,) = self.state["sources"]
        self.assertEqual(source["resolution"], "accepted")
        proposal = self.state["proposals"][int(item["proposal_id"])]
        self.assertEqual(proposal["status"], "confirmed")
        stored = get_feedback(self.conn, int(item["id"]))
        self.assertEqual(stored["status"], "confirmed")
        self.assertEqual(stored["reviewer_label"], "op")
        (entry,) = self.state["history"]
        self.assertEqual(entry["action"], "confirm")
        self.assertEqual(entry["before_json"]["trust"], "provisional")
        self.assertEqual(entry["after_json"]["trust"], "user_confirmed")

    def test_confirm_is_idempotent(self):
        item, _ = self._submit()
        first, _ = confirm_feedback(self.conn, int(item["id"]))
        second, duplicate = confirm_feedback(self.conn, int(item["id"]))
        self.assertTrue(duplicate)
        self.assertEqual(int(first["id"]), int(second["id"]))
        self.assertEqual(len(self.state["history"]), 1)
        self.assertEqual(len(self.state["knowledge"]), 1)

    def test_confirm_update_path_checks_revision(self):
        self.state["knowledge"][50] = {
            "id": 50, "title": "t", "content": "old", "entity_name": "",
            "trust": "user_confirmed", "active": True, "unit_kind": "fact",
            "applicability": {}, "revision": 3,
            "created_at": _now(), "updated_at": _now(),
        }
        item, _ = self._submit(
            key="upd-1", target_knowledge_id=50, expected_revision=3,
        )
        self.assertIsNone(item["knowledge_id"])
        proposal = self.state["proposals"][int(item["proposal_id"])]
        self.assertEqual(proposal["comparison_result"], "ENRICH")
        knowledge, _ = confirm_feedback(
            self.conn, int(item["id"]), confirmed_text="new",
        )
        self.assertEqual(int(knowledge["id"]), 50)
        self.assertEqual(knowledge["content"], "new")
        self.assertEqual(knowledge["revision"], 4)
        self.assertEqual(knowledge["trust"], "user_confirmed")

    def test_confirm_update_with_stale_revision_is_409(self):
        self.state["knowledge"][50] = {
            "id": 50, "title": "t", "content": "old", "entity_name": "",
            "trust": "user_confirmed", "active": True, "unit_kind": "fact",
            "applicability": {}, "revision": 4,
            "created_at": _now(), "updated_at": _now(),
        }
        item, _ = self._submit(
            key="upd-2", target_knowledge_id=50, expected_revision=3,
        )
        with self.assertRaises(StaleRevision):
            confirm_feedback(self.conn, int(item["id"]), confirmed_text="new")
        stored = get_feedback(self.conn, int(item["id"]))
        self.assertEqual(stored["status"], "open")


class GapQueueTest(unittest.TestCase):
    def setUp(self):
        self.state = {"runs": {7: _run()}, "feedback": {}, "evidence": {},
                      "knowledge": {}, "proposals": {}, "sources": [],
                      "history": [], "next_id": 100}
        self.conn = FakeConn(self.state)

    def test_open_gaps_are_listed_and_counted(self):
        create_feedback(self.conn, answer_run_id=7, idempotency_key="g1",
                        feedback_kind="retrieval_failure",
                        correction_text="Знание есть, но не нашлось.")
        create_feedback(self.conn, answer_run_id=7, idempotency_key="g2",
                        feedback_kind="reply_only", correction_text="ok")
        items = list_unresolved_feedback(self.conn)
        self.assertEqual([i["feedback_kind"] for i in items], ["retrieval_failure"])
        self.assertEqual(items[0]["run_question"], "Где включается ONVIF?")
        self.assertEqual(count_unresolved_feedback(self.conn), 1)

    def test_close_leaves_knowledge_alone(self):
        item, _ = create_feedback(
            self.conn, answer_run_id=7, idempotency_key="g1",
            feedback_kind="missing_information", correction_text="Нужна версия.",
        )
        closed = close_feedback(self.conn, int(item["id"]))
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(self.state["knowledge"], {})
        again = close_feedback(self.conn, int(item["id"]))
        self.assertEqual(again["status"], "closed")

    def test_feedback_history_per_run(self):
        create_feedback(self.conn, answer_run_id=7, idempotency_key="g1",
                        feedback_kind="reply_only", correction_text="a")
        create_feedback(self.conn, answer_run_id=7, idempotency_key="g2",
                        feedback_kind="reply_only", correction_text="b")
        items = list_feedback_for_run(self.conn, 7)
        self.assertEqual(len(items), 2)
        self.assertGreater(items[0]["id"], items[1]["id"])


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.state = {"runs": {7: _run()}, "feedback": {}, "evidence": {},
                      "knowledge": {}, "proposals": {}, "sources": [],
                      "history": [], "next_id": 100}
        self.conn = FakeConn(self.state)

    def test_verdict_is_recorded(self):
        set_answer_verdict(self.conn, 7, verdict="pass",
                           reason="Повтор совпал.", reviewer_label="op")
        row = self.state["runs"][7]
        self.assertEqual(row["reviewer_verdict"], "pass")
        self.assertEqual(row["reviewer_reason"], "Повтор совпал.")
        self.assertIsNotNone(row["reviewed_at"])

    def test_bad_verdict_is_rejected(self):
        with self.assertRaises(FeedbackConflict):
            set_answer_verdict(self.conn, 7, verdict="maybe")

    def test_missing_run_is_not_found(self):
        with self.assertRaises(FeedbackNotFound):
            set_answer_verdict(self.conn, 4242, verdict="pass")


class DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):  # pragma: no cover - module functions are patched below
        raise AssertionError("database must not be touched")


def _feedback_item():
    return {
        "id": 11, "answer_run_id": 7, "feedback_kind": "save_experience",
        "correction_text": "text", "applicability": {}, "unit_kind": "experience",
        "target_knowledge_id": None, "expected_revision": None,
        "raw_evidence_id": 3, "proposal_id": 5, "knowledge_id": None,
        "status": "open", "field_result": None, "reviewer_label": "",
        "run_question": "q", "run_answer_status": "unsupported",
        "created_at": None, "updated_at": None,
    }


class FeedbackApiTest(unittest.TestCase):
    """Route-level tests: HTTP mapping, not service logic."""

    def setUp(self):
        from unittest.mock import patch

        import app as app_module
        from app import (
            V2FeedbackConfirmIn,
            V2FeedbackIn,
            V2VerdictIn,
        )

        self.app_module = app_module
        self.V2FeedbackIn = V2FeedbackIn
        self.V2FeedbackConfirmIn = V2FeedbackConfirmIn
        self.V2VerdictIn = V2VerdictIn
        self._previous_api_key = app_module.settings["api_key"]
        app_module.settings["api_key"] = "test-key"
        self.db_patch = patch.object(app_module, "db", return_value=DummyConn())
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(app_module.settings.__setitem__, "api_key", self._previous_api_key)
        self._patches = []

    def _patch(self, name, value):
        patcher = unittest.mock.patch.object(self.app_module, name, value)
        patcher.start()
        self._patches.append(patcher)
        self.addCleanup(patcher.stop)

    def _payload(self, **kwargs):
        params = {"feedback_kind": "reply_only", "correction_text": "ok"}
        params.update(kwargs)
        return self.V2FeedbackIn(**params)

    def test_create_feedback_shape_and_duplicate(self):
        from app import v2_create_feedback

        self._patch("create_feedback", lambda conn, **_: (_feedback_item(), True))
        response = v2_create_feedback(
            7, self._payload(), x_api_key="test-key", idempotency_key="k",
        )
        self.assertEqual(response["feedback_id"], 11)
        self.assertTrue(response["duplicate"])
        self.assertEqual(response["status"], "open")

    def test_create_feedback_maps_errors(self):
        from fastapi import HTTPException

        from app import v2_create_feedback

        self._patch(
            "create_feedback",
            lambda conn, **_: (_ for _ in ()).throw(FeedbackNotFound("no run")),
        )
        with self.assertRaises(HTTPException) as caught:
            v2_create_feedback(7, self._payload(), x_api_key="test-key",
                               idempotency_key="k")
        self.assertEqual(caught.exception.status_code, 404)

    def test_create_feedback_conflict_is_409(self):
        from fastapi import HTTPException

        from app import v2_create_feedback

        def conflict(conn, **_):
            raise FeedbackConflict("bad target")

        self._patch("create_feedback", conflict)
        with self.assertRaises(HTTPException) as caught:
            v2_create_feedback(7, self._payload(), x_api_key="test-key",
                               idempotency_key="k")
        self.assertEqual(caught.exception.status_code, 409)

    def test_confirm_and_close_shapes(self):
        from app import v2_close_feedback, v2_confirm_feedback

        self._patch(
            "confirm_feedback",
            lambda conn, *_, **__: ({"id": 21, "trust": "user_confirmed",
                                     "unit_kind": "experience", "revision": 2}, False),
        )
        confirmed = v2_confirm_feedback(
            11, self.V2FeedbackConfirmIn(confirmed_text="final"), x_api_key="test-key",
        )
        self.assertEqual(confirmed["knowledge_id"], 21)
        self.assertEqual(confirmed["trust"], "user_confirmed")

        item = _feedback_item()
        item["status"] = "closed"
        self._patch("close_feedback", lambda conn, *_, **__: item)
        closed = v2_close_feedback(11, x_api_key="test-key")
        self.assertEqual(closed["status"], "closed")

    def test_retest_returns_new_run_shape(self):
        from app import v2_retest_feedback

        run = {
            "run_id": 31, "idempotency_key": "k", "question": "q",
            "execution_status": "completed", "answer_status": "answered",
            "answer_text": "new", "clarifying_question": "",
            "reason_code": "grounded_answer", "evidence_snapshot": [],
            "retrieval_trace": {}, "model": "m", "prompt_version": "v",
            "llm_requests": 1, "latency_ms": 5, "retest_of": 7,
            "feedback_id": 11, "duplicate": False,
            "created_at": None,
        }
        self._patch("retest_feedback", lambda *_, **__: run)
        response = v2_retest_feedback(11, x_api_key="test-key",
                                      idempotency_key="k")
        self.assertEqual(response["run_id"], 31)
        self.assertEqual(response["retest_of"], 7)
        self.assertEqual(response["answer_status"], "answered")

    def test_verdict_returns_run_shape(self):
        from app import v2_answer_verdict

        row = {"id": 7, "idempotency_key": "k", "question": "q",
               "execution_status": "completed", "answer_status": "answered",
               "answer_text": "a", "clarifying_question": "",
               "reason_code": "grounded_answer", "evidence_snapshot": [],
               "retrieval_trace": {}, "model": "m", "prompt_version": "v",
               "llm_requests": 1, "latency_ms": 5,
               "reviewer_verdict": "pass", "reviewer_reason": "",
               "reviewer_label": "op", "created_at": None,
               "updated_at": None}
        self._patch("set_answer_verdict", lambda conn, *_, **__: row)
        response = v2_answer_verdict(
            7, self.V2VerdictIn(verdict="pass"), x_api_key="test-key",
        )
        self.assertEqual(response["reviewer_verdict"], "pass")

    def test_unresolved_and_run_history_shapes(self):
        from app import v2_list_run_feedback, v2_unresolved_feedback

        self._patch("list_unresolved_feedback", lambda conn, **__: [_feedback_item()])
        self._patch("count_unresolved_feedback", lambda conn: 1)
        queued = v2_unresolved_feedback(x_api_key="test-key")
        self.assertEqual(queued["total"], 1)
        self.assertEqual(len(queued["items"]), 1)
        self._patch("list_feedback_for_run", lambda conn, _: [_feedback_item()])
        history = v2_list_run_feedback(7, x_api_key="test-key")
        self.assertEqual(len(history["items"]), 1)


if __name__ == "__main__":
    unittest.main()
