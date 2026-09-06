"""Unit tests for Phase 4.2 complete document knowledge units.

Context building, extraction shape, deterministic validation, rendering,
and confirm rules are exercised without a database.  Persistence, the
learn worker step, the validation answer gate, and a toy end-to-end
(parse -> extract -> confirm -> answer) run against PostgreSQL when
``V2_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import psycopg
from psycopg.rows import dict_row

from v2.document_learning import (
    DocumentLearnError,
    build_extract_messages,
    build_learning_contexts,
    check_units,
    extract_units,
    render_unit_content,
)


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


def _block(block_id, text, section="Guide", page=1, kind="paragraph"):
    return {
        "id": block_id, "block_key": f"p{page}-{block_id}",
        "section_path": [section], "page_no": page, "slide_no": None,
        "block_type": kind, "evidence_text": text,
    }


def _version():
    return {"document_key": "MANUAL", "title": "Quick Start Guide", "file_name": "m.pdf"}


def _blocks():
    return {
        11: ("p1-11", "Open Users and tap Add to create one."),
        12: ("p1-12", "Enroll two fingerprints per user."),
    }


def _unit(**kwargs):
    params = {
        "title": "Add a user",
        "unit_kind": "procedure",
        "content": "Open Users and tap Add to create one.",
        "applicability": {"models": ["F-NR-208E/2"]},
        "ordered_steps": ["Open Users", "tap Add"],
        "expected_result": "A new user exists.",
        "sources": [{"block_id": 11, "excerpt": "Open Users and tap Add to create one."}],
    }
    params.update(kwargs)
    return params


class ContextTest(unittest.TestCase):
    def test_sections_become_contexts_and_images_stay_out(self):
        blocks = [
            _block(1, "Quick Start Guide", section="Guide", page=1, kind="heading"),
            _block(2, "Open Users and tap Add.", section="Guide", page=1),
            _block(3, "", section="Guide", page=2, kind="image"),
            _block(4, "Enroll fingerprints.", section="Fingerprints", page=2),
        ]
        contexts = build_learning_contexts(blocks)
        self.assertEqual([ctx["context_key"] for ctx in contexts], ["Guide", "Fingerprints"])
        self.assertEqual([item["block_id"] for item in contexts[0]["blocks"]], [1, 2])
        self.assertNotIn(3, [item["block_id"] for ctx in contexts for item in ctx["blocks"]])

    def test_oversize_section_splits_by_block(self):
        import v2.document_learning as learning

        blocks = [_block(i, f"Step number {i} of the onboarding flow. ", section="Big") for i in range(1, 6)]
        old, learning.MAX_CONTEXT_CHARS = learning.MAX_CONTEXT_CHARS, 100
        try:
            contexts = build_learning_contexts(blocks)
        finally:
            learning.MAX_CONTEXT_CHARS = old
        self.assertGreater(len(contexts), 1)
        self.assertTrue(all(ctx["context_key"].startswith("Big") for ctx in contexts))
        total = sum(len(ctx["blocks"]) for ctx in contexts)
        self.assertEqual(total, 5)


class ExtractShapeTest(unittest.TestCase):
    def test_messages_carry_block_ids_and_prompt_version(self):
        import v2.document_learning as learning

        version = _version()
        context = {"context_key": "Guide", "title": "Guide", "blocks": [
            {"block_id": 11, "block_key": "p1-11", "locator": "page 1", "text": "abc"},
        ], "approx_chars": 3}
        messages = build_extract_messages(version, context)
        self.assertEqual(len(messages), 2)
        user = json.loads(messages[1]["content"])
        self.assertEqual(user["blocks"][0]["block_id"], 11)
        self.assertIn("block_id", messages[0]["content"])
        self.assertEqual(learning.V2_DOC_EXTRACT_PROMPT_VERSION, "v2-doc-extract-1")

    def test_extract_parses_units_list(self):
        class FakeExtractor:
            def extract_structured(self, messages, schema, max_tokens):
                assert isinstance(schema, dict)
                return json.dumps({"units": [_unit()]})

        units = extract_units(FakeExtractor(), _version(), {"blocks": []})
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["title"], "Add a user")

    def test_extract_rejects_non_object_output(self):
        class FakeExtractor:
            def extract_structured(self, messages, schema, max_tokens):
                return json.dumps({"units": "nope"})

        with self.assertRaises(DocumentLearnError):
            extract_units(FakeExtractor(), _version(), {"blocks": []})

    def test_extract_requires_a_model(self):
        with self.assertRaises(DocumentLearnError):
            extract_units(None, _version(), {"blocks": []})


class ValidationTest(unittest.TestCase):
    def _check(self, units):
        return check_units(_version(), _blocks(), {"F-NR-208E/2"}, set(), units)

    def test_good_procedure_passes(self):
        valid, errors = self._check([_unit()])
        self.assertEqual(errors, [])
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["details"]["ordered_steps"], ["Open Users", "tap Add"])
        self.assertEqual(valid[0]["sources"][0]["block_id"], 11)

    def test_unknown_block_rejected(self):
        valid, errors = self._check([_unit(sources=[{"block_id": 999, "excerpt": "x"}])])
        self.assertEqual(valid, [])
        self.assertIn("not part of this version", errors[0]["error"])

    def test_nonverbatim_excerpt_rejected(self):
        valid, errors = self._check([_unit(sources=[{"block_id": 11, "excerpt": "invented words"}])])
        self.assertEqual(valid, [])
        self.assertIn("verbatim", errors[0]["error"])

    def test_invented_model_rejected(self):
        valid, errors = self._check([_unit(
            content="ZZ-INVENTED-1 needs a reboot.",
            sources=[{"block_id": 11, "excerpt": "Open Users and tap Add to create one."}],
        )])
        self.assertEqual(valid, [])
        self.assertIn("does not occur", errors[0]["error"])

    def test_invented_number_rejected(self):
        valid, errors = self._check([_unit(
            content="Wait 987654 seconds between steps.",
            sources=[{"block_id": 11, "excerpt": "Open Users and tap Add to create one."}],
        )])
        self.assertEqual(valid, [])
        self.assertIn("does not occur", errors[0]["error"])

    def test_procedure_requires_steps_and_rule_requires_trigger(self):
        valid, errors = self._check([_unit(ordered_steps=[])])
        self.assertEqual(valid, [])
        self.assertIn("ordered_steps", errors[0]["error"])
        valid, errors = self._check([_unit(unit_kind="rule", title="R",
                                           content="Open Users and tap Add.",
                                           ordered_steps=None)])
        # rule without trigger/result fails even with valid sources
        self.assertEqual(valid, [])
        self.assertIn("trigger", errors[0]["error"])

    def test_empty_sources_rejected(self):
        valid, errors = self._check([_unit(sources=[])])
        self.assertEqual(valid, [])
        self.assertIn("source", errors[0]["error"])


class RenderTest(unittest.TestCase):
    def test_procedure_renders_deterministically(self):
        first = render_unit_content("Add a user", "procedure", {
            "applicability": {"models": ["F-NR-208E/2"]},
            "prerequisites": ["Admin rights"],
            "ordered_steps": ["Open Users", "tap Add"],
            "expected_result": "A new user exists.",
            "warnings": ["Do not skip review"],
        })
        second = render_unit_content("Add a user", "procedure", {
            "warnings": ["Do not skip review"],
            "expected_result": "A new user exists.",
            "ordered_steps": ["Open Users", "tap Add"],
            "prerequisites": ["Admin rights"],
            "applicability": {"models": ["F-NR-208E/2"]},
        })
        self.assertEqual(first, second)
        self.assertIn("1. Open Users", first)
        self.assertIn("适用范围", first)

    def test_rule_and_experience_shapes(self):
        rule = render_unit_content("R", "rule", {"trigger": "night", "result": "infrared"})
        self.assertIn("触发条件", rule)
        experience = render_unit_content("E", "experience", {"observation": "worked"})
        self.assertIn("现象与结果", experience)

    def test_overlong_content_is_rejected(self):
        from v2.document_learning import _text

        with self.assertRaises(DocumentLearnError):
            _text("x" * 12001)


class DocumentLearnApiTest(unittest.TestCase):
    """Route-level tests: HTTP mapping, not service logic."""

    def setUp(self):
        from unittest.mock import patch

        import app as app_module

        self.app_module = app_module
        self._previous_api_key = app_module.settings["api_key"]
        app_module.settings["api_key"] = "test-key"
        self.db_patch = patch.object(app_module, "db", return_value=_DummyConn())
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(app_module.settings.__setitem__, "api_key", self._previous_api_key)

    def _patch(self, name, value):
        from unittest.mock import patch

        patcher = patch.object(self.app_module, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _proposal(self):
        return {
            "id": 51, "fact_text": "Do X.", "unit_kind": "procedure",
            "applicability": {}, "details_json": {"title": "Do X"},
            "status": "pending_confirmation", "comparison_result": "NEW",
            "origin_document_version_id": 9, "confirmed_knowledge_id": None,
            "created_at": None, "updated_at": None,
        }

    def test_learn_queues_and_lists_proposals(self):
        from app import v2_document_proposals, v2_learn_document_version

        self._patch("queue_learn_jobs",
                    lambda conn, *_args, **_kwargs: [{"job_id": 3, "context_key": "Guide"}])
        queued = v2_learn_document_version(9, x_api_key="test-key")
        self.assertEqual(queued["total"], 1)
        self.assertEqual(queued["queued"][0]["context_key"], "Guide")

        self._patch("get_version", lambda conn, _: {"id": 9})
        self._patch("list_document_proposals", lambda conn, _: [self._proposal()])
        listed = v2_document_proposals(9, x_api_key="test-key")
        self.assertEqual(listed["total"], 1)
        self.assertEqual(listed["items"][0]["proposal_id"], 51)

        from fastapi import HTTPException

        from app import v2_document_proposal

        self._patch("get_document_proposal", lambda conn, _: None)
        with self.assertRaises(HTTPException) as caught:
            v2_document_proposal(4242, x_api_key="test-key")
        self.assertEqual(caught.exception.status_code, 404)

    def test_confirm_proposal_shape_and_errors(self):
        from fastapi import HTTPException

        from app import V2DocumentProposalConfirmIn, v2_confirm_document_proposal
        from v2.document_learning import DocumentLearnError, DocumentLearnNotFound

        knowledge = {"id": 61, "trust": "user_confirmed", "unit_kind": "procedure",
                     "revision": 1, "validation_status": "validated"}
        self._patch("confirm_document_proposal", lambda conn, *_, **__: (knowledge, False))
        confirmed = v2_confirm_document_proposal(
            51, V2DocumentProposalConfirmIn(content="Do X now."),
            x_api_key="test-key",
        )
        self.assertEqual(confirmed["knowledge_id"], 61)
        self.assertEqual(confirmed["validation_status"], "validated")

        def missing(conn, *_, **__):
            raise DocumentLearnNotFound("gone")

        self._patch("confirm_document_proposal", missing)
        with self.assertRaises(HTTPException) as caught:
            v2_confirm_document_proposal(
                51, V2DocumentProposalConfirmIn(), x_api_key="test-key")
        self.assertEqual(caught.exception.status_code, 404)

        def bad(conn, *_, **__):
            raise DocumentLearnError("no source")

        self._patch("confirm_document_proposal", bad)
        with self.assertRaises(HTTPException) as caught:
            v2_confirm_document_proposal(
                51, V2DocumentProposalConfirmIn(), x_api_key="test-key")
        self.assertEqual(caught.exception.status_code, 409)


class _DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):  # pragma: no cover - module functions are patched above
        raise AssertionError("database must not be touched")


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2DocumentLearningPostgresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def tearDown(self):
        try:
            self.conn.rollback()
            with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM v2_document_jobs WHERE version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_document_blocks WHERE version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_answer_feedback WHERE idempotency_key LIKE 'ZZDOC-%'"
                    )
                    cur.execute(
                        "DELETE FROM v2_learning_proposals WHERE origin_document_version_id IN ("
                        "SELECT id FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge_history WHERE knowledge_id IN ("
                        "SELECT id FROM v2_knowledge WHERE title LIKE 'ZZDOC %')"
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge_sources WHERE knowledge_id IN ("
                        "SELECT id FROM v2_knowledge WHERE title LIKE 'ZZDOC %')"
                    )
                    cur.execute("DELETE FROM v2_knowledge WHERE title LIKE 'ZZDOC %'")
                    cur.execute("DELETE FROM v2_raw_evidence WHERE source_label LIKE 'ZZDOC %'")
                    cur.execute(
                        "DELETE FROM v2_document_versions WHERE document_key LIKE 'ZZDOC %'"
                    )
                    cur.execute("DELETE FROM v2_answer_runs WHERE idempotency_key LIKE 'ZZDOC-%'")
        finally:
            self.conn.rollback()
            self.conn.close()
            self.tmp.cleanup()

    def _factory(self):
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def _parsed_version(self, key="ZZDOC learn"):
        import sys

        sys.path.insert(0, ".")
        from test_v2_documents import _pdf_bytes
        from v2.document_processing import process_document_job
        from v2.documents import claim_document_job, create_version

        version, _ = create_version(
            self.conn, base_dir=self.tmp.name, document_key=key,
            version_label="v1", filename="m.pdf", content=_pdf_bytes(),
        )
        self.conn.commit()
        with self._factory() as conn:
            job = claim_document_job(conn, ("parse",))
            conn.commit()
        process_document_job(int(job["id"]), db_factory=self._factory, base_dir=self.tmp.name)
        return version

    def _extractor_for(self, blocks):
        """Canned extraction citing real blocks with verbatim excerpts."""

        texts = {int(block["id"]): str(block.get("evidence_text") or "") for block in blocks}
        first, second = list(texts)[:2]

        class FakeExtractor:
            def extract_structured(self, messages, schema, max_tokens):
                return json.dumps({"units": [{
                    "title": "ZZDOC Add a user",
                    "unit_kind": "procedure",
                    "content": "ZZDOC " + texts[first][:200],
                    "applicability": {},
                    "ordered_steps": ["ZZDOC step one", "ZZDOC step two"],
                    "expected_result": "ZZDOC done",
                    "sources": [
                        {"block_id": first, "excerpt": texts[first][:120]},
                        {"block_id": second, "excerpt": texts[second][:120]},
                    ],
                }]})

        return FakeExtractor()

    def test_learn_step_extracts_and_confirms_answerable_unit(self):
        from v2.answering import answer_question
        from v2.document_learning import (
            confirm_document_proposal,
            list_document_proposals,
            queue_learn_jobs,
        )
        from v2.document_processing import process_document_job
        from v2.documents import claim_document_job, get_blocks
        from v2.retrieval import retrieve_for_answer

        version = self._parsed_version()
        jobs = queue_learn_jobs(self.conn, int(version["id"]))
        self.conn.commit()
        self.assertGreaterEqual(len(jobs), 1)
        # Queueing twice only returns newly created jobs (idempotent).
        again = queue_learn_jobs(self.conn, int(version["id"]))
        self.conn.commit()
        self.assertEqual(again, [])

        with self._factory() as conn:
            blocks = get_blocks(conn, int(version["id"]))
            extractor = self._extractor_for(blocks)
            job = claim_document_job(conn, ("learn",))
            conn.commit()
        self.assertIsNotNone(job)
        process_document_job(
            int(job["id"]), db_factory=self._factory, base_dir=self.tmp.name,
            llm_service=extractor,
        )
        with self._factory() as conn:
            proposals = list_document_proposals(conn, int(version["id"]))
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0]["status"], "pending_confirmation")
            # Still a proposal: not answer-eligible before confirm.
            retrieved = retrieve_for_answer(conn, "ZZDOC add a user", embedder=None)
            self.assertNotIn(proposals[0]["id"], retrieved["diagnostics"]["eligible_ids"])
            knowledge, duplicate = confirm_document_proposal(conn, int(proposals[0]["id"]))
            conn.commit()
        self.assertFalse(duplicate)
        self.assertEqual(knowledge["trust"], "user_confirmed")
        self.assertEqual(knowledge["validation_status"], "validated")
        self.assertEqual(knowledge["unit_kind"], "procedure")
        self.assertEqual(
            int(knowledge["origin_document_version_id"]), int(version["id"]))
        self.assertIn("1. ZZDOC step one", knowledge["content"])

        class FakeJudge:
            def judge(self, messages, max_tokens=600):
                return json.dumps({
                    "status": "answered", "answer": "ZZDOC open Users and tap Add.",
                    "clarifying_question": "", "source_indexes": [0],
                    "confidence": 0.9,
                })

        with self._factory() as conn:
            retrieved = retrieve_for_answer(conn, "ZZDOC add a user", embedder=None)
            self.assertIn(int(knowledge["id"]), retrieved["diagnostics"]["eligible_ids"])
            run = answer_question(
                "ZZDOC how do I add a user?", context={},
                idempotency_key="ZZDOC-run-1", db_factory=self._factory,
                llm_service=FakeJudge(),
            )
        self.assertEqual(run["answer_status"], "answered")
        self.assertEqual(
            run["evidence_snapshot"][0]["knowledge_id"], int(knowledge["id"]))

        # Confirm is idempotent: no second Knowledge row.
        with self._factory() as conn:
            same, duplicate = confirm_document_proposal(conn, int(proposals[0]["id"]))
            conn.commit()
        self.assertTrue(duplicate)
        self.assertEqual(int(same["id"]), int(knowledge["id"]))

    def test_validation_gate_for_document_units(self):
        from v2.documents import get_blocks
        from v2.retrieval import retrieve_for_answer

        version = self._parsed_version(key="ZZDOC gate")
        with self._factory() as conn:
            blocks = get_blocks(conn, int(version["id"]))
            first = next(block for block in blocks if str(block.get("evidence_text") or "").strip())
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO v2_knowledge(
                        title, content, trust, active, unit_kind,
                        origin_document_version_id, validation_status
                    ) VALUES('ZZDOC pending unit', %s, 'user_confirmed', TRUE,
                             'fact', %s, 'pending')
                    RETURNING id
                    """,
                    ("ZZDOC " + str(first["evidence_text"])[:200], int(version["id"])),
                )
                knowledge_id = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    INSERT INTO v2_knowledge_sources(
                        knowledge_id, raw_evidence_id, source_kind, relation,
                        source_role, excerpt, active, resolution
                    ) VALUES(%s, %s, 'other', 'supports', 'supporting', 'x', TRUE, 'accepted')
                    """,
                    (knowledge_id, int(first["raw_evidence_id"])),
                )
            conn.commit()
            gated = retrieve_for_answer(conn, "ZZDOC pending", embedder=None)
            self.assertNotIn(knowledge_id, gated["diagnostics"]["eligible_ids"])
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE v2_knowledge SET validation_status='validated' WHERE id=%s",
                    (knowledge_id,),
                )
            conn.commit()
            admitted = retrieve_for_answer(conn, "ZZDOC pending", embedder=None)
            self.assertIn(knowledge_id, admitted["diagnostics"]["eligible_ids"])


if __name__ == "__main__":
    unittest.main()
