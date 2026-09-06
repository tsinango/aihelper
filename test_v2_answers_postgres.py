"""PostgreSQL integration tests for Phase 3.1 read-only internal QA.

Set ``V2_TEST_DATABASE_URL`` to an initialized database containing migrations
001-021.  Every test rolls its writes back; only identity sequences may
advance, which is harmless.
"""

from __future__ import annotations

import json
import os
import random
import string
import unittest
import uuid

import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row

from v2.answering import answer_question, get_answer_run
from v2.retrieval import retrieve_for_answer, store_knowledge_embedding
from v2.service import edit_knowledge


DATABASE_URL = os.getenv("V2_TEST_DATABASE_URL", "").strip()


class FakeLLM:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def judge(self, messages, max_tokens=600):
        self.calls.append((messages, max_tokens))
        if self.error:
            raise self.error
        return self.responses.pop(0)


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        vector = [1.0] + [0.0] * 2047
        return [vector for _ in texts]


def answered_json(answer="Ответ на русском языке."):
    return json.dumps({
        "status": "answered", "answer": answer,
        "clarifying_question": "", "source_indexes": [0],
        "confidence": 0.9,
    }, ensure_ascii=False)


@unittest.skipUnless(DATABASE_URL, "set V2_TEST_DATABASE_URL to run PostgreSQL integration tests")
class V2AnswersPostgresTest(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        # Letters-only tag: uuid hex contains digits, which the model-identifier
        # regex would read as product-model scope and pollute conflict checks.
        self.tag = "ZZ" + "".join(random.choice(string.ascii_uppercase) for _ in range(8))

    def tearDown(self):
        # Roll back this connection first: it may hold uncommitted writes
        # (e.g. the embedding UPDATE) that would block the cleanup below.
        # The answer service commits through its own factory connections, so
        # remove this test's letter-tagged rows explicitly (FK children first).
        try:
            self.conn.rollback()
            with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM v2_knowledge_history WHERE knowledge_id IN ("
                        "SELECT id FROM v2_knowledge WHERE title LIKE %s)",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge_sources WHERE excerpt LIKE %s",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_knowledge WHERE title LIKE %s",
                        (f"{self.tag} %",),
                    )
                    cur.execute(
                        "DELETE FROM v2_raw_evidence WHERE source_label LIKE %s",
                        (f"{self.tag}-%",),
                    )
                    cur.execute(
                        "DELETE FROM v2_answer_runs WHERE idempotency_key LIKE %s",
                        (f"{self.tag}-%",),
                    )
        finally:
            self.conn.rollback()
            self.conn.close()

    def _factory(self):
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    def _write(self, sql, params=()):
        with self._factory() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                try:
                    return cur.fetchone()
                except Exception:
                    return None

    def _evidence(self, label="evidence", status="active") -> int:
        row = self._write(
            """
            INSERT INTO v2_raw_evidence(
                evidence_type, author_role, content, source_label, source_locator,
                evidence_status
            ) VALUES('user_input', 'product_expert', %s, %s, %s, %s)
            RETURNING id
            """,
            (f"{self.tag} {label} 内容", f"{self.tag}-{label}", f"locator-{self.tag}", status),
        )
        assert row is not None
        return int(row["id"])

    def _knowledge(self, title, content, entity="", trust="user_confirmed", active=True) -> int:
        row = self._write(
            """
            INSERT INTO v2_knowledge(title, content, entity_name, trust, active)
            VALUES(%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (f"{self.tag} {title}", f"{self.tag} {content}", entity, trust, active),
        )
        assert row is not None
        return int(row["id"])

    def _source(self, knowledge_id, evidence_id, *, relation="supports",
                resolution="accepted", active=True, kind="user_confirmation",
                role="primary", excerpt="摘录 excerpt") -> None:
        self._write(
            """
            INSERT INTO v2_knowledge_sources(
                knowledge_id, raw_evidence_id, source_kind, relation,
                source_role, excerpt, active, resolution
            ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (knowledge_id, evidence_id, kind, relation, role,
             f"{self.tag} {excerpt}", active, resolution),
        )

    def _eligible(self, title, content, entity="") -> int:
        knowledge_id = self._knowledge(title, content, entity)
        self._source(knowledge_id, self._evidence())
        return knowledge_id

    def test_answer_runs_table_enforces_status_vocabulary(self):
        with self.assertRaises(Exception):
            self.conn.execute(
                "INSERT INTO v2_answer_runs(idempotency_key, answer_status) VALUES('k-bad', 'maybe')"
            )
        self.conn.rollback()
        self.conn.execute("SELECT 1")
        with self.assertRaises(UniqueViolation):
            self.conn.execute(
                "INSERT INTO v2_answer_runs(idempotency_key) VALUES('k-dup') RETURNING id"
            )
            self.conn.execute(
                "INSERT INTO v2_answer_runs(idempotency_key) VALUES('k-dup') RETURNING id"
            )

    def test_sql_eligibility_gate(self):
        good = self._eligible("T1234 安装", "T1234 支持机架安装", "T1234")
        provisional = self._knowledge("T1234 安装", "T1234 草稿", "T1234", trust="provisional")
        self._source(provisional, self._evidence("prov"))
        inactive = self._knowledge("T1234 安装", "T1234 旧版", "T1234", active=False)
        self._source(inactive, self._evidence("inact"))
        unsourced = self._knowledge("T1234 安装", "T1234 无来源", "T1234")
        pending_source = self._knowledge("T1234 安装", "T1234 待确认来源", "T1234")
        self._source(pending_source, self._evidence("pend"), resolution="unresolved")
        contradicted = self._knowledge("T1234 安装", "T1234 有争议", "T1234")
        self._source(contradicted, self._evidence("contra-ok"))
        self._source(
            contradicted, self._evidence("contra-bad"), relation="contradicts",
            role="contradicting", resolution="accepted",
        )
        result = retrieve_for_answer(self.conn, "T1234 如何安装？")
        ids = [int(item["id"]) for item in result["candidates"]]
        self.assertEqual(ids, [good])
        self.assertIn(good, result["diagnostics"]["eligible_ids"])
        for excluded_id in (provisional, inactive, unsourced, pending_source, contradicted):
            self.assertNotIn(excluded_id, result["diagnostics"]["eligible_ids"])
        self.assertEqual(result["candidates"][0]["sources"][0]["source_locator"], f"locator-{self.tag}")

    def test_inactive_raw_evidence_is_not_answer_eligible(self):
        superseded_only = self._knowledge("T2468 安装", "T2468 支持机架安装", "T2468")
        self._source(superseded_only, self._evidence("sup", status="superseded"))
        redacted_only = self._knowledge("T1357 安装", "T1357 支持机架安装", "T1357")
        self._source(redacted_only, self._evidence("red", status="redacted"))
        mixed = self._knowledge("T8642 安装", "T8642 支持机架安装", "T8642")
        self._source(mixed, self._evidence("good"), excerpt="可用摘录")
        self._source(mixed, self._evidence("stale", status="superseded"), excerpt="过期摘录")
        result = retrieve_for_answer(self.conn, "T2468 T1357 T8642 如何安装？")
        ids = [int(item["id"]) for item in result["candidates"]]
        self.assertEqual(ids, [mixed])
        self.assertNotIn(superseded_only, result["diagnostics"]["eligible_ids"])
        self.assertNotIn(redacted_only, result["diagnostics"]["eligible_ids"])
        # Mixed Knowledge stays eligible through its good source, but the
        # superseded excerpt must never surface in answer evidence.
        self.assertIn(mixed, result["diagnostics"]["eligible_ids"])
        sources = retrieve_for_answer(self.conn, "T8642 如何安装？")["candidates"][0]["sources"]
        self.assertEqual(len(sources), 1)
        self.assertIn("可用摘录", sources[0]["excerpt"])
        self.assertNotIn("过期摘录", sources[0]["excerpt"])

    def test_embedding_scan_works_over_pgvector(self):
        knowledge_id = self._eligible("T5678 安装", "T5678 支持机架安装", "T5678")
        stored = store_knowledge_embedding(
            self.conn, knowledge_id, "T5678 支持机架安装", embedder=FakeEmbedder(),
        )
        self.assertTrue(stored)
        result = retrieve_for_answer(
            self.conn, "T5678 如何安装？", embedder=FakeEmbedder(),
        )
        self.assertEqual([int(item["id"]) for item in result["candidates"]], [knowledge_id])
        self.assertFalse(result["diagnostics"]["lexical_only"])
        self.assertAlmostEqual(float(result["candidates"][0]["embedding_score"]), 1.0, places=5)

    def test_answer_run_persists_with_snapshot(self):
        knowledge_id = self._eligible("T9012 安装", "T9012 支持机架安装", "T9012")
        llm = FakeLLM([answered_json("T9012 安装在标准机架上。")])
        result = answer_question(
            "T9012 如何安装在机架上？",
            idempotency_key=f"{self.tag}-run-1",
            db_factory=self._factory,
            llm_service=llm,
            embedding_client=None,
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["evidence_snapshot"][0]["knowledge_id"], knowledge_id)
        # A fresh connection sees the committed run with its snapshot.
        with self._factory() as check:
            stored = get_answer_run(check, int(result["run_id"]))
        self.assertIsNotNone(stored)
        self.assertEqual(stored["answer_status"], "answered")
        self.assertIn("机架安装", stored["evidence_snapshot"][0]["content"])

    def test_snapshot_survives_a_real_knowledge_edit(self):
        knowledge_id = self._eligible("T3456 安装", "T3456 支持机架安装", "T3456")
        llm = FakeLLM([answered_json("T3456 安装在机架上。")])
        factory = self._factory
        result = answer_question(
            "T3456 如何安装？",
            idempotency_key=f"{self.tag}-run-2",
            db_factory=factory,
            llm_service=llm,
        )
        self.assertEqual(result["answer_status"], "answered")
        with factory() as conn:
            edit_knowledge(conn, knowledge_id, f"{self.tag} T3456 需要特殊导轨安装", None)
            conn.commit()
        with factory() as conn:
            stored = get_answer_run(conn, int(result["run_id"]))
        snapshot = stored["evidence_snapshot"][0]
        self.assertIn("机架安装", snapshot["content"])
        self.assertNotIn("特殊导轨", snapshot["content"])

    def test_duplicate_key_returns_stored_run(self):
        self._eligible("T7890 安装", "T7890 支持机架安装", "T7890")
        llm = FakeLLM([answered_json("T7890 安装在机架上。")])
        factory = self._factory
        first = answer_question(
            "T7890 如何安装？", idempotency_key=f"{self.tag}-run-3",
            db_factory=factory, llm_service=llm,
        )
        second = answer_question(
            "T7890 如何安装？", idempotency_key=f"{self.tag}-run-3",
            db_factory=factory, llm_service=llm,
        )
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
