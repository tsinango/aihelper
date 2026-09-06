"""Unit tests for the independent Phase 3.1 answer retrieval gate.

The SQL eligibility predicate (active + trusted + accepted supports source)
is authoritative and covered by the PostgreSQL integration tests; the fake
connections here exercise the Python-side trust guard, model/version conflict
exclusion, diagnostics, and the embedding-failure fallback.
"""

from __future__ import annotations

import unittest

from v2.retrieval import (
    _scope_models,
    _version_tokens,
    retrieve_for_answer,
)


class Cursor:
    def __init__(self, rows, sources):
        self.rows = rows
        self.sources = sources

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=()):
        self.query = " ".join(str(query).split())
        self.params = params

    def fetchall(self):
        # The eligibility query also mentions v2_knowledge_sources inside
        # EXISTS subqueries; only the sources query selects source_id.
        if "AS source_id" in self.query:
            return self.sources
        return self.rows


class Connection:
    def __init__(self, rows, sources=()):
        self.rows = rows
        self.sources = sources

    def cursor(self):
        return Cursor(self.rows, self.sources)


def row(identifier, title, content, entity="", *, trust="user_confirmed", active=True):
    return {
        "id": identifier, "title": title, "content": content,
        "entity_name": entity, "legacy_entity_name": entity,
        "trust": trust, "active": active,
        "embedding": None, "embedding_model": None,
        "created_at": None, "updated_at": None,
    }


def source(knowledge_id, identifier=1, excerpt="摘录", locator="locator"):
    return {
        "source_id": identifier, "knowledge_id": knowledge_id,
        "source_kind": "user_confirmation", "source_role": "primary",
        "excerpt": excerpt, "relation": "supports", "resolution": "accepted",
        "raw_evidence_id": identifier, "evidence_type": "user_input",
        "source_label": "label", "source_locator": locator,
        "evidence_status": "active",
    }


class Embedder:
    def __init__(self, vector=None, error=None):
        self.vector, self.error = vector, error

    def encode(self, texts, **kwargs):
        if self.error:
            raise self.error
        return [self.vector]


class V2AnswerScopeTokensTest(unittest.TestCase):
    def test_scope_models_drops_codecs_and_numbers(self):
        models = _scope_models("Камера DS-2CD2386G2-I поддерживает H.265 и ONVIF, 30 кадров")
        self.assertIn("DS-2CD2386G2-I", models)
        self.assertNotIn("H.265", [item for item in models])
        self.assertFalse(any(item.isdigit() for item in models))
        self.assertFalse(any("ONVIF" in item for item in models))

    def test_version_tokens(self):
        versions = _version_tokens("прошивка V5 7.2 build 230427, было 5.6.11")
        self.assertIn("72", versions)
        self.assertIn("5611", versions)
        self.assertIn("230427", versions)


class V2AnswerRetrievalTest(unittest.TestCase):
    def test_confirmed_rows_are_candidates_with_sources(self):
        conn = Connection(
            [row(1, "F-NR-208E/2 机架安装", "F-NR-208E/2 支持标准 19 英寸机架安装", "F-NR-208E/2")],
            [source(1)],
        )
        result = retrieve_for_answer(conn, "F-NR-208E/2 如何安装在机架上？")
        self.assertEqual([item["id"] for item in result["candidates"]], [1])
        self.assertEqual(result["candidates"][0]["sources"][0]["source_locator"], "locator")
        self.assertEqual(result["diagnostics"]["query_models"], ["F-NR-208E/2"])

    def test_provisional_and_inactive_rows_never_rank(self):
        conn = Connection([
            row(1, "F-X 功能", "F-X 支持功能", "F-X", trust="provisional"),
            row(2, "F-X 功能", "F-X 支持功能", "F-X", active=False),
        ])
        result = retrieve_for_answer(conn, "F-X 支持哪些功能？")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["diagnostics"]["eligible_ids"], [])

    def test_wrong_model_is_excluded_with_reason(self):
        conn = Connection([
            row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2"),
            row(2, "F-NR-232X/2 安装", "F-NR-232X/2 支持机架安装", "F-NR-232X/2"),
        ])
        result = retrieve_for_answer(conn, "F-NR-208E/2 如何安装？")
        self.assertEqual([item["id"] for item in result["candidates"]], [1])
        excluded = result["diagnostics"]["topical_excluded"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["knowledge_id"], 2)
        self.assertEqual(excluded[0]["reason"], "model_conflict")

    def test_wrong_version_is_excluded_with_reason(self):
        conn = Connection([
            row(1, "升级到 5.7", "将 IDS-TCM203-A 从 5.6.11 升级到 5.7 版本", "IDS-TCM203-A"),
            row(2, "版本 4.5.7 修复", "IDS-TCM203-A 版本 4.5.7 修复识别问题", "IDS-TCM203-A"),
        ])
        result = retrieve_for_answer(conn, "IDS-TCM203-A 5.6.11 升级失败怎么办？")
        self.assertEqual([item["id"] for item in result["candidates"]], [1])
        excluded = result["diagnostics"]["topical_excluded"]
        self.assertEqual([(item["knowledge_id"], item["reason"]) for item in excluded], [(2, "version_conflict")])

    def test_codec_mentions_do_not_conflict(self):
        conn = Connection([
            row(1, "H.265 设置", "DS-2CD2386G2-I 支持 H.265 编码", "DS-2CD2386G2-I"),
        ])
        result = retrieve_for_answer(conn, "DS-2CD2386G2-I 如何开启 H.264？")
        self.assertEqual([item["id"] for item in result["candidates"]], [1])
        self.assertEqual(result["diagnostics"]["topical_excluded"], [])

    def test_model_scopes_are_reported_not_generalized(self):
        conn = Connection([
            row(1, "F-X1 安装", "F-X1 支持机架安装", "F-X1"),
            row(2, "F-Y2 安装", "F-Y2 支持机架安装", "F-Y2"),
        ])
        result = retrieve_for_answer(conn, "设备 安装 机架 方法")
        self.assertEqual(result["diagnostics"]["topical_scopes"], [["F-X1"], ["F-Y2"]])

    def test_embedding_failure_falls_back_to_lexical(self):
        conn = Connection(
            [row(1, "F-NR-208E/2 安装", "F-NR-208E/2 支持机架安装", "F-NR-208E/2")],
            [source(1)],
        )
        result = retrieve_for_answer(
            conn, "F-NR-208E/2 如何安装？", embedder=Embedder(error=RuntimeError("boom")),
        )
        self.assertTrue(result["diagnostics"]["lexical_only"])
        self.assertEqual([item["id"] for item in result["candidates"]], [1])

    def test_empty_question_returns_nothing(self):
        conn = Connection([row(1, "标题", "内容", "实体")])
        result = retrieve_for_answer(conn, "   ")
        self.assertEqual(result["candidates"], [])


if __name__ == "__main__":
    unittest.main()
