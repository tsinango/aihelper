"""Import per-table Wrangler D1 exports without changing the source D1."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

TABLES = [
    "products", "documents", "document_chunks", "product_attributes", "features",
    "feature_aliases", "product_features", "verified_facts", "questions", "import_batches",
    "support_cases", "support_case_analysis", "support_case_analysis_failures",
]
JSON_COLUMNS = {
    "metadata_json", "value_json", "value", "participants", "models", "domain_tags",
    "intent_tags", "media", "messages", "raw_json", "normalized_json", "answer_json",
    "retrieval_trace_json", "models_json", "products_json", "features_json", "symptoms_json",
    "error_codes_json", "secondary_questions_json",
}
BOOL_COLUMNS = {"verified", "production_answer_allowed"}
PG_COLUMNS = {
    "product_attributes": {"value_json": "value"},
}


def load_sqlite(export_dir: Path) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    for table in TABLES:
        db.executescript((export_dir / f"{table}.sql").read_text(encoding="utf-8"))
    return db


def value(column: str, item):
    if item is None:
        return None
    if column in JSON_COLUMNS:
        try:
            return Jsonb(json.loads(item))
        except (TypeError, json.JSONDecodeError):
            raise ValueError(f"invalid JSON in column {column}: {item[:120]!r}") from None
    if column in BOOL_COLUMNS:
        return bool(item)
    return item


def rows(db: sqlite3.Connection, table: str):
    columns = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
    target = [PG_COLUMNS.get(table, {}).get(column, column) for column in columns]
    for row in db.execute(f'SELECT {", ".join(columns)} FROM "{table}"'):
        yield target, [value(column, item) for column, item in zip(columns, row, strict=True)]


def primary_key_columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in sorted(
        (row for row in db.execute(f'PRAGMA table_info("{table}")') if row[5]),
        key=lambda row: row[5],
    )]


def main() -> None:
    export_dir = Path(__file__).with_name("d1-export")
    database_url = __import__("os").environ["DATABASE_URL"]
    sqlite_db = load_sqlite(export_dir)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for table in TABLES:
                first = True
                count = 0
                conflict_columns = primary_key_columns(sqlite_db, table)
                for columns, values in rows(sqlite_db, table):
                    if first:
                        assignments = [
                            sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(column), sql.Identifier(column))
                            for column in columns if column not in conflict_columns
                        ]
                        if conflict_columns and assignments:
                            conflict_action = sql.SQL("ON CONFLICT ({}) DO UPDATE SET {} ").format(
                                sql.SQL(", ").join(map(sql.Identifier, conflict_columns)),
                                sql.SQL(", ").join(assignments),
                            )
                        else:
                            conflict_action = sql.SQL("ON CONFLICT DO NOTHING ")
                        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) {}").format(
                            sql.Identifier(table),
                            sql.SQL(", ").join(map(sql.Identifier, columns)),
                            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
                            conflict_action,
                        )
                        first = False
                    cur.execute(statement, values)
                    count += cur.rowcount
                print(f"{table}: {count} inserted")
            for table in TABLES:
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
                print(f"{table}: {cur.fetchone()[0]} rows")


if __name__ == "__main__":
    main()
