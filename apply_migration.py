#!/usr/bin/env python3
"""Apply an additive SQL migration from the local migrations directory."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import psycopg

from build_verified_knowledge_pilot import load_env_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("migration", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path("/etc/aihelper.env"))
    args = parser.parse_args()
    load_env_file(args.env_file)
    migration_path = args.migration.resolve()
    migrations_dir = Path(__file__).with_name("migrations").resolve()
    if migration_path.parent != migrations_dir:
        raise SystemExit(f"migration must be inside {migrations_dir}")
    sql = migration_path.read_text(encoding="utf-8")
    migration_name = migration_path.name
    checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              migration_name TEXT PRIMARY KEY,
              checksum TEXT NOT NULL,
              applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        row = conn.execute(
            "SELECT checksum FROM schema_migrations WHERE migration_name=%s",
            (migration_name,),
        ).fetchone()
        if row:
            if row[0] != checksum:
                raise SystemExit(f"checksum mismatch for already applied migration {migration_name}")
            print(f"already_applied={migration_name}")
            return 0
        conn.execute(sql)
        conn.execute(
            "INSERT INTO schema_migrations(migration_name,checksum) VALUES(%s,%s)",
            (migration_name, checksum),
        )
    print(f"applied={migration_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
