"""Populate PostgreSQL search vectors with OpenRouter Nemotron embeddings.

The database remains the durable checkpoint: each committed batch is safe to
resume. A local content-hash cache avoids repeat requests for document chunks
when the same text is encountered again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psycopg

from embeddings import OPENROUTER_EMBEDDING_MODEL, OpenRouterEmbeddingClient, read_openrouter_token


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        keys = [str(key) for key in archive["keys"].tolist()]
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(keys), 2048):
        raise RuntimeError("OpenRouter embedding cache has an unexpected shape")
    return {key: vectors[index] for index, key in enumerate(keys)}


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    vectors = np.vstack([cache[key] for key in keys]) if keys else np.empty((0, 2048), dtype=np.float32)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, keys=np.asarray(keys, dtype="U64"), embeddings=vectors)
    os.replace(temporary, path)


def content_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def main() -> None:
    load_env_file(Path(os.getenv("AI_SALES_ENGINEER_ENV_FILE", "/etc/ai-sales-engineer.env")))
    database_url = os.environ["DATABASE_URL"]
    token = read_openrouter_token(os.getenv("OPENROUTER_TOKEN_FILE", str(Path(__file__).with_name("openrouter"))))
    if not token:
        raise SystemExit("OpenRouter token is required")
    model_name = OPENROUTER_EMBEDDING_MODEL
    batch_size = max(1, int(os.getenv("OPENROUTER_EMBED_BATCH_SIZE", "32")))
    cache_path = Path(os.getenv("OPENROUTER_EMBED_CACHE", "/home/ubuntu/ai-sales-engineer-knowledge/cache/openrouter_embeddings.npz"))
    cache = load_cache(cache_path)
    embedder = OpenRouterEmbeddingClient(
        token,
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
    )

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content FROM document_chunks "
                "WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s ORDER BY id",
                (model_name,),
            )
            document_rows = cur.fetchall()
            knowledge_rows = []
            for table in ("case_knowledge_memory", "verified_knowledge", "knowledge_learning_examples"):
                id_column = "verified_knowledge_id" if table == "verified_knowledge" else "id"
                condition = "embedding IS NULL OR embedding_status IS DISTINCT FROM 'ready'" if table == "verified_knowledge" else (
                    "embedding IS NULL OR embedding_model IS DISTINCT FROM %s"
                )
                parameters = () if table == "verified_knowledge" else (model_name,)
                cur.execute(
                    f"SELECT {id_column}, searchable_text FROM {table} "
                    f"WHERE searchable_text <> '' AND ({condition}) ORDER BY {id_column}",
                    parameters,
                )
                knowledge_rows.extend((table, row[0], row[1]) for row in cur.fetchall())

        document_pending = []
        document_cached = 0
        for row in document_rows:
            key = content_key(row[1])
            if key in cache:
                document_cached += 1
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE document_chunks SET embedding=%s::vector, embedding_model=%s WHERE id=%s",
                        (str(cache[key].tolist()), model_name, row[0]),
                    )
            else:
                document_pending.append((row[0], row[1], key))
        conn.commit()
        print(f"cache_hits={document_cached} uncached={len(document_pending)}", flush=True)

        for start in range(0, len(document_pending), batch_size):
            batch = document_pending[start:start + batch_size]
            vectors = embedder.encode([row[1] for row in batch], batch_size=batch_size, normalize_embeddings=True)
            with conn.cursor() as cur:
                for (row_id, _text, key), vector in zip(batch, vectors, strict=True):
                    cache[key] = np.asarray(vector, dtype=np.float32)
                    cur.execute(
                        "UPDATE document_chunks SET embedding=%s::vector, embedding_model=%s WHERE id=%s",
                        (str(vector), model_name, row_id),
                    )
            conn.commit()
            save_cache(cache_path, cache)
            print(f"embedded documents {min(start + len(batch), len(document_pending))}/{len(document_pending)}", flush=True)

        for start in range(0, len(knowledge_rows), batch_size):
            batch = knowledge_rows[start:start + batch_size]
            vectors = embedder.encode([row[2] for row in batch], batch_size=batch_size, normalize_embeddings=True)
            with conn.cursor() as cur:
                for (table, row_id, _text), vector in zip(batch, vectors, strict=True):
                    model_column = ", embedding_model=%s" if table != "verified_knowledge" else ""
                    timestamp = ", updated_at=CURRENT_TIMESTAMP" if table != "knowledge_learning_examples" else ""
                    status_columns = (
                        ", embedding_status='ready',embedding_error='',embedding_updated_at=CURRENT_TIMESTAMP"
                        if table == "verified_knowledge" else ""
                    )
                    parameters = (
                        (str(vector), model_name, row_id)
                        if table != "verified_knowledge" else (str(vector), row_id)
                    )
                    cur.execute(
                        f"UPDATE {table} SET embedding=%s::vector{model_column}{status_columns}{timestamp} "
                        f"WHERE {'verified_knowledge_id' if table == 'verified_knowledge' else 'id'}=%s",
                        parameters,
                    )
            conn.commit()
            print(f"embedded knowledge {min(start + len(batch), len(knowledge_rows))}/{len(knowledge_rows)}", flush=True)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL")
            print(json.dumps({"embedding_model": model_name, "embedding_rows": cur.fetchone()[0]}))


if __name__ == "__main__":
    main()
