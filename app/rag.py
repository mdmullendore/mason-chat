"""Retrieves the top-k most similar chunks via a pgvector ANN query (HNSW index),
rather than loading everything into memory — scales with the content in the
`chunks` table instead of what fits comfortably in process memory."""

from pgvector import Vector

from app.db import get_pool


def retrieve(query_vector: list[float], k: int = 4) -> list[dict]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT text, source FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (Vector(query_vector), k),
        ).fetchall()
    return [{"text": row[0], "source": row[1]} for row in rows]
