# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL BM25 search over LightRAG document chunks."""

from __future__ import annotations

import inspect
from typing import Any

from dlightrag.storage.sql_identifiers import pg_identifier

BM25_INDEX = pg_identifier("idx_lightrag_doc_chunks_bm25")
BM25_TABLE = pg_identifier("LIGHTRAG_DOC_CHUNKS")
BM25_TEXT_CONFIGS = frozenset(
    {
        "simple",
        "english",
        "jiebacfg",
        "french",
        "german",
        "spanish",
        "italian",
        "portuguese",
        "dutch",
        "russian",
        "swedish",
        "norwegian",
        "danish",
        "finnish",
        "hungarian",
        "turkish",
        "romanian",
    }
)


def build_bm25_sql(*, candidate_ids: set[str] | None, limit: int) -> str:
    """Build a pg_textsearch BM25 query with optional hard candidate filter."""
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    candidate_clause = "AND id = ANY($3::text[])" if candidate_ids is not None else ""
    limit_placeholder = "$4" if candidate_ids is not None else "$3"
    return f"""
        SELECT id, content, file_path,
               -(content <@> to_bm25query($1, '{BM25_INDEX}')) AS score
        FROM {BM25_TABLE}
        WHERE workspace = $2
        {candidate_clause}
        ORDER BY content <@> to_bm25query($1, '{BM25_INDEX}')
        LIMIT {limit_placeholder}
    """


class PostgresBM25:
    """BM25 retriever backed by pg_textsearch."""

    def __init__(self, *, pool: Any, workspace: str, top_k: int = 40) -> None:
        self._pool = pool
        self._workspace = workspace
        self._top_k = top_k

    async def _run(self, operation):
        run = getattr(self._pool, "run", None)
        if callable(run) and inspect.iscoroutinefunction(run):
            return await run(operation)
        async with self._pool.acquire() as conn:
            return await operation(conn)

    async def ensure_index(self, *, text_config: str = "simple") -> None:
        if text_config not in BM25_TEXT_CONFIGS:
            raise ValueError(f"unsupported BM25 text_config: {text_config}")

        async def _operation(conn: Any) -> None:
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {BM25_INDEX} "
                f"ON {BM25_TABLE} USING bm25(content) "
                f"WITH (text_config='{text_config}')"
            )

        await self._run(_operation)

    async def verify_index(self) -> None:
        """Verify the BM25 index exists without attempting DDL."""

        async def _operation(conn: Any) -> bool:
            return await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
                BM25_INDEX,
            )

        exists = await self._run(_operation)
        if not exists:
            raise RuntimeError(f"{BM25_INDEX} is missing; create it on the primary first")

    async def search(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if candidate_ids is not None and len(candidate_ids) == 0:
            return []
        limit = self._top_k if top_k is None else top_k
        sql = build_bm25_sql(candidate_ids=candidate_ids, limit=limit)

        async def _operation(conn: Any) -> list[Any]:
            if candidate_ids is None:
                return await conn.fetch(sql, query, self._workspace, int(limit))
            return await conn.fetch(
                sql,
                query,
                self._workspace,
                list(candidate_ids),
                int(limit),
            )

        rows = await self._run(_operation)
        return [
            {
                "chunk_id": row["id"],
                "content": row["content"],
                "file_path": row["file_path"],
                "score": float(row["score"]),
            }
            for row in rows
        ]
