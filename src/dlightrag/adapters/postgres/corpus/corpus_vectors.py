# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL filtered vector search for LightRAG chunks."""

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from dlightrag.adapters.postgres.core.identifiers import pg_identifier, pg_qualified_identifier
from dlightrag.engine.rag.retrieval import MetadataScope

logger = logging.getLogger(__name__)
EXACT_FILTER_THRESHOLD = 8192


class PGFilteredVectorSearch:
    """Strict document-scoped pgvector search and supporting index DDL."""

    def __init__(self, original: Any, *, exact_threshold: int = EXACT_FILTER_THRESHOLD) -> None:
        self._original = original
        self._backend = type(original).__name__
        self._exact_threshold = exact_threshold

    async def search(
        self,
        embedding: list[float],
        *,
        scope: MetadataScope,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self._backend != "PGVectorStorage":
            raise RuntimeError(
                f"Filtered vector search requires PGVectorStorage, got {self._backend}"
            )
        if not scope:
            return []
        table_name = pg_qualified_identifier(self._original.table_name)
        workspace = self._original.workspace
        cosine_threshold = self._original.cosine_better_than_threshold
        vector_cast = (
            "halfvec"
            if getattr(self._original.db, "vector_index_type", None) == "HNSW_HALFVEC"
            else "vector"
        )
        embedding_vec = np.array(embedding, dtype=np.float32)
        doc_ids = scope.as_list()

        if scope.chunk_count <= self._exact_threshold:
            rows = await self._run(
                lambda conn: conn.fetch(
                    f"WITH candidate_rows AS MATERIALIZED ("  # noqa: S608
                    f"  SELECT v.id, v.content, v.file_path, v.full_doc_id, v.content_vector "
                    f"  FROM {table_name} v "
                    f"  WHERE v.workspace = $2 AND v.full_doc_id = ANY($3::text[])"
                    f") "
                    f"SELECT id, content, file_path, full_doc_id, "
                    f"1 - (content_vector <=> $1::{vector_cast}) AS score "
                    f"FROM candidate_rows "
                    f"WHERE 1 - (content_vector <=> $1::{vector_cast}) > $4 "
                    f"ORDER BY content_vector <=> $1::{vector_cast} "
                    f"LIMIT $5",
                    embedding_vec,
                    workspace,
                    doc_ids,
                    cosine_threshold,
                    top_k,
                )
            )
            logger.info(
                "Exact in-filtered PG search: %d results from %d doc(s)/%d chunk(s)",
                len(rows),
                len(doc_ids),
                scope.chunk_count,
            )
            return self._format_rows(rows)

        async def iterative_search(conn: Any) -> Any:
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                await conn.execute("SET LOCAL hnsw.max_scan_tuples = 20000")
                return await conn.fetch(
                    f"WITH nearest_results AS MATERIALIZED ("  # noqa: S608
                    f"  SELECT id, content, file_path, full_doc_id, "
                    f"  1 - (content_vector <=> $1::{vector_cast}) AS score, "
                    f"  content_vector <=> $1::{vector_cast} AS distance "
                    f"  FROM {table_name} "
                    f"  WHERE workspace = $2 "
                    f"  AND full_doc_id = ANY($3::text[]) "
                    f"  ORDER BY content_vector <=> $1::{vector_cast} "
                    f"  LIMIT $5"
                    f") "
                    f"SELECT id, content, file_path, full_doc_id, score "
                    f"FROM nearest_results "
                    f"WHERE score > $4 "
                    f"ORDER BY distance + 0",
                    embedding_vec,
                    workspace,
                    doc_ids,
                    cosine_threshold,
                    top_k,
                )

        rows = await self._run(iterative_search)
        logger.info(
            "HNSW in-filtered PG search: %d results from %d doc(s)/%d chunk(s)",
            len(rows),
            len(doc_ids),
            scope.chunk_count,
        )
        return self._format_rows(rows)

    async def ensure_document_scope_index(self) -> None:
        table = self._original.table_name
        table_name = pg_qualified_identifier(table)
        digest = hashlib.md5(table.encode(), usedforsecurity=False).hexdigest()[:12]
        index_name = pg_identifier(f"idx_dlightrag_full_doc_id_{digest}")
        try:
            await self._run(
                lambda conn: conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    f"ON {table_name}(workspace, full_doc_id)"
                )
            )
        except Exception:
            logger.warning(
                "Could not create %s on %s; document-scoped vector filters will scan sequentially",
                index_name,
                table,
                exc_info=True,
            )

    async def _run(self, operation: Callable[[Any], Awaitable[Any]]) -> Any:
        return await self._original.db._run_with_retry(operation)

    @staticmethod
    def _format_rows(rows: Any) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for row in rows:
            chunk = {
                "id": row["id"],
                "content": row.get("content", ""),
                "file_path": row.get("file_path", ""),
                "distance": 1 - row["score"],
            }
            if row.get("full_doc_id"):
                chunk["full_doc_id"] = row["full_doc_id"]
            chunks.append(chunk)
        return chunks


__all__ = ["EXACT_FILTER_THRESHOLD", "PGFilteredVectorSearch"]
