# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL filtered vector search for LightRAG chunks."""

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from dlightrag.adapters.postgres.core.identifiers import pg_identifier, pg_qualified_identifier
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    METADATA_TABLE,
    metadata_match_conditions,
)
from dlightrag.engine.rag.retrieval import MetadataScope
from dlightrag.engine.rag.retrieval.filtering import current_filter_stats

logger = logging.getLogger(__name__)
EXACT_FILTER_THRESHOLD = 8192


def _metadata_doc_subquery(
    workspace: str,
    scope: MetadataScope,
    *,
    start_index: int,
) -> tuple[str, list[Any]]:
    """One database-side metadata semi-join source with shifted placeholders."""
    conditions, params = metadata_match_conditions(
        workspace,
        scope.filters,
        filename_mode=scope.filename_mode,
        start_index=start_index,
        alias="m",
    )
    subquery = f"SELECT m.doc_id FROM {METADATA_TABLE} m WHERE {' AND '.join(conditions)}"  # noqa: S608
    return subquery, params


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

        if scope.candidate_count <= self._exact_threshold:
            strategy = "exact_vector"
            rows = await self._run(
                lambda conn: self._exact_search(
                    conn,
                    embedding_vec,
                    workspace,
                    scope,
                    cosine_threshold,
                    top_k,
                    table_name=table_name,
                    vector_cast=vector_cast,
                )
            )
        else:
            strategy = "hnsw"
            rows = await self._run(
                lambda conn: self._hnsw_search(
                    conn,
                    embedding_vec,
                    workspace,
                    scope,
                    cosine_threshold,
                    top_k,
                    table_name=table_name,
                    vector_cast=vector_cast,
                )
            )
        stats = current_filter_stats()
        if stats is not None:
            stats.vector_strategy = strategy
            if scope.candidate_count_exact:
                shortfall = max(0, min(top_k, scope.candidate_count) - len(rows))
                if shortfall:
                    stats.vector_candidate_shortfall = shortfall
        logger.info(
            "%s in-filtered PG search: %d results from %s chunk(s)",
            "Exact" if strategy == "exact_vector" else "HNSW",
            len(rows),
            scope.render_candidate_count(),
        )
        return self._format_rows(rows)

    async def _exact_search(
        self,
        conn: Any,
        embedding_vec: Any,
        workspace: str,
        scope: MetadataScope,
        cosine_threshold: float,
        top_k: int,
        *,
        table_name: str,
        vector_cast: str,
    ) -> list[Any]:
        # A bounded candidate set: every candidate row is materialized once,
        # then exact distance ordering runs over it. The metadata filter stays
        # a database-side semi-join before the distance order, never a Python
        # document-id array.
        subquery, params = _metadata_doc_subquery(workspace, scope, start_index=3)
        threshold_slot = len(params) + 3
        limit_slot = threshold_slot + 1
        sql = (
            f"WITH candidate_rows AS MATERIALIZED ("  # noqa: S608
            f"SELECT v.id, v.content, v.file_path, v.full_doc_id, v.content_vector "
            f"FROM {table_name} v "
            f"WHERE v.workspace = $2 AND v.full_doc_id IN ({subquery})"
            f") "
            f"SELECT id, content, file_path, full_doc_id, "
            f"1 - (content_vector <=> $1::{vector_cast}) AS score "
            f"FROM candidate_rows "
            f"WHERE 1 - (content_vector <=> $1::{vector_cast}) > ${threshold_slot} "
            f"ORDER BY content_vector <=> $1::{vector_cast} "
            f"LIMIT ${limit_slot}"
        )
        return await conn.fetch(
            sql,
            embedding_vec,
            workspace,
            *params,
            cosine_threshold,
            top_k,
        )

    async def _hnsw_search(
        self,
        conn: Any,
        embedding_vec: Any,
        workspace: str,
        scope: MetadataScope,
        cosine_threshold: float,
        top_k: int,
        *,
        table_name: str,
        vector_cast: str,
    ) -> list[Any]:
        async def iterative_search(conn: Any) -> Any:
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                await conn.execute("SET LOCAL hnsw.max_scan_tuples = 20000")
                # The ranked stream stays the HNSW source; the metadata
                # semi-join filters before the outer LIMIT collects top_k
                # matching rows, with the existing ANN search budget intact.
                subquery, params = _metadata_doc_subquery(workspace, scope, start_index=3)
                threshold_slot = len(params) + 3
                limit_slot = threshold_slot + 1
                sql = (
                    f"WITH nearest_results AS MATERIALIZED ("  # noqa: S608
                    f"SELECT id, content, file_path, full_doc_id, "
                    f"1 - (content_vector <=> $1::{vector_cast}) AS score, "
                    f"content_vector <=> $1::{vector_cast} AS distance "
                    f"FROM {table_name} "
                    f"WHERE workspace = $2 AND full_doc_id IN ({subquery}) "
                    f"ORDER BY content_vector <=> $1::{vector_cast} "
                    f"LIMIT ${limit_slot}"
                    f") "
                    f"SELECT id, content, file_path, full_doc_id, score "
                    f"FROM nearest_results "
                    f"WHERE score > ${threshold_slot} "
                    f"ORDER BY distance + 0"
                )
                return await conn.fetch(
                    sql,
                    embedding_vec,
                    workspace,
                    *params,
                    cosine_threshold,
                    top_k,
                )

        return await iterative_search(conn)

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
