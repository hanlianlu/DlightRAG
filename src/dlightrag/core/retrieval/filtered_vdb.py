# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Filtered storage wrappers for metadata-aware in-filtering.

Wraps the two LightRAG collaborators a document scope has to reach: ``chunks_vdb``
for the vector leg, and the ``text_chunks`` KV store for the knowledge-graph legs.
Uses contextvars for async-safe per-request state — concurrent requests don't
interfere, and ingest/delete paths run outside the scope so they always pass through.

DlightRAG supports PostgreSQL as the storage ecosystem. Metadata filtering is
a hard in-filter constraint, not a post-filter hint.
"""

import contextvars
import hashlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dlightrag_rag.retrieval import MetadataScope

from dlightrag.storage.sql_identifiers import pg_identifier, pg_qualified_identifier

logger = logging.getLogger(__name__)

# Max chunks the exact branch will brute-force before falling back to HNSW.
EXACT_FILTER_THRESHOLD = 8192


@dataclass(slots=True)
class MetadataFilterStats:
    """Out-of-scope chunks the knowledge-graph legs asked for and did not get."""

    kg_chunks_dropped: int = 0


# Per-request filter state (async-safe: each coroutine gets its own value)
_active_filter: contextvars.ContextVar[MetadataScope | None] = contextvars.ContextVar(
    "_active_filter", default=None
)
_active_stats: contextvars.ContextVar[MetadataFilterStats | None] = contextvars.ContextVar(
    "_active_stats", default=None
)


@asynccontextmanager
async def metadata_filter_scope(
    scope: MetadataScope | None,
) -> AsyncIterator[MetadataFilterStats]:
    """Set metadata filter for the duration of a retrieval request.

    Within this scope chunks_vdb.query() and text_chunks.get_by_ids() only yield
    chunks belonging to the scope's documents. The yielded stats stay all-zero
    when no filter is active.
    """
    stats = MetadataFilterStats()
    if scope is None:
        yield stats
        return
    filter_token = _active_filter.set(scope)
    stats_token = _active_stats.set(stats)
    try:
        yield stats
    finally:
        _active_filter.reset(filter_token)
        _active_stats.reset(stats_token)


class FilteredVectorStorage:
    """Wraps LightRAG's chunks_vdb to inject metadata filtering.

    When _active_filter contextvar is set (via metadata_filter_scope),
    query() runs native filtered search for the detected backend.
    When unset, delegates to original query() — zero overhead.
    """

    def __init__(
        self,
        original: Any,
        embedding_func: Callable[..., Any],
        *,
        exact_threshold: int = EXACT_FILTER_THRESHOLD,
    ) -> None:
        self._original = original
        self._embedding_func = embedding_func
        self._backend = type(original).__name__
        self._exact_threshold = exact_threshold

    async def query(
        self, query: str | Any, top_k: int, query_embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Query with optional in-filtering via contextvar."""
        scope = _active_filter.get()
        if scope is None:
            return await self._original.query(query, top_k, query_embedding)

        # Compute embedding if not provided
        if query_embedding is None:
            if isinstance(query, str):
                embeddings = await self._embedding_func([query], context="query")
                emb = embeddings[0]
                query_embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            else:
                query_embedding = query

        if query_embedding is None:
            raise RuntimeError("Filtered vector search requires a query embedding")
        if self._backend == "PGVectorStorage":
            return await self._pg_filtered_search(query_embedding, scope, top_k)
        raise RuntimeError(f"Filtered vector search requires PGVectorStorage, got {self._backend}")

    async def _pg_filtered_search(
        self,
        embedding: list[float],
        scope: MetadataScope,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """pgvector SQL with strict metadata in-filtering.

        Scopes with a small chunk fan-out use exact scans to avoid the HNSW
        post-filter recall trap. Large ones use pgvector iterative scans.
        """
        import numpy as np

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
            rows = await self._run_pg_operation(
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

        async def _iterative_search(conn: Any) -> Any:
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

        rows = await self._run_pg_operation(_iterative_search)

        logger.info(
            "HNSW in-filtered PG search: %d results from %d doc(s)/%d chunk(s)",
            len(rows),
            len(doc_ids),
            scope.chunk_count,
        )
        return self._format_rows(rows)

    async def ensure_doc_scope_index(self) -> None:
        """Index full_doc_id so document-scoped exact scans avoid a seq scan.

        LightRAG only indexes (workspace, id) on the vector table; metadata
        filtering looks rows up by document instead. Writer-only — a reader owns
        no corpus schema and must never issue DDL.
        """
        table = self._original.table_name
        table_name = pg_qualified_identifier(table)
        # Hash the table name: it embeds the embedding model, so a plain suffix
        # would overflow PostgreSQL's 63-char identifier limit and two models
        # sharing a long prefix would silently share one index.
        digest = hashlib.md5(table.encode(), usedforsecurity=False).hexdigest()[:12]
        index_name = pg_identifier(f"idx_dlightrag_full_doc_id_{digest}")
        try:
            await self._run_pg_operation(
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

    async def _run_pg_operation(self, operation: Callable[[Any], Awaitable[Any]]) -> Any:
        db = self._original.db
        return await db._run_with_retry(operation)

    @staticmethod
    def _format_rows(rows: Any) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for r in rows:
            chunk = {
                "id": r["id"],
                "content": r.get("content", ""),
                "file_path": r.get("file_path", ""),
                "distance": 1 - r["score"],
            }
            if r.get("full_doc_id"):
                chunk["full_doc_id"] = r["full_doc_id"]
            chunks.append(chunk)
        return chunks

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to original (table_name, workspace, etc.)."""
        return getattr(self._original, name)


class FilteredChunkStore:
    """Wraps LightRAG's text_chunks KV store so a document scope reaches the KG legs.

    LightRAG's entity and relation legs never vector-search for their chunks: they
    read the chunk ids baked into graph nodes at ingest time and resolve them by
    primary key, so the chunks_vdb in-filter cannot see them. Dropping out-of-scope
    rows here is the storage's own contract — get_by_ids already returns None for
    ids it cannot resolve, and every caller skips None entries.

    Filtering the vector lookup that *selects* those ids is not an option: LightRAG
    reads a short result from chunks_vdb.get_vectors_by_ids as storage corruption
    and falls back to an unfiltered ranking method.
    """

    def __init__(self, original: Any) -> None:
        self._original = original

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        rows = await self._original.get_by_ids(ids)
        scope = _active_filter.get()
        if scope is None:
            return rows
        scoped: list[dict[str, Any] | None] = []
        dropped = 0
        for row in rows:
            # A row with no document attribution cannot satisfy a hard filter.
            if row is not None and row.get("full_doc_id") not in scope.doc_ids:
                scoped.append(None)
                dropped += 1
            else:
                scoped.append(row)
        if dropped:
            stats = _active_stats.get()
            if stats is not None:
                stats.kg_chunks_dropped += dropped
            logger.info(
                "Metadata scope dropped %d of %d graph-referenced chunk(s)",
                dropped,
                len(rows),
            )
        return scoped

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to original (global_config, embedding_func, etc.)."""
        return getattr(self._original, name)
