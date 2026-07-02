# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Filtered vector storage wrapper for metadata-aware in-filtering.

Wraps LightRAG's chunks_vdb to inject WHERE/filter clause into vector searches
when a per-request metadata filter is active. Uses contextvars for async-safe
per-request state — concurrent requests don't interfere.

DlightRAG supports PostgreSQL as the storage ecosystem. Metadata filtering is
a hard in-filter constraint, not a post-filter hint.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.storage.sql_identifiers import pg_qualified_identifier

logger = logging.getLogger(__name__)

EXACT_FILTER_THRESHOLD = 8192

# Per-request filter state (async-safe: each coroutine gets its own value)
_active_filter: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "_active_filter", default=None
)


@asynccontextmanager
async def metadata_filter_scope(candidate_ids: set[str] | None) -> AsyncIterator[None]:
    """Set metadata filter for the duration of a retrieval request.

    All chunks_vdb.query() calls within this scope will be filtered to only
    return chunks whose IDs are in candidate_ids.
    """
    if candidate_ids is None:
        yield
        return
    token = _active_filter.set(candidate_ids)
    try:
        yield
    finally:
        _active_filter.reset(token)


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
        candidates = _active_filter.get()
        if candidates is None:
            return await self._original.query(query, top_k, query_embedding)

        # Compute embedding if not provided
        if query_embedding is None:
            if isinstance(query, str):
                embeddings = await self._embedding_func([query])
                emb = embeddings[0]
                query_embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            else:
                query_embedding = query

        assert query_embedding is not None
        if self._backend == "PGVectorStorage":
            return await self._pg_filtered_search(query_embedding, candidates, top_k)
        raise RuntimeError(f"Filtered vector search requires PGVectorStorage, got {self._backend}")

    async def _pg_filtered_search(
        self,
        embedding: list[float],
        candidate_ids: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """pgvector SQL with strict metadata in-filtering.

        Small fragmented candidate sets use exact candidate scans to avoid the
        HNSW post-filter recall trap. Large sets use pgvector iterative scans.
        """
        import numpy as np

        if not candidate_ids:
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

        if len(candidate_ids) <= self._exact_threshold:
            rows = await self._run_pg_operation(
                lambda conn: conn.fetch(
                    f"WITH candidate_ids(id) AS (SELECT unnest($3::text[])), "
                    f"candidate_rows AS MATERIALIZED ("
                    f"  SELECT v.id, v.content, v.file_path, v.content_vector "
                    f"  FROM {table_name} v "
                    f"  JOIN candidate_ids c ON c.id = v.id "
                    f"  WHERE v.workspace = $2"
                    f") "
                    f"SELECT id, content, file_path, "
                    f"1 - (content_vector <=> $1::{vector_cast}) AS score "
                    f"FROM candidate_rows "
                    f"WHERE 1 - (content_vector <=> $1::{vector_cast}) > $4 "
                    f"ORDER BY content_vector <=> $1::{vector_cast} "
                    f"LIMIT $5",
                    embedding_vec,
                    workspace,
                    list(candidate_ids),
                    cosine_threshold,
                    top_k,
                )
            )
            logger.info(
                "Exact in-filtered PG search: %d results from %d candidates",
                len(rows),
                len(candidate_ids),
            )
            return self._format_rows(rows)

        async def _iterative_search(conn: Any) -> Any:
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                await conn.execute("SET LOCAL hnsw.max_scan_tuples = 20000")
                return await conn.fetch(
                    f"WITH nearest_results AS MATERIALIZED ("
                    f"  SELECT id, content, file_path, "
                    f"  1 - (content_vector <=> $1::{vector_cast}) AS score, "
                    f"  content_vector <=> $1::{vector_cast} AS distance "
                    f"  FROM {table_name} "
                    f"  WHERE workspace = $2 "
                    f"  AND id = ANY($3::text[]) "
                    f"  ORDER BY content_vector <=> $1::{vector_cast} "
                    f"  LIMIT $5"
                    f") "
                    f"SELECT id, content, file_path, score "
                    f"FROM nearest_results "
                    f"WHERE score > $4 "
                    f"ORDER BY distance + 0",
                    embedding_vec,
                    workspace,
                    list(candidate_ids),
                    cosine_threshold,
                    top_k,
                )

        rows = await self._run_pg_operation(_iterative_search)

        logger.info(
            "HNSW in-filtered PG search: %d results from %d candidates",
            len(rows),
            len(candidate_ids),
        )
        return self._format_rows(rows)

    async def _run_pg_operation(self, operation: Callable[[Any], Awaitable[Any]]) -> Any:
        db = self._original.db
        return await db._run_with_retry(operation)

    @staticmethod
    def _format_rows(rows: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "content": r.get("content", ""),
                "file_path": r.get("file_path", ""),
                "distance": 1 - r["score"],
            }
            for r in rows
        ]

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to original (table_name, workspace, etc.)."""
        return getattr(self._original, name)


# ── Shared force-injection helper ─────────────────────────────────
# Used by service-level candidate injection to recover metadata-resolved chunks
# that the KG/vector retrieval did not return in the first pass.


async def fetch_missing_chunks(
    text_chunks: Any,
    seen_ids: set[str],
    max_count: int = 30,
) -> list[dict[str, Any]]:
    """Fetch chunks matching the active metadata filter but missing from *seen_ids*.

    Reads the per-request ``_active_filter`` contextvar and fetches any
    chunk_ids that are in the filter set but absent from *seen_ids*.

    Returns basic chunk dicts (chunk_id, content, reference_id, file_path).
    ``file_path`` is populated from the LightRAG text_chunks payload when
    available; this is required so downstream
    ``canonicalize_reference_ids`` can assign a stable doc-level
    ``reference_id`` (the upstream helper skips chunks with empty
    file_path). Callers may further enrich with image data afterward.
    """
    active_ids = _active_filter.get()
    if active_ids is None:
        return []
    missing = active_ids - seen_ids
    if not missing:
        return []

    inject_ids = sorted(missing)[:max_count]
    chunks = await fetch_chunks_by_ids(text_chunks, inject_ids)
    if chunks:
        logger.info("Force-injected %d metadata-resolved chunks", len(chunks))
    return chunks


async def fetch_chunks_by_ids(text_chunks: Any, chunk_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch explicit LightRAG text chunk ids and format them for retrieval contexts."""
    if not chunk_ids:
        return []
    inject_ids = list(dict.fromkeys(chunk_ids))
    raw_contents = await text_chunks.get_by_ids(inject_ids)

    chunks: list[dict[str, Any]] = []
    for cid, content_raw in zip(inject_ids, raw_contents, strict=False):
        if content_raw is None:
            continue
        if isinstance(content_raw, str):
            content = content_raw
            file_path = ""
        else:
            content = content_raw.get("content", "")
            file_path = content_raw.get("file_path", "") or ""
        chunks.append(
            {
                "chunk_id": cid,
                "content": content,
                "reference_id": "",
                "file_path": file_path,
            }
        )
    return chunks
