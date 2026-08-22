# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Filtered storage wrappers for metadata-aware in-filtering.

Wraps the two LightRAG collaborators a document scope has to reach: ``chunks_vdb``
for the vector leg, and the ``text_chunks`` KV store for the knowledge-graph legs.
Uses contextvars for async-safe per-request state — concurrent requests don't
interfere, and ingest/delete paths run outside the scope so they always pass through.

Metadata filtering is a hard adapter-level in-filter constraint, not a
post-filter hint.
"""

import contextvars
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dlightrag.rag.ports import FilteredVectorSearch
from dlightrag.rag.retrieval import MetadataScope

logger = logging.getLogger(__name__)


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
        filtered_search: FilteredVectorSearch,
    ) -> None:
        self._original = original
        self._embedding_func = embedding_func
        self._filtered_search = filtered_search

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
        return await self._filtered_search.search(query_embedding, scope=scope, top_k=top_k)

    async def ensure_doc_scope_index(self) -> None:
        await self._filtered_search.ensure_document_scope_index()

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
