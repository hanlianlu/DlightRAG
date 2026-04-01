# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Filtered vector storage wrapper for metadata-aware in-filtering.

Wraps LightRAG's chunks_vdb to inject WHERE/filter clause into vector searches
when a per-request metadata filter is active. Uses contextvars for async-safe
per-request state — concurrent requests don't interfere.

Supports PostgreSQL (pgvector), Milvus, and Qdrant backends. Falls back to
post-filter for unknown backends.

Backported from ArtRAG's filtered_vdb.py, extended for multi-backend support.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

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
    if not candidate_ids:
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
    ) -> None:
        self._original = original
        self._embedding_func = embedding_func
        self._backend = type(original).__name__

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
        if self._backend == "MilvusVectorDBStorage":
            return await self._milvus_filtered_search(query_embedding, candidates, top_k)
        if self._backend == "QdrantVectorDBStorage":
            return await self._qdrant_filtered_search(query_embedding, candidates, top_k)

        # Unknown backend: delegate to original, post-filter
        logger.info("In-filter: unknown backend %s, falling back to post-filter", self._backend)
        results = await self._original.query(query, top_k * 3, query_embedding)
        return [r for r in results if r.get("id") in candidates][:top_k]

    async def _pg_filtered_search(
        self,
        embedding: list[float],
        candidate_ids: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """pgvector SQL with in-filtering + iterative scanning.

        Uses native pgvector parameter binding via register_vector codec
        (registered on LightRAG's pool init), avoiding SQL string interpolation.
        """
        import numpy as np

        table_name = self._original.table_name
        workspace = self._original.workspace
        cosine_threshold = self._original.cosine_better_than_threshold
        pool = self._original.db.pool

        embedding_vec = np.array(embedding, dtype=np.float32)

        async with pool.acquire() as conn:
            await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
            rows = await conn.fetch(
                f"SELECT id, content, file_path, "
                f"1 - (content_vector <=> $1) AS score "
                f"FROM {table_name} "
                f"WHERE workspace = $2 "
                f"AND id = ANY($3) "
                f"AND 1 - (content_vector <=> $1) > $4 "
                f"ORDER BY content_vector <=> $1 "
                f"LIMIT $5",
                embedding_vec,
                workspace,
                list(candidate_ids),
                cosine_threshold,
                top_k,
            )

        logger.info(
            "In-filtered PG search: %d results from %d candidates", len(rows), len(candidate_ids)
        )
        return [
            {
                "id": r["id"],
                "content": r.get("content", ""),
                "file_path": r.get("file_path", ""),
                "distance": 1 - r["score"],
            }
            for r in rows
        ]

    async def _milvus_filtered_search(
        self,
        embedding: list[float],
        candidate_ids: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Milvus native filtered vector search (experimental — not production-tested)."""
        try:
            client = self._original._client
            collection = self._original.final_namespace

            # Escape double-quotes inside chunk_ids to prevent filter injection
            escaped = [cid.replace("\\", "\\\\").replace('"', '\\"') for cid in candidate_ids]
            id_list_str = ", ".join(f'"{cid}"' for cid in escaped)
            filter_expr = f"id in [{id_list_str}]"

            results = client.search(
                collection_name=collection,
                data=[embedding],
                filter=filter_expr,
                limit=top_k,
                output_fields=["id", "content", "file_path"],
            )

            if results and len(results) > 0:
                return [
                    {
                        "id": hit["entity"]["id"],
                        "content": hit["entity"].get("content", ""),
                        "file_path": hit["entity"].get("file_path", ""),
                        "distance": hit.get("distance", 0),
                    }
                    for hit in results[0]
                ]
            return []
        except Exception as exc:
            logger.warning("In-filtered Milvus search failed: %s, falling back to post-filter", exc)
            results = await self._original.query("", top_k * 3, embedding)
            return [r for r in results if r.get("id") in candidate_ids][:top_k]

    async def _qdrant_filtered_search(
        self,
        embedding: list[float],
        candidate_ids: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Qdrant native filtered vector search (experimental — not production-tested)."""
        try:
            from qdrant_client import models  # type: ignore[import-not-found]

            client = self._original._client
            collection = self._original.final_namespace

            results = client.query_points(
                collection_name=collection,
                query=embedding,
                query_filter=models.Filter(
                    must=[models.HasIdCondition(has_id=list(candidate_ids))]
                ),
                limit=top_k,
            )

            if hasattr(results, "points"):
                return [
                    {
                        "id": str(p.id),
                        "content": p.payload.get("content", "") if p.payload else "",
                        "file_path": p.payload.get("file_path", "") if p.payload else "",
                        "distance": p.score if hasattr(p, "score") else 0,
                    }
                    for p in results.points
                ]
            return []
        except Exception as exc:
            logger.warning("In-filtered Qdrant search failed: %s, falling back to post-filter", exc)
            results = await self._original.query("", top_k * 3, embedding)
            return [r for r in results if r.get("id") in candidate_ids][:top_k]

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to original (table_name, workspace, etc.)."""
        return getattr(self._original, name)


# ── Shared force-injection helper ─────────────────────────────────
# Used by both RAGService (caption mode, post-rerank) and
# VisualRetriever (unified mode, pre-rerank) to inject
# metadata-resolved chunks that the KG retrieval missed.


async def fetch_missing_chunks(
    text_chunks: Any,
    seen_ids: set[str],
    max_count: int = 30,
) -> list[dict[str, Any]]:
    """Fetch chunks matching the active metadata filter but missing from *seen_ids*.

    Reads the per-request ``_active_filter`` contextvar and fetches any
    chunk_ids that are in the filter set but absent from *seen_ids*.

    Returns a list of basic chunk dicts (chunk_id, content, reference_id,
    file_path).  Callers may enrich with visual data or metadata afterward.
    """
    active_ids = _active_filter.get()
    if not active_ids:
        return []
    missing = active_ids - seen_ids
    if not missing:
        return []

    inject_ids = sorted(missing)[:max_count]
    raw_contents = await text_chunks.get_by_ids(inject_ids)

    chunks: list[dict[str, Any]] = []
    for cid, content_raw in zip(inject_ids, raw_contents, strict=False):
        if content_raw is None:
            continue
        content = content_raw if isinstance(content_raw, str) else content_raw.get("content", "")
        chunks.append(
            {
                "chunk_id": cid,
                "content": content,
                "reference_id": "",
                "file_path": "",
            }
        )
    if chunks:
        logger.info("Force-injected %d metadata-resolved chunks", len(chunks))
    return chunks
