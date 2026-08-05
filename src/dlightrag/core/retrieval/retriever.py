# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified retrieval orchestration for LightRAG mix + DlightRAG BM25."""

import asyncio
import logging
from typing import Any

from dlightrag.core.retrieval.filtered_vdb import metadata_filter_scope
from dlightrag.core.retrieval.fusion import format_bm25_top, rrf_fuse
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter, MetadataScope
from dlightrag.core.retrieval.protocols import (
    BM25Retriever,
    ContextRow,
    MetadataChunkStore,
    RetrievalBackend,
    RetrievalResult,
)
from dlightrag.storage.protocols import MetadataIndexProtocol

logger = logging.getLogger(__name__)


def _chunk_ids(chunks: list[ContextRow]) -> set[str]:
    return {
        str(chunk_id) for chunk in chunks if (chunk_id := chunk.get("chunk_id") or chunk.get("id"))
    }


def _count_fused_sources(
    fused_chunks: list[ContextRow],
    *,
    semantic_chunks: list[ContextRow],
    bm25_chunks: list[ContextRow],
) -> dict[str, int]:
    semantic_ids = _chunk_ids(semantic_chunks)
    bm25_ids = _chunk_ids(bm25_chunks)
    counts = {"semantic_only": 0, "bm25_only": 0, "both": 0}

    for chunk_id in _chunk_ids(fused_chunks):
        in_semantic = chunk_id in semantic_ids
        in_bm25 = chunk_id in bm25_ids
        if in_semantic and in_bm25:
            counts["both"] += 1
        elif in_semantic:
            counts["semantic_only"] += 1
        elif in_bm25:
            counts["bm25_only"] += 1
    return counts


class UnifiedRetriever:
    """Run retrieval-wide metadata filtering, LightRAG mix, BM25, and fusion."""

    def __init__(
        self,
        *,
        backend: RetrievalBackend,
        bm25: BM25Retriever | None,
        metadata_index: MetadataIndexProtocol,
        stores: MetadataChunkStore,
        rrf_k: int = 60,
    ) -> None:
        self._backend = backend
        self._bm25 = bm25
        self._metadata_index = metadata_index
        self._stores = stores
        self._rrf_k = rrf_k

    async def aretrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        metadata_filter_source: str | None = None,
        bm25_query: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        scope = await self._resolve_candidates(metadata_filter)
        trace: dict[str, Any] = {
            "metadata_filter_source": metadata_filter_source,
            "metadata_doc_count": len(scope.doc_ids) if scope is not None else None,
            "metadata_candidate_count": scope.chunk_count if scope is not None else None,
            "metadata_filter_relaxed": False,
        }
        if scope is not None and not scope:
            if metadata_filter_source == "llm_inferred":
                scope = None
                trace["metadata_filter_relaxed"] = True
            else:
                trace["strict_filter_empty"] = True
                return RetrievalResult(trace=trace)

        chunk_candidate_limit = chunk_top_k or top_k
        lexical_query = bm25_query or query
        async with metadata_filter_scope(scope) as filter_stats:
            lightrag_task = asyncio.create_task(
                self._backend.aretrieve(
                    query,
                    mode="mix",
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    **kwargs,
                )
            )
            bm25_task = (
                asyncio.create_task(
                    self._bm25.search(
                        lexical_query,
                        scope=scope,
                        top_k=chunk_candidate_limit,
                    )
                )
                if self._bm25 is not None
                else None
            )

            lightrag_error: Exception | None = None
            try:
                lightrag_result = await lightrag_task
            except Exception as exc:
                lightrag_error = exc
                if bm25_task is None:
                    logger.error(
                        "LightRAG retrieval failed and BM25 is disabled",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "LightRAG retrieval failed; falling back to BM25-only",
                        exc_info=True,
                    )
                lightrag_result = RetrievalResult(
                    trace={
                        "lightrag_error": True,
                        "lightrag_error_type": type(exc).__name__,
                    },
                )
            bm25_error: Exception | None = None
            try:
                bm25_chunks = await bm25_task if bm25_task is not None else []
            except Exception as exc:
                bm25_error = exc
                bm25_chunks = []
                logger.warning(
                    "BM25 retrieval failed; falling back to semantic-only", exc_info=True
                )
            if lightrag_error is not None:
                if bm25_task is None:
                    raise lightrag_error
                if bm25_error is not None:
                    raise lightrag_error from bm25_error

        trace.update(getattr(lightrag_result, "trace", {}) or {})
        trace["bm25_enabled"] = self._bm25 is not None
        trace["bm25_query"] = lexical_query if self._bm25 is not None else None
        trace["metadata_kg_chunks_dropped"] = filter_stats.kg_chunks_dropped
        if bm25_error is not None:
            trace["bm25_error_type"] = type(bm25_error).__name__
        trace["bm25_chunk_count"] = len(bm25_chunks)
        trace["semantic_chunk_count"] = len(lightrag_result.contexts.get("chunks", []))
        semantic_chunks = lightrag_result.contexts.get("chunks", [])
        if bm25_chunks:
            fused = rrf_fuse([semantic_chunks, bm25_chunks], k=self._rrf_k)
        else:
            fused = list(semantic_chunks)
        lightrag_result.contexts["chunks"] = fused
        trace["fused_chunk_count"] = len(lightrag_result.contexts["chunks"])
        if scope is not None and metadata_filter_source == "llm_inferred" and not fused:
            relaxed = await self.aretrieve(
                query,
                metadata_filter=None,
                metadata_filter_source=None,
                bm25_query=bm25_query,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                **kwargs,
            )
            relaxed.trace["metadata_filter_source"] = metadata_filter_source
            relaxed.trace["metadata_doc_count"] = len(scope.doc_ids)
            relaxed.trace["metadata_candidate_count"] = scope.chunk_count
            relaxed.trace["metadata_filter_relaxed"] = True
            return relaxed
        fused_source_counts = _count_fused_sources(
            lightrag_result.contexts["chunks"],
            semantic_chunks=semantic_chunks,
            bm25_chunks=bm25_chunks,
        )
        trace["fused_source_counts"] = fused_source_counts
        logger.info(
            "[Retriever] mix: bm25_enabled=%s bm25_query=%r filter_source=%s "
            "metadata_scope=%s filter_relaxed=%s kg_chunks_dropped=%d semantic_chunks=%d "
            "bm25_chunks=%d fused_chunks=%d "
            "fused_sources=semantic_only=%d bm25_only=%d both=%d bm25_top=%s",
            self._bm25 is not None,
            lexical_query if self._bm25 is not None else None,
            metadata_filter_source,
            f"{len(scope.doc_ids)}doc/{scope.chunk_count}chunk" if scope is not None else "all",
            trace.get("metadata_filter_relaxed", False),
            filter_stats.kg_chunks_dropped,
            trace["semantic_chunk_count"],
            trace["bm25_chunk_count"],
            trace["fused_chunk_count"],
            fused_source_counts["semantic_only"],
            fused_source_counts["bm25_only"],
            fused_source_counts["both"],
            format_bm25_top(bm25_chunks),
        )
        lightrag_result.trace = trace
        return lightrag_result

    async def _resolve_candidates(
        self, metadata_filter: MetadataFilter | None
    ) -> MetadataScope | None:
        if metadata_filter is None or metadata_filter.is_empty():
            return None
        return await metadata_retrieve(
            metadata_index=self._metadata_index,
            stores=self._stores,
            filters=metadata_filter,
        )
