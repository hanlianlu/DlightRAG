# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified retrieval orchestration for LightRAG mix + DlightRAG BM25."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dlightrag.core.retrieval.filtered_vdb import metadata_filter_scope
from dlightrag.core.retrieval.fusion import dedup_chunks_by_content, rrf_fuse
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalResult

logger = logging.getLogger(__name__)


def _format_bm25_top(chunks: list[dict[str, Any]], *, limit: int = 3) -> str:
    parts: list[str] = []
    for chunk in chunks[:limit]:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "?")
        profile = str(chunk.get("bm25_profile") or "?")
        score = chunk.get("score")
        if score is None:
            score_text = "?"
        else:
            try:
                score_text = f"{float(score):.3f}"
            except (TypeError, ValueError):
                score_text = "?"
        parts.append(f"{chunk_id}:{profile}:{score_text}")
    if len(chunks) > limit:
        parts.append(f"+{len(chunks) - limit}")
    return ",".join(parts) if parts else "none"


def _chunk_ids(chunks: list[dict[str, Any]]) -> set[str]:
    return {
        str(chunk_id) for chunk in chunks if (chunk_id := chunk.get("chunk_id") or chunk.get("id"))
    }


def _count_fused_sources(
    fused_chunks: list[dict[str, Any]],
    *,
    semantic_chunks: list[dict[str, Any]],
    bm25_chunks: list[dict[str, Any]],
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
        backend: Any,
        bm25: Any | None,
        metadata_index: Any,
        stores: Any,
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
        multimodal_content: list[dict[str, Any]] | None = None,
        bm25_query: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        candidate_ids = await self._resolve_candidates(metadata_filter)
        trace: dict[str, Any] = {
            "metadata_filter_source": metadata_filter_source,
            "metadata_candidate_count": len(candidate_ids) if candidate_ids is not None else None,
            "metadata_filter_relaxed": False,
        }
        if candidate_ids is not None and not candidate_ids:
            if metadata_filter_source == "llm_inferred":
                candidate_ids = None
                trace["metadata_filter_relaxed"] = True
            else:
                trace["strict_filter_empty"] = True
                return RetrievalResult(trace=trace)

        chunk_candidate_limit = chunk_top_k or top_k
        lexical_query = bm25_query or query
        async with metadata_filter_scope(candidate_ids):
            lightrag_task = asyncio.create_task(
                self._backend.aretrieve(
                    query,
                    multimodal_content=multimodal_content,
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
                        candidate_ids=candidate_ids,
                        top_k=chunk_candidate_limit,
                    )
                )
                if self._bm25 is not None
                else None
            )

            try:
                lightrag_result = await lightrag_task
            except Exception:
                logger.warning(
                    "LightRAG retrieval failed; falling back to BM25-only", exc_info=True
                )
                lightrag_result = RetrievalResult(
                    trace={"lightrag_error": True},
                )
            bm25_chunks = await bm25_task if bm25_task is not None else []

        trace.update(getattr(lightrag_result, "trace", {}) or {})
        trace["bm25_enabled"] = self._bm25 is not None
        trace["bm25_query"] = lexical_query if self._bm25 is not None else None
        trace["bm25_chunk_count"] = len(bm25_chunks)
        trace["semantic_chunk_count"] = len(lightrag_result.contexts.get("chunks", []))
        semantic_chunks = lightrag_result.contexts.get("chunks", [])
        if bm25_chunks:
            fused = rrf_fuse([semantic_chunks, bm25_chunks], k=self._rrf_k)
        else:
            fused = list(semantic_chunks)
        fused = dedup_chunks_by_content(fused)
        lightrag_result.contexts["chunks"] = fused
        trace["fused_chunk_count"] = len(lightrag_result.contexts["chunks"])
        fused_source_counts = _count_fused_sources(
            lightrag_result.contexts["chunks"],
            semantic_chunks=semantic_chunks,
            bm25_chunks=bm25_chunks,
        )
        trace["fused_source_counts"] = fused_source_counts
        logger.info(
            "[Retriever] mix: bm25_enabled=%s bm25_query=%r filter_source=%s "
            "metadata_candidates=%s filter_relaxed=%s semantic_chunks=%d bm25_chunks=%d "
            "fused_chunks=%d fused_sources=semantic_only=%d bm25_only=%d both=%d "
            "bm25_top=%s",
            self._bm25 is not None,
            lexical_query if self._bm25 is not None else None,
            metadata_filter_source,
            len(candidate_ids) if candidate_ids is not None else "all",
            trace.get("metadata_filter_relaxed", False),
            trace["semantic_chunk_count"],
            trace["bm25_chunk_count"],
            trace["fused_chunk_count"],
            fused_source_counts["semantic_only"],
            fused_source_counts["bm25_only"],
            fused_source_counts["both"],
            _format_bm25_top(bm25_chunks),
        )
        lightrag_result.trace = trace
        return lightrag_result

    async def _resolve_candidates(self, metadata_filter: MetadataFilter | None) -> set[str] | None:
        if metadata_filter is None or metadata_filter.is_empty():
            return None
        chunk_ids = await metadata_retrieve(
            metadata_index=self._metadata_index,
            stores=self._stores,
            filters=metadata_filter,
        )
        return set(chunk_ids)
