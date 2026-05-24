# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified retrieval orchestration for LightRAG mix + DlightRAG BM25."""

from __future__ import annotations

import asyncio
from typing import Any

from dlightrag.core.retrieval.filtered_vdb import metadata_filter_scope
from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalResult


class UnifiedRetriever:
    """Run retrieval-wide metadata filtering, LightRAG mix, BM25, and fusion."""

    def __init__(
        self,
        *,
        backend: Any,
        bm25: Any | None,
        metadata_index: Any,
        chunk_provenance: Any,
        rrf_k: int = 60,
    ) -> None:
        self._backend = backend
        self._bm25 = bm25
        self._metadata_index = metadata_index
        self._chunk_provenance = chunk_provenance
        self._rrf_k = rrf_k

    async def aretrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        metadata_filter_source: str | None = None,
        multimodal_content: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        candidate_ids = await self._resolve_candidates(metadata_filter)
        if candidate_ids is not None and not candidate_ids:
            if metadata_filter_source == "llm_inferred":
                candidate_ids = None
            else:
                return RetrievalResult()

        chunk_limit = chunk_top_k or top_k
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
                    self._bm25.search(query, candidate_ids=candidate_ids, top_k=chunk_limit)
                )
                if self._bm25 is not None
                else None
            )

            lightrag_result = await lightrag_task
            bm25_chunks = await bm25_task if bm25_task is not None else []

        if bm25_chunks:
            semantic_chunks = lightrag_result.contexts.get("chunks", [])
            fused = rrf_fuse([semantic_chunks, bm25_chunks], k=self._rrf_k)
            lightrag_result.contexts["chunks"] = fused[: chunk_limit or len(fused)]
        return lightrag_result

    async def _resolve_candidates(
        self, metadata_filter: MetadataFilter | None
    ) -> set[str] | None:
        if metadata_filter is None or metadata_filter.is_empty():
            return None
        chunk_ids = await metadata_retrieve(
            metadata_index=self._metadata_index,
            chunk_provenance=self._chunk_provenance,
            filters=metadata_filter,
        )
        return set(chunk_ids)
