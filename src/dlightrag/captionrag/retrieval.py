# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval engine — composition-based wrapper over RAGAnything.

Provides structured retrieval (RetrievalResult) by composing a RAGAnything
instance rather than inheriting from it. Calls LightRAG's aquery_data
for structured data and RAGAnything's multimodal query enhancement.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from lightrag import QueryParam

from dlightrag.core.retrieval.protocols import RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Structured retrieval over a RAGAnything instance (composition, not inheritance).

    Uses:
    - rag.lightrag.aquery_data() for data-only retrieval
    - rag._process_multimodal_query_content() for multimodal query enhancement
    """

    def __init__(self, rag: Any, config: Any) -> None:
        self.rag = rag
        self.config = config
        self._path_resolver: Any | None = None

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer."""
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            logger.warning("LightRAG not initialized - no documents ingested yet")
            return RetrievalResult(answer=None)

        enhanced_query = query
        if multimodal_content and hasattr(self.rag, "_process_multimodal_query_content"):
            enhanced_query = await self.rag._process_multimodal_query_content(
                query, multimodal_content
            )

        adjusted_top_k = top_k or self.config.top_k
        adjusted_chunk_top_k = chunk_top_k or self.config.chunk_top_k

        if is_reretrieve:
            enable_rerank = False
        else:
            enable_rerank = kwargs.pop("enable_rerank", self.config.rerank.enabled)

        query_kwargs = {
            "top_k": adjusted_top_k,
            "chunk_top_k": adjusted_chunk_top_k,
            "enable_rerank": enable_rerank,
            "max_entity_tokens": kwargs.pop("max_entity_tokens", self.config.max_entity_tokens),
            "max_relation_tokens": kwargs.pop(
                "max_relation_tokens", self.config.max_relation_tokens
            ),
            "max_total_tokens": kwargs.pop("max_total_tokens", self.config.max_total_tokens),
            **kwargs,
        }

        query_param = QueryParam(mode=mode or self.config.default_mode, **query_kwargs)
        retrieval_data = await lightrag.aquery_data(enhanced_query, param=query_param)

        contexts = retrieval_data.get("data", {})

        await self._attach_page_idx(contexts)

        return RetrievalResult(answer=None, contexts=contexts)

    async def _attach_page_idx(self, contexts: dict) -> None:
        """Attach 1-based page_idx from text_chunks KV store to chunk contexts."""
        chunk_contexts = contexts.get("chunks", [])
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag and hasattr(lightrag, "text_chunks") and chunk_contexts:
            chunk_ids = [ctx.get("chunk_id") for ctx in chunk_contexts if ctx.get("chunk_id")]
            if chunk_ids:
                try:
                    chunk_data_list = await lightrag.text_chunks.get_by_ids(chunk_ids)
                    for cid, cdata in zip(chunk_ids, chunk_data_list, strict=True):
                        if cdata and cdata.get("page_idx") is not None:
                            for ctx in chunk_contexts:
                                if ctx.get("chunk_id") == cid:
                                    ctx["page_idx"] = cdata["page_idx"] + 1  # 0→1-based
                except Exception as e:
                    logger.warning(f"Failed to load page_idx: {e}")


__all__ = [
    "RetrievalEngine",
    "RetrievalResult",
]
