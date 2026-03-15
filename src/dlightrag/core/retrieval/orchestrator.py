# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multi-path retrieval orchestrator."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dlightrag.core.retrieval.fusion import reciprocal_rank_fusion
from dlightrag.core.retrieval.metadata_index import PGMetadataIndex
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import RetrievalPlan
from dlightrag.core.retrieval.vector_path import vector_retrieve

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Dispatches retrieval across multiple paths and fuses results.

    Paths:
    - metadata: structured metadata index queries (mode-agnostic)
    - kg: LightRAG knowledge graph search (existing, unchanged)
    - vector: direct visual embedding search (unified mode only)
    """

    def __init__(
        self,
        metadata_index: PGMetadataIndex | None = None,
        lightrag: Any = None,
        rag_mode: str = "caption",
        chunks_vdb: Any = None,
        rrf_k: int = 60,
    ) -> None:
        self._metadata_index = metadata_index
        self._lightrag = lightrag
        self._rag_mode = rag_mode
        self._chunks_vdb = chunks_vdb
        self._rrf_k = rrf_k

    async def orchestrate(
        self,
        plan: RetrievalPlan,
        top_k: int = 60,
    ) -> list[str]:
        """Execute retrieval plan and return fused chunk_ids.

        Dispatches activated paths in parallel, then fuses via RRF.
        Returns chunk_ids ordered by fused score.
        """
        tasks: dict[str, asyncio.Task[list[str]]] = {}

        if "metadata" in plan.paths and plan.metadata_filters and self._metadata_index:
            tasks["metadata"] = asyncio.create_task(
                metadata_retrieve(
                    self._metadata_index,
                    plan.metadata_filters,
                    self._lightrag,
                    self._rag_mode,
                )
            )

        # VectorPath: unified mode only, and only when there's a semantic query
        if self._rag_mode == "unified" and plan.semantic_query and self._chunks_vdb:
            tasks["vector"] = asyncio.create_task(
                vector_retrieve(
                    plan.semantic_query,
                    self._chunks_vdb,
                    top_k=top_k,
                )
            )

        if not tasks:
            return []

        # Await all paths in parallel
        results: dict[str, list[str]] = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as exc:
                logger.warning("[Orchestrator] path %s failed: %s", name, exc)
                results[name] = []

        fused = reciprocal_rank_fusion(results, k=self._rrf_k)
        logger.info(
            "[Orchestrator] fused %d chunk(s) from %d path(s): %s",
            len(fused),
            len(results),
            list(results.keys()),
        )
        return fused
