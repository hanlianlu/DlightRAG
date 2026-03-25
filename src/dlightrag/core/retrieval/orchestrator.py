# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multi-path retrieval orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from dlightrag.core.retrieval.metadata_index import PGMetadataIndex
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import RetrievalPlan

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Dispatches retrieval to MetadataPath and returns ranked chunk_ids.

    MetadataPath handles its own scoped vector search internally when
    result sets are large — no separate VectorPath needed.
    """

    def __init__(
        self,
        metadata_index: PGMetadataIndex | None = None,
        lightrag: Any = None,
        rag_mode: str = "caption",
        chunks_vdb: Any = None,
    ) -> None:
        self._metadata_index = metadata_index
        self._lightrag = lightrag
        self._rag_mode = rag_mode
        self._chunks_vdb = chunks_vdb

    async def orchestrate(
        self,
        plan: RetrievalPlan,
        top_k: int = 60,
    ) -> list[str]:
        """Execute retrieval plan and return chunk_ids.

        Dispatches MetadataPath with scoped search params.
        """
        if not plan.metadata_filters or not self._metadata_index:
            return []

        try:
            chunk_ids = await metadata_retrieve(
                self._metadata_index,
                plan.metadata_filters,
                self._lightrag,
                self._rag_mode,
                query=plan.query,
                chunks_vdb=self._chunks_vdb,
                top_k=top_k,
            )
            logger.info(
                "[Orchestrator] MetadataPath returned %d chunk(s)",
                len(chunk_ids),
            )
            return chunk_ids
        except Exception as exc:
            logger.warning("[Orchestrator] MetadataPath failed: %s", exc)
            return []
