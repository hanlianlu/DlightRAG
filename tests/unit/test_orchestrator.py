# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RetrievalOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.core.retrieval.models import MetadataFilter, RetrievalPlan
from dlightrag.core.retrieval.orchestrator import RetrievalOrchestrator


@pytest.fixture
def mock_metadata_index():
    idx = AsyncMock()
    idx.query = AsyncMock(return_value=["doc-1"])
    return idx


@pytest.fixture
def mock_lightrag():
    lr = MagicMock()
    lr.doc_status = AsyncMock()
    lr.doc_status.get_by_id = AsyncMock(return_value={"chunks_list": ["chunk-a", "chunk-b"]})
    return lr


class TestOrchestratorMetadataOnly:
    @pytest.mark.asyncio
    async def test_metadata_path_returns_chunks(self, mock_metadata_index, mock_lightrag) -> None:
        orch = RetrievalOrchestrator(
            metadata_index=mock_metadata_index,
            lightrag=mock_lightrag,
            rag_mode="caption",
        )
        plan = RetrievalPlan(
            semantic_query="",
            metadata_filters=MetadataFilter(filename="test.pdf"),
            paths=["metadata"],
        )
        result = await orch.orchestrate(plan)
        assert "chunk-a" in result
        assert "chunk-b" in result

    @pytest.mark.asyncio
    async def test_no_metadata_index_returns_empty(self) -> None:
        orch = RetrievalOrchestrator(
            metadata_index=None,
            lightrag=MagicMock(),
            rag_mode="caption",
        )
        plan = RetrievalPlan(
            semantic_query="test",
            metadata_filters=MetadataFilter(filename="test.pdf"),
            paths=["metadata"],
        )
        result = await orch.orchestrate(plan)
        assert result == []


class TestOrchestratorEmptyPlan:
    @pytest.mark.asyncio
    async def test_no_paths_returns_empty(self) -> None:
        orch = RetrievalOrchestrator(rag_mode="caption")
        plan = RetrievalPlan(
            semantic_query="test",
            metadata_filters=None,
            paths=["kg"],  # kg is not handled by orchestrator
        )
        result = await orch.orchestrate(plan)
        assert result == []

    @pytest.mark.asyncio
    async def test_path_failure_graceful(self, mock_metadata_index) -> None:
        mock_metadata_index.query.side_effect = RuntimeError("DB down")
        orch = RetrievalOrchestrator(
            metadata_index=mock_metadata_index,
            lightrag=MagicMock(),
            rag_mode="caption",
        )
        plan = RetrievalPlan(
            semantic_query="",
            metadata_filters=MetadataFilter(filename="test.pdf"),
            paths=["metadata"],
        )
        result = await orch.orchestrate(plan)
        assert result == []
