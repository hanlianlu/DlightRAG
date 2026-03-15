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


class TestOrchestratorVectorPath:
    @pytest.mark.asyncio
    async def test_vector_path_unified_mode(self) -> None:
        mock_vdb = AsyncMock()
        mock_vdb.query = AsyncMock(return_value=[{"id": "chunk-v1"}, {"id": "chunk-v2"}])

        orch = RetrievalOrchestrator(
            rag_mode="unified",
            chunks_vdb=mock_vdb,
        )
        plan = RetrievalPlan(
            semantic_query="visual content",
            metadata_filters=None,
            paths=["kg"],  # kg not executed here, but vector auto-activates for unified
        )
        result = await orch.orchestrate(plan)
        assert "chunk-v1" in result
        assert "chunk-v2" in result

    @pytest.mark.asyncio
    async def test_vector_path_not_activated_caption_mode(self) -> None:
        mock_vdb = AsyncMock()

        orch = RetrievalOrchestrator(
            rag_mode="caption",
            chunks_vdb=mock_vdb,
        )
        plan = RetrievalPlan(
            semantic_query="test query",
            metadata_filters=None,
            paths=["kg"],
        )
        result = await orch.orchestrate(plan)
        assert result == []
        mock_vdb.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_vector_path_not_activated_without_semantic_query(self) -> None:
        mock_vdb = AsyncMock()

        orch = RetrievalOrchestrator(
            rag_mode="unified",
            chunks_vdb=mock_vdb,
        )
        plan = RetrievalPlan(
            semantic_query="",
            metadata_filters=MetadataFilter(filename="test.pdf"),
            paths=["metadata"],
        )
        await orch.orchestrate(plan)
        # No vector path since semantic_query is empty
        mock_vdb.query.assert_not_called()


class TestOrchestratorMultiPath:
    @pytest.mark.asyncio
    async def test_metadata_plus_vector_fusion(self, mock_metadata_index) -> None:
        from lightrag.utils import compute_mdhash_id

        # Unified mode: metadata_index.get() returns page_count
        chunk_0 = compute_mdhash_id("doc-1:page:0", prefix="chunk-")
        chunk_1 = compute_mdhash_id("doc-1:page:1", prefix="chunk-")
        mock_metadata_index.get = AsyncMock(return_value={"page_count": 2})

        # Vector returns one overlapping chunk (chunk_0) + one unique
        mock_vdb = AsyncMock()
        mock_vdb.query = AsyncMock(return_value=[{"id": chunk_0}, {"id": "chunk-v1"}])

        orch = RetrievalOrchestrator(
            metadata_index=mock_metadata_index,
            rag_mode="unified",
            chunks_vdb=mock_vdb,
        )
        plan = RetrievalPlan(
            semantic_query="visual content",
            metadata_filters=MetadataFilter(filename="test.pdf"),
            paths=["metadata"],
        )
        result = await orch.orchestrate(plan)
        # chunk_0 should be boosted (found in both paths)
        assert result[0] == chunk_0
        assert len(set(result)) == len(result)  # no duplicates


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
