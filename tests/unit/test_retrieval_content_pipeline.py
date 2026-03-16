# tests/unit/test_retrieval_content_pipeline.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval content pipeline changes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.retrieval.models import MetadataFilter, RetrievalPlan
from dlightrag.core.retrieval.orchestrator import RetrievalOrchestrator

import pytest


class TestVisualRetrieverIncludesTextOnlyChunks:
    """_retrieve() must include chunks that have text content but no visual data."""

    @pytest.mark.asyncio
    async def test_text_only_chunks_not_dropped(self) -> None:
        """Chunks with text content but no visual_chunks entry must appear in output."""
        from dlightrag.unifiedrepresent.retriever import VisualRetriever

        mock_lightrag = MagicMock()
        mock_visual_chunks = AsyncMock()
        mock_config = MagicMock()
        mock_config.rerank_score_threshold = 0.5

        # LightRAG returns 3 chunks
        mock_lightrag.aquery_data = AsyncMock(return_value={
            "data": {
                "chunks": [
                    {"chunk_id": "c1", "content": "text A", "reference_id": "1", "file_path": "/a.pdf"},
                    {"chunk_id": "c2", "content": "text B", "reference_id": "2", "file_path": "/b.pdf"},
                    {"chunk_id": "c3", "content": "text C", "reference_id": "3", "file_path": "/c.pdf"},
                ],
                "entities": [],
                "relationships": [],
            }
        })
        mock_lightrag.text_chunks = AsyncMock()

        # visual_chunks only has c1 (c2, c3 are text-only)
        # Use side_effect to return correct data regardless of set ordering
        visual_map = {
            "c1": {"image_data": "base64data", "page_index": 0, "file_path": "/a.pdf"},
        }
        mock_visual_chunks.get_by_ids = AsyncMock(
            side_effect=lambda ids: [visual_map.get(cid) for cid in ids]
        )

        retriever = VisualRetriever(
            lightrag=mock_lightrag,
            visual_chunks=mock_visual_chunks,
            config=mock_config,
            rerank_backend=None,  # skip reranking
        )
        result = await retriever._retrieve("test query", top_k=60, chunk_top_k=10)

        chunk_ids = {c["chunk_id"] for c in result["contexts"]["chunks"]}
        assert "c1" in chunk_ids, "Visual chunk should be included"
        assert "c2" in chunk_ids, "Text-only chunk should NOT be dropped"
        assert "c3" in chunk_ids, "Text-only chunk should NOT be dropped"

        # Verify text-only chunks have content but no image_data
        for chunk in result["contexts"]["chunks"]:
            if chunk["chunk_id"] in ("c2", "c3"):
                assert chunk["content"] != ""
                assert chunk.get("image_data") is None


class TestOrchestratorSimplified:
    """Orchestrator runs only MetadataPath (no VectorPath)."""

    @pytest.mark.asyncio
    async def test_orchestrate_metadata_only(self) -> None:
        """Orchestrator dispatches metadata path only, returns chunk_ids directly."""
        mock_index = AsyncMock()
        plan = RetrievalPlan(
            semantic_query="key info",
            metadata_filters=MetadataFilter(filename="test.pdf"),
            paths=["metadata"],
            original_query="what is key info from test.pdf",
        )

        with patch(
            "dlightrag.core.retrieval.orchestrator.metadata_retrieve",
            new_callable=AsyncMock,
            return_value=["c1", "c2", "c3"],
        ):
            orch = RetrievalOrchestrator(
                metadata_index=mock_index,
                lightrag=MagicMock(),
                rag_mode="unified",
                chunks_vdb=MagicMock(),
            )
            result = await orch.orchestrate(plan, top_k=20)
            assert result == ["c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_orchestrate_no_metadata_returns_empty(self) -> None:
        """No metadata filters -> empty result."""
        plan = RetrievalPlan(
            semantic_query="query",
            metadata_filters=None,
            paths=[],
            original_query="query",
        )
        orch = RetrievalOrchestrator()
        result = await orch.orchestrate(plan, top_k=20)
        assert result == []


# ---------------------------------------------------------------------------
# TestSystemPromptUpdated
# ---------------------------------------------------------------------------


class TestResolveChunkContextsVisualEnrichment:
    """_resolve_chunk_contexts() must enrich with visual data in unified mode."""

    @pytest.mark.asyncio
    async def test_unified_mode_adds_image_data(self) -> None:
        from dlightrag.core.service import RAGService

        svc = RAGService.__new__(RAGService)
        svc.rag = None

        # Mock config
        mock_config = MagicMock()
        mock_config.rag_mode = "unified"
        svc.config = mock_config

        # Mock LightRAG text_chunks
        mock_lr = MagicMock()
        mock_lr.text_chunks.get_by_ids = AsyncMock(return_value=[
            {"content": "Revenue grew", "file_path": "/report.pdf", "full_doc_id": "doc1"},
        ])
        svc._lightrag = mock_lr

        # Mock unified engine with visual_chunks
        mock_unified = MagicMock()
        mock_visual_store = AsyncMock()
        mock_visual_store.get_by_ids = AsyncMock(return_value=[
            {"image_data": "base64img", "page_index": 2},
        ])
        mock_unified.visual_chunks = mock_visual_store
        svc.unified = mock_unified

        contexts = await svc._resolve_chunk_contexts(["chunk-1"])

        assert len(contexts) == 1
        assert contexts[0]["image_data"] == "base64img"
        assert contexts[0]["page_idx"] == 3  # 0-based → 1-based

    @pytest.mark.asyncio
    async def test_caption_mode_no_visual_enrichment(self) -> None:
        from dlightrag.core.service import RAGService

        svc = RAGService.__new__(RAGService)
        svc.rag = None

        mock_config = MagicMock()
        mock_config.rag_mode = "caption"
        svc.config = mock_config

        mock_lr = MagicMock()
        mock_lr.text_chunks.get_by_ids = AsyncMock(return_value=[
            {"content": "text", "file_path": "/doc.pdf", "full_doc_id": "doc1"},
        ])
        svc._lightrag = mock_lr
        svc.unified = None

        contexts = await svc._resolve_chunk_contexts(["chunk-1"])

        assert len(contexts) == 1
        assert "image_data" not in contexts[0]


class TestMergeRoundRobin:
    """_merge_round_robin interleaves Step 1 and Step 2 chunks."""

    def test_round_robin_interleaving(self) -> None:
        from dlightrag.core.service import RAGService
        from dlightrag.core.retrieval.protocols import RetrievalResult

        primary = RetrievalResult(
            contexts={
                "chunks": [
                    {"chunk_id": "p1", "content": "primary1"},
                    {"chunk_id": "p2", "content": "primary2"},
                    {"chunk_id": "p3", "content": "primary3"},
                ],
                "entities": [],
                "relationships": [],
            }
        )
        supplementary = [
            {"chunk_id": "s1", "content": "supp1"},
            {"chunk_id": "s2", "content": "supp2"},
        ]
        RAGService._merge_round_robin(primary, supplementary, top_k=10)

        ids = [c["chunk_id"] for c in primary.contexts["chunks"]]
        assert ids == ["p1", "s1", "p2", "s2", "p3"]

    def test_dedup_step1_wins(self) -> None:
        """Duplicate chunk_ids in supplementary are excluded."""
        from dlightrag.core.service import RAGService
        from dlightrag.core.retrieval.protocols import RetrievalResult

        primary = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": "c1", "content": "original"}],
                "entities": [],
                "relationships": [],
            }
        )
        supplementary = [
            {"chunk_id": "c1", "content": "duplicate"},
            {"chunk_id": "c2", "content": "new"},
        ]
        RAGService._merge_round_robin(primary, supplementary, top_k=10)

        ids = [c["chunk_id"] for c in primary.contexts["chunks"]]
        assert ids == ["c1", "c2"]
        # Step 1 version kept
        assert primary.contexts["chunks"][0]["content"] == "original"

    def test_top_k_cutoff(self) -> None:
        from dlightrag.core.service import RAGService
        from dlightrag.core.retrieval.protocols import RetrievalResult

        primary = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": f"p{i}"} for i in range(5)],
                "entities": [],
                "relationships": [],
            }
        )
        supplementary = [{"chunk_id": f"s{i}"} for i in range(5)]
        RAGService._merge_round_robin(primary, supplementary, top_k=4)

        assert len(primary.contexts["chunks"]) == 4


class TestSystemPromptUpdated:
    """System prompt must mention text excerpts alongside images."""

    def test_prompt_mentions_text_excerpts(self) -> None:
        from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt

        prompt = get_answer_system_prompt(structured=True)
        assert "text excerpts" in prompt.lower() or "document text" in prompt.lower()

    def test_prompt_mentions_images_when_available(self) -> None:
        from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt

        prompt = get_answer_system_prompt(structured=False)
        assert "when available" in prompt.lower() or "if available" in prompt.lower()
