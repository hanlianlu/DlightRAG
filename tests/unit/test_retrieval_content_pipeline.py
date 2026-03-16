# tests/unit/test_retrieval_content_pipeline.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval content pipeline changes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
        mock_visual_chunks.get_by_ids = AsyncMock(return_value=[
            {"image_data": "base64data", "page_index": 0, "file_path": "/a.pdf"},
            None,
            None,
        ])

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
