# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multi-path retrieval integration in RAGService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.core.retrieval.protocols import RetrievalResult


class TestMergeSupplementaryChunks:
    def test_merge_no_duplicates(self) -> None:
        from dlightrag.core.service import RAGService

        result = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": "c1", "content": "a"}],
                "entities": [],
                "relationships": [],
            }
        )
        supplementary = [
            {"chunk_id": "c2", "content": "b"},
            {"chunk_id": "c3", "content": "c"},
        ]
        RAGService._merge_supplementary_chunks(result, supplementary)
        assert len(result.contexts["chunks"]) == 3

    def test_merge_skips_duplicates(self) -> None:
        from dlightrag.core.service import RAGService

        result = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": "c1", "content": "a"}],
                "entities": [],
                "relationships": [],
            }
        )
        supplementary = [
            {"chunk_id": "c1", "content": "duplicate"},
            {"chunk_id": "c2", "content": "new"},
        ]
        RAGService._merge_supplementary_chunks(result, supplementary)
        assert len(result.contexts["chunks"]) == 2
        chunk_ids = [c["chunk_id"] for c in result.contexts["chunks"]]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids

    def test_merge_empty_supplementary(self) -> None:
        from dlightrag.core.service import RAGService

        result = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": "c1", "content": "a"}],
                "entities": [],
                "relationships": [],
            }
        )
        RAGService._merge_supplementary_chunks(result, [])
        assert len(result.contexts["chunks"]) == 1


class TestResolveChunkContexts:
    @pytest.mark.asyncio
    async def test_resolve_returns_contexts(self) -> None:
        from dlightrag.core.service import RAGService

        svc = RAGService.__new__(RAGService)
        svc._lightrag = MagicMock()
        svc.rag = None
        svc.config = MagicMock()
        svc.config.rag_mode = "caption"
        svc.unified = None
        svc._lightrag.text_chunks = AsyncMock()
        svc._lightrag.text_chunks.get_by_ids = AsyncMock(
            return_value=[
                {"content": "hello", "full_doc_id": "doc-1", "file_path": "/a.pdf", "page_idx": 0},
                None,
                {"content": "world", "full_doc_id": "doc-2", "file_path": "/b.pdf"},
            ]
        )

        contexts = await svc._resolve_chunk_contexts(["c1", "c2", "c3"])
        assert len(contexts) == 2
        assert contexts[0]["chunk_id"] == "c1"
        assert contexts[0]["page_idx"] == 1  # 0-based to 1-based
        assert contexts[1]["chunk_id"] == "c3"
        assert "page_idx" not in contexts[1]

    @pytest.mark.asyncio
    async def test_resolve_no_lightrag(self) -> None:
        from dlightrag.core.service import RAGService

        svc = RAGService.__new__(RAGService)
        svc._lightrag = None
        svc.rag = None
        svc.config = MagicMock()
        svc.config.rag_mode = "caption"
        svc.unified = None

        contexts = await svc._resolve_chunk_contexts(["c1"])
        assert contexts == []
