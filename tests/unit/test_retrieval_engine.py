# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval engine: RetrievalEngine methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.captionrag.retrieval import RetrievalEngine
from dlightrag.core.retrieval.protocols import RetrievalResult

# ---------------------------------------------------------------------------
# TestRetrievalEngineAretrieve
# ---------------------------------------------------------------------------


class TestRetrievalEngineAretrieve:
    """Test data-only retrieval via composition."""

    def _make_engine(self, config=None) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = config or DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={"data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []}}
        )
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced query")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    async def test_aretrieve_calls_lightrag_aquery_data(self) -> None:
        engine, mock_rag = self._make_engine()
        result = await engine.aretrieve("test query")
        mock_rag.lightrag.aquery_data.assert_awaited_once()
        assert isinstance(result, RetrievalResult)

    async def test_aretrieve_with_multimodal_enhances_query(self) -> None:
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query", multimodal_content=[{"type": "image"}])
        mock_rag._process_multimodal_query_content.assert_awaited_once()

    async def test_aretrieve_without_multimodal_no_enhancement(self) -> None:
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query")
        mock_rag._process_multimodal_query_content.assert_not_awaited()

    async def test_aretrieve_no_lightrag_returns_empty(self) -> None:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = None
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        result = await engine.aretrieve("query")
        assert result.answer is None
        assert result.contexts == {"chunks": [], "entities": [], "relationships": []}

    async def test_aretrieve_page_idx_injected(self) -> None:
        """page_idx from text_chunks KV store is attached to chunk contexts (0->1-based)."""
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={
                "data": {
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "file_path": "/storage/doc.pdf",
                            "reference_id": "ref-001",
                            "content": "text",
                        },
                    ],
                    "entities": [],
                    "relationships": [],
                }
            }
        )
        mock_rag.lightrag.text_chunks = MagicMock()
        mock_rag.lightrag.text_chunks.get_by_ids = AsyncMock(
            return_value=[{"page_idx": 2, "content": "text"}]
        )
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        result = await engine.aretrieve("query")
        chunk = result.contexts["chunks"][0]
        assert chunk["page_idx"] == 3  # 0-based 2 -> 1-based 3


# ---------------------------------------------------------------------------
# TestRetrievalEngineAanswer
# ---------------------------------------------------------------------------


class TestRetrievalEngineAanswer:
    """Test LLM answer retrieval."""

    def _make_engine(self) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()
        mock_rag.lightrag.aquery_llm = AsyncMock(
            return_value={
                "llm_response": {"content": "The answer is 42"},
                "data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []},
            }
        )
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    async def test_aanswer_returns_answer(self) -> None:
        engine, _ = self._make_engine()
        result = await engine.aanswer("query")
        assert result.answer == "The answer is 42"

    async def test_aanswer_calls_aquery_llm(self) -> None:
        engine, mock_rag = self._make_engine()
        await engine.aanswer("query")
        mock_rag.lightrag.aquery_llm.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRetrievalEngineAanswerStream
# ---------------------------------------------------------------------------


class TestRetrievalEngineAanswerStream:
    """Test streaming answer retrieval."""

    def _make_engine(self) -> tuple[RetrievalEngine, MagicMock]:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = MagicMock()

        # aquery_data returns retrieval contexts
        mock_rag.lightrag.aquery_data = AsyncMock(
            return_value={
                "data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []},
            }
        )

        # aquery returns an async iterator when stream=True
        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        mock_rag.lightrag.aquery = AsyncMock(return_value=mock_stream())
        mock_rag._process_multimodal_query_content = AsyncMock(return_value="enhanced")
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        return engine, mock_rag

    async def test_aanswer_stream_returns_contexts_and_iterator(self) -> None:
        engine, mock_rag = self._make_engine()
        contexts, token_iter = await engine.aanswer_stream("query")
        assert "chunks" in contexts
        tokens = [t async for t in token_iter]
        assert tokens == ["Hello", " ", "world"]

    async def test_aanswer_stream_calls_aquery_with_stream_true(self) -> None:
        engine, mock_rag = self._make_engine()
        await engine.aanswer_stream("query")
        mock_rag.lightrag.aquery.assert_awaited_once()
        mock_rag.lightrag.aquery_data.assert_awaited_once()

    async def test_aanswer_stream_no_lightrag_raises(self) -> None:
        from dlightrag.config import DlightragConfig

        cfg = DlightragConfig(openai_api_key="test")  # type: ignore[call-arg]
        mock_rag = MagicMock()
        mock_rag.lightrag = None
        engine = RetrievalEngine(rag=mock_rag, config=cfg)
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.aanswer_stream("query")
