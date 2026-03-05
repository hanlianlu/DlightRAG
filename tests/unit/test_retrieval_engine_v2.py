# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for composition-based RetrievalEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.retrieval.engine import RetrievalEngine, RetrievalResult


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

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_calls_lightrag_aquery_data(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        result = await engine.aretrieve("test query")
        mock_rag.lightrag.aquery_data.assert_awaited_once()
        assert isinstance(result, RetrievalResult)

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_with_multimodal_enhances_query(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aretrieve("query", multimodal_content=[{"type": "image"}])
        mock_rag._process_multimodal_query_content.assert_awaited_once()

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aretrieve_without_multimodal_no_enhancement(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
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
        assert result.contexts == {}


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

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aanswer_returns_answer(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, _ = self._make_engine()
        result = await engine.aanswer("query")
        assert result.answer == "The answer is 42"

    @patch("dlightrag.core.retrieval.engine.augment_retrieval_result", new_callable=AsyncMock)
    async def test_aanswer_calls_aquery_llm(self, mock_augment) -> None:
        mock_augment.side_effect = lambda r, *a, **kw: r
        engine, mock_rag = self._make_engine()
        await engine.aanswer("query")
        mock_rag.lightrag.aquery_llm.assert_awaited_once()
