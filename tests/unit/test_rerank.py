# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified reranker strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.models.rerank import _fallback_ranking, build_llm_rerank_func


class TestFallbackRanking:
    def test_returns_descending_scores(self):
        result = _fallback_ranking(3)
        assert len(result) == 3
        assert result[0]["relevance_score"] > result[2]["relevance_score"]

    def test_indices_sequential(self):
        result = _fallback_ranking(4)
        assert [r["index"] for r in result] == [0, 1, 2, 3]


class TestLlmListwiseRerank:
    @pytest.mark.asyncio
    async def test_ranks_documents(self):
        mock_llm = AsyncMock(
            return_value='{"ranked_chunks": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]}'
        )
        rerank = build_llm_rerank_func(mock_llm)
        result = await rerank("query", ["doc0", "doc1"])
        assert result[0]["index"] == 1
        assert result[0]["relevance_score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        mock_llm = AsyncMock(side_effect=RuntimeError("API down"))
        rerank = build_llm_rerank_func(mock_llm)
        result = await rerank("query", ["doc0", "doc1"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        mock_llm = AsyncMock()
        rerank = build_llm_rerank_func(mock_llm)
        result = await rerank("query", [])
        assert result == []
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_chunks_appended(self):
        # LLM only returns index 0, should append index 1 with lower score
        mock_llm = AsyncMock(
            return_value='{"ranked_chunks": [{"index": 0, "relevance_score": 0.9}]}'
        )
        rerank = build_llm_rerank_func(mock_llm)
        result = await rerank("query", ["doc0", "doc1"])
        assert len(result) == 2
        assert result[1]["index"] == 1
        assert result[1]["relevance_score"] < result[0]["relevance_score"]
