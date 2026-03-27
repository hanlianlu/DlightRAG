# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified reranker strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.models.rerank import (
    _fallback_ranking,
    _parse_listwise_scores,
    build_llm_rerank_func,
)


class TestParseListwiseScores:
    def test_json_array(self):
        assert _parse_listwise_scores("[0.9, 0.5, 0.3]", 3) == [0.9, 0.5, 0.3]

    def test_regex_fallback(self):
        text = "Scores: 0.95 for first, 0.40 for second"
        result = _parse_listwise_scores(text, 2)
        assert result == [0.95, 0.40]

    def test_clamp_above_one(self):
        assert _parse_listwise_scores("[1.5, 0.5]", 2) == [1.0, 0.5]

    def test_clamp_below_zero(self):
        assert _parse_listwise_scores("[-0.3, 0.5]", 2) == [0.0, 0.5]

    def test_unparseable_returns_zeros(self):
        assert _parse_listwise_scores("no numbers here", 3) == [0.0, 0.0, 0.0]

    def test_truncates_to_expected(self):
        assert _parse_listwise_scores("[0.9, 0.8, 0.7, 0.6]", 2) == [0.9, 0.8]


class TestFallbackRanking:
    def test_returns_descending_scores(self):
        result = _fallback_ranking(3)
        assert len(result) == 3
        assert result[0]["relevance_score"] > result[2]["relevance_score"]

    def test_indices_sequential(self):
        result = _fallback_ranking(4)
        assert [r["index"] for r in result] == [0, 1, 2, 3]


class TestLlmListwiseRerank:
    async def test_ranks_documents(self):
        mock_llm = AsyncMock(return_value="[0.5, 0.9]")
        rerank = build_llm_rerank_func(mock_llm, score_threshold=0.3)
        result = await rerank("query", ["doc0", "doc1"])
        # Sorted descending by score
        assert result[0]["index"] == 1
        assert result[0]["relevance_score"] == pytest.approx(0.9)
        assert result[1]["index"] == 0
        assert result[1]["relevance_score"] == pytest.approx(0.5)

    async def test_fallback_on_error(self):
        mock_llm = AsyncMock(side_effect=RuntimeError("API down"))
        rerank = build_llm_rerank_func(mock_llm)
        result = await rerank("query", ["doc0", "doc1"])
        # Fallback keeps top-5 (all zeros → fallback_ranking)
        assert len(result) == 2

    async def test_empty_documents(self):
        mock_llm = AsyncMock()
        rerank = build_llm_rerank_func(mock_llm)
        result = await rerank("query", [])
        assert result == []
        mock_llm.assert_not_called()

    async def test_score_threshold_filters(self):
        mock_llm = AsyncMock(return_value="[0.1, 0.9, 0.2]")
        rerank = build_llm_rerank_func(mock_llm, score_threshold=0.5)
        result = await rerank("query", ["a", "b", "c"])
        # Only doc1 (0.9) passes threshold
        assert len(result) == 1
        assert result[0]["index"] == 1

    async def test_threshold_fallback_keeps_top5(self):
        mock_llm = AsyncMock(return_value="[0.05, 0.08, 0.03]")
        rerank = build_llm_rerank_func(mock_llm, score_threshold=0.5)
        result = await rerank("query", ["a", "b", "c"])
        # All below threshold → fallback keeps top-5 (here 3)
        assert len(result) == 3
        assert result[0]["relevance_score"] == pytest.approx(0.08)

    async def test_batched_scoring(self):
        call_count = 0

        async def mock_llm(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            # Count chunks in prompt by counting "--- Chunk" markers
            n = prompt.count("--- Chunk")
            return "[" + ", ".join(["0.7"] * n) + "]"

        rerank = build_llm_rerank_func(mock_llm, batch_size=2, score_threshold=0.3)
        result = await rerank("query", ["a", "b", "c", "d", "e"])
        # 5 docs, batch_size=2 → 3 batches (2+2+1)
        assert call_count == 3
        assert len(result) == 5

    async def test_no_mutation_of_input(self):
        mock_llm = AsyncMock(return_value="[0.9, 0.5]")
        docs = ["doc0", "doc1"]
        original = docs.copy()
        rerank = build_llm_rerank_func(mock_llm, score_threshold=0.3)
        await rerank("query", docs)
        assert docs == original

    async def test_domain_knowledge_in_prompt(self):
        mock_llm = AsyncMock(return_value="[0.9]")
        rerank = build_llm_rerank_func(mock_llm, domain_knowledge_hints="Art domain")
        await rerank("query", ["doc0"])
        call_args = mock_llm.call_args[0][0]
        assert "Art domain" in call_args
