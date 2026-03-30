# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal reranker strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.models.rerank import (
    _chat_llm_rerank,
    _parse_listwise_scores,
    build_lightrag_rerank_adapter,
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


class TestChatLlmRerank:
    async def test_ranks_chunks(self):
        mock_vlm = AsyncMock(return_value="[0.5, 0.9]")
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, vlm_func=mock_vlm, score_threshold=0.3
        )
        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert result[0]["content"] == "doc1"
        assert result[1]["rerank_score"] == pytest.approx(0.5)

    async def test_multimodal_chunks(self):
        """Chunks with image_data should include images in VLM messages."""
        received_messages = []

        async def mock_vlm(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8, 0.6]"

        chunks = [
            {"content": "text only"},
            {"content": "with image", "image_data": "aW1hZ2U="},  # base64 "image"
        ]
        await _chat_llm_rerank("query", chunks, top_k=10, vlm_func=mock_vlm, score_threshold=0.3)

        # Should have sent one VLM call with multimodal content
        assert len(received_messages) == 1
        content = received_messages[0][0]["content"]
        # Check that image_url type is present for the chunk with image_data
        has_image = any(c.get("type") == "image_url" for c in content)
        assert has_image

    async def test_fallback_on_error(self):
        mock_vlm = AsyncMock(side_effect=RuntimeError("API down"))
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, vlm_func=mock_vlm, score_threshold=0.3
        )
        # Fallback keeps top-5 (all zeros)
        assert len(result) == 2
        assert all(c["rerank_score"] == 0.0 for c in result)

    async def test_empty_chunks(self):
        mock_vlm = AsyncMock()
        result = await _chat_llm_rerank(
            "query", [], top_k=10, vlm_func=mock_vlm, score_threshold=0.3
        )
        assert result == []
        mock_vlm.assert_not_called()

    async def test_score_threshold_filters(self):
        mock_vlm = AsyncMock(return_value="[0.1, 0.9, 0.2]")
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, vlm_func=mock_vlm, score_threshold=0.5
        )
        assert len(result) == 1
        assert result[0]["content"] == "b"

    async def test_threshold_fallback_keeps_top5(self):
        mock_vlm = AsyncMock(return_value="[0.05, 0.08, 0.03]")
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, vlm_func=mock_vlm, score_threshold=0.5
        )
        assert len(result) == 3
        assert result[0]["rerank_score"] == pytest.approx(0.08)

    async def test_batched_scoring(self):
        call_count = 0

        async def mock_vlm(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            # Count items by "--- Item" markers in the content
            content_text = "".join(
                c.get("text", "") for c in messages[0]["content"] if c.get("type") == "text"
            )
            n = content_text.count("--- Item")
            return "[" + ", ".join(["0.7"] * n) + "]"

        chunks = [{"content": c} for c in ["a", "b", "c", "d", "e"]]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, vlm_func=mock_vlm, batch_size=2, score_threshold=0.3
        )
        # 5 chunks, batch_size=2 → 3 batches (2+2+1)
        assert call_count == 3
        assert len(result) == 5

    async def test_no_mutation_of_input(self):
        mock_vlm = AsyncMock(return_value="[0.9, 0.5]")
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        original = [c.copy() for c in chunks]
        await _chat_llm_rerank("query", chunks, top_k=10, vlm_func=mock_vlm, score_threshold=0.3)
        assert chunks == original


class TestLightragAdapter:
    async def test_converts_text_to_chunks(self):
        """Adapter should convert list[str] to list[dict] and back."""

        async def mock_rerank(query, chunks, top_k):
            # Verify chunks have content field
            assert all("content" in c for c in chunks)
            # Return scored chunks
            return [{**c, "rerank_score": 0.9 - i * 0.1} for i, c in enumerate(chunks[:top_k])]

        adapter = build_lightrag_rerank_adapter(mock_rerank)
        result = await adapter("test query", ["doc A", "doc B", "doc C"])

        assert len(result) == 3
        assert result[0]["index"] == 0
        assert result[0]["relevance_score"] == pytest.approx(0.9)
        assert result[1]["index"] == 1

    async def test_forwards_top_n(self):
        """LightRAG passes top_n; adapter should forward as top_k."""
        received_top_k = None

        async def mock_rerank(query, chunks, top_k):
            nonlocal received_top_k
            received_top_k = top_k
            return [{**c, "rerank_score": 0.9} for c in chunks[:top_k]]

        adapter = build_lightrag_rerank_adapter(mock_rerank)
        await adapter("query", ["a", "b", "c"], top_n=2)
        assert received_top_k == 2

    async def test_empty_documents(self):
        mock_rerank = AsyncMock()
        adapter = build_lightrag_rerank_adapter(mock_rerank)
        result = await adapter("query", [])
        assert result == []
        mock_rerank.assert_not_called()
