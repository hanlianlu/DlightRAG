# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal reranker strategies."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from dlightrag.models.rerank import (
    _chat_llm_rerank,
    _http_rerank,
    _parse_listwise_scores,
)

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
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
        mock_scoring = AsyncMock(return_value="[0.5, 0.9]")
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.3
        )
        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert result[0]["content"] == "doc1"
        assert result[1]["rerank_score"] == pytest.approx(0.5)

    async def test_listwise_prompt_comes_from_central_guidance(self, monkeypatch):
        import dlightrag.models.rerank as rerank_module

        prompt_template = "Central listwise prompt: {n} items for {query}"
        monkeypatch.setattr(rerank_module, "LISTWISE_RERANK_PROMPT", prompt_template)
        received_messages = []

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8]"

        await _chat_llm_rerank(
            "market query",
            [{"content": "candidate text"}],
            top_k=10,
            scoring_func=mock_scoring,
            score_threshold=0.3,
        )

        assert received_messages[0][0]["content"][0]["text"] == (
            "Central listwise prompt: 1 items for market query"
        )

    async def test_multimodal_chunks(self):
        """Chunks with image_data should include images in scoring messages."""
        received_messages = []

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8, 0.6]"

        chunks = [
            {"content": "text only"},
            {"content": "with image", "image_data": _PNG_B64},
        ]
        await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.3
        )

        # Should have sent one scoring call with multimodal content
        assert len(received_messages) == 1
        content = received_messages[0][0]["content"]
        # Check that image_url type is present for the chunk with image_data
        has_image = any(c.get("type") == "image_url" for c in content)
        assert has_image

    async def test_multimodal_chunks_are_bounded_before_scoring(self, monkeypatch):
        """Rerank image payloads should not bypass the rerank-stage budget."""
        import dlightrag.utils.image_budget as image_budget_module

        raw_image = "RAW_ORIGINAL_IMAGE_PAYLOAD"
        bounded_uri = "data:image/jpeg;base64,BOUNDED"
        calls = []
        received_messages = []

        def fake_bounded_image_data_uri(value, **kwargs):
            calls.append((value, kwargs))
            return bounded_uri, 123

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8]"

        monkeypatch.setattr(
            image_budget_module,
            "bounded_image_data_uri",
            fake_bounded_image_data_uri,
        )

        await _chat_llm_rerank(
            "query",
            [{"content": "with image", "image_data": raw_image}],
            top_k=10,
            scoring_func=mock_scoring,
            score_threshold=0.3,
            image_max_bytes=456,
            image_max_total_bytes=789,
            image_max_px=1024,
            image_min_px=768,
            image_quality=86,
            image_min_quality=76,
        )

        assert calls == [
            (
                raw_image,
                {
                    "max_bytes": 456,
                    "max_px": 1024,
                    "min_px": 768,
                    "quality": 86,
                    "min_quality": 76,
                },
            )
        ]
        serialized = str(received_messages)
        assert bounded_uri in serialized
        assert raw_image not in serialized

    async def test_multimodal_chunks_skip_images_when_batch_budget_is_exhausted(self, monkeypatch):
        """Reranker should omit images that do not fit the batch request budget."""
        import dlightrag.utils.image_budget as image_budget_module

        received_messages = []

        def fake_bounded_image_data_uri(value, **kwargs):
            return "data:image/jpeg;base64,TOO_LARGE", 100

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8]"

        monkeypatch.setattr(
            image_budget_module,
            "bounded_image_data_uri",
            fake_bounded_image_data_uri,
        )

        await _chat_llm_rerank(
            "query",
            [{"content": "text fallback", "image_data": "raw-image"}],
            top_k=10,
            scoring_func=mock_scoring,
            score_threshold=0.3,
            image_max_bytes=456,
            image_max_total_bytes=50,
            image_max_px=1024,
            image_min_px=768,
            image_quality=86,
            image_min_quality=76,
        )

        content = received_messages[0][0]["content"]
        assert not any(c.get("type") == "image_url" for c in content)
        assert any(c.get("text") == "text fallback" for c in content)

    async def test_fallback_on_error(self):
        mock_scoring = AsyncMock(side_effect=RuntimeError("API down"))
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.3
        )
        # Fallback keeps top-5 (all zeros)
        assert len(result) == 2
        assert all(c["rerank_score"] == 0.0 for c in result)

    async def test_empty_chunks(self):
        mock_scoring = AsyncMock()
        result = await _chat_llm_rerank(
            "query", [], top_k=10, scoring_func=mock_scoring, score_threshold=0.3
        )
        assert result == []
        mock_scoring.assert_not_called()

    async def test_score_threshold_filters(self):
        mock_scoring = AsyncMock(return_value="[0.1, 0.9, 0.2]")
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.5
        )
        assert len(result) == 1
        assert result[0]["content"] == "b"

    async def test_threshold_fallback_keeps_top5(self):
        mock_scoring = AsyncMock(return_value="[0.05, 0.08, 0.03]")
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.5
        )
        assert len(result) == 3
        assert result[0]["rerank_score"] == pytest.approx(0.08)

    async def test_batched_scoring(self):
        call_count = 0

        async def mock_scoring(messages, **kwargs):
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
            "query",
            chunks,
            top_k=10,
            scoring_func=mock_scoring,
            batch_size=2,
            score_threshold=0.3,
        )
        # 5 chunks, batch_size=2 → 3 batches (2+2+1)
        assert call_count == 3
        assert len(result) == 5

    async def test_logs_listwise_schedule_summary(self, caplog):
        caplog.set_level(logging.INFO, logger="dlightrag.models.rerank")

        async def mock_scoring(messages, **kwargs):
            content_text = "".join(
                c.get("text", "") for c in messages[0]["content"] if c.get("type") == "text"
            )
            n = content_text.count("--- Item")
            return "[" + ", ".join(["0.7"] * n) + "]"

        chunks = [{"content": f"doc{i}"} for i in range(46)]
        await _chat_llm_rerank(
            "query",
            chunks,
            top_k=10,
            scoring_func=mock_scoring,
            batch_size=8,
            max_concurrency=8,
            score_threshold=0.3,
        )

        assert (
            "Rerank listwise schedule: chunks=46 batch_size=8 batches=6 "
            "max_concurrency=8 active_batches=6"
        ) in caplog.text

    async def test_max_concurrency_runs_batches_in_parallel(self):
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release = asyncio.Event()
        starts = 0

        async def mock_scoring(messages, **kwargs):
            nonlocal starts
            starts += 1
            if starts == 1:
                first_entered.set()
            elif starts == 2:
                second_entered.set()
            await release.wait()
            return "[0.7]"

        chunks = [{"content": c} for c in ["a", "b", "c"]]
        task = asyncio.create_task(
            _chat_llm_rerank(
                "query",
                chunks,
                top_k=10,
                scoring_func=mock_scoring,
                batch_size=1,
                max_concurrency=2,
                score_threshold=0.3,
            )
        )

        await asyncio.wait_for(first_entered.wait(), timeout=0.5)
        try:
            await asyncio.wait_for(second_entered.wait(), timeout=0.1)
        finally:
            release.set()
            await task

    async def test_no_mutation_of_input(self):
        mock_scoring = AsyncMock(return_value="[0.9, 0.5]")
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        original = [c.copy() for c in chunks]
        await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.3
        )
        assert chunks == original


class TestHttpRerank:
    async def test_bounds_image_payloads(self, monkeypatch):
        import dlightrag.utils.image_budget as image_budget_module

        raw_image = "RAW_ORIGINAL_IMAGE_PAYLOAD"
        bounded_uri = "data:image/jpeg;base64,BOUNDED"
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.9}]})

        def fake_bounded_image_data_uri(value, **kwargs):
            assert value == raw_image
            assert kwargs["max_bytes"] == 456
            return bounded_uri, 123

        monkeypatch.setattr(
            image_budget_module,
            "bounded_image_data_uri",
            fake_bounded_image_data_uri,
        )

        result = await _http_rerank(
            "query",
            [{"content": "with image", "image_data": raw_image}],
            top_k=1,
            url="https://rerank.example",
            model="reranker",
            score_threshold=0.3,
            client=client,
            image_max_bytes=456,
            image_max_total_bytes=789,
            image_max_px=1024,
            image_min_px=768,
            image_quality=86,
            image_min_quality=76,
        )

        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert client.payload["documents"] == [{"image": bounded_uri}]
        assert raw_image not in str(client.payload)


class _CaptureClient:
    def __init__(self, data):
        self.data = data
        self.payload = None

    async def post(self, url, *, json, headers):
        self.payload = json
        return _CaptureResponse(self.data)


class _CaptureResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data
