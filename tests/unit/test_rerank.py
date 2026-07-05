# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal reranker strategies."""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import dlightrag.models.rerank as rerank_module
from dlightrag.config import RerankConfig
from dlightrag.models.rerank import (
    _aliyun_rerank,
    _azure_cohere_rerank,
    _azure_cohere_rerank_url,
    _build_scored_chunks,
    _chat_llm_rerank,
    _cohere_rerank,
    _http_rerank,
    _jina_rerank,
    _parse_listwise_scores,
    _resolve_input_modality,
    _voyage_rerank,
    build_rerank_func,
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


class TestBuildScoredChunks:
    def test_preserves_chunk_metadata(self):
        chunks = [
            {
                "chunk_id": "c0",
                "source_id": "s0",
                "reference_id": "1",
                "file_path": "a.pdf",
                "content": "first",
            },
            {
                "chunk_id": "c1",
                "source_id": "s1",
                "reference_id": "2",
                "file_path": "b.pdf",
                "content": "second",
            },
        ]

        result = _build_scored_chunks(
            chunks,
            [{"index": 1, "relevance_score": 0.9}],
            score_threshold=0.5,
            top_k=10,
        )

        assert result == [
            {
                "chunk_id": "c1",
                "source_id": "s1",
                "reference_id": "2",
                "file_path": "b.pdf",
                "content": "second",
                "rerank_score": 0.9,
            }
        ]
        assert "rerank_score" not in chunks[1]


class TestBuildRerankFunc:
    @staticmethod
    def _capture_threshold(monkeypatch, func_name: str) -> dict[str, float | None]:
        captured: dict[str, float | None] = {}

        async def fake_rerank(query, chunks, top_k, **kwargs):
            captured["score_threshold"] = kwargs["score_threshold"]
            return chunks[:top_k]

        monkeypatch.setattr(
            "dlightrag.observability.wrap_rerank_func",
            lambda fn, *, name: fn,
        )
        monkeypatch.setattr(rerank_module, func_name, fake_rerank)
        return captured

    @staticmethod
    def _stub_http_client(monkeypatch) -> None:
        monkeypatch.setattr(
            rerank_module.httpx,
            "AsyncClient",
            lambda *args, **kwargs: object(),
        )

    async def test_chat_llm_default_threshold_filters_weak_scores(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_chat_llm_rerank")

        fn = build_rerank_func(
            RerankConfig(strategy="chat_llm_reranker"),
            ingest_func=AsyncMock(),
        )
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] == 0.5

    async def test_provider_default_threshold_keeps_all_scored_candidates(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_voyage_rerank")
        self._stub_http_client(monkeypatch)

        fn = build_rerank_func(
            RerankConfig(strategy="voyage_reranker", api_key="voyage-key"),
        )
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] is None

    async def test_explicit_provider_threshold_is_preserved(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_voyage_rerank")
        self._stub_http_client(monkeypatch)

        fn = build_rerank_func(
            RerankConfig(
                strategy="voyage_reranker",
                api_key="voyage-key",
                score_threshold=0.42,
            ),
        )
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] == 0.42


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
        with pytest.raises(RuntimeError, match="All rerank batches failed"):
            await _chat_llm_rerank(
                "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.3
            )

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

    async def test_score_threshold_drops_all_below_threshold(self):
        mock_scoring = AsyncMock(return_value="[0.05, 0.08, 0.03]")
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = await _chat_llm_rerank(
            "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.5
        )
        assert result == []

    async def test_batched_scoring(self):
        call_count = 0

        async def mock_scoring(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            content_text = "".join(
                c.get("text", "") for c in messages[0]["content"] if c.get("type") == "text"
            )
            n = content_text.count("\nCandidate ")
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
            n = content_text.count("\nCandidate ")
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

    async def test_multimodal_false_skips_images_uses_full_text(self):
        """When multimodal=False, image_url blocks are omitted and full VLM text is sent."""
        received_messages = []

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8, 0.6]"

        long_vlm_text = "VLM analysis of image: " + "x" * 600
        chunks = [
            {"content": "text only chunk", "image_data": _PNG_B64},
            {"content": long_vlm_text, "image_data": _PNG_B64},
        ]
        await _chat_llm_rerank(
            "query",
            chunks,
            top_k=10,
            scoring_func=mock_scoring,
            score_threshold=0.3,
            multimodal=False,
        )

        content = received_messages[0][0]["content"]
        # No image_url blocks at all
        assert not any(c.get("type") == "image_url" for c in content)
        # Full VLM text (not truncated to 500) — text-only fallback uses full content
        assert long_vlm_text in [c.get("text", "") for c in content]

    async def test_multimodal_true_default_keeps_images(self):
        """Default multimodal=True sends images and truncated text."""
        received_messages = []

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8]"

        chunks = [{"content": "text", "image_data": _PNG_B64}]
        await _chat_llm_rerank(
            "query",
            chunks,
            top_k=10,
            scoring_func=mock_scoring,
            score_threshold=0.3,
        )
        content = received_messages[0][0]["content"]
        marker_idx = next(i for i, c in enumerate(content) if c.get("text") == "\nCandidate 1:")
        image_idx = next(i for i, c in enumerate(content) if c.get("type") == "image_url")
        text_idx = next(i for i, c in enumerate(content) if c.get("text") == "text")
        assert marker_idx < image_idx < text_idx

    async def test_multimodal_false_concurrent_batches(self):
        """Text-only fallback works correctly with batched concurrent scoring."""
        call_count = 0

        async def mock_scoring(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            content_parts = [
                c.get("text", "") for c in messages[0]["content"] if c.get("type") == "text"
            ]
            n = sum(1 for t in content_parts if t.startswith("\nCandidate "))
            return "[" + ", ".join(["0.7"] * n) + "]"

        chunks = [{"content": f"doc{i}", "image_data": _PNG_B64} for i in range(5)]
        result = await _chat_llm_rerank(
            "query",
            chunks,
            top_k=10,
            scoring_func=mock_scoring,
            batch_size=2,
            score_threshold=0.3,
            multimodal=False,
        )
        assert call_count == 3  # 5 chunks, batch_size=2 → 3 batches
        assert len(result) == 5
        # All scores should be valid (not zeros from API error)
        assert all(c["rerank_score"] > 0 for c in result)


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
            client=cast(Any, client),
            image_max_bytes=456,
            image_max_total_bytes=789,
            image_max_px=1024,
            image_min_px=768,
            image_quality=86,
            image_min_quality=76,
        )

        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert client.payload is not None
        assert client.payload["documents"] == [{"image": bounded_uri}]
        assert raw_image not in str(client.payload)

    async def test_multimodal_false_skips_images_sends_text_only(self):
        """When multimodal=False, http reranker only sends {"text": ...} documents."""
        raw_image = "RAW_ORIGINAL_IMAGE_PAYLOAD"
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.85}]})

        result = await _http_rerank(
            "query",
            [{"content": "VLM text description", "image_data": raw_image}],
            top_k=1,
            url="https://rerank.example",
            model="text-only-reranker",
            score_threshold=0.3,
            client=cast(Any, client),
            multimodal=False,
        )

        assert result[0]["rerank_score"] == pytest.approx(0.85)
        # Should only have {"text": ...}, no {"image": ...}
        assert client.payload is not None
        assert client.payload["documents"] == [{"text": "VLM text description"}]
        assert raw_image not in str(client.payload)

    async def test_score_threshold_drops_all_http_results_below_threshold(self):
        client = _CaptureClient(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.2},
                ]
            }
        )

        result = await _http_rerank(
            "query",
            [{"content": "low"}, {"content": "also low"}],
            top_k=10,
            url="https://rerank.example",
            model="reranker",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert result == []


class TestRerankInputModality:
    def test_auto_uses_known_model_capability(self):
        assert _resolve_input_modality("auto", "jina-reranker-v3") == "text"
        assert _resolve_input_modality("auto", "jina-reranker-m0") == "multimodal"
        assert _resolve_input_modality("auto", "qwen3-vl-rerank") == "multimodal"
        assert _resolve_input_modality("auto", "rerank-2.5-lite") == "text"
        assert _resolve_input_modality("auto", "rerank-v4.0") == "text"
        assert _resolve_input_modality("auto", "rerank-v4.0-fast") == "text"
        assert _resolve_input_modality("auto", "unknown-reranker") == "text"

    def test_known_text_models_stay_text_when_multimodal_is_forced(self):
        assert _resolve_input_modality("multimodal", "jina-reranker-v3") == "text"
        assert _resolve_input_modality("multimodal", "rerank-2.5-lite") == "text"
        assert _resolve_input_modality("multimodal", "rerank-v4.0") == "text"
        assert _resolve_input_modality("multimodal", "rerank-v4.0-pro") == "text"


class TestJinaRerank:
    async def test_default_latest_model_uses_text(self):
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.9}]})

        result = await _jina_rerank(
            "query",
            [{"content": "VLM text", "image_data": _PNG_B64}],
            top_k=1,
            url="https://api.jina.ai/v1/rerank",
            api_key="jina-key",
            input_modality="auto",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert client.payload == {
            "model": "jina-reranker-v3",
            "query": "query",
            "documents": ["VLM text"],
            "top_n": 1,
            "return_documents": False,
        }
        assert _PNG_B64 not in str(client.payload)

    async def test_text_model_auto_uses_vlm_text(self):
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.8}]})

        await _jina_rerank(
            "query",
            [{"content": "VLM text", "image_data": _PNG_B64}],
            top_k=1,
            url="https://api.jina.ai/v1/rerank",
            model="jina-reranker-v3",
            api_key="jina-key",
            input_modality="auto",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert client.payload is not None
        assert client.payload["documents"] == ["VLM text"]
        assert _PNG_B64 not in str(client.payload)

    async def test_m0_auto_sends_multimodal_image_document(self):
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.8}]})

        await _jina_rerank(
            "query",
            [{"content": "VLM text", "image_data": _PNG_B64}],
            top_k=1,
            url="https://api.jina.ai/v1/rerank",
            model="jina-reranker-m0",
            api_key="jina-key",
            input_modality="auto",
            client=cast(Any, client),
        )

        assert client.payload is not None
        assert client.payload["documents"] == [{"image": f"data:image/png;base64,{_PNG_B64}"}]


class TestAliyunRerank:
    async def test_qwen3_vl_auto_sends_dashscope_multimodal_payload(self):
        client = _CaptureClient({"output": {"results": [{"index": 0, "relevance_score": 0.93}]}})

        result = await _aliyun_rerank(
            "query",
            [{"content": "VLM text", "image_data": _PNG_B64}],
            top_k=1,
            url="https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            model="qwen3-vl-rerank",
            api_key="aliyun-key",
            input_modality="auto",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert result[0]["rerank_score"] == pytest.approx(0.93)
        assert client.payload == {
            "model": "qwen3-vl-rerank",
            "input": {
                "query": {"text": "query"},
                "documents": [{"image": f"data:image/png;base64,{_PNG_B64}"}],
            },
            "parameters": {"top_n": 1},
        }

    async def test_qwen3_text_uses_compatible_payload(self):
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.86}]})

        await _aliyun_rerank(
            "query",
            [{"content": "VLM text", "image_data": _PNG_B64}],
            top_k=1,
            url="https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/reranks",
            model="qwen3-rerank",
            api_key="aliyun-key",
            input_modality="auto",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert client.payload == {
            "model": "qwen3-rerank",
            "query": "query",
            "documents": ["VLM text"],
            "top_n": 1,
        }
        assert _PNG_B64 not in str(client.payload)


class TestVoyageRerank:
    async def test_sends_voyage_text_payload(self):
        client = _CaptureClient(
            {
                "data": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        )

        result = await _voyage_rerank(
            "query",
            [{"content": "first", "image_data": "RAW_IMAGE"}, {"content": "second"}],
            top_k=1,
            url="https://api.voyageai.com/v1/rerank",
            model="rerank-2.5",
            api_key="voyage-key",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert result == [{"content": "second", "rerank_score": 0.91}]
        assert client.url == "https://api.voyageai.com/v1/rerank"
        assert client.headers is not None
        assert client.headers["Authorization"] == "Bearer voyage-key"
        assert client.payload == {
            "model": "rerank-2.5",
            "query": "query",
            "documents": ["first", "second"],
            "top_k": 1,
            "truncation": True,
        }
        assert "RAW_IMAGE" not in str(client.payload)

    async def test_default_threshold_keeps_negative_scores(self):
        client = _CaptureClient(
            {
                "data": [
                    {"index": 0, "relevance_score": -0.1},
                    {"index": 1, "relevance_score": -0.2},
                ]
            }
        )

        result = await _voyage_rerank(
            "query",
            [{"content": "first"}, {"content": "second"}],
            top_k=2,
            api_key="voyage-key",
            client=cast(Any, client),
        )

        assert result == [
            {"content": "first", "rerank_score": -0.1},
            {"content": "second", "rerank_score": -0.2},
        ]


class TestCohereRerank:
    async def test_sends_cohere_text_payload(self):
        client = _CaptureClient(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.88},
                    {"index": 1, "relevance_score": 0.4},
                ]
            }
        )

        result = await _cohere_rerank(
            "query",
            [{"content": "first"}, {"content": "second", "image_data": "RAW_IMAGE"}],
            top_k=2,
            url="https://api.cohere.com/v2/rerank",
            model="rerank-v4.0-fast",
            api_key="cohere-key",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert result == [{"content": "first", "rerank_score": 0.88}]
        assert client.url == "https://api.cohere.com/v2/rerank"
        assert client.headers is not None
        assert client.headers["Authorization"] == "Bearer cohere-key"
        assert client.payload == {
            "model": "rerank-v4.0-fast",
            "query": "query",
            "documents": ["first", "second"],
            "top_n": 2,
        }
        assert "RAW_IMAGE" not in str(client.payload)


class TestAzureCohereRerank:
    def test_project_endpoint_matching_uses_hostname_only(self):
        assert (
            _azure_cohere_rerank_url("https://project.services.ai.azure.com")
            == "https://project.services.ai.azure.com/providers/cohere/v2/rerank"
        )
        assert (
            _azure_cohere_rerank_url("https://example.com/path/.services.ai.azure.com")
            == "https://example.com/path/.services.ai.azure.com/v1/rerank"
        )
        assert (
            _azure_cohere_rerank_url("https://project.services.ai.azure.com.example.com")
            == "https://project.services.ai.azure.com.example.com/v1/rerank"
        )

    async def test_project_endpoint_uses_provider_route(self):
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.88}]})

        result = await _azure_cohere_rerank(
            "query",
            [{"content": "first"}, {"content": "second"}],
            top_k=1,
            endpoint="https://project.services.ai.azure.com",
            api_key="azure-key",
            deployment="Cohere-rerank-v4.0-pro",
            score_threshold=0.5,
            client=cast(Any, client),
        )

        assert result == [{"content": "first", "rerank_score": 0.88}]
        assert client.url == "https://project.services.ai.azure.com/providers/cohere/v2/rerank"
        assert client.headers is not None
        assert client.headers["Authorization"] == "azure-key"
        assert client.payload == {
            "model": "Cohere-rerank-v4.0-pro",
            "query": "query",
            "documents": ["first", "second"],
            "top_n": 1,
        }

    async def test_model_endpoint_uses_v1_rerank_route(self):
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.88}]})

        await _azure_cohere_rerank(
            "query",
            [{"content": "first"}],
            top_k=1,
            endpoint="https://cohere-rerank-v4-pro.example.eastus.models.ai.azure.com/",
            api_key="azure-key",
            deployment="model",
            client=cast(Any, client),
        )

        assert (
            client.url
            == "https://cohere-rerank-v4-pro.example.eastus.models.ai.azure.com/v1/rerank"
        )


class _CaptureClient:
    def __init__(self, data):
        self.data = data
        self.payload = None
        self.url = None
        self.headers = None

    async def post(self, url, *, json, headers):
        self.url = url
        self.payload = json
        self.headers = headers
        return _CaptureResponse(self.data)


class _CaptureResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data
