# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared rerank fallback and multimodal provider strategies."""

import asyncio
import json
import logging
from functools import partial
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import dlightrag.models.rerank as rerank_module
from dlightrag.config import RerankConfig
from dlightrag.core.retrieval.rerank import rerank_with_fallback
from dlightrag.models.providers.rerank_base import resolve_rerank_input_modality
from dlightrag.models.providers.rerank_providers import (
    AliyunRerankProvider,
    AzureCohereRerankProvider,
    CohereRerankProvider,
    HttpRerankProvider,
    JinaRerankProvider,
    VoyageRerankProvider,
    _azure_cohere_rerank_url,
)
from dlightrag.models.rerank import (
    _build_scored_chunks,
    _chat_llm_rerank,
    _parse_listwise_scores,
    _run_http_rerank,
    build_rerank_func,
)
from dlightrag.utils.image_budget import ImagePayloadBudget


def _chunks() -> list[dict[str, Any]]:
    return [
        {"chunk_id": "a", "content": "first"},
        {"chunk_id": "b", "content": "second"},
        {"chunk_id": "c", "content": "third"},
    ]


async def test_rerank_with_fallback_caps_successful_provider_result() -> None:
    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        assert query == "query"
        assert top_k == 2
        return list(reversed(chunks))

    outcome = await rerank_with_fallback(
        query="query",
        chunks=_chunks(),
        top_k=2,
        rerank_func=_rerank,
    )

    assert [chunk["chunk_id"] for chunk in outcome.chunks] == ["c", "b"]
    assert outcome.reranked is True
    assert outcome.error_type is None


async def test_rerank_with_fallback_truncates_rrf_when_provider_absent() -> None:
    outcome = await rerank_with_fallback(
        query="query",
        chunks=_chunks(),
        top_k=2,
        rerank_func=None,
    )

    assert [chunk["chunk_id"] for chunk in outcome.chunks] == ["a", "b"]
    assert outcome.reranked is False
    assert outcome.error_type is None


async def test_rerank_with_fallback_returns_rrf_top_k_and_logs_failure_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("provider unavailable")

    async def _fail(**_kwargs: Any) -> Any:
        raise failure

    with caplog.at_level(logging.WARNING, logger="dlightrag.core.retrieval.rerank"):
        outcome = await rerank_with_fallback(
            query="query",
            chunks=_chunks(),
            top_k=2,
            rerank_func=_fail,
        )

    assert [chunk["chunk_id"] for chunk in outcome.chunks] == ["a", "b"]
    assert outcome.reranked is False
    assert outcome.error_type == "RuntimeError"
    records = [
        record for record in caplog.records if record.name == "dlightrag.core.retrieval.rerank"
    ]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is failure


async def test_rerank_with_fallback_propagates_cancellation() -> None:
    async def _cancel(**_kwargs: Any) -> Any:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await rerank_with_fallback(
            query="query",
            chunks=_chunks(),
            top_k=2,
            rerank_func=_cancel,
        )


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _rerank_user_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        json.loads(block["text"]) for block in messages[1]["content"] if block.get("type") == "text"
    ]


class TestParseListwiseScores:
    def test_json_array(self):
        assert _parse_listwise_scores("[0.9, 0.5, 0.3]", 3) == [0.9, 0.5, 0.3]

    def test_non_json_response_returns_zeros(self):
        text = "Scores: 0.95 for first, 0.40 for second"
        result = _parse_listwise_scores(text, 2)
        assert result == [0.0, 0.0]

    @pytest.mark.parametrize("text", ("[1.5, 0.5]", "[-0.3, 0.5]"))
    def test_out_of_domain_score_invalidates_batch(self, text):
        assert _parse_listwise_scores(text, 2) == [0.0, 0.0]

    def test_unparseable_returns_zeros(self):
        assert _parse_listwise_scores("no numbers here", 3) == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("text", ("[0.9]", "[0.9, 0.8, 0.7]"))
    def test_wrong_score_count_returns_zeros(self, text):
        assert _parse_listwise_scores(text, 2) == [0.0, 0.0]


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

    async def test_chat_llm_default_threshold_keeps_all_scored_candidates(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_chat_llm_rerank")

        fn = build_rerank_func(
            RerankConfig(strategy="chat_llm_reranker"),
            ingest_func=AsyncMock(),
        )
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] is None

    async def test_provider_default_threshold_keeps_all_scored_candidates(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_run_http_rerank")
        self._stub_http_client(monkeypatch)

        fn = build_rerank_func(
            RerankConfig(strategy="voyage_reranker", api_key="voyage-key"),
        )
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] is None

    async def test_explicit_provider_threshold_is_preserved(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_run_http_rerank")
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

    def test_provider_requires_api_key(self):
        with pytest.raises(ValueError, match="requires api_key"):
            build_rerank_func(RerankConfig(strategy="voyage_reranker"))

    def test_provider_requires_base_url(self):
        with pytest.raises(ValueError, match="requires base_url"):
            build_rerank_func(RerankConfig(strategy="aliyun_reranker", api_key="k"))

    def test_multimodal_on_text_only_provider_raises(self):
        with pytest.raises(ValueError, match="text-only"):
            build_rerank_func(
                RerankConfig(strategy="voyage_reranker", api_key="k", input_modality="multimodal")
            )


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

    async def test_listwise_prompt_separates_rules_from_json_data(self, monkeypatch):
        import dlightrag.models.rerank as rerank_module

        prompt_template = "Central listwise prompt for {n} items"
        monkeypatch.setattr(rerank_module, "LISTWISE_RERANK_SYSTEM_PROMPT", prompt_template)
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

        messages = received_messages[0]
        assert messages[0] == {"role": "system", "content": "Central listwise prompt for 1 items"}
        content = messages[1]["content"]
        assert json.loads(content[0]["text"]) == {"query": "market query"}
        assert json.loads(content[1]["text"]) == {
            "candidate": 1,
            "text": "candidate text",
        }

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
        content = received_messages[0][1]["content"]
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
                    "max_pixels": 40_000_000,
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

        content = received_messages[0][1]["content"]
        assert not any(c.get("type") == "image_url" for c in content)
        assert any(
            payload.get("text") == "text fallback"
            for payload in _rerank_user_payloads(received_messages[0])
        )

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
            n = sum("candidate" in payload for payload in _rerank_user_payloads(messages))
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
            n = sum("candidate" in payload for payload in _rerank_user_payloads(messages))
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

        content = received_messages[0][1]["content"]
        # No image_url blocks at all
        assert not any(c.get("type") == "image_url" for c in content)
        # Full VLM text (not truncated to 500) — text-only fallback uses full content
        assert long_vlm_text in [
            payload.get("text", "") for payload in _rerank_user_payloads(received_messages[0])
        ]

    async def test_multimodal_true_keeps_image_and_full_text(self):
        """Default multimodal=True fuses the image with the FULL VLM description."""
        received_messages = []

        async def mock_scoring(messages, **kwargs):
            received_messages.append(messages)
            return "[0.8]"

        long_vlm_text = "VLM analysis of image: " + "x" * 600
        chunks = [{"content": long_vlm_text, "image_data": _PNG_B64}]
        await _chat_llm_rerank(
            "query",
            chunks,
            top_k=10,
            scoring_func=mock_scoring,
            score_threshold=0.3,
        )
        content = received_messages[0][1]["content"]
        candidate_idx = next(
            i
            for i, block in enumerate(content)
            if block.get("type") == "text" and json.loads(block["text"]).get("candidate") == 1
        )
        image_idx = next(i for i, c in enumerate(content) if c.get("type") == "image_url")
        assert candidate_idx < image_idx
        # Full description is fused with the image -- no arbitrary truncation.
        assert long_vlm_text in [
            payload.get("text", "") for payload in _rerank_user_payloads(received_messages[0])
        ]

    async def test_multimodal_false_concurrent_batches(self):
        """Text-only fallback works correctly with batched concurrent scoring."""
        call_count = 0

        async def mock_scoring(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            n = sum("candidate" in payload for payload in _rerank_user_payloads(messages))
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


class TestRerankInputModality:
    def test_auto_defaults_to_text(self):
        assert resolve_rerank_input_modality("auto") == "text"

    def test_explicit_modality_is_honored(self):
        assert resolve_rerank_input_modality("text") == "text"
        assert resolve_rerank_input_modality("multimodal") == "multimodal"


def _budget_factory():
    return partial(
        ImagePayloadBudget,
        max_total_bytes=8_000_000,
        max_bytes_per_image=1_500_000,
        max_pixels=40_000_000,
        max_px=1280,
        min_px=768,
        quality=86,
        min_quality=76,
    )


class TestRunHttpRerank:
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

        result = await _run_http_rerank(
            "query",
            [{"content": "with image", "image_data": raw_image}],
            top_k=1,
            provider=HttpRerankProvider(),
            model="reranker",
            base_url="https://rerank.example",
            api_key=None,
            modality="multimodal",
            score_threshold=0.3,
            client=cast(Any, client),
            budget_factory=partial(
                ImagePayloadBudget,
                max_total_bytes=789,
                max_bytes_per_image=456,
                max_pixels=40_000_000,
                max_px=1024,
                min_px=768,
                quality=86,
                min_quality=76,
            ),
        )

        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert client.payload is not None
        assert client.payload["documents"] == [{"image": bounded_uri}]
        assert client.url == "https://rerank.example/rerank"
        assert raw_image not in str(client.payload)

    async def test_text_modality_skips_images_sends_text_only(self):
        raw_image = "RAW_ORIGINAL_IMAGE_PAYLOAD"
        client = _CaptureClient({"results": [{"index": 0, "relevance_score": 0.85}]})

        result = await _run_http_rerank(
            "query",
            [{"content": "VLM text description", "image_data": raw_image}],
            top_k=1,
            provider=HttpRerankProvider(),
            model="text-only-reranker",
            base_url="https://rerank.example",
            api_key=None,
            modality="text",
            score_threshold=0.3,
            client=cast(Any, client),
            budget_factory=_budget_factory(),
        )

        assert result[0]["rerank_score"] == pytest.approx(0.85)
        assert client.payload is not None
        assert client.payload["documents"] == [{"text": "VLM text description"}]
        assert raw_image not in str(client.payload)

    async def test_score_threshold_drops_all_results_below_threshold(self):
        client = _CaptureClient(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.2},
                ]
            }
        )

        result = await _run_http_rerank(
            "query",
            [{"content": "low"}, {"content": "also low"}],
            top_k=10,
            provider=HttpRerankProvider(),
            model="reranker",
            base_url="https://rerank.example",
            api_key=None,
            modality="text",
            score_threshold=0.5,
            client=cast(Any, client),
            budget_factory=_budget_factory(),
        )

        assert result == []

    async def test_empty_chunks_short_circuits(self):
        client = _CaptureClient({"results": []})
        result = await _run_http_rerank(
            "query",
            [],
            top_k=5,
            provider=HttpRerankProvider(),
            model="reranker",
            base_url="https://rerank.example",
            api_key=None,
            modality="text",
            score_threshold=None,
            client=cast(Any, client),
            budget_factory=_budget_factory(),
        )
        assert result == []
        assert client.payload is None


class TestRerankProviders:
    def test_jina_text_and_image_documents(self):
        provider = JinaRerankProvider()
        assert provider.accepts_images is True
        assert provider.request_url(None, "m") == "https://api.jina.ai/v1/rerank"
        text_payload = provider.build_payload(
            model="jina-reranker-v3", query="q", documents=[("VLM text", None)], top_n=1
        )
        assert text_payload == {
            "model": "jina-reranker-v3",
            "query": "q",
            "documents": ["VLM text"],
            "top_n": 1,
            "return_documents": False,
        }
        image_payload = provider.build_payload(
            model="jina-reranker-m0",
            query="q",
            documents=[("VLM text", "data:image/png;base64,ABC")],
            top_n=1,
        )
        assert image_payload["documents"] == [{"image": "data:image/png;base64,ABC"}]

    def test_voyage_text_only_top_k_and_data_key(self):
        provider = VoyageRerankProvider()
        assert provider.accepts_images is False
        payload = provider.build_payload(
            model="rerank-2.5",
            query="q",
            documents=[("first", "data:image/png;base64,ABC"), ("second", None)],
            top_n=1,
        )
        assert payload == {
            "model": "rerank-2.5",
            "query": "q",
            "documents": ["first", "second"],
            "top_k": 1,
            "truncation": True,
        }
        assert provider.parse_results({"data": [{"index": 0, "relevance_score": 0.9}]}) == [
            {"index": 0, "relevance_score": 0.9}
        ]

    def test_cohere_text_only(self):
        provider = CohereRerankProvider()
        assert provider.accepts_images is False
        payload = provider.build_payload(
            model="rerank-v4.0-fast", query="q", documents=[("first", None)], top_n=2
        )
        assert payload == {
            "model": "rerank-v4.0-fast",
            "query": "q",
            "documents": ["first"],
            "top_n": 2,
        }

    def test_aliyun_qwen3_text_flat_body(self):
        provider = AliyunRerankProvider()
        payload = provider.build_payload(
            model="qwen3-rerank",
            query="q",
            documents=[("VLM text", "data:image/png;base64,ABC")],
            top_n=1,
        )
        assert payload == {
            "model": "qwen3-rerank",
            "query": "q",
            "documents": ["VLM text"],
            "top_n": 1,
        }
        assert provider.parse_results({"results": [{"index": 0, "relevance_score": 0.8}]}) == [
            {"index": 0, "relevance_score": 0.8}
        ]

    def test_aliyun_qwen3_vl_nested_multimodal_body(self):
        provider = AliyunRerankProvider()
        assert provider.accepts_images is True
        payload = provider.build_payload(
            model="qwen3-vl-rerank",
            query="q",
            documents=[("VLM text", "data:image/png;base64,ABC")],
            top_n=1,
        )
        assert payload == {
            "model": "qwen3-vl-rerank",
            "input": {
                "query": {"text": "q"},
                "documents": [{"image": "data:image/png;base64,ABC"}],
            },
            "parameters": {"top_n": 1},
        }
        assert provider.parse_results(
            {"output": {"results": [{"index": 0, "relevance_score": 0.93}]}}
        ) == [{"index": 0, "relevance_score": 0.93}]

    def test_http_generic_appends_rerank_path_and_dict_documents(self):
        provider = HttpRerankProvider()
        assert provider.requires_api_key is False
        assert (
            provider.request_url("https://rerank.example/", "m") == "https://rerank.example/rerank"
        )
        payload = provider.build_payload(
            model="m",
            query="q",
            documents=[("text a", None), ("text b", "data:image/png;base64,ABC")],
            top_n=2,
        )
        assert payload == {
            "model": "m",
            "query": "q",
            "documents": [{"text": "text a"}, {"image": "data:image/png;base64,ABC"}],
            "top_n": 2,
        }

    def test_azure_cohere_url_derivation(self):
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

    def test_azure_cohere_raw_authorization_and_route(self):
        provider = AzureCohereRerankProvider()
        assert provider.accepts_images is False
        assert provider.request_headers("azure-key") == {
            "Content-Type": "application/json",
            "Authorization": "azure-key",
        }
        assert (
            provider.request_url("https://project.services.ai.azure.com", "m")
            == "https://project.services.ai.azure.com/providers/cohere/v2/rerank"
        )
        payload = provider.build_payload(
            model="Cohere-rerank-v4.0-pro", query="q", documents=[("first", None)], top_n=1
        )
        assert payload == {
            "model": "Cohere-rerank-v4.0-pro",
            "query": "q",
            "documents": ["first"],
            "top_n": 1,
        }


class TestRunHttpRerankIntegration:
    async def test_voyage_end_to_end_url_headers_and_parse(self):
        client = _CaptureClient(
            {
                "data": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        )

        result = await _run_http_rerank(
            "query",
            [{"content": "first", "image_data": "RAW_IMAGE"}, {"content": "second"}],
            top_k=1,
            provider=VoyageRerankProvider(),
            model="rerank-2.5",
            base_url=None,
            api_key="voyage-key",
            modality="text",
            score_threshold=0.5,
            client=cast(Any, client),
            budget_factory=_budget_factory(),
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
