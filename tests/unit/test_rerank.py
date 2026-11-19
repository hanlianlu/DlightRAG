# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared rerank fallback and multimodal provider strategies."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager, nullcontext
from functools import partial
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import httpx2
import openai
import pytest

import dlightrag.engine.rag.retrieval.rerank as rerank_module
from dlightrag.engine.ai.media import ImagePayloadBudget
from dlightrag.engine.ai.providers.rerank_base import resolve_rerank_input_modality
from dlightrag.engine.ai.providers.rerank_providers import (
    AliyunRerankProvider,
    AzureCohereRerankProvider,
    CohereRerankProvider,
    HttpRerankProvider,
    JinaRerankProvider,
    VoyageRerankProvider,
    _azure_cohere_rerank_url,
)
from dlightrag.engine.ai.rerank import RerankModel
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings, RerankSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.rag.retrieval.rerank import (
    ListwiseScoreValidationError,
    RerankBatchError,
    _build_scored_chunks,
    _chat_llm_rerank,
    _parse_listwise_scores,
    _run_http_rerank,
    build_rerank_func,
)
from dlightrag.engine.rag.retrieval.rerank_breaker import RerankBreaker, RerankCircuitOpen
from dlightrag.engine.rag.retrieval.rerank_fallback import rerank_with_fallback
from dlightrag.engine.rag.workspace.settings import RERANK_DEADLINE_SECONDS, RetrievalSettings


def _chunks() -> list[dict[str, Any]]:
    return [
        {"chunk_id": "a", "content": "first"},
        {"chunk_id": "b", "content": "second"},
        {"chunk_id": "c", "content": "third"},
    ]


async def test_ai_rerank_model_owns_http_provider_execution(monkeypatch) -> None:
    from dlightrag.engine.ai import rerank

    provider = MagicMock()
    provider.requires_api_key = True
    provider.requires_base_url = False
    provider.default_model = "rerank-default"
    provider.request_url.return_value = "https://rerank.example/v1/rerank"
    provider.request_headers.return_value = {"Authorization": "Bearer key"}
    provider.build_payload.return_value = {"query": "query", "documents": ["text"]}
    provider.parse_results.return_value = [{"index": 0, "relevance_score": 0.8}]
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"results": []}
    client = AsyncMock()
    client.post.return_value = response
    monkeypatch.setattr(rerank, "RERANK_PROVIDERS", {"voyage_reranker": provider})
    monkeypatch.setattr("dlightrag.engine.ai.rerank.httpx.AsyncClient", lambda **_kwargs: client)

    model = rerank.create_rerank_model(
        RerankSettings(
            strategy="voyage_reranker",
            model="rerank-3-lite",
            api_key="key",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NOOP_TELEMETRY,
    )
    scores = await model.score("query", [("text", None)], top_n=1)
    await model.aclose()

    assert scores == [{"index": 0, "relevance_score": 0.8}]
    client.post.assert_awaited_once()
    client.aclose.assert_awaited_once()


async def test_rerank_requests_share_scheduler_limit() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0
    provider = MagicMock()
    provider.default_model = "rerank-default"
    provider.build_payload.return_value = {"query": "query", "documents": ["text"]}
    provider.request_url.return_value = "https://rerank.example/v1/rerank"
    provider.request_headers.return_value = {}
    provider.parse_results.return_value = [{"index": 0, "relevance_score": 0.8}]
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"results": []}
    model = RerankModel(
        RerankSettings(strategy="voyage_reranker", model="rerank-3-lite", api_key="key"),
        provider,
        scheduler=scheduler,
        telemetry=NOOP_TELEMETRY,
    )

    async def post(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        return response

    model._client.post = post  # pyright: ignore[reportPrivateUsage]
    first = asyncio.create_task(model.score("first", [("text", None)], top_n=1))
    await first_started.wait()
    second = asyncio.create_task(model.score("second", [("text", None)], top_n=1))
    await asyncio.sleep(0)
    assert calls == 1

    release_first.set()
    assert await asyncio.gather(first, second) == [
        [{"index": 0, "relevance_score": 0.8}],
        [{"index": 0, "relevance_score": 0.8}],
    ]
    assert calls == 2


async def test_rerank_error_states_what_failed() -> None:
    class Observation:
        updates: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    class Telemetry:
        observation = Observation()

        def trace(self, **_kwargs: Any):
            return nullcontext()

        @asynccontextmanager
        async def observe(self, name: str, **_kwargs: Any):
            del name
            yield self.observation

    model = RerankModel(
        RerankSettings(
            strategy="voyage_reranker",
            model="rerank-3-lite",
            api_key="key",
        ),
        VoyageRerankProvider(),
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=Telemetry(),
    )
    model._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=RuntimeError("upstream echoed secret rerank query")
    )

    try:
        with pytest.raises(RuntimeError, match="secret rerank query"):
            await model.score("secret", [("document", None)], top_n=1)
    finally:
        await model.aclose()

    assert Telemetry.observation.updates == [
        {
            "level": "ERROR",
            "status_message": "upstream echoed secret rerank query",
            "error_type": "RuntimeError",
        }
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
    assert outcome.failed_batch is None


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
    assert outcome.failed_batch is None


async def test_rerank_with_fallback_returns_rrf_top_k_and_logs_failure_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("provider unavailable")

    async def _fail(**_kwargs: Any) -> Any:
        raise failure

    with caplog.at_level(logging.WARNING, logger="dlightrag.engine.rag.retrieval.rerank_fallback"):
        outcome = await rerank_with_fallback(
            query="query",
            chunks=_chunks(),
            top_k=2,
            rerank_func=_fail,
        )

    assert [chunk["chunk_id"] for chunk in outcome.chunks] == ["a", "b"]
    assert outcome.reranked is False
    assert outcome.error_type == "RuntimeError"
    assert outcome.failed_batch is None
    records = [
        record
        for record in caplog.records
        if record.name == "dlightrag.engine.rag.retrieval.rerank_fallback"
    ]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is failure


async def test_malformed_middle_chat_batch_falls_back_atomically() -> None:
    call_count = 0
    chunks = _chunks()
    original = [dict(chunk) for chunk in chunks]

    async def scoring(messages, **_kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        count = sum("candidate" in item for item in _rerank_user_payloads(messages))
        return "not-json" if current == 1 else json.dumps([0.9] * count)

    async def rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        return await _chat_llm_rerank(
            query,
            chunks,
            top_k,
            scoring_func=scoring,
            batch_size=1,
            max_concurrency=1,
        )

    outcome = await rerank_with_fallback(
        query="query",
        chunks=chunks,
        top_k=2,
        rerank_func=rerank,
    )

    assert [chunk["chunk_id"] for chunk in outcome.chunks] == ["a", "b"]
    assert outcome.reranked is False
    assert outcome.error_type == "ListwiseScoreValidationError"
    assert outcome.failed_batch == 2
    assert chunks == original


async def test_generic_exception_metadata_cannot_leak_into_fallback_trace() -> None:
    class HostileError(RuntimeError):
        error_type = "secret query text"
        batch_ordinal = 99

    async def fail(**_kwargs: Any) -> Any:
        raise HostileError("provider failed")

    outcome = await rerank_with_fallback(
        query="query",
        chunks=_chunks(),
        top_k=2,
        rerank_func=fail,
    )

    assert outcome.error_type == "HostileError"
    assert outcome.failed_batch is None


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
    def test_exact_json_array(self):
        assert _parse_listwise_scores("[0.9, 0.5, 0.3]", 3) == [0.9, 0.5, 0.3]

    def test_exact_all_zero_array_is_valid(self):
        assert _parse_listwise_scores("[0, 0]", 2) == [0.0, 0.0]

    @pytest.mark.parametrize(
        "text",
        (
            "Scores: 0.95 for first, 0.40 for second",
            "no numbers here",
            "[]",
            "[0.9]",
            "[0.9, 0.8, 0.7]",
            "[1.5, 0.5]",
            "[-0.3, 0.5]",
            "[true, 0.5]",
            "[NaN, 0.5]",
            '[{"index":0,"relevance_score":0.9},{"index":0,"relevance_score":0.8}]',
            '[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.8}]',
            '{"scores":[0.9,0.8]}',
        ),
    )
    def test_malformed_or_non_exact_response_fails_batch(self, text):
        with pytest.raises(ListwiseScoreValidationError):
            _parse_listwise_scores(text, 2)

    def test_non_string_response_fails_batch(self):
        with pytest.raises(ListwiseScoreValidationError):
            _parse_listwise_scores(cast(Any, [0.9, 0.8]), 2)


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

        monkeypatch.setattr(rerank_module, func_name, fake_rerank)
        return captured

    @staticmethod
    def _stub_http_model(monkeypatch) -> AsyncMock:
        model = AsyncMock()
        model.aclose = AsyncMock()
        monkeypatch.setattr(rerank_module, "create_rerank_model", MagicMock(return_value=model))
        return model

    async def test_chat_llm_default_threshold_keeps_all_scored_candidates(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_chat_llm_rerank")
        scoring_model = AsyncMock()
        scoring_model.aclose = AsyncMock()
        monkeypatch.setattr(
            rerank_module,
            "CompletionModel",
            MagicMock(return_value=scoring_model),
        )

        fn = build_rerank_func(
            RerankSettings(strategy="chat_llm_reranker"),
            scheduler=ModelScheduler(max_concurrency=1),
            scoring_settings=ModelSettings(provider="openai", model="scoring-model"),
            telemetry=NOOP_TELEMETRY,
            breaker=RerankBreaker(),
        )
        assert fn is not None
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] is None

    async def test_provider_default_threshold_keeps_all_scored_candidates(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_run_http_rerank")
        self._stub_http_model(monkeypatch)

        fn = build_rerank_func(
            RerankSettings(strategy="voyage_reranker", api_key="voyage-key"),
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=NOOP_TELEMETRY,
            breaker=RerankBreaker(),
        )
        assert fn is not None
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] is None

    async def test_explicit_provider_threshold_is_preserved(self, monkeypatch):
        captured = self._capture_threshold(monkeypatch, "_run_http_rerank")
        self._stub_http_model(monkeypatch)

        fn = build_rerank_func(
            RerankSettings(
                strategy="voyage_reranker",
                api_key="voyage-key",
                score_threshold=0.42,
            ),
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=NOOP_TELEMETRY,
            breaker=RerankBreaker(),
        )
        assert fn is not None
        await fn("query", [{"content": "chunk"}], 1)

        assert captured["score_threshold"] == 0.42

    def test_provider_requires_api_key(self):
        with pytest.raises(ValueError, match="requires api_key"):
            build_rerank_func(
                RerankSettings(strategy="voyage_reranker"),
                scheduler=ModelScheduler(max_concurrency=1),
                telemetry=NOOP_TELEMETRY,
                breaker=RerankBreaker(),
            )

    def test_provider_requires_base_url(self):
        with pytest.raises(ValueError, match="requires base_url"):
            build_rerank_func(
                RerankSettings(strategy="aliyun_reranker", api_key="k"),
                scheduler=ModelScheduler(max_concurrency=1),
                telemetry=NOOP_TELEMETRY,
                breaker=RerankBreaker(),
            )

    def test_multimodal_on_text_only_provider_raises(self):
        with pytest.raises(ValueError, match="text-only"):
            build_rerank_func(
                RerankSettings(
                    strategy="voyage_reranker",
                    api_key="k",
                    input_modality="multimodal",
                ),
                scheduler=ModelScheduler(max_concurrency=1),
                telemetry=NOOP_TELEMETRY,
                breaker=RerankBreaker(),
            )

    async def test_http_rerank_keeps_bounded_orchestration_observation(
        self,
        monkeypatch,
    ) -> None:
        calls: list[dict[str, Any]] = []

        class Telemetry:
            def trace(self, **_kwargs: Any):
                return nullcontext()

            @asynccontextmanager
            async def observe(self, name: str, **kwargs: Any):
                calls.append({"name": name, **kwargs})
                observation = MagicMock()
                yield observation

        model = AsyncMock()
        model.score.return_value = [{"index": 0, "relevance_score": 0.8}]
        model.aclose = AsyncMock()
        monkeypatch.setattr(
            rerank_module,
            "create_rerank_model",
            MagicMock(return_value=model),
        )
        fn = build_rerank_func(
            RerankSettings(strategy="voyage_reranker", api_key="key"),
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=Telemetry(),
            breaker=RerankBreaker(),
        )
        assert fn is not None

        await fn("q" * 1200, [{"content": "candidate"}], 1)

        assert calls[0]["name"] == "rerank-passages"
        assert calls[0]["metadata"] == {"chunk_count": 1, "top_k": 1}
        traced_query = calls[0]["input"]["query"]
        assert len(traced_query) < 1100
        assert "truncated 200 chars" in traced_query

    async def test_a_down_reranker_is_skipped_by_every_workspace_without_reaching_it(
        self,
        monkeypatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        models = [AsyncMock(), AsyncMock()]
        for model in models:
            model.score.side_effect = httpx.ConnectError("connection refused")
        monkeypatch.setattr(rerank_module, "create_rerank_model", MagicMock(side_effect=models))
        settings = RerankSettings(strategy="voyage_reranker", api_key="key")
        # Two workspaces rerank through one endpoint; the composition root gives both one breaker.
        workspaces = [
            build_rerank_func(
                settings,
                scheduler=ModelScheduler(max_concurrency=1),
                telemetry=NOOP_TELEMETRY,
                breaker=breaker,
            )
            for _ in models
        ]
        assert workspaces[0] is not None and workspaces[1] is not None

        async def rerank(workspace: int):
            return await rerank_with_fallback(
                query="query",
                chunks=_chunks(),
                top_k=2,
                rerank_func=workspaces[workspace],
            )

        with caplog.at_level(logging.INFO, logger="dlightrag.engine.rag.retrieval"):
            for _ in range(3):
                failed = await rerank(0)
                assert failed.error_type == "ConnectError"
            logged = len(caplog.records)

            skipped = await rerank(1)

            assert skipped.reranked is False
            assert skipped.error_type == "RerankCircuitOpen"
            assert [chunk["chunk_id"] for chunk in skipped.chunks] == ["a", "b"]
            assert models[0].score.await_count == 3
            assert models[1].score.await_count == 0
            assert len(caplog.records) == logged  # a skipped query logs nothing
            assert sum(1 for record in caplog.records if record.exc_info) == 3
            assert [
                record.getMessage() for record in caplog.records if "opened" in record.getMessage()
            ] == ["Rerank circuit opened after 3 consecutive failures; skipping rerank for 30s"]

            # Once the wait is over, a query with nothing to rerank is not the trial. The next
            # one with candidates is, and the reranker is still down, so the circuit reopens.
            now = 30.0
            assert await workspaces[1]("query", [], 2) == []
            assert models[1].score.await_count == 0
            assert (await rerank(1)).error_type == "ConnectError"
            assert models[1].score.await_count == 1
            assert (await rerank(1)).error_type == "RerankCircuitOpen"

            # The reranker is back; the trial after the doubled wait resumes reranking.
            now = 90.0
            models[1].score.side_effect = None
            models[1].score.return_value = [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
            ]
            resumed = await rerank(1)

        assert resumed.reranked is True
        assert [chunk["chunk_id"] for chunk in resumed.chunks] == ["b", "a"]
        assert models[1].score.await_count == 2

    @pytest.mark.parametrize("strategy", ["voyage_reranker", "chat_llm_reranker"])
    async def test_a_hanging_reranker_is_cut_at_the_deadline_and_the_cut_counts_as_a_failure(
        self, monkeypatch, strategy: str
    ) -> None:
        started = asyncio.Event()

        async def hang(*_args: Any, **_kwargs: Any) -> None:
            started.set()
            await asyncio.Event().wait()

        endpoint = AsyncMock(side_effect=hang)
        scoring_settings = None
        if strategy == "voyage_reranker":
            model = AsyncMock()
            model.score = endpoint
            monkeypatch.setattr(rerank_module, "create_rerank_model", MagicMock(return_value=model))
            settings = RerankSettings(strategy="voyage_reranker", api_key="key")
        else:
            endpoint.aclose = AsyncMock()
            monkeypatch.setattr(rerank_module, "CompletionModel", MagicMock(return_value=endpoint))
            settings = RerankSettings(strategy="chat_llm_reranker")
            scoring_settings = ModelSettings(provider="openai", model="scoring-model")
        fn = build_rerank_func(
            settings,
            scheduler=ModelScheduler(max_concurrency=1),
            scoring_settings=scoring_settings,
            telemetry=NOOP_TELEMETRY,
            breaker=RerankBreaker(),
        )

        async def rerank():
            return await rerank_with_fallback(
                query="query", chunks=_chunks(), top_k=2, rerank_func=fn
            )

        async with asyncio.timeout(20):  # a guard: with no deadline these calls would hang
            # A cancel from outside is not the deadline: it propagates, and counts as nothing.
            outside = asyncio.create_task(rerank())
            await started.wait()
            outside.cancel()
            with pytest.raises(asyncio.CancelledError):
                await outside

            monkeypatch.setattr(rerank_module, "RERANK_DEADLINE_SECONDS", 0.05)
            for _ in range(3):
                cut = await rerank()
                assert cut.reranked is False
                assert cut.error_type == "TimeoutError"
                assert [chunk["chunk_id"] for chunk in cut.chunks] == ["a", "b"]
            reached = endpoint.await_count
            skipped = await rerank()

        assert skipped.error_type == "RerankCircuitOpen"  # the three cuts opened the breaker
        assert endpoint.await_count == reached

    def test_the_rerank_deadline_is_inside_the_default_retrieval_deadline(self) -> None:
        # Otherwise the run cancels a hanging call first, and the breaker sees no failure.
        assert RERANK_DEADLINE_SECONDS < RetrievalSettings().timeout


class _Service:
    """A reranker endpoint that counts the calls that reach it."""

    def __init__(self) -> None:
        self.calls = 0
        self.error: BaseException | None = None

    async def __call__(self) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "scored"


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://reranker.example/rerank")
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=httpx.Response(status, request=request)
    )


def _sdk_error(status: int) -> openai.APIStatusError:
    """What the chat reranker's scoring model raises when its endpoint answers ``status``."""
    request = httpx2.Request("POST", "https://scorer.example/v1/chat/completions")
    return openai.APIStatusError(
        f"HTTP {status}", response=httpx2.Response(status, request=request), body=None
    )


def _failed_batch(cause: Exception) -> RerankBatchError:
    error = RerankBatchError(batch_ordinal=1, batch_start=0, error_type=type(cause).__name__)
    error.__cause__ = cause
    return error


async def _fail_until_open(breaker: RerankBreaker, service: _Service) -> None:
    # Not a RuntimeError, so a refused call (a RuntimeError) can never pass for a failed one.
    service.error = ConnectionError("reranker down")
    for _ in range(3):
        with pytest.raises(ConnectionError):
            await breaker.call(service)


class TestRerankBreaker:
    async def test_third_consecutive_failure_opens_it_and_open_calls_never_reach_the_reranker(
        self,
    ) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        service = _Service()
        service.error = ConnectionError("reranker down")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(service)
        service.error = None
        assert await breaker.call(service) == "scored"  # an answer resets the count
        service.error = ConnectionError("reranker down")
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(service)
        assert service.calls == 5  # two failures, an answer, two failures: still closed

        with pytest.raises(ConnectionError):  # the third failure in a row
            await breaker.call(service)
        reached = service.calls
        now = 29.9
        for _ in range(3):
            with pytest.raises(RerankCircuitOpen):
                await breaker.call(service)
        assert service.calls == reached

    async def test_after_thirty_seconds_one_trial_goes_through_and_its_answer_closes_it(
        self,
    ) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        service = _Service()
        await _fail_until_open(breaker, service)
        service.error = None
        reached = service.calls
        release = asyncio.Event()

        async def trial_call() -> str:
            await release.wait()
            return "scored"

        now = 30.0
        trial = asyncio.create_task(breaker.call(trial_call))
        await asyncio.sleep(0)
        for _ in range(3):  # while the trial is in flight, everyone else still fails at once
            with pytest.raises(RerankCircuitOpen):
                await breaker.call(service)
        assert service.calls == reached

        release.set()
        assert await trial == "scored"
        assert await breaker.call(service) == "scored"
        assert service.calls == reached + 1

    async def test_failed_trial_reopens_for_twice_as_long_up_to_five_minutes_until_an_answer(
        self,
    ) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        service = _Service()
        await _fail_until_open(breaker, service)

        for wait in (30.0, 60.0, 120.0, 240.0, 300.0, 300.0):
            now += wait - 1
            with pytest.raises(RerankCircuitOpen):
                await breaker.call(service)
            now += 1
            with pytest.raises(ConnectionError):  # the trial reaches the reranker and fails
                await breaker.call(service)

        service.error = None
        now += 300
        assert await breaker.call(service) == "scored"  # an answered trial closes it
        await _fail_until_open(breaker, service)
        now += 29
        with pytest.raises(RerankCircuitOpen):
            await breaker.call(service)
        now += 1
        with pytest.raises(ConnectionError):  # the wait is 30 s again, not 300 s
            await breaker.call(service)

    async def test_cancellation_is_not_a_failure_and_a_cancelled_trial_frees_its_place(
        self,
    ) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        service = _Service()

        async def hang() -> str:
            await asyncio.Event().wait()
            return "scored"

        for _ in range(3):
            cancelled = asyncio.create_task(breaker.call(hang))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
        assert await breaker.call(service) == "scored"  # three cancellations did not open it

        await _fail_until_open(breaker, service)
        service.error = None
        now = 30.0
        trial = asyncio.create_task(breaker.call(hang))
        await asyncio.sleep(0)
        with pytest.raises(RerankCircuitOpen):  # the trial holds the only place
            await breaker.call(service)
        trial.cancel()
        with pytest.raises(asyncio.CancelledError):
            await trial

        # The cancelled trial gave its place back without closing the breaker: the next call
        # is the trial, at once, and when the reranker is still down it opens again.
        service.error = ConnectionError("reranker down")
        with pytest.raises(ConnectionError):
            await breaker.call(service)
        with pytest.raises(RerankCircuitOpen):
            await breaker.call(service)

    @pytest.mark.parametrize(
        ("error", "opens"),
        [
            pytest.param(_http_error(400), False, id="400"),
            pytest.param(_http_error(401), False, id="401"),
            pytest.param(_failed_batch(_sdk_error(400)), False, id="listwise-batch-400"),
            pytest.param(_http_error(408), True, id="408"),
            pytest.param(_http_error(429), True, id="429"),
            pytest.param(_http_error(500), True, id="500"),
            pytest.param(httpx.ReadTimeout("timed out"), True, id="read-timeout"),
            pytest.param(ValueError("not json"), True, id="unreadable-answer"),
        ],
    )
    async def test_only_a_4xx_other_than_408_and_429_is_an_answer(
        self, error: Exception, opens: bool
    ) -> None:
        # One federated query reranks once per workspace through one breaker, so three
        # refusals of the same query must not take the reranker out for everyone.
        breaker = RerankBreaker(clock=lambda: 0.0)
        service = _Service()
        service.error = error
        for _ in range(3):
            with pytest.raises(type(error)):
                await breaker.call(service)

        with pytest.raises(RerankCircuitOpen if opens else type(error)):
            await breaker.call(service)
        assert service.calls == (3 if opens else 4)

    async def test_a_refusal_resets_the_count_and_closes_the_trial_it_answers(self) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        service = _Service()
        down = ConnectionError("reranker down")

        # Failure, failure, refusal, failure, failure: the refusal broke the run of failures.
        for error in (down, down, _http_error(400), down, down):
            service.error = error
            with pytest.raises(type(error)):
                await breaker.call(service)

        # A third failure opens it. The first trial fails, so the wait doubles to 60 s; the
        # second is answered with a refusal, which closes the breaker and resets the wait.
        service.error = down
        with pytest.raises(ConnectionError):
            await breaker.call(service)
        now += 30
        with pytest.raises(ConnectionError):
            await breaker.call(service)
        now += 60
        service.error = _http_error(413)
        with pytest.raises(httpx.HTTPStatusError):
            await breaker.call(service)

        await _fail_until_open(breaker, service)
        now += 29
        with pytest.raises(RerankCircuitOpen):
            await breaker.call(service)
        now += 1
        with pytest.raises(ConnectionError):  # the wait is 30 s again, not 60 s
            await breaker.call(service)

    async def test_calls_that_began_before_it_opened_change_nothing_when_they_finish(
        self,
    ) -> None:
        now = 0.0
        breaker = RerankBreaker(clock=lambda: now)
        service = _Service()
        finish = {name: asyncio.Event() for name in ("answer", "failure", "late failure")}

        async def slow(name: str, *, fails: bool) -> str:
            await finish[name].wait()
            if fails:
                raise TimeoutError("timed out")
            return "scored"

        answer = asyncio.create_task(breaker.call(lambda: slow("answer", fails=False)))
        failure = asyncio.create_task(breaker.call(lambda: slow("failure", fails=True)))
        late_failure = asyncio.create_task(breaker.call(lambda: slow("late failure", fails=True)))
        await asyncio.sleep(0)
        await _fail_until_open(breaker, service)  # opens at 0 s, for 30 s
        service.error = None

        now = 20.0
        finish["failure"].set()  # an old failure that lands mid-wait does not lengthen it
        with pytest.raises(TimeoutError):
            await failure
        finish["answer"].set()  # and an old answer does not close the open breaker
        assert await answer == "scored"
        with pytest.raises(RerankCircuitOpen):
            await breaker.call(service)

        now = 30.0  # so the wait ends on time, and the trial begins and holds the place
        release = asyncio.Event()

        async def trial_call() -> str:
            await release.wait()
            return "scored"

        trial = asyncio.create_task(breaker.call(trial_call))
        await asyncio.sleep(0)
        with pytest.raises(RerankCircuitOpen):  # the trial began on time and holds the place
            await breaker.call(service)
        release.set()
        assert await trial == "scored"

        finish["late failure"].set()  # nor does an old failure count against the closed breaker
        with pytest.raises(TimeoutError):
            await late_failure
        service.error = ConnectionError("reranker down")
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(service)
        service.error = None
        assert await breaker.call(service) == "scored"


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
        import dlightrag.engine.rag.retrieval.rerank as rerank_module

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
        import dlightrag.engine.ai.media as image_budget_module

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
        import dlightrag.engine.ai.media as image_budget_module

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

    async def test_batch_error_identifies_the_failed_batch(self):
        mock_scoring = AsyncMock(side_effect=RuntimeError("API down"))
        chunks = [{"content": "doc0"}, {"content": "doc1"}]
        with pytest.raises(RerankBatchError) as caught:
            await _chat_llm_rerank(
                "query", chunks, top_k=10, scoring_func=mock_scoring, score_threshold=0.3
            )

        assert caught.value.batch_ordinal == 1
        assert caught.value.batch_start == 0
        assert caught.value.error_type == "RuntimeError"

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

    @pytest.mark.parametrize("failed_call", (0, 1, 2))
    async def test_any_failed_batch_discards_the_atomic_pass(self, failed_call: int):
        call_count = 0
        chunks = [{"chunk_id": str(index), "content": str(index)} for index in range(5)]
        original = [dict(chunk) for chunk in chunks]

        async def mock_scoring(messages, **_kwargs):
            nonlocal call_count
            current = call_count
            call_count += 1
            if current == failed_call:
                raise RuntimeError(f"batch {current + 1} failed")
            count = sum("candidate" in item for item in _rerank_user_payloads(messages))
            return json.dumps([0.5] * count)

        with pytest.raises(RerankBatchError) as caught:
            await _chat_llm_rerank(
                "query",
                chunks,
                top_k=5,
                scoring_func=mock_scoring,
                batch_size=2,
                max_concurrency=1,
            )

        assert caught.value.batch_ordinal == failed_call + 1
        assert caught.value.batch_start == failed_call * 2
        assert caught.value.error_type == "RuntimeError"
        assert chunks == original

    async def test_valid_zero_scores_preserve_stable_rrf_order(self):
        async def zero_scores(messages, **_kwargs):
            count = sum("candidate" in item for item in _rerank_user_payloads(messages))
            return json.dumps([0.0] * count)

        chunks = [{"chunk_id": str(index), "content": str(index)} for index in range(5)]
        result = await _chat_llm_rerank(
            "query",
            chunks,
            top_k=5,
            scoring_func=zero_scores,
            batch_size=2,
        )

        assert [chunk["chunk_id"] for chunk in result] == ["0", "1", "2", "3", "4"]
        assert all(chunk["rerank_score"] == 0.0 for chunk in result)

    async def test_successful_batches_sort_globally_and_stably(self):
        responses = iter(("[0.1, 0.8]", "[0.7, 0.9]"))

        async def scores(**_kwargs):
            return next(responses)

        chunks = [{"chunk_id": value, "content": value} for value in ("a", "b", "c", "d")]
        result = await _chat_llm_rerank(
            "query",
            chunks,
            top_k=4,
            scoring_func=scores,
            batch_size=2,
            max_concurrency=1,
        )

        assert [chunk["chunk_id"] for chunk in result] == ["d", "b", "c", "a"]

    async def test_scoring_cancellation_propagates(self):
        async def cancel(**_kwargs):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _chat_llm_rerank(
                "query",
                [{"content": "a"}, {"content": "b"}],
                top_k=2,
                scoring_func=cancel,
                batch_size=1,
            )

    async def test_cancellation_wins_over_an_earlier_batch_failure(self):
        async def mixed_failure(messages, **_kwargs):
            candidate = next(
                payload for payload in _rerank_user_payloads(messages) if "candidate" in payload
            )
            if candidate["text"] == "ordinary failure":
                raise RuntimeError("provider failed")
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _chat_llm_rerank(
                "query",
                [{"content": "ordinary failure"}, {"content": "cancelled"}],
                top_k=2,
                scoring_func=mixed_failure,
                batch_size=1,
                max_concurrency=2,
            )

    async def test_logs_listwise_schedule_summary(self, caplog):
        caplog.set_level(logging.INFO, logger="dlightrag.engine.rag.retrieval.rerank")

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


class _ScoreModel:
    def __init__(self, scores: list[dict[str, Any]]) -> None:
        self.scores = scores
        self.query: str | None = None
        self.documents: list[tuple[str, str | None]] | None = None
        self.top_n: int | None = None

    async def score(
        self,
        query: str,
        documents: list[tuple[str, str | None]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        self.query = query
        self.documents = documents
        self.top_n = top_n
        return self.scores


class TestRunHttpRerank:
    async def test_bounds_image_payloads(self, monkeypatch):
        import dlightrag.engine.ai.media as image_budget_module

        raw_image = "RAW_ORIGINAL_IMAGE_PAYLOAD"
        bounded_uri = "data:image/jpeg;base64,BOUNDED"
        model = _ScoreModel([{"index": 0, "relevance_score": 0.9}])

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
            model=cast(Any, model),
            modality="multimodal",
            score_threshold=0.3,
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
        assert model.documents == [("with image", bounded_uri)]
        assert raw_image not in str(model.documents)

    async def test_text_modality_skips_images_sends_text_only(self):
        raw_image = "RAW_ORIGINAL_IMAGE_PAYLOAD"
        model = _ScoreModel([{"index": 0, "relevance_score": 0.85}])

        result = await _run_http_rerank(
            "query",
            [{"content": "VLM text description", "image_data": raw_image}],
            top_k=1,
            model=cast(Any, model),
            modality="text",
            score_threshold=0.3,
            budget_factory=_budget_factory(),
        )

        assert result[0]["rerank_score"] == pytest.approx(0.85)
        assert model.documents == [("VLM text description", None)]
        assert raw_image not in str(model.documents)

    async def test_provider_top_n_subset_remains_valid(self):
        model = _ScoreModel([{"index": 2, "relevance_score": 0.95}])

        result = await _run_http_rerank(
            "query",
            [{"content": "a"}, {"content": "b"}, {"content": "c"}],
            top_k=1,
            model=cast(Any, model),
            modality="text",
            score_threshold=None,
            budget_factory=_budget_factory(),
        )

        assert [chunk["content"] for chunk in result] == ["c"]
        assert model.top_n == 1

    async def test_score_threshold_drops_all_results_below_threshold(self):
        model = _ScoreModel(
            [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.2},
            ]
        )

        result = await _run_http_rerank(
            "query",
            [{"content": "low"}, {"content": "also low"}],
            top_k=10,
            model=cast(Any, model),
            modality="text",
            score_threshold=0.5,
            budget_factory=_budget_factory(),
        )

        assert result == []


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
            model="rerank-3-lite",
            query="q",
            documents=[("first", "data:image/png;base64,ABC"), ("second", None)],
            top_n=1,
        )
        assert payload == {
            "model": "rerank-3-lite",
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

        model = RerankModel(
            RerankSettings(
                strategy="voyage_reranker",
                model="rerank-3-lite",
                api_key="voyage-key",
            ),
            VoyageRerankProvider(),
            scheduler=ModelScheduler(max_concurrency=1),
            telemetry=NOOP_TELEMETRY,
        )
        model._client = cast(Any, client)

        result = await _run_http_rerank(
            "query",
            [{"content": "first", "image_data": "RAW_IMAGE"}, {"content": "second"}],
            top_k=1,
            model=model,
            modality="text",
            score_threshold=0.5,
            budget_factory=_budget_factory(),
        )
        await model.aclose()

        assert result == [{"content": "second", "rerank_score": 0.91}]
        assert client.url == "https://api.voyageai.com/v1/rerank"
        assert client.headers is not None
        assert client.headers["Authorization"] == "Bearer voyage-key"
        assert client.payload == {
            "model": "rerank-3-lite",
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
        self.closed = False

    async def post(self, url, *, json, headers):
        self.url = url
        self.payload = json
        self.headers = headers
        return _CaptureResponse(self.data)

    async def aclose(self) -> None:
        self.closed = True


class _CaptureResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data
