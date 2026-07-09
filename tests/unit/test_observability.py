# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Langfuse observability wrappers."""

import uuid
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest

from dlightrag import observability


class _RecordingObservation:
    def __init__(self, client: _RecordingLangfuse, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.parent = client.active[-1] if client.active else None
        self.updates: list[dict[str, Any]] = []

    def __enter__(self) -> _RecordingObservation:
        self.client.active.append(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        popped = self.client.active.pop()
        assert popped is self

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _RecordingLangfuse:
    def __init__(self) -> None:
        self.observations: list[_RecordingObservation] = []
        self.active: list[_RecordingObservation] = []
        self.flushed = False
        self.shutdown_called = False

    def start_as_current_observation(self, **kwargs: Any) -> _RecordingObservation:
        obs = _RecordingObservation(self, kwargs)
        self.observations.append(obs)
        return obs

    def create_trace_id(self, *, seed: str | None = None) -> str:
        return uuid.uuid4().hex

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture(autouse=True)
def reset_langfuse_client() -> Generator[None]:
    previous = observability._client
    observability._client = None
    yield
    observability._client = previous


async def test_chat_wrapper_uses_generation_observation() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "answer"

    wrapped = observability.wrap_chat_func(complete, name="llm_gpt-5.4-mini", model="gpt-5.4-mini")

    result = await wrapped(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    assert result == "answer"
    assert len(client.observations) == 1
    obs = client.observations[0]
    assert obs.kwargs["as_type"] == "generation"
    assert obs.kwargs["name"] == "llm_gpt-5.4-mini"
    assert obs.kwargs["model"] == "gpt-5.4-mini"
    assert obs.kwargs["metadata"] == {"temperature": 0.2}
    assert obs.updates == [{"output": "answer"}]


async def test_chat_wrapper_updates_generation_usage_and_cost_details() -> None:
    from dlightrag.models.providers.base import CompletionOutput

    client = _RecordingLangfuse()
    observability._client = client

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> CompletionOutput:
        return CompletionOutput(
            "answer",
            usage_details={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            cost_details={"total": 0.001},
        )

    wrapped = observability.wrap_chat_func(complete, name="llm_gpt-5.4-mini", model="gpt-5.4-mini")

    result = await wrapped(messages=[{"role": "user", "content": "hi"}])

    assert result == "answer"
    assert client.observations[0].updates == [
        {
            "output": "answer",
            "usage_details": {"input": 3, "output": 2, "total": 5},
            "cost_details": {"total": 0.001},
        }
    ]


def test_langfuse_usage_details_normalizes_overlapping_provider_keys() -> None:
    # DeepSeek-style usage mixes components, an aggregate, and cache counters;
    # Langfuse sums every value into total, so forwarding raw triple-counts.
    raw = {
        "prompt_tokens": 3911,
        "completion_tokens": 254,
        "total_tokens": 4165,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 3911,
    }
    assert observability._langfuse_usage_details(raw) == {
        "input": 3911,
        "output": 254,
        "total": 4165,
    }


def test_langfuse_usage_details_derives_total_when_absent() -> None:
    assert observability._langfuse_usage_details({"input_tokens": 10, "output_tokens": 4}) == {
        "input": 10,
        "output": 4,
        "total": 14,
    }


async def test_chat_wrapper_does_not_retry_model_call_on_error() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    calls = 0

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider down")

    wrapped = observability.wrap_chat_func(complete, name="llm_test", model="test-model")

    with pytest.raises(RuntimeError, match="provider down"):
        await wrapped(messages=[{"role": "user", "content": "hi"}])

    assert calls == 1
    assert len(client.observations) == 1
    assert client.observations[0].updates[-1]["level"] == "ERROR"


async def test_chat_wrapper_traces_streaming_generation() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    calls = 0

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1

        async def stream() -> Any:
            yield "hel"
            yield "lo"

        return stream()

    wrapped = observability.wrap_chat_func(complete, name="llm_stream", model="stream-model")

    token_iterator = await wrapped(messages=[{"role": "user", "content": "hi"}], stream=True)
    result = [chunk async for chunk in token_iterator]

    assert result == ["hel", "lo"]
    assert calls == 1
    assert len(client.observations) == 1
    obs = client.observations[0]
    assert obs.kwargs["as_type"] == "generation"
    assert obs.kwargs["name"] == "llm_stream"
    assert obs.updates == [{"output": "hello"}]
    assert client.active == []


async def test_streaming_generation_attaches_usage_and_cost() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        holder = kwargs.get("usage_holder")

        async def stream() -> Any:
            yield "hel"
            yield "lo"
            if holder is not None:  # provider records usage at stream end
                holder["usage_details"] = {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                }
                holder["cost_details"] = {"total": 0.001}

        return stream()

    wrapped = observability.wrap_chat_func(complete, name="llm_stream", model="stream-model")

    token_iterator = await wrapped(messages=[{"role": "user", "content": "hi"}], stream=True)
    result = [chunk async for chunk in token_iterator]

    assert result == ["hel", "lo"]
    assert client.observations[0].updates == [
        {"output": "hello"},
        {"usage_details": {"input": 5, "output": 2, "total": 7}, "cost_details": {"total": 0.001}},
    ]


async def test_streaming_generation_no_usage_is_graceful() -> None:
    # A provider that records no streaming usage must not add a usage update.
    client = _RecordingLangfuse()
    observability._client = client

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        async def stream() -> Any:
            yield "x"

        return stream()

    wrapped = observability.wrap_chat_func(complete, name="llm_stream", model="stream-model")

    token_iterator = await wrapped(messages=[{"role": "user", "content": "hi"}], stream=True)
    assert [chunk async for chunk in token_iterator] == ["x"]
    assert client.observations[0].updates == [{"output": "x"}]


async def test_chat_func_root_starts_new_trace() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async def complete(messages: Any, **kwargs: Any) -> str:
        return "ok"

    wrapped = observability.wrap_chat_func(complete, name="llm_x", model="x", root=True)
    await wrapped([{"role": "user", "content": "q"}])

    tc = client.observations[0].kwargs.get("trace_context")
    assert tc is not None
    assert isinstance(tc["trace_id"], str) and len(tc["trace_id"]) == 32


async def test_chat_func_default_has_no_trace_context() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async def complete(messages: Any, **kwargs: Any) -> str:
        return "ok"

    wrapped = observability.wrap_chat_func(complete, name="llm_x", model="x")
    await wrapped([{"role": "user", "content": "q"}])

    assert "trace_context" not in client.observations[0].kwargs


async def test_embedding_func_root_starts_new_trace() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async def embed(inputs: list[str], **kwargs: Any) -> list[list[float]]:
        return [[0.1]]

    wrapped = observability.wrap_embedding_func(embed, name="embed_x", root=True)
    await wrapped(["hello"])

    tc = client.observations[0].kwargs.get("trace_context")
    assert tc is not None
    assert isinstance(tc["trace_id"], str) and len(tc["trace_id"]) == 32


async def test_embedding_wrapper_uses_embedding_observation() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async def embed(inputs: list[str], **kwargs: Any) -> list[list[float]]:
        return [[0.1, 0.2]]

    wrapped = observability.wrap_embedding_func(embed, name="embed_text-embedding-3-large")

    result = await wrapped(["hello"], context="document")

    assert result == [[0.1, 0.2]]
    assert len(client.observations) == 1
    obs = client.observations[0]
    assert obs.kwargs["as_type"] == "embedding"
    assert obs.kwargs["name"] == "embed_text-embedding-3-large"
    assert obs.kwargs["metadata"] == {"context": "document", "input_count": 1}
    assert obs.updates == [{"output": {"embedding_count": 1}}]


async def test_embedding_wrapper_keeps_carrier_out_of_provider_and_metadata() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    seen_kwargs: dict[str, Any] = {}

    async def embed(inputs: list[str], **kwargs: Any) -> list[list[float]]:
        seen_kwargs.update(kwargs)
        return [[0.1, 0.2]]

    traced = observability.wrap_embedding_func(embed, name="embed_test")
    bound = observability.bind_trace_context(traced)

    result = await bound(["hello"], context="document")

    assert result == [[0.1, 0.2]]
    # the OTEL context carrier is consumed by the embedding wrapper: it reaches
    # neither the provider call nor the observation metadata
    assert observability._CONTEXT_CARRIER_KEY not in seen_kwargs
    assert observability._CONTEXT_CARRIER_KEY not in client.observations[0].kwargs["metadata"]


async def test_trace_observation_nests_child_observations() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async with observability.trace_observation(
        "answer_pipeline",
        as_type="chain",
        input={"query": "q"},
        metadata={"workspaces": ["default"]},
    ) as trace:
        trace.update(output={"answer_len": 12})
        async with observability.trace_observation(
            "retrieve",
            as_type="retriever",
            input={"query": "q"},
            metadata={"workspaces": ["default"]},
        ):
            pass

    assert [obs.kwargs["name"] for obs in client.observations] == [
        "answer_pipeline",
        "retrieve",
    ]
    assert [obs.kwargs["as_type"] for obs in client.observations] == ["chain", "retriever"]
    assert client.observations[0].kwargs["input"] == {"query": "q"}
    assert client.observations[0].kwargs["metadata"] == {"workspaces": ["default"]}
    assert client.observations[0].updates == [{"output": {"answer_len": 12}}]
    assert client.observations[1].parent is client.observations[0]


async def test_trace_observation_update_is_noop_without_client() -> None:
    observability._client = None

    async with observability.trace_observation("disabled", as_type="chain") as trace:
        trace.update(output={"answer_len": 12})


def test_init_tracing_filters_external_spans_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)

    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
        langfuse_export_external_spans=False,
    )
    observability.init_tracing(config)

    should_export_span = captured["should_export_span"]

    assert captured["base_url"] == "https://cloud.langfuse.com"
    assert should_export_span(
        SimpleNamespace(instrumentation_scope=SimpleNamespace(name="langfuse-sdk"))
    )
    assert not should_export_span(
        SimpleNamespace(instrumentation_scope=SimpleNamespace(name="openai"))
    )


def test_init_tracing_does_not_call_blocking_auth_check(monkeypatch: pytest.MonkeyPatch) -> None:
    auth_called = False

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def auth_check(self) -> bool:
            nonlocal auth_called
            auth_called = True
            raise AssertionError("auth_check should not run during production startup")

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)

    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
        langfuse_export_external_spans=False,
    )
    observability.init_tracing(config)

    assert auth_called is False
    assert isinstance(observability._client, FakeLangfuse)


def test_init_tracing_forwards_v4_client_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)

    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
        langfuse_export_external_spans=False,
        langfuse_environment="production",
        langfuse_release="2026.06.06",
        langfuse_sample_rate=0.25,
        langfuse_timeout=7,
        langfuse_flush_at=16,
        langfuse_flush_interval=2.5,
    )
    observability.init_tracing(config)

    assert captured["environment"] == "production"
    assert captured["release"] == "2026.06.06"
    assert captured["sample_rate"] == 0.25
    assert captured["timeout"] == 7
    assert captured["flush_at"] == 16
    assert captured["flush_interval"] == 2.5
    assert callable(captured["mask"])


def test_langfuse_mask_redacts_secrets_and_omits_images(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)
    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
        langfuse_export_external_spans=False,
    )

    observability.init_tracing(config)
    mask = captured["mask"]
    masked = mask(
        {
            "api_key": "sk-secret",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    )

    assert masked == {
        "api_key": "[redacted]",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": "[image omitted]"},
        ],
    }


def test_init_tracing_clears_previous_client_when_keys_missing() -> None:
    observability._client = _RecordingLangfuse()

    config = SimpleNamespace(
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    observability.init_tracing(config)

    assert observability._client is None


def test_shutdown_tracing_uses_sdk_shutdown_and_clears_client() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    observability.shutdown_tracing()

    assert client.shutdown_called is True
    assert client.flushed is False
    assert observability._client is None


def test_bind_trace_context_noop_when_disabled() -> None:
    async def fn() -> None: ...

    # _client is None (autouse fixture) -> returned unchanged, zero overhead
    assert observability.bind_trace_context(fn) is fn


async def test_bind_trace_context_injects_context_carrier() -> None:
    observability._client = _RecordingLangfuse()
    seen_kwargs: dict[str, Any] = {}

    async def inner(**kwargs: Any) -> str:
        seen_kwargs.update(kwargs)
        return "ok"

    bound = observability.bind_trace_context(inner)
    assert await bound(messages=[]) == "ok"
    # the caller's OTEL context is captured and carried to the (detached) worker
    assert observability._CONTEXT_CARRIER_KEY in seen_kwargs


async def test_streaming_generation_keeps_carrier_out_of_provider_and_metadata() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    seen_kwargs: dict[str, Any] = {}

    async def complete(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        seen_kwargs.update(kwargs)

        async def stream() -> Any:
            yield "hi"

        return stream()

    traced = observability.wrap_chat_func(complete, name="llm_stream", model="stream-model")
    bound = observability.bind_trace_context(traced)

    token_iterator = await bound(messages=[{"role": "user", "content": "q"}], stream=True)
    assert [chunk async for chunk in token_iterator] == ["hi"]

    # the OTEL context carrier is consumed by the wrapper: it never reaches the
    # provider call nor the observation metadata
    assert observability._CONTEXT_CARRIER_KEY not in seen_kwargs
    assert observability._CONTEXT_CARRIER_KEY not in client.observations[0].kwargs["metadata"]
