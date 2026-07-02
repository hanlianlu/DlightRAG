# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Langfuse observability wrappers."""

from __future__ import annotations

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
            "usage_details": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "cost_details": {"total": 0.001},
        }
    ]


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


async def test_trace_pipeline_nests_child_observations() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async with observability.trace_pipeline("answer_pipeline", query="q"):
        async with observability.trace_pipeline("retrieve", workspaces=["default"]):
            pass

    assert [obs.kwargs["name"] for obs in client.observations] == [
        "answer_pipeline",
        "retrieve",
    ]
    assert client.observations[1].parent is client.observations[0]


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
