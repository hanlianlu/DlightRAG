# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Langfuse observability wrappers."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from dlightrag_ai.telemetry import NoopTelemetry

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
    previous_sensitive = observability._trace_sensitive
    observability._client = None
    observability._trace_sensitive = True
    yield
    observability._client = previous
    observability._trace_sensitive = previous_sensitive


async def test_trace_observation_captures_input_when_enabled() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    observability._trace_sensitive = True

    async with observability.trace_observation(
        "answer_pipeline", as_type="chain", input={"query": "q"}
    ):
        pass

    assert client.observations[-1].kwargs.get("input") == {"query": "q"}


async def test_noop_telemetry_accepts_updates_without_product_dependencies() -> None:
    async with NoopTelemetry().observe("standalone", metadata={"source": "test"}) as observation:
        observation.update(output={"ok": True})


async def test_langfuse_telemetry_adapts_neutral_observation() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async with observability.LangfuseTelemetry().observe(
        "agent_tool",
        as_type="tool",
        metadata={"tool": "search"},
    ) as observation:
        observation.update(output={"outcome": "ok"})

    assert client.observations[0].kwargs == {
        "as_type": "tool",
        "name": "agent_tool",
        "metadata": {"tool": "search"},
    }
    assert client.observations[0].updates == [{"output": {"outcome": "ok"}}]


async def test_langfuse_telemetry_normalizes_provider_usage_and_cost() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    async with observability.LangfuseTelemetry().observe(
        "llm_model",
        as_type="generation",
    ) as observation:
        observation.update(
            usage_details={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            cost_details={"total": 0.001},
        )

    assert client.observations[0].updates == [
        {
            "usage_details": {"input": 3, "output": 2, "total": 5},
            "cost_details": {"total": 0.001},
        }
    ]


async def test_trace_observation_redacts_input_in_privacy_mode() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    observability._trace_sensitive = False

    async with observability.trace_observation(
        "answer_pipeline", as_type="chain", input={"query": "secret prompt"}
    ):
        pass

    assert "input" not in client.observations[-1].kwargs


async def test_answer_output_follows_the_same_privacy_switch_as_the_query() -> None:
    from dlightrag.core.servicemanager import answer_trace_output

    observability._trace_sensitive = True
    assert answer_trace_output("the answer", [], {})["answer"] == "the answer"

    observability._trace_sensitive = False
    assert "answer" not in answer_trace_output("the answer", [], {})


async def test_trace_observation_records_error_text_when_enabled() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    observability._trace_sensitive = True

    with pytest.raises(RuntimeError):
        async with observability.trace_observation("answer_pipeline", as_type="chain"):
            raise RuntimeError("secret provider detail")

    assert client.observations[-1].updates[-1]["status_message"] == "secret provider detail"


async def test_langfuse_telemetry_cancellation_is_not_an_error() -> None:
    client = _RecordingLangfuse()
    observability._client = client

    with pytest.raises(asyncio.CancelledError):
        async with observability.LangfuseTelemetry().observe(
            "llm_model",
            as_type="generation",
        ):
            raise asyncio.CancelledError

    assert client.observations[-1].updates == []


async def test_trace_observation_redacts_error_text_in_privacy_mode() -> None:
    client = _RecordingLangfuse()
    observability._client = client
    observability._trace_sensitive = False

    with pytest.raises(RuntimeError):
        async with observability.trace_observation("answer_pipeline", as_type="chain"):
            raise RuntimeError("secret provider detail")

    update = client.observations[-1].updates[-1]
    assert update["status_message"] == "error"
    assert "secret provider detail" not in str(update)


def _record_propagation(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    @contextmanager
    def fake_propagate(**kwargs: Any) -> Generator[None]:
        order.append(f"session={kwargs['session_id']}")
        yield

    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagate)


async def test_trace_observation_opens_the_session_before_the_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Langfuse only propagates to spans opened after the session context."""
    order: list[str] = []
    _record_propagation(monkeypatch, order)

    class _OrderedClient(_RecordingLangfuse):
        def start_as_current_observation(self, **kwargs: Any) -> _RecordingObservation:
            order.append("span")
            return super().start_as_current_observation(**kwargs)

    observability._client = _OrderedClient()

    async with observability.trace_observation(
        "answer_pipeline", as_type="chain", session_id="conv-1"
    ):
        pass

    assert order == ["session=conv-1", "span"]


async def test_trace_observation_without_a_session_claims_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    _record_propagation(monkeypatch, order)
    observability._client = _RecordingLangfuse()

    async with observability.trace_observation("ingest_pipeline", as_type="chain"):
        pass

    assert order == []


def test_init_tracing_reads_trace_sensitive_flag() -> None:
    observability.init_tracing(
        SimpleNamespace(
            langfuse_public_key=None,
            langfuse_secret_key=None,
            langfuse_trace_sensitive_data=False,
        )
    )
    assert observability._trace_sensitive is False


def test_trace_sensitive_enabled_reflects_flag() -> None:
    observability._trace_sensitive = False
    assert observability.trace_sensitive_enabled() is False
    observability._trace_sensitive = True
    assert observability.trace_sensitive_enabled() is True


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
