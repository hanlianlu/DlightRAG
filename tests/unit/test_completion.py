# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AI-owned completion streaming behavior."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pytest
from dlightrag_ai.completion import CompletionModel
from dlightrag_ai.settings import ModelSettings


class RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingTelemetry:
    capture_sensitive_data = True

    def __init__(self) -> None:
        self.observation = RecordingObservation()
        self.calls: list[dict[str, Any]] = []

    @asynccontextmanager
    async def observe(self, name: str, **_kwargs: Any):
        self.calls.append({"name": name, **_kwargs})
        yield self.observation


async def test_provider_error_text_is_redacted_when_sensitive_capture_is_disabled(
    monkeypatch,
) -> None:
    class Provider:
        async def complete(self, **_kwargs: Any) -> str:
            raise RuntimeError("upstream echoed secret prompt fragment")

        async def aclose(self) -> None:
            return None

    telemetry = RecordingTelemetry()
    telemetry.capture_sensitive_data = False
    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="model"),
        telemetry=telemetry,
    )

    with pytest.raises(RuntimeError, match="secret prompt fragment"):
        await model(messages=[{"role": "user", "content": "secret"}])

    assert telemetry.observation.updates == [{"level": "ERROR", "status_message": "RuntimeError"}]


async def test_stream_records_ttft_usage_cost_and_sensitive_text(monkeypatch) -> None:
    class Provider:
        def stream(self, **kwargs: Any):
            holder = kwargs["usage_holder"]
            assert "usage_holder" not in kwargs["model_kwargs"]

            async def tokens():
                yield "hel"
                yield "lo"
                holder["usage_details"] = {"prompt_tokens": 5, "completion_tokens": 2}
                holder["cost_details"] = {"total": 0.001}

            return tokens()

        async def aclose(self) -> None:
            return None

    telemetry = RecordingTelemetry()
    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="stream-model"),
        telemetry=telemetry,
    )

    usage_holder: dict[str, Any] = {}
    stream = await model(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,SECRET"},
                    },
                ],
            }
        ],
        stream=True,
        usage_holder=usage_holder,
    )
    result = [token async for token in stream]

    assert result == ["hel", "lo"]
    assert "SECRET" not in str(telemetry.calls[0]["input"])
    assert "[image omitted]" in str(telemetry.calls[0]["input"])
    assert usage_holder == {
        "usage_details": {"prompt_tokens": 5, "completion_tokens": 2},
        "cost_details": {"total": 0.001},
    }
    assert isinstance(telemetry.observation.updates[0]["completion_start_time"], datetime)
    assert telemetry.observation.updates[1] == {
        "output": {"text_length": 5, "text": "hello"},
        "usage_details": {"prompt_tokens": 5, "completion_tokens": 2},
        "cost_details": {"total": 0.001},
    }


async def test_stream_propagates_cancellation_without_false_error(monkeypatch) -> None:
    class Provider:
        def stream(self, **_kwargs: Any):
            async def tokens():
                yield "first"
                raise asyncio.CancelledError

            return tokens()

        async def aclose(self) -> None:
            return None

    telemetry = RecordingTelemetry()
    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="stream-model"),
        telemetry=telemetry,
    )
    stream = await model(messages=[{"role": "user", "content": "hi"}], stream=True)

    with pytest.raises(asyncio.CancelledError):
        _ = [token async for token in stream]

    assert not any(update.get("level") == "ERROR" for update in telemetry.observation.updates)
    assert telemetry.observation.updates[1]["output"] == {
        "text_length": 5,
        "text": "first",
    }


async def test_stream_consumer_abandonment_closes_provider_iterator(monkeypatch) -> None:
    finalized = asyncio.Event()

    class Provider:
        def stream(self, **_kwargs: Any):
            async def tokens():
                try:
                    yield "first"
                    yield "second"
                finally:
                    finalized.set()

            return tokens()

        async def aclose(self) -> None:
            return None

    telemetry = RecordingTelemetry()
    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="stream-model"),
        telemetry=telemetry,
    )
    stream = await model(messages=[{"role": "user", "content": "hi"}], stream=True)

    assert await anext(stream) == "first"
    await stream.aclose()

    assert finalized.is_set()
    assert telemetry.observation.updates[-1]["output"] == {
        "text_length": 5,
        "text": "first",
    }
