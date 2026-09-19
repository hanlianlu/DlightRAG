# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AI-owned completion streaming behavior."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.completion import CompletionModel, structured_response_format
from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile
from dlightrag.engine.ai.response_policy import ResponseRequestError
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.structured import StructuredOutput


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

    def trace(self, **_kwargs: Any):
        return nullcontext()

    @asynccontextmanager
    async def observe(self, name: str, **_kwargs: Any):
        self.calls.append({"name": name, **_kwargs})
        yield self.observation


async def test_response_family_uses_response_reasoning_shape_in_completion_requests(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "answer"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="deepseek-reasoner",
            api_family="response",
            reasoning="high",
            model_kwargs={"top_p": 0.9},
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    profile = ModelProfile(
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        reasoning=ReasoningProfile(
            format="deepseek",
            levels=ReasoningLevels(
                off="disabled",
                minimal=None,
                low="low",
                medium=None,
                high="high",
                xhigh=None,
                max="max",
            ),
        ),
    )

    result = await model(
        messages=[{"role": "user", "content": "hi"}],
        model_profile=profile,
    )

    assert result == "answer"
    assert calls[0]["model_kwargs"] == {
        "top_p": 0.9,
        "reasoning": {"effort": "high"},
    }


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "input",
        "messages",
        "instructions",
        "tools",
        "tool_choice",
        "previous_response_id",
        "conversation",
        "store",
        "background",
        "truncation",
        "stream",
        "stream_options",
        "reasoning",
        "text",
        "response_format",
        "max_output_tokens",
        "max_tokens",
        "temperature",
    ],
)
async def test_response_family_rejects_product_owned_raw_model_kwargs(
    monkeypatch,
    field: str,
) -> None:
    class Provider:
        complete = AsyncMock(return_value="answer")

        async def aclose(self) -> None:
            return None

    provider = Provider()
    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="model",
            api_family="response",
            model_kwargs={field: "override"},
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )

    with pytest.raises(ResponseRequestError, match=field):
        await model(messages=[{"role": "user", "content": "hi"}])

    provider.complete.assert_not_awaited()


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
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="model"),
        scheduler=ModelScheduler(max_concurrency=1),
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
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="stream-model"),
        scheduler=ModelScheduler(max_concurrency=1),
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
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="stream-model"),
        scheduler=ModelScheduler(max_concurrency=1),
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
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="stream-model"),
        scheduler=ModelScheduler(max_concurrency=1),
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


class _ModeDecision(BaseModel):
    mode: str


_MODE_OUTPUT = StructuredOutput(name="answer_mode", schema=_ModeDecision)


def test_auto_openai_compat_prefers_json_schema() -> None:
    settings = ModelSettings(
        provider="openai",
        model="compat",
        base_url="https://openrouter.ai/api/v1",
        structured_output="auto",
    )
    fmt = structured_response_format(_MODE_OUTPUT, settings)
    assert fmt["type"] == "json_schema"


def test_explicit_json_object_stays_json_object() -> None:
    settings = ModelSettings(
        provider="openai",
        model="compat",
        structured_output="json_object",
    )
    assert structured_response_format(_MODE_OUTPUT, settings) == {"type": "json_object"}


async def test_json_object_folds_hint_into_system(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            seen.update(kwargs)
            return "{}"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="compat", structured_output="json_object"),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    await model(
        messages=[
            {"role": "system", "content": "Pick a mode."},
            {"role": "user", "content": "q"},
        ],
        structured_output=_MODE_OUTPUT,
    )
    sent = seen["messages"]
    assert sent[0]["role"] == "system"
    assert "json" in sent[0]["content"].casefold()
    assert sent[1]["content"] == "q"
    assert seen["response_format"] == {"type": "json_object"}


_DEEPSEEK_JSON_SCHEMA_TYPE_ERROR = (
    "Error code: 400 - {'error': {'message': 'This response_format type is unavailable now', "
    "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}"
)


async def test_json_schema_failure_retries_json_object_with_system_hint(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            if kwargs.get("response_format", {}).get("type") == "json_schema":
                raise RuntimeError(_DEEPSEEK_JSON_SCHEMA_TYPE_ERROR)
            return "{}"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="compat",
            base_url="https://openrouter.ai/api/v1",
            structured_output="auto",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    await model(
        messages=[{"role": "system", "content": "Pick a mode."}],
        structured_output=_MODE_OUTPUT,
    )
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert "json" not in calls[0]["messages"][0]["content"].casefold()
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "json" in calls[1]["messages"][0]["content"].casefold()


async def test_json_schema_unrelated_failure_is_not_retried(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            raise RuntimeError("context length exceeded")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="compat",
            base_url="https://openrouter.ai/api/v1",
            structured_output="auto",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )

    with pytest.raises(RuntimeError, match="context length exceeded"):
        await model(
            messages=[{"role": "system", "content": "Pick a mode."}],
            structured_output=_MODE_OUTPUT,
        )

    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"


async def test_stream_retries_json_object_when_json_schema_type_is_unavailable(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        def stream(self, **kwargs: Any):
            calls.append(kwargs)

            async def tokens():
                if kwargs.get("response_format", {}).get("type") == "json_schema":
                    raise RuntimeError(_DEEPSEEK_JSON_SCHEMA_TYPE_ERROR)
                yield "{"
                yield "}"

            return tokens()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="compat",
            base_url="https://openrouter.ai/api/v1",
            structured_output="auto",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    stream = await model(
        messages=[{"role": "system", "content": "Pick a mode."}],
        structured_output=_MODE_OUTPUT,
        stream=True,
    )
    result = [token async for token in stream]

    assert result == ["{", "}"]
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "json" in calls[1]["messages"][0]["content"].casefold()


async def test_json_schema_type_rejection_does_not_poison_other_endpoints(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            calls.append((kwargs["model"], kwargs["response_format"]["type"]))
            if kwargs.get("response_format", {}).get("type") == "json_schema":
                if kwargs["model"] == "broken":
                    raise RuntimeError(_DEEPSEEK_JSON_SCHEMA_TYPE_ERROR)
            return "{}"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    broken = CompletionModel(
        ModelSettings(
            provider="openai",
            model="broken",
            base_url="https://api.deepseek.com",
            structured_output="auto",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    healthy = CompletionModel(
        ModelSettings(
            provider="openai",
            model="healthy",
            base_url="https://openrouter.ai/api/v1",
            structured_output="auto",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    messages = [{"role": "system", "content": "Pick a mode."}]

    await broken(messages=messages, structured_output=_MODE_OUTPUT)
    await healthy(messages=messages, structured_output=_MODE_OUTPUT)

    assert calls == [
        ("broken", "json_schema"),
        ("broken", "json_object"),
        ("healthy", "json_schema"),
    ]


async def test_schema_validation_rejection_retries_without_being_remembered(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            if kwargs.get("response_format", {}).get("type") == "json_schema":
                raise RuntimeError("Invalid schema for response_format 'answer_mode'")
            return "{}"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    settings = ModelSettings(
        provider="openai",
        model="compat",
        base_url="https://api.openai.com/v1",
        structured_output="auto",
    )
    model = CompletionModel(settings, scheduler=ModelScheduler(max_concurrency=1))
    messages = [{"role": "system", "content": "Pick a mode."}]

    await model(messages=messages, structured_output=_MODE_OUTPUT)
    await model(messages=messages, structured_output=_MODE_OUTPUT)

    # A schema complaint earns one retry each time, but no standing capability
    # verdict: the endpoint keeps getting a chance to serve the strict schema.
    assert [call["response_format"]["type"] for call in calls] == [
        "json_schema",
        "json_object",
        "json_schema",
        "json_object",
    ]


async def test_remembered_rejection_is_reused_across_instances_and_modes(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Provider:
        async def complete(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            if kwargs.get("response_format", {}).get("type") == "json_schema":
                raise RuntimeError(_DEEPSEEK_JSON_SCHEMA_TYPE_ERROR)
            return "{}"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag.engine.ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    common = {"provider": "openai", "model": "compat", "base_url": "https://api.deepseek.com"}
    auto = CompletionModel(
        ModelSettings(**common, structured_output="auto"),  # type: ignore[arg-type]
        scheduler=ModelScheduler(max_concurrency=1),
    )
    explicit = CompletionModel(
        ModelSettings(**common, structured_output="json_schema"),  # type: ignore[arg-type]
        scheduler=ModelScheduler(max_concurrency=1),
    )
    messages = [{"role": "system", "content": "Pick a mode."}]

    await auto(messages=messages, structured_output=_MODE_OUTPUT)
    await explicit(messages=messages, structured_output=_MODE_OUTPUT)

    # The first call learns the endpoint's transport; later calls skip the 400.
    assert [call["response_format"]["type"] for call in calls] == [
        "json_schema",
        "json_object",
        "json_object",
    ]
