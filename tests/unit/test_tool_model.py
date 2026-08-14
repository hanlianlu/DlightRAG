# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the optional query-role tool model."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast
from unittest.mock import AsyncMock

import pytest
from dlightrag_ai.messages import AssistantTurn, ToolDefinition
from dlightrag_ai.settings import ModelSettings
from dlightrag_ai.tool_model import ToolModel


async def test_ai_tool_model_accepts_settings_and_owns_provider(monkeypatch) -> None:
    provider = AsyncMock()
    provider.complete_tool_turn.return_value = AssistantTurn(
        text="done",
        tool_calls=(),
        stop_reason="stop",
    )
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = ToolModel(
        ModelSettings(
            provider="openai",
            model="query-model",
            model_kwargs={"reasoning": {"enabled": False}},
            agentic_model_kwargs={"reasoning": {"enabled": True}},
        )
    )

    turn = await model(messages=[{"role": "user", "content": "q"}], tools=[])
    await model.aclose()

    assert turn.text == "done"
    assert provider.complete_tool_turn.await_args.kwargs["model_kwargs"] == {
        "reasoning": {"enabled": True}
    }
    provider.aclose.assert_awaited_once()


async def test_tool_model_error_uses_privacy_safe_status(monkeypatch) -> None:
    class Observation:
        updates: list[dict[str, object]] = []

        def update(self, **kwargs: object) -> None:
            self.updates.append(kwargs)

    class Telemetry:
        capture_sensitive_data = False
        observation = Observation()

        @asynccontextmanager
        async def observe(self, name: str, **_kwargs: object):
            del name
            yield self.observation

    provider = AsyncMock()
    provider.complete_tool_turn.side_effect = RuntimeError("echoed secret tool transcript")
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = ToolModel(_query_settings(), telemetry=Telemetry())

    with pytest.raises(RuntimeError, match="secret tool transcript"):
        await model(messages=[{"role": "user", "content": "secret"}], tools=[])

    assert Telemetry.observation.updates == [{"level": "ERROR", "status_message": "RuntimeError"}]


def _query_settings(
    *,
    model_kwargs: dict[str, object] | None = None,
    agentic_model_kwargs: dict[str, object] | None = None,
) -> ModelSettings:
    return ModelSettings(
        provider="openai",
        model="query-model",
        api_key="key",
        model_kwargs=model_kwargs or {},
        agentic_model_kwargs=agentic_model_kwargs or {},
    )


async def test_tool_model_passes_provider_settings_and_agentic_options(monkeypatch) -> None:
    seen: dict[str, object] = {}
    provider = AsyncMock()
    provider.complete_tool_turn.return_value = AssistantTurn(
        text="done",
        tool_calls=(),
        stop_reason="stop",
    )

    def get_provider(name: str, **kwargs):
        seen.update({"provider": name, **kwargs})
        return provider

    monkeypatch.setattr("dlightrag_ai.tool_model.get_provider", get_provider)
    settings = ModelSettings(
        provider="openai",
        model="default-model",
        api_key="default-key",
        model_kwargs={"thinking": {"type": "disabled"}},
        agentic_model_kwargs={"thinking": {"type": "enabled"}},
    )

    model = ToolModel(settings)
    await model(messages=[{"role": "user", "content": "q"}], tools=[])

    assert seen["provider"] == "openai"
    assert seen["api_key"] == "default-key"
    await_args = provider.complete_tool_turn.await_args
    assert await_args is not None
    assert await_args.kwargs["model_kwargs"] == {"thinking": {"type": "enabled"}}


async def test_query_tool_model_owns_query_role_provider_and_closes_it(monkeypatch) -> None:
    provider = AsyncMock()
    provider.complete_tool_turn.return_value = AssistantTurn(
        text="done",
        tool_calls=(),
        stop_reason="stop",
    )
    seen: dict[str, object] = {}

    def get_provider(name: str, **kwargs):
        seen.update({"provider": name, **kwargs})
        return provider

    monkeypatch.setattr("dlightrag_ai.tool_model.get_provider", get_provider)
    settings = ModelSettings(
        provider="openai",
        model="query-model",
        api_key="query-key",
        temperature=0.2,
        model_kwargs={"enable_thinking": False},
        agentic_model_kwargs={"enable_thinking": True},
    )
    model = ToolModel(settings)
    tool = ToolDefinition(name="search", description="Search.", parameters={"type": "object"})

    turn = await model(
        messages=[{"role": "user", "content": "q"}],
        tools=[tool],
        tool_choice="required",
    )
    await model.aclose()

    assert turn.text == "done"
    assert seen["provider"] == "openai"
    assert seen["api_key"] == "query-key"
    provider.complete_tool_turn.assert_awaited_once_with(
        [{"role": "user", "content": "q"}],
        "query-model",
        tools=[tool],
        tool_choice="required",
        temperature=0.2,
        model_kwargs={"enable_thinking": True},
    )
    assert settings.model_kwargs == {"enable_thinking": False}
    provider.aclose.assert_awaited_once()


async def test_query_tool_model_streams_final_text_through_owned_provider(monkeypatch) -> None:
    provider = AsyncMock()
    seen: dict[str, object] = {}

    async def tokens():
        yield "final "
        yield "answer"

    def stream_tool_text(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return tokens()

    provider.stream_tool_text = stream_tool_text
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = ToolModel(_query_settings())
    messages = [{"role": "user", "content": "answer now"}]

    output = [token async for token in model.stream_text(messages=messages)]

    assert output == ["final ", "answer"]
    assert seen["args"] == (messages, "query-model")
    assert seen["kwargs"] == {
        "temperature": None,
        "model_kwargs": {},
        "usage_holder": {},
    }


async def test_query_tool_model_retries_empty_final_stream_with_ordinary_kwargs(
    monkeypatch,
) -> None:
    provider = AsyncMock()
    calls: list[dict[str, object]] = []

    async def empty_tokens():
        if False:
            yield ""

    async def answer_tokens():
        yield "final answer"

    def stream_tool_text(*_args, **kwargs):
        calls.append(kwargs)
        return empty_tokens() if len(calls) == 1 else answer_tokens()

    provider.stream_tool_text = stream_tool_text
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    settings = _query_settings(
        model_kwargs={"thinking": {"type": "disabled"}},
        agentic_model_kwargs={"thinking": {"type": "enabled"}},
    )
    model = ToolModel(settings)

    output = [
        token
        async for token in model.stream_text(messages=[{"role": "user", "content": "answer now"}])
    ]

    assert output == ["final answer"]
    assert [call["model_kwargs"] for call in calls] == [
        {"thinking": {"type": "enabled"}},
        {"thinking": {"type": "disabled"}},
    ]


async def test_query_tool_model_rejects_repeated_empty_final_stream(monkeypatch) -> None:
    provider = AsyncMock()
    call_count = 0

    async def empty_tokens():
        if False:
            yield ""

    def stream_tool_text(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return empty_tokens()

    provider.stream_tool_text = stream_tool_text
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = ToolModel(_query_settings())

    with pytest.raises(RuntimeError, match="empty final answer"):
        _ = [
            token
            async for token in model.stream_text(
                messages=[{"role": "user", "content": "answer now"}]
            )
        ]

    assert call_count == 2


async def test_tool_model_stream_abandonment_closes_provider_iterator(monkeypatch) -> None:
    finalized = asyncio.Event()
    provider = AsyncMock()

    def stream_tool_text(*_args, **_kwargs):
        async def tokens():
            try:
                yield "first"
                yield "second"
            finally:
                finalized.set()

        return tokens()

    provider.stream_tool_text = stream_tool_text
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = ToolModel(_query_settings())
    stream = model.stream_text(messages=[{"role": "user", "content": "answer now"}])

    assert await anext(stream) == "first"
    await cast(AsyncGenerator[str], stream).aclose()

    assert finalized.is_set()


async def test_query_tool_model_completes_final_text_through_owned_provider(monkeypatch) -> None:
    provider = AsyncMock()
    provider.complete_tool_turn = AsyncMock(
        return_value=AssistantTurn(text="final answer", tool_calls=(), stop_reason="stop")
    )
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = ToolModel(_query_settings())
    messages = [{"role": "user", "content": "answer now"}]

    output = await model.complete_text(messages=messages)

    assert output == "final answer"
    provider.complete_tool_turn.assert_awaited_once_with(
        messages,
        "query-model",
        tools=[],
        temperature=None,
        model_kwargs={},
    )


async def test_query_tool_model_retries_empty_final_completion_with_ordinary_kwargs(
    monkeypatch,
) -> None:
    provider = AsyncMock()
    provider.complete_tool_turn.side_effect = [
        AssistantTurn(text="", tool_calls=(), stop_reason="stop"),
        AssistantTurn(text="final answer", tool_calls=(), stop_reason="stop"),
    ]
    monkeypatch.setattr(
        "dlightrag_ai.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    settings = _query_settings(
        model_kwargs={"thinking": {"type": "disabled"}},
        agentic_model_kwargs={"thinking": {"type": "enabled"}},
    )
    model = ToolModel(settings)

    output = await model.complete_text(messages=[{"role": "user", "content": "answer now"}])

    assert output == "final answer"
    assert [
        call.kwargs["model_kwargs"] for call in provider.complete_tool_turn.await_args_list
    ] == [
        {"thinking": {"type": "enabled"}},
        {"thinking": {"type": "disabled"}},
    ]
