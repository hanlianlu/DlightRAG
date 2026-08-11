# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the optional query-role tool model."""

from unittest.mock import AsyncMock

import pytest

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
)
from dlightrag.models.tool_model import create_query_tool_model
from dlightrag.models.tool_turn import AssistantTurn, ToolDefinition


def _query_config(
    *,
    model_kwargs: dict[str, object] | None = None,
    agentic_model_kwargs: dict[str, object] | None = None,
) -> DlightragConfig:
    return DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(
                provider="openai",
                model="query-model",
                api_key="key",
                model_kwargs=model_kwargs or {},
                agentic_model_kwargs=agentic_model_kwargs or {},
            )
        ),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="embedding-key",
            startup_probe=False,
        ),
    )


async def test_query_tool_model_uses_default_when_query_role_is_absent(monkeypatch) -> None:
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

    monkeypatch.setattr("dlightrag.models.tool_model.get_provider", get_provider)
    cfg = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(
                provider="openai",
                model="default-model",
                api_key="default-key",
                model_kwargs={"thinking": {"type": "disabled"}},
                agentic_model_kwargs={"thinking": {"type": "enabled"}},
            )
        ),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="embedding-key",
            startup_probe=False,
        ),
    )

    model = create_query_tool_model(cfg)
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

    monkeypatch.setattr("dlightrag.models.tool_model.get_provider", get_provider)
    cfg = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(provider="openai", model="default", api_key="default-key"),
            roles=LLMRolesConfig(
                query=ModelConfig(
                    provider="openai",
                    model="query-model",
                    api_key="query-key",
                    temperature=0.2,
                    model_kwargs={"enable_thinking": False},
                    agentic_model_kwargs={"enable_thinking": True},
                )
            ),
        ),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="embedding-key",
            startup_probe=False,
        ),
    )
    model = create_query_tool_model(cfg)
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
    assert cfg.llm.roles.query is not None
    assert cfg.llm.roles.query.model_kwargs == {"enable_thinking": False}
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
        "dlightrag.models.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    cfg = DlightragConfig(
        llm=LLMConfig(default=ModelConfig(provider="openai", model="query-model", api_key="key")),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="embedding-key",
            startup_probe=False,
        ),
    )
    model = create_query_tool_model(cfg)
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
        "dlightrag.models.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    cfg = _query_config(
        model_kwargs={"thinking": {"type": "disabled"}},
        agentic_model_kwargs={"thinking": {"type": "enabled"}},
    )
    model = create_query_tool_model(cfg)

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
        "dlightrag.models.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    cfg = _query_config()
    model = create_query_tool_model(cfg)

    with pytest.raises(RuntimeError, match="empty final answer"):
        _ = [
            token
            async for token in model.stream_text(
                messages=[{"role": "user", "content": "answer now"}]
            )
        ]

    assert call_count == 2


async def test_query_tool_model_completes_final_text_through_owned_provider(monkeypatch) -> None:
    provider = AsyncMock()
    provider.complete_tool_turn = AsyncMock(
        return_value=AssistantTurn(text="final answer", tool_calls=(), stop_reason="stop")
    )
    monkeypatch.setattr(
        "dlightrag.models.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    cfg = DlightragConfig(
        llm=LLMConfig(default=ModelConfig(provider="openai", model="query-model", api_key="key")),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="embedding-key",
            startup_probe=False,
        ),
    )
    model = create_query_tool_model(cfg)
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
        "dlightrag.models.tool_model.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    cfg = _query_config(
        model_kwargs={"thinking": {"type": "disabled"}},
        agentic_model_kwargs={"thinking": {"type": "enabled"}},
    )
    model = create_query_tool_model(cfg)

    output = await model.complete_text(messages=[{"role": "user", "content": "answer now"}])

    assert output == "final answer"
    assert [
        call.kwargs["model_kwargs"] for call in provider.complete_tool_turn.await_args_list
    ] == [
        {"thinking": {"type": "enabled"}},
        {"thinking": {"type": "disabled"}},
    ]
