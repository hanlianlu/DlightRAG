# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the optional query-role tool model."""

from unittest.mock import AsyncMock

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
)
from dlightrag.models.tool_model import create_query_tool_model
from dlightrag.models.tool_turn import AssistantTurn, ToolDefinition


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
        model_kwargs={"enable_thinking": False},
    )
    provider.aclose.assert_awaited_once()
