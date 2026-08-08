# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned query model for optional tool-capable turns."""

from typing import Any

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.llm_roles import model_for_role
from dlightrag.models.providers import get_provider
from dlightrag.models.providers.base import CompletionProvider
from dlightrag.models.tool_turn import AssistantTurn, ToolChoice, ToolDefinition


class QueryToolModel:
    """Own one query-role provider without changing the text completion path."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._provider: CompletionProvider = get_provider(
            config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        from dlightrag.observability import trace_observation

        async with trace_observation(
            "agent_model_turn",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={
                "model": self._config.model,
                "tool_names": [tool.name for tool in tools],
                "tool_choice": tool_choice,
            },
        ) as trace:
            turn = await self._provider.complete_tool_turn(
                messages,
                self._config.model,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self._config.temperature,
                model_kwargs=self._config.model_kwargs,
            )
            trace.update(
                output={
                    "stop_reason": turn.stop_reason,
                    "tool_calls": len(turn.tool_calls),
                    "text_length": len(turn.text),
                },
                usage_details=turn.usage_details,
                cost_details=turn.cost_details,
            )
            return turn

    async def aclose(self) -> None:
        await self._provider.aclose()


def create_query_tool_model(config: DlightragConfig) -> QueryToolModel:
    return QueryToolModel(model_for_role(config, "query"))


__all__ = ["QueryToolModel", "create_query_tool_model"]
