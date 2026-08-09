# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned query model for optional tool-capable turns."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.llm import structured_response_format
from dlightrag.models.llm_roles import model_for_role
from dlightrag.models.providers import get_provider
from dlightrag.models.providers.base import CompletionProvider
from dlightrag.models.structured import StructuredOutput
from dlightrag.models.tool_turn import AssistantTurn, ToolChoice, ToolDefinition


class QueryToolModel:
    """Own one query-role provider without changing the text completion path."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model_kwargs = {
            **config.model_kwargs,
            **config.agentic_model_kwargs,
        }
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
                model_kwargs=self._model_kwargs,
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

    async def stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream a tools-none final answer from a rich tool transcript."""
        from dlightrag.observability import trace_observation

        usage_holder: dict[str, Any] = {}
        text_length = 0
        async with trace_observation(
            "agent_final_answer",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={"model": self._config.model},
        ) as trace:
            try:
                async for token in self._provider.stream_tool_text(
                    messages,
                    self._config.model,
                    temperature=self._config.temperature,
                    model_kwargs=self._model_kwargs,
                    usage_holder=usage_holder,
                ):
                    text_length += len(token)
                    yield token
            finally:
                trace.update(
                    output={"text_length": text_length},
                    usage_details=usage_holder.get("usage_details"),
                    cost_details=usage_holder.get("cost_details"),
                )

    async def complete_text(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        """Return a tools-disabled final answer from a rich tool transcript.

        Non-streaming analogue of :meth:`stream_text`. Reuses the provider's
        tool-transcript path with no tools offered, so provider-native reasoning
        signatures in the transcript are preserved while the model can only emit
        text.
        """
        from dlightrag.observability import trace_observation

        async with trace_observation(
            "agent_final_answer",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={"model": self._config.model},
        ) as trace:
            turn = await self._provider.complete_tool_turn(
                messages,
                self._config.model,
                tools=[],
                temperature=self._config.temperature,
                model_kwargs=self._model_kwargs,
            )
            trace.update(
                output={"text_length": len(turn.text)},
                usage_details=turn.usage_details,
                cost_details=turn.cost_details,
            )
            return turn.text

    async def complete_structured(
        self,
        *,
        messages: list[dict[str, Any]],
        structured_output: StructuredOutput,
    ) -> BaseModel:
        """Return one validated control decision without inventing a tool action."""
        from dlightrag.observability import trace_observation

        response_format = structured_response_format(
            structured_output,
            self._config,
            provider=self._provider,
        )
        async with trace_observation(
            "agent_control_decision",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={
                "model": self._config.model,
                "schema": structured_output.name,
            },
        ) as trace:
            try:
                raw = await self._provider.complete(
                    messages,
                    self._config.model,
                    temperature=self._config.temperature,
                    response_format=response_format,
                    model_kwargs=self._model_kwargs,
                )
            except Exception:
                if not (
                    self._config.provider == "openai"
                    and response_format.get("type") == "json_schema"
                ):
                    raise
                raw = await self._provider.complete(
                    messages,
                    self._config.model,
                    temperature=self._config.temperature,
                    response_format={"type": "json_object"},
                    model_kwargs=self._model_kwargs,
                )
            decision = structured_output.parse(raw)
            trace.update(
                output=decision.model_dump(mode="json"),
                usage_details=getattr(raw, "usage_details", None),
                cost_details=getattr(raw, "cost_details", None),
            )
            return decision

    async def aclose(self) -> None:
        await self._provider.aclose()


def create_query_tool_model(config: DlightragConfig) -> QueryToolModel:
    return QueryToolModel(model_for_role(config, "query"))


__all__ = ["QueryToolModel", "create_query_tool_model"]
