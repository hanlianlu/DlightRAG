# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned query model for optional tool-capable turns."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from dlightrag_ai.messages import AssistantTurn, ToolChoice, ToolDefinition
from dlightrag_ai.providers import get_provider
from dlightrag_ai.providers.base import CompletionProvider

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.llm_roles import model_for_role

logger = logging.getLogger(__name__)


class QueryToolModel:
    """Own one query-role provider without changing the text completion path."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._ordinary_model_kwargs = dict(config.model_kwargs)
        # Research calls inherit ordinary options; same-named top-level keys
        # from the endpoint-specific agentic overlay replace them.
        self._agentic_model_kwargs = {
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
                model_kwargs=self._agentic_model_kwargs,
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
        from dlightrag.observability import trace_observation, trace_sensitive_enabled

        record_text = trace_sensitive_enabled()
        streamed: list[str] = []
        usage_details: dict[str, int | float] = {}
        cost_details: dict[str, int | float] = {}
        text_length = 0
        attempts = 0
        async with trace_observation(
            "agent_final_answer",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={"model": self._config.model},
        ) as trace:
            try:
                for model_kwargs in self._final_attempt_kwargs():
                    attempts += 1
                    attempt_usage: dict[str, Any] = {}
                    substantive_text = False
                    async for token in self._provider.stream_tool_text(
                        messages,
                        self._config.model,
                        temperature=self._config.temperature,
                        model_kwargs=model_kwargs,
                        usage_holder=attempt_usage,
                    ):
                        text_length += len(token)
                        substantive_text = substantive_text or bool(token.strip())
                        if record_text:
                            streamed.append(token)
                        yield token
                    _accumulate_metrics(usage_details, attempt_usage.get("usage_details"))
                    _accumulate_metrics(cost_details, attempt_usage.get("cost_details"))
                    if substantive_text:
                        return
                    if attempts == 1:
                        logger.warning(
                            "Agent final answer returned no text; retrying with ordinary model options"
                        )
                raise RuntimeError("Query model returned an empty final answer after retry")
            finally:
                output: dict[str, Any] = {"text_length": text_length, "attempts": attempts}
                if record_text:
                    output["text"] = "".join(streamed)
                trace.update(
                    output=output,
                    usage_details=usage_details or None,
                    cost_details=cost_details or None,
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
        from dlightrag.observability import trace_observation, trace_sensitive_enabled

        usage_details: dict[str, int | float] = {}
        cost_details: dict[str, int | float] = {}
        text = ""
        attempts = 0
        async with trace_observation(
            "agent_final_answer",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={"model": self._config.model},
        ) as trace:
            try:
                for model_kwargs in self._final_attempt_kwargs():
                    attempts += 1
                    turn = await self._provider.complete_tool_turn(
                        messages,
                        self._config.model,
                        tools=[],
                        temperature=self._config.temperature,
                        model_kwargs=model_kwargs,
                    )
                    _accumulate_metrics(usage_details, turn.usage_details)
                    _accumulate_metrics(cost_details, turn.cost_details)
                    text = turn.text
                    if text.strip():
                        return text
                    if attempts == 1:
                        logger.warning(
                            "Agent final answer returned no text; "
                            "retrying with ordinary model options"
                        )
                raise RuntimeError("Query model returned an empty final answer after retry")
            finally:
                output: dict[str, Any] = {"text_length": len(text), "attempts": attempts}
                if trace_sensitive_enabled():
                    output["text"] = text
                trace.update(
                    output=output,
                    usage_details=usage_details or None,
                    cost_details=cost_details or None,
                )

    def _final_attempt_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._agentic_model_kwargs, self._ordinary_model_kwargs

    async def aclose(self) -> None:
        await self._provider.aclose()


def create_query_tool_model(config: DlightragConfig) -> QueryToolModel:
    return QueryToolModel(model_for_role(config, "query"))


def _accumulate_metrics(
    total: dict[str, int | float],
    current: dict[str, Any] | None,
) -> None:
    for key, value in (current or {}).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        total[key] = total.get(key, 0) + value


__all__ = ["QueryToolModel", "create_query_tool_model"]
