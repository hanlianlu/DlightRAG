# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closeable tool-capable model execution over provider-neutral settings."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from dlightrag_ai.fingerprints import model_fingerprint
from dlightrag_ai.messages import AssistantTurn, ToolChoice, ToolDefinition
from dlightrag_ai.providers import get_provider
from dlightrag_ai.providers.base import CompletionProvider
from dlightrag_ai.scheduler import ModelScheduler
from dlightrag_ai.settings import ModelSettings
from dlightrag_ai.telemetry import NOOP_TELEMETRY, Telemetry, telemetry_error_message

logger = logging.getLogger(__name__)


class ToolModel:
    """Own one provider for tool turns and final transcript generation."""

    def __init__(
        self,
        settings: ModelSettings,
        *,
        scheduler: ModelScheduler,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self.settings = settings
        self.fingerprint = model_fingerprint(settings)
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._ordinary_model_kwargs = settings.model_kwargs_copy()
        self._agentic_model_kwargs = settings.agentic_model_kwargs_copy()
        self._provider: CompletionProvider = get_provider(
            settings.provider,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        return await self._scheduler.run(
            lambda: self._complete_tool_turn(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        )

    async def _complete_tool_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice,
    ) -> AssistantTurn:
        async with self._telemetry.observe(
            "agent_model_turn",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={
                "model": self.fingerprint.model,
                "provider": self.fingerprint.provider,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
                "tool_names": [tool.name for tool in tools],
                "tool_choice": tool_choice,
            },
            model=self.settings.model,
        ) as observation:
            try:
                turn = await self._provider.complete_tool_turn(
                    messages,
                    self.settings.model,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self.settings.temperature,
                    model_kwargs=self._agentic_model_kwargs,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            observation.update(
                output={
                    "stop_reason": turn.stop_reason,
                    "tool_calls": len(turn.tool_calls),
                    "text_length": len(turn.text),
                },
                usage_details=turn.usage_details,
                cost_details=turn.cost_details,
            )
            return turn

    def stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        """Stream a tools-disabled final answer from a rich tool transcript.

        An explicit ``model_kwargs`` replaces the agentic-then-ordinary retry
        sequence with one single attempt using exactly those kwargs: the
        compaction summarizer uses this to force thinking off and a profile
        output cap, with no silent empty-output retry.
        """
        return self._scheduler.stream(
            lambda: self._stream_text(messages=messages, model_kwargs=model_kwargs)
        )

    async def _stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        record_text = self._telemetry.capture_sensitive_data
        streamed: list[str] = []
        usage_details: dict[str, int | float] = {}
        cost_details: dict[str, int | float] = {}
        text_length = 0
        attempts = 0
        async with self._telemetry.observe(
            "agent_final_answer",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={
                "model": self.fingerprint.model,
                "provider": self.fingerprint.provider,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
            },
            model=self.settings.model,
        ) as observation:
            try:
                attempts_kwargs = (
                    (model_kwargs,) if model_kwargs is not None else self._final_attempt_kwargs()
                )
                for attempt_kwargs in attempts_kwargs:
                    attempts += 1
                    attempt_usage: dict[str, Any] = {}
                    substantive_text = False
                    stream = self._provider.stream_tool_text(
                        messages,
                        self.settings.model,
                        temperature=self.settings.temperature,
                        model_kwargs=attempt_kwargs,
                        usage_holder=attempt_usage,
                    )
                    async with aclosing(stream):
                        async for token in stream:
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
            except asyncio.CancelledError, GeneratorExit:
                raise
            except BaseException as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            finally:
                output: dict[str, Any] = {"text_length": text_length, "attempts": attempts}
                if record_text:
                    output["text"] = "".join(streamed)
                observation.update(
                    output=output,
                    usage_details=usage_details or None,
                    cost_details=cost_details or None,
                )

    async def complete_text(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        """Return a tools-disabled final answer from a rich tool transcript."""
        return await self._scheduler.run(lambda: self._complete_text(messages=messages))

    async def _complete_text(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        usage_details: dict[str, int | float] = {}
        cost_details: dict[str, int | float] = {}
        text = ""
        attempts = 0
        async with self._telemetry.observe(
            "agent_final_answer",
            as_type="generation",
            input={"message_count": len(messages)},
            metadata={
                "model": self.fingerprint.model,
                "provider": self.fingerprint.provider,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
            },
            model=self.settings.model,
        ) as observation:
            try:
                for model_kwargs in self._final_attempt_kwargs():
                    attempts += 1
                    turn = await self._provider.complete_tool_turn(
                        messages,
                        self.settings.model,
                        tools=[],
                        temperature=self.settings.temperature,
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
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            finally:
                output: dict[str, Any] = {"text_length": len(text), "attempts": attempts}
                if self._telemetry.capture_sensitive_data:
                    output["text"] = text
                observation.update(
                    output=output,
                    usage_details=usage_details or None,
                    cost_details=cost_details or None,
                )

    def _final_attempt_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._agentic_model_kwargs, self._ordinary_model_kwargs

    async def aclose(self) -> None:
        """Release the provider SDK client and its connection pools."""
        await self._provider.aclose()


def _accumulate_metrics(
    total: dict[str, int | float],
    current: dict[str, Any] | None,
) -> None:
    for key, value in (current or {}).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        total[key] = total.get(key, 0) + value


__all__ = ["ToolModel"]
