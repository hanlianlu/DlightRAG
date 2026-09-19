# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closeable tool-capable model execution over provider-neutral settings."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

from dlightrag.engine.ai.capacity import ModelCapabilityError, ModelProfile
from dlightrag.engine.ai.catalog import resolve_model_profile
from dlightrag.engine.ai.fingerprints import model_invocation_fingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolChoice, ToolDefinition
from dlightrag.engine.ai.providers import get_provider
from dlightrag.engine.ai.providers.base import CompletionProvider
from dlightrag.engine.ai.reasoning import (
    ReasoningLevel,
    ResolvedReasoning,
    merge_reasoning_kwargs,
    resolve_reasoning,
)
from dlightrag.engine.ai.replay import bind_provider_replay, messages_for_model
from dlightrag.engine.ai.response_policy import validate_response_extensions
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY, Telemetry, telemetry_error_message

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
        self.fingerprint = model_invocation_fingerprint(settings)
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._ordinary_model_kwargs = settings.model_kwargs_copy()
        self._agentic_model_kwargs = settings.agentic_model_kwargs_copy()
        self._provider: CompletionProvider = get_provider(
            settings.provider,
            api_key=settings.api_key,
            base_url=settings.base_url,
            api_family=settings.api_family,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
        model_profile: ModelProfile | None = None,
    ) -> AssistantTurn:
        return await self._scheduler.run(
            lambda: self._complete_tool_turn(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                model_profile=model_profile,
            )
        )

    async def stream_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        emit_text: Callable[[str], Awaitable[None]],
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
        model_profile: ModelProfile | None = None,
    ) -> AssistantTurn:
        """Run one tool-capable turn and forward native provider text deltas."""
        return await self._scheduler.run(
            lambda: self._complete_tool_turn(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                model_profile=model_profile,
                emit_text=emit_text,
            )
        )

    async def _complete_tool_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice,
        max_tokens: int | None,
        model_profile: ModelProfile | None,
        emit_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> AssistantTurn:
        self._validate_image_inputs(messages, model_profile)
        resolved = self._resolve_reasoning(
            self.settings.effective_agentic_reasoning,
            model_profile,
        )
        model_kwargs = self._merge_model_kwargs(self._agentic_model_kwargs, resolved)
        async with self._telemetry.observe(
            "generate-agent-turn",
            input={"message_count": len(messages)},
            metadata={
                "model": self.fingerprint.model,
                "provider": self.fingerprint.provider,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
                "api_family": self.fingerprint.api_family,
                "tool_names": [tool.name for tool in tools],
                "tool_choice": tool_choice,
                **self._reasoning_metadata(resolved),
            },
            model=self.settings.model,
        ) as observation:
            try:
                provider_method = (
                    self._provider.complete_tool_turn
                    if emit_text is None
                    else self._provider.complete_tool_turn_streaming
                )
                provider_kwargs: dict[str, Any] = {
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "temperature": self.settings.temperature,
                    "max_tokens": max_tokens,
                    "model_kwargs": model_kwargs,
                }
                if emit_text is not None:
                    provider_kwargs["emit_text"] = emit_text
                turn = await provider_method(
                    messages_for_model(messages, self.fingerprint),
                    self.settings.model,
                    **provider_kwargs,
                )
                turn = bind_provider_replay(turn, self.fingerprint)
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
        reasoning: ReasoningLevel | None = None,
        model_profile: ModelProfile | None = None,
    ) -> AsyncGenerator[str]:
        """Stream the compaction summary of a rich tool transcript.

        Explicit ``model_kwargs`` selects one attempt. Typed ``reasoning`` is
        still resolved from the pinned profile and owns its provider fields.
        """
        return self._scheduler.stream(
            lambda: self._stream_text(
                messages=messages,
                model_kwargs=model_kwargs,
                reasoning=reasoning,
                model_profile=model_profile,
            )
        )

    async def _stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
        model_kwargs: dict[str, Any] | None = None,
        reasoning: ReasoningLevel | None = None,
        model_profile: ModelProfile | None = None,
    ) -> AsyncGenerator[str]:
        record_text = self._telemetry.capture_sensitive_data
        streamed: list[str] = []
        usage_details: dict[str, int | float] = {}
        cost_details: dict[str, int | float] = {}
        text_length = 0
        attempts = 0
        reasoning_attempts: list[dict[str, str]] = []
        attempt_options = self._final_attempt_options(
            model_kwargs=model_kwargs,
            reasoning=reasoning,
            model_profile=model_profile,
        )
        self._validate_image_inputs(messages, model_profile)
        prepared_messages = messages_for_model(messages, self.fingerprint)
        async with self._telemetry.observe(
            "compact-session",
            input={"message_count": len(messages)},
            metadata={
                "model": self.fingerprint.model,
                "provider": self.fingerprint.provider,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
                "api_family": self.fingerprint.api_family,
            },
            model=self.settings.model,
        ) as observation:
            try:
                for attempt_kwargs, resolved in attempt_options:
                    attempts += 1
                    if resolved is not None:
                        reasoning_attempts.append(self._reasoning_metadata(resolved))
                    attempt_usage: dict[str, Any] = {}
                    substantive_text = False
                    stream = self._provider.stream_tool_text(
                        prepared_messages,
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
                    if attempts == 1 and len(attempt_options) > 1:
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
                if reasoning_attempts:
                    output["reasoning"] = reasoning_attempts
                if record_text:
                    output["text"] = "".join(streamed)
                observation.update(
                    output=output,
                    usage_details=usage_details or None,
                    cost_details=cost_details or None,
                )

    def _final_attempt_options(
        self,
        *,
        model_kwargs: dict[str, Any] | None,
        reasoning: ReasoningLevel | None,
        model_profile: ModelProfile | None,
    ) -> tuple[tuple[dict[str, Any], ResolvedReasoning | None], ...]:
        if model_kwargs is not None or reasoning is not None:
            requested = reasoning if reasoning is not None else self.settings.reasoning
            resolved = self._resolve_reasoning(requested, model_profile)
            return ((self._merge_model_kwargs(model_kwargs or {}, resolved), resolved),)
        agentic = self._resolve_reasoning(
            self.settings.effective_agentic_reasoning,
            model_profile,
        )
        ordinary = self._resolve_reasoning(self.settings.reasoning, model_profile)
        return (
            (self._merge_model_kwargs(self._agentic_model_kwargs, agentic), agentic),
            (self._merge_model_kwargs(self._ordinary_model_kwargs, ordinary), ordinary),
        )

    def _merge_model_kwargs(
        self,
        raw: dict[str, Any],
        resolved: ResolvedReasoning | None,
    ) -> dict[str, Any]:
        if self.settings.api_family == "response":
            validate_response_extensions(raw)
        return merge_reasoning_kwargs(
            raw,
            resolved,
            api_family=self.settings.api_family,
        )

    def _validate_image_inputs(
        self, messages: list[dict[str, Any]], profile: ModelProfile | None
    ) -> None:
        if (
            profile is not None
            and not profile.supports_images
            and any(
                isinstance(block, dict) and block.get("type") == "image_url"
                for message in messages
                for block in (
                    message["content"] if isinstance(message.get("content"), list) else []
                )
            )
        ):
            raise ModelCapabilityError(role=self.settings.model, capability="image inputs")

    def _resolve_reasoning(
        self,
        requested: ReasoningLevel | None,
        model_profile: ModelProfile | None,
    ) -> ResolvedReasoning | None:
        profile = model_profile or resolve_model_profile(self.fingerprint.endpoint)
        resolved = resolve_reasoning(profile.reasoning, requested)
        if resolved is not None and resolved.requested != resolved.effective:
            logger.info(
                "Clamped reasoning level for %s from %s to %s",
                self.settings.model,
                resolved.requested,
                resolved.effective,
            )
        return resolved

    @staticmethod
    def _reasoning_metadata(resolved: ResolvedReasoning | None) -> dict[str, str]:
        if resolved is None:
            return {}
        return {
            "reasoning_requested": resolved.requested,
            "reasoning_effective": resolved.effective,
        }

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
