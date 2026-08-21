# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and execute one batch of provider-neutral tool calls."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from dlightrag_ai.messages import AssistantTurn, ToolCall, ToolChoice
from dlightrag_ai.telemetry import NOOP_TELEMETRY, Telemetry
from dlightrag_ai.tokens import estimate_tokens, truncate_to_estimated_tokens
from pydantic import BaseModel, ValidationError

from dlightrag_agent.session.effects import (
    EffectIntent,
    ToolResultEntry,
    canonical_json,
)
from dlightrag_agent.session.ids import IntentId
from dlightrag_agent.tools.contracts import (
    AgentTool,
    ExecutedTurn,
    ToolExecution,
    ToolModelFunc,
    ToolObservation,
    ToolResult,
    ToolResultCapacityError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolPreflight:
    """Ordered intents for valid calls plus deterministic validation results."""

    intents: tuple[EffectIntent, ...]
    validation_results: tuple[ToolResultEntry, ...]


def preflight_tool_calls(
    assistant: AssistantTurn,
    tools: Sequence[AgentTool],
    *,
    intent_factory: Callable[[], IntentId] = IntentId.new,
) -> ToolPreflight:
    """Create ordered intents for valid calls and validation results for invalid ones.

    Intents keep assistant source order (M3-D12). Unknown tools and invalid
    arguments produce deterministic validation-result entries instead of intents
    (M3-D26); a length-stopped response must not be preflighted at all because
    its calls are never executed (M3-D11).
    """
    tools_by_name = {tool.name: tool for tool in tools}
    intents: list[EffectIntent] = []
    validation_results: list[ToolResultEntry] = []
    for call in assistant.tool_calls:
        tool, _, outcome, message = _validate_call(call, tools_by_name)
        if outcome is not None:
            if message is None:
                raise RuntimeError("validated tool call lost its error message")
            validation_results.append(
                ToolResultEntry(
                    tool_name=call.name,
                    call_id=call.id,
                    outcome=outcome,  # type: ignore[arg-type]
                    content=message,
                )
            )
            continue
        if tool is None:
            raise RuntimeError("valid tool call lost its tool")
        intents.append(
            EffectIntent(
                intent_id=intent_factory(),
                tool_name=tool.name,
                replay_policy=tool.replay_policy,
                contract_version=tool.contract_version,
                input_schema_digest=tool.input_schema_digest,
                canonical_input=canonical_json(call.arguments),
                source_call_id=call.id,
            )
        )
    return ToolPreflight(intents=tuple(intents), validation_results=tuple(validation_results))


def _validate_call(
    call: ToolCall,
    tools: dict[str, AgentTool],
) -> tuple[AgentTool | None, BaseModel | None, str | None, str | None]:
    """Validate one call: (tool, arguments, outcome, message) on success/error."""
    tool = tools.get(call.name)
    if tool is None:
        return None, None, "unknown_tool", f'Tool "{call.name}" is not available.'
    if call.argument_error:
        return (
            None,
            None,
            "invalid_arguments",
            f'Arguments for tool "{call.name}" are invalid: {call.argument_error}',
        )
    try:
        arguments = tool.input_model.model_validate(call.arguments)
    except ValidationError as exc:
        return (
            None,
            None,
            "invalid_arguments",
            f'Arguments for tool "{call.name}" are invalid: {exc}',
        )
    return tool, arguments, None, None


@dataclass(frozen=True, slots=True)
class PreparedToolTurn:
    """One model response with its preflight, before any tool executes.

    The host may durably persist the assistant entry and its EffectIntents
    between ``prepare_turn`` and ``execute_prepared`` so no external effect
    can run without a durable intent behind it (M3-D12 ordering).
    """

    assistant: AssistantTurn
    preflight: ToolPreflight
    transcript: list[dict[str, Any]]


class ToolTurnExecutor:
    """Run one model turn and execute its valid tool calls in parallel."""

    def __init__(
        self,
        model_func: ToolModelFunc,
        *,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self._model_func = model_func
        self._telemetry = telemetry

    async def prepare_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[AgentTool],
        *,
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
    ) -> PreparedToolTurn:
        """Call the model once and preflight its tool calls; no side effects.

        A length-stopped response must not be preflighted at all because its
        calls are never executed (M3-D11); its results are fabricated at
        execution time as before.
        """
        model_kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": [tool.definition for tool in tools],
            "tool_choice": tool_choice,
        }
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        assistant = await self._model_func(**model_kwargs)
        transcript = [*messages, _assistant_message(assistant)]
        preflight = (
            ToolPreflight(intents=(), validation_results=())
            if assistant.stop_reason == "length"
            else preflight_tool_calls(assistant, tools)
        )
        return PreparedToolTurn(assistant=assistant, preflight=preflight, transcript=transcript)

    async def execute_prepared(
        self,
        prepared: PreparedToolTurn,
        tools: list[AgentTool],
        *,
        observation_budget: Callable[[list[dict[str, Any]]], int] | None = None,
        on_result: (
            Callable[[EffectIntent, ToolExecution | None, bool], Awaitable[None]] | None
        ) = None,
    ) -> ExecutedTurn:
        """Execute one prepared turn's tool batch and settle in source order.

        Tools still run in parallel; ``on_result`` is awaited per intent in
        assistant source order as each call's result becomes available, so a
        host can settle durable EffectIntents incrementally (first-completed
        settles first, but never out of order). A missing execution settles
        as interrupted. Validation failures and length-stop fabrications never
        reach ``on_result`` because they carry no intent.
        """
        assistant = prepared.assistant
        transcript = prepared.transcript
        if not assistant.tool_calls:
            return ExecutedTurn(
                assistant=assistant,
                results=(),
                messages=transcript,
                intents=prepared.preflight.intents,
                validation_results=prepared.preflight.validation_results,
            )

        max_observation_tokens = (
            observation_budget(transcript) if observation_budget is not None else None
        )
        if max_observation_tokens is not None and max_observation_tokens < 0:
            raise ValueError("observation budget cannot be negative")

        tools_by_name = {tool.name: tool for tool in tools}
        intents = prepared.preflight.intents
        if assistant.stop_reason == "length":
            results = tuple(
                _error(
                    call,
                    (
                        f'Tool "{call.name}" was not executed because the model hit its '
                        "output token limit and the arguments may be truncated."
                    ),
                    outcome="truncated_arguments",
                    started=time.perf_counter(),
                )
                for call in assistant.tool_calls
            )
            return _assemble_turn(
                assistant,
                transcript,
                results,
                prepared.preflight,
                max_observation_tokens,
            )

        tasks = {
            call.id: asyncio.create_task(_execute_call(call, tools_by_name, self._telemetry))
            for call in assistant.tool_calls
        }
        executions: dict[str, ToolExecution] = {}
        completed = False
        try:
            for position, intent in enumerate(intents):
                task = tasks.get(intent.source_call_id or "")
                execution = None
                if task is not None:
                    execution = await task
                    executions[intent.source_call_id or ""] = execution
                if on_result is not None:
                    await on_result(intent, execution, position == len(intents) - 1)
            ordered: list[ToolExecution] = []
            for call in assistant.tool_calls:
                execution = executions.get(call.id)
                if execution is None:
                    execution = await tasks[call.id]
                    executions[call.id] = execution
                ordered.append(execution)
            completed = True
        finally:
            if not completed:
                for task in tasks.values():
                    task.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)
        return _assemble_turn(
            assistant,
            transcript,
            tuple(ordered),
            prepared.preflight,
            max_observation_tokens,
        )

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[AgentTool],
        *,
        tool_choice: ToolChoice = "auto",
        observation_budget: Callable[[list[dict[str, Any]]], int] | None = None,
        max_tokens: int | None = None,
    ) -> ExecutedTurn:
        """Convenience for hosts without a durable boundary between the model

        call and tool execution: prepare then execute in one step.
        """
        prepared = await self.prepare_turn(
            messages, tools, tool_choice=tool_choice, max_tokens=max_tokens
        )
        return await self.execute_prepared(prepared, tools, observation_budget=observation_budget)


def _assemble_turn(
    assistant: AssistantTurn,
    transcript: list[dict[str, Any]],
    results: tuple[ToolExecution, ...],
    preflight: ToolPreflight,
    max_observation_tokens: int | None,
) -> ExecutedTurn:
    """Fit results to the observation budget and complete the turn."""
    if max_observation_tokens is not None:
        results = _fit_results(results, max_tokens=max_observation_tokens)
    transcript.extend(_tool_message(result) for result in results)
    return ExecutedTurn(
        assistant=assistant,
        results=results,
        messages=transcript,
        intents=preflight.intents,
        validation_results=preflight.validation_results,
    )


def _fit_results(
    results: tuple[ToolExecution, ...],
    *,
    max_tokens: int,
) -> tuple[ToolExecution, ...]:
    remaining = max_tokens
    fitted: list[ToolExecution] = []
    for index, execution in enumerate(results):
        result_count = len(results) - index
        allowance = remaining // result_count
        content = _fit_result_content(execution.result, max_tokens=allowance)
        remaining = max(0, remaining - estimate_tokens(content))
        if content == execution.result.content:
            fitted.append(execution)
            continue
        fitted.append(
            replace(
                execution,
                result=replace(execution.result, content=content),
                observation=replace(execution.observation, content_chars=len(content)),
            )
        )
    return tuple(fitted)


def _fit_result_content(result: ToolResult, *, max_tokens: int) -> str:
    if estimate_tokens(result.content) <= max_tokens:
        return result.content
    if max_tokens < 1:
        raise ToolResultCapacityError("tool result has no residual model input capacity")
    marker = "[tool result truncated to the shared model residual]"
    suffix = result.protected_suffix.strip()
    suffix_tokens = estimate_tokens(suffix)
    if suffix and suffix_tokens > max_tokens:
        raise ToolResultCapacityError("tool result continuation does not fit the model residual")
    fixed = "\n".join(part for part in (marker, suffix) if part)
    fixed_tokens = estimate_tokens(fixed)
    if fixed_tokens >= max_tokens:
        return suffix or truncate_to_estimated_tokens(marker, max_tokens)
    body = result.content
    if suffix and body.endswith(result.protected_suffix):
        body = body[: -len(result.protected_suffix)].rstrip()
    body_tokens = max_tokens - fixed_tokens
    while body_tokens >= 0:
        truncated = truncate_to_estimated_tokens(body, body_tokens)
        fitted = "\n".join(part for part in (truncated, fixed) if part)
        if estimate_tokens(fitted) <= max_tokens:
            return fitted
        body_tokens -= 1
    return suffix


async def _execute_call(
    call: ToolCall,
    tools: dict[str, AgentTool],
    telemetry: Telemetry,
) -> ToolExecution:
    """Execute one model tool call under exactly one observation span."""
    started = time.perf_counter()
    async with telemetry.observe(
        "agent_tool",
        as_type="tool",
        metadata={"tool": call.name, "call_id": call.id},
    ) as span:
        execution = await _dispatch_call(call, tools, started=started)
        span.update(output=execution.observation.as_dict())
        return execution


async def _dispatch_call(
    call: ToolCall,
    tools: dict[str, AgentTool],
    *,
    started: float,
) -> ToolExecution:
    tool, arguments, outcome, message = _validate_call(call, tools)
    if outcome is not None:
        if message is None:
            raise RuntimeError("invalid tool call lost its error message")
        return _error(call, message, outcome=outcome, started=started)
    if tool is None or arguments is None:
        raise RuntimeError("valid tool call lost its tool or arguments")
    from dlightrag_agent.tools.context import bind_tool_call, reset_tool_call

    token = bind_tool_call(call.id, call.name)
    try:
        result = await tool.execute(arguments)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The model only ever sees the message; the traceback belongs to the operator.
        logger.warning("Agent tool %r failed", call.name, exc_info=True)
        return _error(
            call,
            f'Tool "{call.name}" failed: {exc}',
            outcome="failed",
            started=started,
        )
    finally:
        reset_tool_call(token)
    return ToolExecution(
        call=call,
        result=result,
        observation=_observe(
            call,
            outcome="cached" if result.cached else "ok",
            started=started,
            cached=result.cached,
            is_error=False,
            content=result.content,
        ),
    )


def _error(call: ToolCall, message: str, *, outcome: str, started: float) -> ToolExecution:
    return ToolExecution(
        call=call,
        result=ToolResult(content=message),
        observation=_observe(
            call,
            outcome=outcome,
            started=started,
            cached=False,
            is_error=True,
            content=message,
        ),
        is_error=True,
    )


def _observe(
    call: ToolCall,
    *,
    outcome: str,
    started: float,
    cached: bool,
    is_error: bool,
    content: str,
) -> ToolObservation:
    return ToolObservation(
        tool=call.name,
        call_id=call.id,
        outcome=outcome,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        cached=cached,
        is_error=is_error,
        content_chars=len(content),
    )


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.text,
        "tool_calls": [_tool_call_message(call) for call in turn.tool_calls],
    }
    if turn.provider_state is not None:
        message["provider_state"] = turn.provider_state
    return message


def _tool_call_message(call: ToolCall) -> dict[str, Any]:
    message: dict[str, Any] = {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }
    if call.thought_signature is not None:
        message["thought_signature"] = call.thought_signature
    return message


def _tool_message(execution: ToolExecution) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": execution.call.id,
        "name": execution.call.name,
        "content": execution.result.content,
        "is_error": execution.is_error,
    }


__all__ = [
    "PreparedToolTurn",
    "ToolPreflight",
    "ToolTurnExecutor",
    "preflight_tool_calls",
]
