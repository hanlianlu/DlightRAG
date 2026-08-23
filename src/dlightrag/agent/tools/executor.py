# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and execute one batch of provider-neutral tool calls."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel, ValidationError

from dlightrag.agent.events import AgentEvent
from dlightrag.agent.session.effects import (
    EffectIntent,
    ToolResultEntry,
    canonical_json,
)
from dlightrag.agent.session.ids import IntentId
from dlightrag.agent.tool_content import (
    ToolTextPart,
    tool_content_attachments,
    tool_content_message_fields,
)
from dlightrag.agent.tools.contracts import (
    AgentTool,
    ExecutedTurn,
    ToolExecution,
    ToolModelFunc,
    ToolObservation,
    ToolResult,
    ToolResultCapacityError,
    ToolRuntime,
)
from dlightrag.ai.messages import AssistantTurn, ToolCall, ToolChoice
from dlightrag.ai.telemetry import NOOP_TELEMETRY, Telemetry
from dlightrag.ai.tokens import estimate_tokens, truncate_to_estimated_tokens

logger = logging.getLogger(__name__)


class _LatestToolUpdatePump:
    """Deliver latest snapshots without applying sink backpressure to a tool."""

    def __init__(self, sink: Callable[[AgentEvent], Awaitable[None]]) -> None:
        self._sink = sink
        self._wake = asyncio.Event()
        self._pending: AgentEvent | None = None
        self._closed = False
        self._task = asyncio.create_task(self._run())

    def publish(self, event: AgentEvent) -> None:
        self._pending = event
        self._wake.set()

    async def close(self, *, wait: bool) -> None:
        if not wait:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            return
        self._closed = True
        self._wake.set()
        await self._task

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            event = self._pending
            self._pending = None
            if event is not None:
                await self._sink(event)
            if self._closed and self._pending is None:
                return


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
                    parts=(ToolTextPart(message),),
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
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> None:
        self._model_func = model_func
        self._telemetry = telemetry
        self._on_event = on_event

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
        await self._emit(
            AgentEvent(
                "model_start",
                data={"tool_names": [tool.name for tool in tools]},
            )
        )
        try:
            assistant = await self._model_func(**model_kwargs)
        except BaseException:
            await self._emit(AgentEvent("model_end", data={"outcome": "error"}))
            raise
        await self._emit(
            AgentEvent(
                "model_end",
                data={
                    "outcome": "ok",
                    "stop_reason": assistant.stop_reason,
                    "tool_calls": len(assistant.tool_calls),
                },
            )
        )
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
        execution_scope: str = "",
        on_result: (
            Callable[[EffectIntent, ToolExecution | None, bool], Awaitable[None]] | None
        ) = None,
    ) -> ExecutedTurn:
        """Execute one prepared turn's tool batch and settle in source order.

        Tools run in parallel. Their complete batch is fitted to the shared
        observation residual before ``on_result`` receives the model-visible
        results, so durable replay stores exactly the same bounded content as
        the live next turn. Validation failures and length-stop fabrications
        never reach ``on_result`` because they carry no intent.
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

        intents_by_call = {
            intent.source_call_id: intent for intent in intents if intent.source_call_id is not None
        }
        tasks = {
            call.id: asyncio.create_task(
                _execute_call(
                    call,
                    tools_by_name,
                    self._telemetry,
                    intent=intents_by_call.get(call.id),
                    execution_scope=execution_scope,
                    source_position=position,
                    on_event=self._emit,
                )
            )
            for position, call in enumerate(assistant.tool_calls)
        }
        completed = False
        try:
            ordered = tuple(
                await asyncio.gather(*(tasks[call.id] for call in assistant.tool_calls))
            )
            fitted = (
                _fit_results(ordered, max_tokens=max_observation_tokens)
                if max_observation_tokens is not None
                else ordered
            )
            by_call_id = {execution.call.id: execution for execution in fitted}
            if on_result is not None:
                for position, intent in enumerate(intents):
                    await on_result(
                        intent,
                        by_call_id.get(intent.source_call_id or ""),
                        position == len(intents) - 1,
                    )
            completed = True
        finally:
            if not completed:
                for task in tasks.values():
                    task.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)
        return _assemble_turn(
            assistant,
            transcript,
            fitted,
            prepared.preflight,
            None,
        )

    async def _emit(self, event: AgentEvent) -> None:
        if self._on_event is None:
            return
        try:
            await self._on_event(event)
        except Exception:
            logger.warning("Agent event sink failed", exc_info=True)


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
        result = fit_tool_result(execution.result, max_tokens=allowance)
        remaining = max(0, remaining - estimate_tokens(result.text_content))
        if result == execution.result:
            fitted.append(execution)
            continue
        fitted.append(
            replace(
                execution,
                result=result,
                observation=replace(
                    execution.observation,
                    content_chars=len(result.text_content),
                ),
            )
        )
    return tuple(fitted)


def fit_tool_result(result: ToolResult, *, max_tokens: int) -> ToolResult:
    """Fit result text while preserving typed attachments and continuation."""
    text = result.text_content
    if estimate_tokens(text) <= max_tokens:
        return result
    if max_tokens < 1:
        raise ToolResultCapacityError("tool result has no residual model input capacity")
    marker = "[tool result truncated to the shared model residual]"
    protected = result.protected_text.strip()
    protected_tokens = estimate_tokens(protected)
    if protected and protected_tokens > max_tokens:
        raise ToolResultCapacityError("tool result continuation does not fit the model residual")
    fixed = "\n".join(part for part in (marker, protected) if part)
    fixed_tokens = estimate_tokens(fixed)
    if fixed_tokens >= max_tokens:
        fitted_text = protected or truncate_to_estimated_tokens(marker, max_tokens)
    else:
        body = text
        if protected and body.endswith(result.protected_text):
            body = body[: -len(result.protected_text)].rstrip()
        body_tokens = max_tokens - fixed_tokens
        fitted_text = protected
        while body_tokens >= 0:
            truncated = truncate_to_estimated_tokens(body, body_tokens)
            candidate = "\n".join(part for part in (truncated, fixed) if part)
            if estimate_tokens(candidate) <= max_tokens:
                fitted_text = candidate
                break
            body_tokens -= 1
    attachments = tool_content_attachments(result.parts)
    return replace(result, parts=(ToolTextPart(fitted_text), *attachments))


async def _execute_call(
    call: ToolCall,
    tools: dict[str, AgentTool],
    telemetry: Telemetry,
    *,
    intent: EffectIntent | None,
    execution_scope: str,
    source_position: int,
    on_event: Callable[[AgentEvent], Awaitable[None]],
) -> ToolExecution:
    """Execute one model tool call under one observation and event lifecycle."""
    started = time.perf_counter()
    update_sequence = 0
    updates = _LatestToolUpdatePump(on_event)
    await on_event(
        AgentEvent(
            "tool_start",
            data={
                "tool_name": call.name,
                "call_id": call.id,
                "source_position": source_position,
            },
        )
    )

    async def emit_update(result: ToolResult) -> None:
        nonlocal update_sequence
        update_sequence += 1
        updates.publish(
            AgentEvent(
                "tool_update",
                data={
                    "tool_name": call.name,
                    "call_id": call.id,
                    "source_position": source_position,
                    "update_sequence": update_sequence,
                    "text_chars": len(result.text_content),
                    "output_bytes": _output_bytes(result),
                    "spill_state": _spill_state(result),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "attachment_count": len(tool_content_attachments(result.parts)),
                    "snapshot": result,
                },
            )
        )

    try:
        async with telemetry.observe(
            "agent_tool",
            as_type="tool",
            metadata={"tool": call.name, "call_id": call.id},
        ) as span:
            execution = await _dispatch_call(
                call,
                tools,
                started=started,
                intent=intent,
                execution_scope=execution_scope,
                emit_update=emit_update,
            )
            span.update(output=execution.observation.as_dict())
    except BaseException as exc:
        await updates.close(wait=False)
        await on_event(
            AgentEvent(
                "tool_end",
                data={
                    "tool_name": call.name,
                    "call_id": call.id,
                    "source_position": source_position,
                    "outcome": (
                        "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                    ),
                },
            )
        )
        raise
    await updates.close(wait=True)
    await on_event(
        AgentEvent(
            "tool_end",
            data={
                "tool_name": call.name,
                "call_id": call.id,
                "source_position": source_position,
                "outcome": execution.observation.outcome,
                "duration_ms": execution.observation.duration_ms,
                "output_bytes": _output_bytes(execution.result),
                "spill_state": _spill_state(execution.result),
            },
        )
    )
    return execution


async def _dispatch_call(
    call: ToolCall,
    tools: dict[str, AgentTool],
    *,
    started: float,
    intent: EffectIntent | None,
    execution_scope: str,
    emit_update: Callable[[ToolResult], Awaitable[None]],
) -> ToolExecution:
    tool, arguments, outcome, message = _validate_call(call, tools)
    if outcome is not None:
        if message is None:
            raise RuntimeError("invalid tool call lost its error message")
        return _error(call, message, outcome=outcome, started=started)
    if tool is None or arguments is None:
        raise RuntimeError("valid tool call lost its tool or arguments")
    if intent is None:
        raise RuntimeError("valid tool call lost its effect intent")
    runtime = ToolRuntime(
        call_id=call.id,
        tool_name=call.name,
        intent_id=intent.intent_id,
        execution_scope=execution_scope,
        _update_sink=emit_update,
    )
    try:
        result = await tool.execute(arguments, runtime)
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
    return ToolExecution(
        call=call,
        result=result,
        observation=_observe(
            call,
            outcome=("failed" if result.is_error else "cached" if result.cached else "ok"),
            started=started,
            cached=result.cached,
            is_error=result.is_error,
            content=result.text_content,
        ),
        is_error=result.is_error,
    )


def _output_bytes(result: ToolResult) -> int:
    if isinstance(result.details, dict):
        value = result.details.get("output_bytes")
        if isinstance(value, int) and value >= 0:
            return value
    return len(result.text_content.encode("utf-8"))


def _spill_state(result: ToolResult) -> str:
    if isinstance(result.details, dict):
        value = result.details.get("spill_state")
        if value in {"none", "staging", "committed"}:
            return str(value)
    return "committed" if result.effects.committed_outputs else "none"


def _error(call: ToolCall, message: str, *, outcome: str, started: float) -> ToolExecution:
    return ToolExecution(
        call=call,
        result=ToolResult.text(message),
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
        **tool_content_message_fields(execution.result.parts),
        "is_error": execution.is_error,
    }


__all__ = [
    "PreparedToolTurn",
    "ToolPreflight",
    "ToolTurnExecutor",
    "preflight_tool_calls",
]
