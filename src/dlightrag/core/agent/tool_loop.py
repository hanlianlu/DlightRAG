# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and execute one batch of provider-neutral tool calls."""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from dlightrag.models.tool_turn import (
    AssistantTurn,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

type ToolModelFunc = Callable[..., Awaitable[AssistantTurn]]
type ToolExecute = Callable[[BaseModel], Awaitable["ToolResult"]]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Text returned to the model plus transport-private details."""

    content: str
    details: dict[str, Any] | None = None
    cached: bool = False


@dataclass(frozen=True, slots=True)
class AgentTool:
    """Executable tool with a Pydantic argument contract."""

    name: str
    description: str
    input_model: type[BaseModel]
    execute: ToolExecute

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """What one tool execution did, with nothing it carried.

    Tool payloads can hold attachment text, provider responses, and redacted
    failure detail, so an observation records only the call's shape: which tool
    ran, how long it took, whether the run had already answered that exact call,
    and how it ended.
    """

    tool: str
    call_id: str
    outcome: str
    duration_ms: float
    cached: bool
    is_error: bool
    content_chars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "call_id": self.call_id,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "cached": self.cached,
            "is_error": self.is_error,
            "content_chars": self.content_chars,
        }


@dataclass(frozen=True, slots=True)
class ToolExecution:
    call: ToolCall
    result: ToolResult
    observation: ToolObservation
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ExecutedTurn:
    assistant: AssistantTurn
    results: tuple[ToolExecution, ...]
    messages: list[dict[str, Any]]


class ToolTurnExecutor:
    """Run one model turn and execute its valid tool calls in parallel."""

    def __init__(self, model_func: ToolModelFunc) -> None:
        self._model_func = model_func

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[AgentTool],
        *,
        tool_choice: ToolChoice = "auto",
    ) -> ExecutedTurn:
        assistant = await self._model_func(
            messages=messages,
            tools=[tool.definition for tool in tools],
            tool_choice=tool_choice,
        )
        transcript = [*messages, _assistant_message(assistant)]
        if not assistant.tool_calls:
            return ExecutedTurn(assistant=assistant, results=(), messages=transcript)

        tools_by_name = {tool.name: tool for tool in tools}
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
        else:
            tasks = [
                asyncio.create_task(_execute_call(call, tools_by_name))
                for call in assistant.tool_calls
            ]
            completed = False
            try:
                results = tuple(await asyncio.gather(*tasks))
                completed = True
            finally:
                if not completed:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        transcript.extend(_tool_message(result) for result in results)
        return ExecutedTurn(assistant=assistant, results=results, messages=transcript)


async def _execute_call(
    call: ToolCall,
    tools: dict[str, AgentTool],
) -> ToolExecution:
    from dlightrag.observability import trace_observation

    started = time.perf_counter()
    tool = tools.get(call.name)
    if tool is None:
        return _error(
            call,
            f'Tool "{call.name}" is not available.',
            outcome="unknown_tool",
            started=started,
        )
    if call.argument_error:
        return _error(
            call,
            f'Arguments for tool "{call.name}" are invalid: {call.argument_error}',
            outcome="invalid_arguments",
            started=started,
        )
    try:
        arguments = tool.input_model.model_validate(call.arguments)
    except ValidationError as exc:
        return _error(
            call,
            f'Arguments for tool "{call.name}" are invalid: {exc}',
            outcome="invalid_arguments",
            started=started,
        )
    async with trace_observation(
        "agent_tool",
        as_type="tool",
        metadata={"tool": call.name, "call_id": call.id},
    ) as span:
        try:
            result = await tool.execute(arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The model only ever sees the message; the traceback belongs to the operator.
            logger.warning("Agent tool %r failed", call.name, exc_info=True)
            execution = _error(
                call,
                f'Tool "{call.name}" failed: {exc}',
                outcome="failed",
                started=started,
            )
        else:
            execution = ToolExecution(
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
        span.update(output=execution.observation.as_dict())
        return execution


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
    "AgentTool",
    "ExecutedTurn",
    "ToolExecution",
    "ToolObservation",
    "ToolResult",
    "ToolTurnExecutor",
]
