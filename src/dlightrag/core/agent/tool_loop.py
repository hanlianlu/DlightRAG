# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and execute one batch of provider-neutral tool calls."""

import asyncio
import json
import logging
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
class ToolExecution:
    call: ToolCall
    result: ToolResult
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
                ToolExecution(
                    call=call,
                    result=ToolResult(
                        content=(
                            f'Tool "{call.name}" was not executed because the model hit its '
                            "output token limit and the arguments may be truncated."
                        )
                    ),
                    is_error=True,
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
    tool = tools.get(call.name)
    if tool is None:
        return _error(call, f'Tool "{call.name}" is not available.')
    if call.argument_error:
        return _error(call, f'Arguments for tool "{call.name}" are invalid: {call.argument_error}')
    try:
        arguments = tool.input_model.model_validate(call.arguments)
    except ValidationError as exc:
        return _error(call, f'Arguments for tool "{call.name}" are invalid: {exc}')
    try:
        return ToolExecution(call=call, result=await tool.execute(arguments))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The model only ever sees the message; the traceback belongs to the operator.
        logger.warning("Agent tool %r failed", call.name, exc_info=True)
        return _error(call, f'Tool "{call.name}" failed: {exc}')


def _error(call: ToolCall, message: str) -> ToolExecution:
    return ToolExecution(call=call, result=ToolResult(content=message), is_error=True)


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
    "ToolResult",
    "ToolTurnExecutor",
]
