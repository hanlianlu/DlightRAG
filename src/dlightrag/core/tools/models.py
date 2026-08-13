# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The contracts one model-visible tool call is made of."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from dlightrag.models.tool_turn import AssistantTurn, ToolCall, ToolDefinition

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


__all__ = [
    "AgentTool",
    "ExecutedTurn",
    "ToolExecute",
    "ToolExecution",
    "ToolModelFunc",
    "ToolObservation",
    "ToolResult",
]
