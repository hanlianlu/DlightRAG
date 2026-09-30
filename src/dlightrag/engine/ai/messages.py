# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral contracts for one tool-capable model turn."""

import json
from dataclasses import dataclass
from typing import Any, Literal

type ToolChoice = Literal["auto", "required", "none"]
type ToolStopReason = Literal["stop", "length", "tool_use"]


class ToolCallingUnavailableError(RuntimeError):
    """Raised when the configured query model cannot execute tool turns."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool exposed to a model as a JSON-schema function."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One normalized model request to execute a tool."""

    id: str
    name: str
    arguments: dict[str, Any]
    argument_error: str | None = None
    thought_signature: Any | None = None


def tool_call_message(call: ToolCall) -> dict[str, Any]:
    """Project one normalized tool call to its model-message shape.

    The arguments are serialized with sorted keys, so a replayed call is the same bytes
    whichever order its arguments were read back in. A Session read back from
    PostgreSQL returns them in jsonb's order (shorter keys first), not the order the
    model wrote or the Run that made the call still holds in memory, and a follow-up
    Run's first request would otherwise diverge from the previous Run's at the first
    call with more than one argument.
    """
    message: dict[str, Any] = {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    }
    if call.thought_signature is not None:
        message["thought_signature"] = call.thought_signature
    return message


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """Complete provider response with text, reasoning, or tool calls."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: ToolStopReason
    reasoning: str = ""
    usage_details: dict[str, int] | None = None
    cost_details: dict[str, float] | None = None
    provider_state: dict[str, Any] | None = None


__all__ = [
    "AssistantTurn",
    "ToolCall",
    "ToolCallingUnavailableError",
    "ToolChoice",
    "ToolDefinition",
    "ToolStopReason",
    "tool_call_message",
]
