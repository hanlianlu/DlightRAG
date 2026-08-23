# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral events emitted by the Agent loop."""

from dataclasses import dataclass, field
from typing import Any, Literal

type AgentEventKind = Literal[
    "agent_start",
    "turn_start",
    "model_start",
    "model_end",
    "tool_start",
    "tool_update",
    "tool_end",
    "turn_end",
    "agent_end",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One immutable lifecycle observation.

    Events report what the kernel did; they do not grant extensions authority to
    alter execution. Product telemetry and transports may project this stream.
    """

    kind: AgentEventKind
    turn_number: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


__all__ = ["AgentEvent", "AgentEventKind"]
