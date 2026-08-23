# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The in-flight tool call visible to execute() without widening its signature."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Identity of the model tool call currently executing."""

    call_id: str
    tool_name: str


_TOOL_CALL: ContextVar[ToolCallContext | None] = ContextVar("tool_call", default=None)
_TOOL_EXECUTION_SCOPE: ContextVar[str] = ContextVar("tool_execution_scope", default="")


def current_tool_call() -> ToolCallContext | None:
    return _TOOL_CALL.get()


def bind_tool_call(call_id: str, tool_name: str) -> object:
    """Install the current call. Return a token for :func:`reset_tool_call`."""
    return _TOOL_CALL.set(ToolCallContext(call_id=call_id, tool_name=tool_name))


def current_tool_execution_scope() -> str:
    """Return the host scope inherited by parallel tool tasks."""
    return _TOOL_EXECUTION_SCOPE.get()


def bind_tool_execution_scope(scope: str) -> object:
    """Bind a run/session scope before tool tasks are created."""
    return _TOOL_EXECUTION_SCOPE.set(scope)


def reset_tool_execution_scope(token: object) -> None:
    _TOOL_EXECUTION_SCOPE.reset(token)  # type: ignore[arg-type]


def reset_tool_call(token: object) -> None:
    _TOOL_CALL.reset(token)  # type: ignore[arg-type]


__all__ = [
    "ToolCallContext",
    "bind_tool_call",
    "bind_tool_execution_scope",
    "current_tool_call",
    "current_tool_execution_scope",
    "reset_tool_call",
    "reset_tool_execution_scope",
]
