# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public-seam helpers for exercising Agent Tool Contract v2."""

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tools import ToolResult, ToolRuntime


async def _ignore_update(_result: ToolResult) -> None:
    return None


def tool_runtime(
    *,
    call_id: str = "test-call",
    tool_name: str = "test-tool",
    execution_scope: str = "test-scope",
) -> ToolRuntime:
    return ToolRuntime(
        call_id=call_id,
        tool_name=tool_name,
        intent_id=IntentId.new(),
        execution_scope=execution_scope,
        _update_sink=_ignore_update,
    )


__all__ = ["tool_runtime"]
