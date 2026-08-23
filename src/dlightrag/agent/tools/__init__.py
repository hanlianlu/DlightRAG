# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral tool contracts and deterministic turn execution."""

from dlightrag.agent.tools.context import current_tool_call
from dlightrag.agent.tools.contracts import (
    AgentTool,
    ExecutedTurn,
    ToolExecution,
    ToolResult,
    ToolResultCapacityError,
)
from dlightrag.agent.tools.executor import (
    PreparedToolTurn,
    ToolPreflight,
    ToolTurnExecutor,
    fit_tool_result_content,
    preflight_tool_calls,
)
from dlightrag.agent.tools.files import (
    bash_tool,
    edit_tool,
    grep_tool,
    path_tools,
    preview_or_spill,
    read_tool,
    write_tool,
)
from dlightrag.agent.tools.registry import DuplicateToolError, ToolRegistry

__all__ = [
    "AgentTool",
    "DuplicateToolError",
    "ExecutedTurn",
    "PreparedToolTurn",
    "ToolExecution",
    "ToolPreflight",
    "ToolResult",
    "ToolRegistry",
    "ToolResultCapacityError",
    "ToolTurnExecutor",
    "current_tool_call",
    "bash_tool",
    "edit_tool",
    "fit_tool_result_content",
    "grep_tool",
    "path_tools",
    "preflight_tool_calls",
    "preview_or_spill",
    "read_tool",
    "write_tool",
]
