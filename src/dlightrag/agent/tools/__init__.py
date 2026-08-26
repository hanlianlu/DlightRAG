# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral tool contracts and deterministic turn execution."""

from dlightrag.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.agent.tools.contracts import (
    AgentTool,
    EvidenceSourceFact,
    ExecutedTurn,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolExecution,
    ToolResult,
    ToolResultCapacityError,
    ToolRuntime,
)
from dlightrag.agent.tools.executor import (
    DuplicateToolCallIdError,
    PreparedToolTurn,
    ToolPreflight,
    ToolTurnExecutor,
    fit_tool_result,
    preflight_tool_calls,
)
from dlightrag.agent.tools.files import (
    bash_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    path_tools,
    preview_or_spill,
    read_tool,
    write_tool,
)
from dlightrag.agent.tools.registry import DuplicateToolError, ToolRegistry

__all__ = [
    "AgentTool",
    "DuplicateToolCallIdError",
    "DuplicateToolError",
    "EvidenceSourceFact",
    "ExecutedTurn",
    "PreparedToolTurn",
    "ResourceAttachmentBytes",
    "ToolEffects",
    "ToolExecution",
    "ToolPreflight",
    "ToolResourceAttachmentPart",
    "ToolResult",
    "ToolRegistry",
    "ToolResultCapacityError",
    "ToolRuntime",
    "ToolTextPart",
    "ToolTurnExecutor",
    "bash_tool",
    "edit_tool",
    "find_tool",
    "fit_tool_result",
    "grep_tool",
    "ls_tool",
    "path_tools",
    "preflight_tool_calls",
    "preview_or_spill",
    "read_tool",
    "write_tool",
]
