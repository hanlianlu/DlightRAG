# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral tool contracts and deterministic turn execution."""

from dlightrag.engine.agent.tools.contracts import (
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
from dlightrag.engine.agent.tools.executor import (
    DuplicateToolCallIdError,
    PreparedToolTurn,
    ToolPreflight,
    ToolTurnExecutor,
    fit_tool_result,
    preflight_tool_calls,
)
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry

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
    "ToolResult",
    "ToolRegistry",
    "ToolResultCapacityError",
    "ToolRuntime",
    "ToolTurnExecutor",
    "fit_tool_result",
    "preflight_tool_calls",
]
