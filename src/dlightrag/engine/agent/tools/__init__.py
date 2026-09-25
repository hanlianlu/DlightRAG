# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral tool contracts and deterministic turn execution."""

from dlightrag.engine.agent.tools.capacity import fit_tool_result
from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    EvidenceSourceFact,
    ExecutedTurn,
    ResourceAttachmentBytes,
    ToolDeclaration,
    ToolEffects,
    ToolExecute,
    ToolResult,
    ToolResultCapacityError,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry

__all__ = [
    "AgentTool",
    "DuplicateToolError",
    "EvidenceSourceFact",
    "ExecutedTurn",
    "ResourceAttachmentBytes",
    "ToolEffects",
    "ToolDeclaration",
    "ToolExecute",
    "ToolRegistry",
    "ToolResult",
    "ToolResultCapacityError",
    "ToolRuntime",
    "fit_tool_result",
]
