# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reusable product-neutral Agent kernel for DlightRAG."""

from dlightrag.agent.context import ContextContribution, ContextProjector, ProjectedContext
from dlightrag.agent.events import AgentEvent, AgentEventKind
from dlightrag.agent.extensions import (
    ContextExtension,
    ExecutionExtension,
    ToolExtension,
    TrustedExtensions,
)
from dlightrag.agent.loop import AgentLoop, AgentLoopCancelled, AgentLoopResult
from dlightrag.agent.tools.registry import DuplicateToolError, ToolRegistry

__all__ = [
    "AgentEvent",
    "AgentEventKind",
    "AgentLoop",
    "AgentLoopCancelled",
    "AgentLoopResult",
    "ContextContribution",
    "ContextExtension",
    "ContextProjector",
    "DuplicateToolError",
    "ExecutionExtension",
    "ProjectedContext",
    "ToolExtension",
    "ToolRegistry",
    "TrustedExtensions",
]
