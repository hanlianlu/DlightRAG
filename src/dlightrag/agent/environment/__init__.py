# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Local execution environment, child env, and access scheduling."""

from dlightrag.agent.environment.access import (
    AccessScheduler,
    ExternalAccess,
    PathAccess,
    ToolAccess,
    WorkspaceAccess,
)
from dlightrag.agent.environment.child import build_child_environment, looks_like_secret_name
from dlightrag.agent.environment.errors import (
    TOOL_RESULT_CHAR_LIMIT,
    TOOL_RESULT_PREVIEW_CHARS,
    WORKSPACE_MAX_BYTES,
    WORKSPACE_MAX_ENTRIES,
    FullOutputUnavailable,
    PathRejected,
    WorkspaceQuotaExceeded,
)
from dlightrag.agent.environment.execution import (
    ExecutionEnvironment,
    ExecutionEnvironmentAdapter,
    ExecutionMode,
    SandboxUnavailableError,
    TrustExecutionAdapter,
    resolve_execution_adapter,
)
from dlightrag.agent.environment.local import (
    CompletedProcess,
    DirectoryEntry,
    LocalExecutionEnvironment,
)

__all__ = [
    "TOOL_RESULT_CHAR_LIMIT",
    "TOOL_RESULT_PREVIEW_CHARS",
    "WORKSPACE_MAX_BYTES",
    "WORKSPACE_MAX_ENTRIES",
    "AccessScheduler",
    "WorkspaceAccess",
    "CompletedProcess",
    "DirectoryEntry",
    "ExecutionEnvironment",
    "ExecutionEnvironmentAdapter",
    "ExecutionMode",
    "ExternalAccess",
    "FullOutputUnavailable",
    "LocalExecutionEnvironment",
    "PathAccess",
    "PathRejected",
    "SandboxUnavailableError",
    "ToolAccess",
    "TrustExecutionAdapter",
    "WorkspaceQuotaExceeded",
    "build_child_environment",
    "looks_like_secret_name",
    "resolve_execution_adapter",
]
