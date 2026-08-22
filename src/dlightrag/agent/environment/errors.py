# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Execution-environment and generic tool-result failures."""

WORKSPACE_MAX_BYTES = 2 * 1024 * 1024 * 1024
WORKSPACE_MAX_ENTRIES = 10_000
TOOL_RESULT_CHAR_LIMIT = 50_000
TOOL_RESULT_PREVIEW_CHARS = 2_000


class PathRejected(ValueError):
    """A model path is absolute, escapes the root, or names a special file."""


class WorkspaceQuotaExceeded(ValueError):
    """A rooted write would exceed the fixed workspace byte or entry limit."""


class FullOutputUnavailable(RuntimeError):
    """An oversized result has neither cursor backing nor spill."""


__all__ = [
    "TOOL_RESULT_CHAR_LIMIT",
    "TOOL_RESULT_PREVIEW_CHARS",
    "WORKSPACE_MAX_BYTES",
    "WORKSPACE_MAX_ENTRIES",
    "FullOutputUnavailable",
    "PathRejected",
    "WorkspaceQuotaExceeded",
]
