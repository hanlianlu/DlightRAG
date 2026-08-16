# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared utilities."""

import re

_WORKSPACE_FORBIDDEN_RE = re.compile(r'[/\\<>"\']')


def validate_workspace_name(name: str, *, max_length: int = 64) -> str:
    """Validate and trim a user-facing workspace name.

    The returned value is still a display label. RAG owns conversion to the
    canonical internal workspace identifier.
    """
    label = name.strip()
    if not label:
        raise ValueError("Workspace name cannot be empty")
    if len(label) > max_length:
        raise ValueError(f"Workspace name too long (max {max_length} characters)")
    if _WORKSPACE_FORBIDDEN_RE.search(label):
        raise ValueError("Workspace name contains forbidden characters")
    return label
