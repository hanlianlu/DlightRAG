# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical workspace identity accepted by RAG interfaces."""

import re
from collections.abc import Iterable

_CANONICAL_WORKSPACE_RE = re.compile(r"[a-z_][a-z0-9_]{0,63}")


def normalize_workspace(name: str) -> str:
    """Normalize a display name into one canonical workspace identifier."""
    workspace_id = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).lower()
    if workspace_id and workspace_id[0].isdigit():
        workspace_id = f"_{workspace_id}"
    return workspace_id


def normalize_workspace_ids(workspaces: Iterable[str]) -> list[str]:
    """Canonicalize and stably de-duplicate workspace identifiers."""
    seen: set[str] = set()
    result: list[str] = []
    for workspace in workspaces:
        workspace_id = normalize_workspace(workspace)
        if workspace_id and workspace_id not in seen:
            seen.add(workspace_id)
            result.append(workspace_id)
    return result


def require_canonical_workspace_id(workspace_id: str) -> str:
    """Return a canonical workspace id or reject display-name input."""
    if not isinstance(workspace_id, str) or not _CANONICAL_WORKSPACE_RE.fullmatch(workspace_id):
        raise ValueError(f"canonical workspace id required, got {workspace_id!r}")
    return workspace_id


__all__ = ["normalize_workspace", "normalize_workspace_ids", "require_canonical_workspace_id"]
