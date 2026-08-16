# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorization-aware Answer result projection policy."""


def can_project_workspace_visual(workspace: str | None, allowed: set[str] | None) -> bool:
    """Allow trusted calls and request-owned evidence; otherwise require workspace ACL."""
    return allowed is None or bool(
        workspace and (workspace.startswith("__") or workspace in allowed)
    )


__all__ = ["can_project_workspace_visual"]
