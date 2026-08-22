# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate optional local execution paths without importing Application."""

from __future__ import annotations

import os
from pathlib import Path

from dlightrag.agent.environment import WORKSPACE_MAX_BYTES

DEFAULT_LOCAL_WORKSPACE_ROOT = Path.home() / ".dlightrag" / "agent_workspaces"


def default_local_workspace_root() -> Path:
    """Single-machine default: outside the repo, never the corpus working_dir."""
    return DEFAULT_LOCAL_WORKSPACE_ROOT.expanduser().resolve()


def validate_agent_execution(
    *,
    execution_environment: str,
    workspace_root: str | None,
    working_dir: str,
) -> Path | None:
    """Reject unsafe local_trusted settings before the coordinator starts."""
    if execution_environment == "disabled":
        return None
    raw = (workspace_root or "").strip()
    if not raw or raw in {"null", "None"}:
        root = default_local_workspace_root()
    else:
        root = Path(raw).expanduser()
        if not root.is_absolute():
            raise ValueError(
                "agent.workspace_root must be an absolute path when execution is local_trusted"
            )
    root.mkdir(parents=True, exist_ok=True)
    working = Path(working_dir).expanduser().resolve()
    resolved = root.resolve()
    if resolved == working or resolved.is_relative_to(working) or working.is_relative_to(resolved):
        raise ValueError("agent.workspace_root must not overlap working_dir")
    usage = os.statvfs(resolved)
    if usage.f_bavail * usage.f_frsize < WORKSPACE_MAX_BYTES:
        raise ValueError("agent.workspace_root does not have headroom for one maximum epoch copy")
    return resolved


__all__ = [
    "DEFAULT_LOCAL_WORKSPACE_ROOT",
    "default_local_workspace_root",
    "validate_agent_execution",
]
