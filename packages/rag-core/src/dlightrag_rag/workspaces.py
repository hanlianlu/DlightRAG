# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical workspace identity accepted by RAG interfaces."""

import re

_CANONICAL_WORKSPACE_RE = re.compile(r"[a-z_][a-z0-9_]{0,63}")


def require_canonical_workspace_id(workspace_id: str) -> str:
    """Return a canonical workspace id or reject display-name input."""
    if not isinstance(workspace_id, str) or not _CANONICAL_WORKSPACE_RE.fullmatch(workspace_id):
        raise ValueError(f"canonical workspace id required, got {workspace_id!r}")
    return workspace_id


__all__ = ["require_canonical_workspace_id"]
