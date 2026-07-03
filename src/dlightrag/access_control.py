# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Small authorization layer for DlightRAG resources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from dlightrag.config import DlightragConfig
from dlightrag.utils import normalize_workspace


class AccessAction:
    WORKSPACE_QUERY = "workspace.query"
    WORKSPACE_INGEST = "workspace.ingest"
    WORKSPACE_LIST_FILES = "workspace.list_files"
    WORKSPACE_DELETE_FILES = "workspace.delete_files"
    WORKSPACE_DOWNLOAD_SOURCE = "workspace.download_source"
    WORKSPACE_READ_METADATA = "workspace.read_metadata"
    WORKSPACE_UPDATE_METADATA = "workspace.update_metadata"
    WORKSPACE_READ_VISUAL_ASSET = "workspace.read_visual_asset"
    WORKSPACE_CREATE = "workspace.create"
    WORKSPACE_DELETE = "workspace.delete"
    WORKSPACE_RESET = "workspace.reset"
    JOB_READ = "job.read"


class AccessDeniedError(PermissionError):
    """Raised when an authenticated user is not authorized for a resource."""


class AccessControl(Protocol):
    async def check(self, user: Any, action: str, *, workspace: str | None = None) -> None: ...

    async def filter_workspaces(
        self,
        user: Any,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]: ...


class AllowAllAccessControl:
    async def check(self, user: Any, action: str, *, workspace: str | None = None) -> None:
        return None

    async def filter_workspaces(
        self,
        user: Any,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        return list(workspaces)


class JwtClaimsAccessControl:
    def __init__(self, config: DlightragConfig) -> None:
        self._rules = tuple(config.access_control.rules)

    async def check(self, user: Any, action: str, *, workspace: str | None = None) -> None:
        if self._allows(user, action, workspace):
            return
        target = f" workspace={workspace}" if workspace else ""
        raise AccessDeniedError(f"Access denied for action={action}{target}")

    async def filter_workspaces(
        self,
        user: Any,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        return [workspace for workspace in workspaces if self._allows(user, action, workspace)]

    def _allows(self, user: Any, action: str, workspace: str | None) -> bool:
        if getattr(user, "auth_mode", None) != "jwt":
            return False
        claims = getattr(user, "claims", None)
        if not isinstance(claims, Mapping):
            return False
        normalized_workspace = normalize_workspace(workspace) if workspace else None
        return any(
            _claim_matches(claims, rule.claim, rule.value)
            and _action_matches(rule.actions, action)
            and _workspace_matches(rule.workspaces, normalized_workspace)
            for rule in self._rules
        )


def access_control_from_config(config: DlightragConfig) -> AccessControl:
    if config.access_control.mode == "jwt_claims":
        return JwtClaimsAccessControl(config)
    return AllowAllAccessControl()


def _claim_matches(claims: Mapping[str, Any], claim_name: str, expected: str) -> bool:
    raw = claims.get(claim_name)
    if isinstance(raw, str):
        return raw == expected
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, Mapping)):
        return expected in {str(value) for value in raw}
    return str(raw) == expected if raw is not None else False


def _action_matches(patterns: Sequence[str], action: str) -> bool:
    return any(
        pattern == "*"
        or pattern == action
        or (pattern.endswith(".*") and action.startswith(pattern[:-1]))
        for pattern in patterns
    )


def _workspace_matches(patterns: Sequence[str], workspace: str | None) -> bool:
    return any(pattern == "*" or normalize_workspace(pattern) == workspace for pattern in patterns)


__all__ = [
    "AccessAction",
    "AccessControl",
    "AccessDeniedError",
    "AllowAllAccessControl",
    "JwtClaimsAccessControl",
    "access_control_from_config",
]
