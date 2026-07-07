# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Small authorization layer for DlightRAG resources."""

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


# Named action bundles usable anywhere an action pattern is accepted (e.g. a
# jwt_claims rule's ``actions``), so rules need not enumerate every action.
# ``reader``, ``editor``, and ``admin`` are reserved names.
_READER_ACTIONS: tuple[str, ...] = (
    AccessAction.WORKSPACE_QUERY,
    AccessAction.WORKSPACE_LIST_FILES,
    AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    AccessAction.WORKSPACE_READ_METADATA,
    AccessAction.WORKSPACE_READ_VISUAL_ASSET,
)
_EDITOR_ACTIONS: tuple[str, ...] = (
    *_READER_ACTIONS,
    AccessAction.WORKSPACE_INGEST,
    AccessAction.WORKSPACE_UPDATE_METADATA,
    AccessAction.WORKSPACE_DELETE_FILES,
    AccessAction.JOB_READ,
)
ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "reader": _READER_ACTIONS,
    "editor": _EDITOR_ACTIONS,
    "admin": ("*",),
}


class AccessDeniedError(PermissionError):
    """Raised when an authenticated user is not authorized for a resource."""


class AccessControl(Protocol):
    async def check(self, user: Any, action: str, *, workspace: str | None = None) -> None:
        raise NotImplementedError

    async def filter_workspaces(
        self,
        user: Any,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        raise NotImplementedError


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
    return any(_pattern_allows_action(pattern, action) for pattern in patterns)


def _pattern_allows_action(pattern: str, action: str) -> bool:
    preset = ACTION_PRESETS.get(pattern)
    if preset is not None:
        return any(_pattern_allows_action(entry, action) for entry in preset)
    return (
        pattern == "*"
        or pattern == action
        or (pattern.endswith(".*") and action.startswith(pattern[:-1]))
    )


def _workspace_matches(patterns: Sequence[str], workspace: str | None) -> bool:
    return any(pattern == "*" or normalize_workspace(pattern) == workspace for pattern in patterns)


__all__ = [
    "ACTION_PRESETS",
    "AccessAction",
    "AccessControl",
    "AccessDeniedError",
    "AllowAllAccessControl",
    "JwtClaimsAccessControl",
    "access_control_from_config",
]
