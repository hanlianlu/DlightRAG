# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorized workspace selection and catalog filtering."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NotRequired, Protocol, Required, TypedDict

from dlightrag.application.access.control import AccessAction, AccessControl, AccessSubject


class WorkspaceSelectionConflictError(ValueError):
    """Raised when all-workspace and explicit selection conflict."""


class NoQueryableWorkspacesError(ValueError):
    """Raised when all-workspace selection has no available candidates."""


class WorkspaceRecord(TypedDict):
    """Canonical workspace facts shared with authorization callers."""

    workspace: Required[str]
    display_name: NotRequired[str | None]
    embedding_model: NotRequired[str]
    created_at: NotRequired[str | None]
    updated_at: NotRequired[str | None]


class WorkspaceCatalog(Protocol):
    """Enumerate the workspace records Access may authorize."""

    async def alist_workspace_records(self) -> Sequence[WorkspaceRecord]: ...


def validate_query_workspace_selection(
    *,
    all_workspaces: bool,
    workspace: str | None = None,
    workspaces: Sequence[str] | None = None,
) -> None:
    has_singular = bool(workspace and workspace.strip())
    if all_workspaces and (has_singular or bool(workspaces)):
        raise WorkspaceSelectionConflictError(
            "all_workspaces cannot be combined with an explicit workspace selection"
        )


def _deduplicate_workspace_ids(workspaces: Iterable[str]) -> list[str]:
    """De-duplicate canonical workspace ids without changing their order."""
    seen: set[str] = set()
    result: list[str] = []
    for workspace in workspaces:
        if workspace and workspace not in seen:
            seen.add(workspace)
            result.append(workspace)
    return result


def resolve_query_workspaces(
    *,
    default_workspace: str,
    workspace: str | None = None,
    workspaces: Sequence[str] | None = None,
    all_workspaces: bool = False,
    available_workspaces: Sequence[str] | None = None,
) -> list[str]:
    """Resolve one concrete normalized workspace list for a query."""
    validate_query_workspace_selection(
        all_workspaces=all_workspaces,
        workspace=workspace,
        workspaces=workspaces,
    )
    if all_workspaces:
        if available_workspaces is None:
            raise ValueError("available_workspaces is required when all_workspaces is true")
        resolved = _deduplicate_workspace_ids(available_workspaces)
        if not resolved:
            raise NoQueryableWorkspacesError("No workspaces are available for query")
        return resolved

    requested = list(workspaces) if workspaces else [workspace or default_workspace]
    resolved = _deduplicate_workspace_ids(requested)
    if not resolved:
        raise WorkspaceSelectionConflictError("At least one query workspace is required")
    return resolved


@dataclass(frozen=True, slots=True)
class AccessGate:
    """Bind one policy and principal for a transport request."""

    access_control: AccessControl
    subject: AccessSubject

    async def check(self, action: str, *, workspace: str | None = None) -> None:
        await self.access_control.check(self.subject, action, workspace=workspace)

    async def filter_workspace_records(
        self,
        action: str,
        records: Sequence[WorkspaceRecord],
    ) -> list[WorkspaceRecord]:
        workspaces = [record["workspace"] for record in records]
        allowed = set(await self.access_control.filter_workspaces(self.subject, action, workspaces))
        return [record for record in records if record["workspace"] in allowed]

    async def authorized_workspace_ids(
        self,
        action: str,
        workspace_ids: Iterable[str],
    ) -> set[str]:
        allowed = await self.access_control.filter_workspaces(
            self.subject,
            action,
            _deduplicate_workspace_ids(workspace_ids),
        )
        return set(allowed)

    async def resolve_query_workspaces(
        self,
        catalog: WorkspaceCatalog,
        *,
        default_workspace: str,
        workspaces: list[str] | None,
        all_workspaces: bool,
    ) -> list[str]:
        available: list[str] | None = None
        if all_workspaces:
            visible = await self.filter_workspace_records(
                AccessAction.WORKSPACE_QUERY,
                await catalog.alist_workspace_records(),
            )
            available = [record["workspace"] for record in visible]

        resolved = resolve_query_workspaces(
            default_workspace=default_workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            available_workspaces=available,
        )
        if not all_workspaces:
            for workspace in resolved:
                await self.check(AccessAction.WORKSPACE_QUERY, workspace=workspace)
        return resolved


__all__ = [
    "AccessGate",
    "NoQueryableWorkspacesError",
    "WorkspaceCatalog",
    "WorkspaceRecord",
    "WorkspaceSelectionConflictError",
    "resolve_query_workspaces",
    "validate_query_workspace_selection",
]
