# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-agnostic workspace access-control orchestration.

REST, Web, and MCP each resolve their own access-control provider and
authenticated subject, then delegate the shared authorization logic here.
Surfaces own only provider/subject resolution and the mapping of the raised
domain errors (:class:`~dlightrag.access_control.AccessDeniedError`,
:class:`~dlightrag.core.request.workspaces.NoQueryableWorkspacesError`,
:class:`~dlightrag.core.request.workspaces.WorkspaceSelectionConflictError`)
onto their transport.
"""

from typing import Any, Protocol

from dlightrag.access_control import AccessAction, AccessControl
from dlightrag.core.request.workspaces import resolve_query_workspaces


class WorkspaceRecordLister(Protocol):
    """Minimal manager contract needed to enumerate workspace records."""

    async def alist_workspace_records(self) -> list[dict[str, Any]]: ...


async def filter_workspace_records(
    access_control: AccessControl,
    subject: Any,
    action: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only records whose workspace ``subject`` may access for ``action``."""
    workspaces = [str(row["workspace"]) for row in records]
    allowed = set(await access_control.filter_workspaces(subject, action, workspaces))
    return [row for row in records if str(row["workspace"]) in allowed]


async def resolve_authorized_query_workspaces(
    access_control: AccessControl,
    subject: Any,
    manager: WorkspaceRecordLister,
    *,
    default_workspace: str,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve concrete query targets after applying ``subject``'s ACL.

    Propagates :class:`NoQueryableWorkspacesError`,
    :class:`WorkspaceSelectionConflictError` and :class:`AccessDeniedError` so
    each surface can map them onto its own transport.
    """
    available: list[str] | None = None
    if all_workspaces:
        records = await manager.alist_workspace_records()
        visible = await filter_workspace_records(
            access_control, subject, AccessAction.WORKSPACE_QUERY, records
        )
        available = [str(row["workspace"]) for row in visible]

    resolved = resolve_query_workspaces(
        default_workspace=default_workspace,
        workspaces=workspaces,
        all_workspaces=all_workspaces,
        available_workspaces=available,
    )

    if not all_workspaces:
        # ``resolve_query_workspaces`` already normalized the list, and
        # ``check`` normalizes each workspace again internally.
        for workspace in resolved:
            await access_control.check(subject, AccessAction.WORKSPACE_QUERY, workspace=workspace)
    return resolved


__all__ = [
    "WorkspaceRecordLister",
    "filter_workspace_records",
    "resolve_authorized_query_workspaces",
]
