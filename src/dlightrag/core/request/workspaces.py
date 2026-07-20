# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral query workspace selection."""

from collections.abc import Sequence
from typing import Self

from pydantic import model_validator

from dlightrag.core.client_contracts import ClientContractModel
from dlightrag.utils import normalize_workspace


class WorkspaceSelectionConflictError(ValueError):
    """Raised when all-workspace and explicit selection conflict."""


class NoQueryableWorkspacesError(ValueError):
    """Raised when all-workspace selection has no available candidates."""


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


def _normalize_unique(workspaces: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for workspace in workspaces:
        normalized = normalize_workspace(workspace)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
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
        resolved = _normalize_unique(available_workspaces)
        if not resolved:
            raise NoQueryableWorkspacesError("No workspaces are available for query")
        return resolved

    requested = list(workspaces) if workspaces else [workspace or default_workspace]
    resolved = _normalize_unique(requested)
    if not resolved:
        raise WorkspaceSelectionConflictError("At least one query workspace is required")
    return resolved


class QueryWorkspaceSelection(ClientContractModel):
    """Shared REST/MCP query workspace selector."""

    workspaces: list[str] | None = None
    all_workspaces: bool = False

    @model_validator(mode="after")
    def _validate_workspace_selection(self) -> Self:
        validate_query_workspace_selection(
            all_workspaces=self.all_workspaces,
            workspaces=self.workspaces,
        )
        return self


__all__ = [
    "NoQueryableWorkspacesError",
    "QueryWorkspaceSelection",
    "WorkspaceSelectionConflictError",
    "resolve_query_workspaces",
    "validate_query_workspace_selection",
]
