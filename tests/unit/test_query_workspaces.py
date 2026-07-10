# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for query workspace selection semantics."""

import pytest
from pydantic import ValidationError

from dlightrag.core.query_workspaces import (
    NoQueryableWorkspacesError,
    QueryWorkspaceSelection,
    WorkspaceSelectionConflictError,
    resolve_query_workspaces,
)


def test_query_workspace_selection_defaults_to_default_scope() -> None:
    selection = QueryWorkspaceSelection()

    assert selection.all_workspaces is False
    assert selection.workspaces is None
    assert resolve_query_workspaces(default_workspace="Default") == ["default"]


@pytest.mark.parametrize("workspaces", [None, []])
def test_all_workspaces_accepts_omitted_or_empty_explicit_list(
    workspaces: list[str] | None,
) -> None:
    selection = QueryWorkspaceSelection(
        all_workspaces=True,
        workspaces=workspaces,
    )

    assert selection.all_workspaces is True
    assert resolve_query_workspaces(
        default_workspace="default",
        all_workspaces=True,
        workspaces=workspaces,
        available_workspaces=["Research Notes", "default", "Research Notes"],
    ) == ["research_notes", "default"]


def test_all_workspaces_rejects_non_empty_explicit_list() -> None:
    with pytest.raises(ValidationError, match="all_workspaces"):
        QueryWorkspaceSelection(
            all_workspaces=True,
            workspaces=["finance"],
        )

    with pytest.raises(WorkspaceSelectionConflictError, match="all_workspaces"):
        resolve_query_workspaces(
            default_workspace="default",
            workspace="finance",
            all_workspaces=True,
            available_workspaces=["finance"],
        )


def test_all_workspaces_rejects_empty_available_set() -> None:
    with pytest.raises(NoQueryableWorkspacesError, match="No workspaces"):
        resolve_query_workspaces(
            default_workspace="default",
            all_workspaces=True,
            available_workspaces=[],
        )


def test_explicit_workspaces_are_normalized_and_stably_deduplicated() -> None:
    assert resolve_query_workspaces(
        default_workspace="default",
        workspaces=["Research Notes", "default", "research_notes"],
    ) == ["research_notes", "default"]
