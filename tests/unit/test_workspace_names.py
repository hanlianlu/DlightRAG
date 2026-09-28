# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A caller-named workspace resolves to one canonical id the same way everywhere."""

import pytest
from fastapi import HTTPException

from dlightrag.adapters.http.rest.models import WorkspaceCreateRequest
from dlightrag.adapters.http.rest.routes.workspaces import _normalize_create_body
from dlightrag.adapters.mcp.contracts import CreateWorkspaceInput
from dlightrag.adapters.mcp.server import _normalize_workspace_argument
from dlightrag.application.corpus_admin import (
    WorkspaceNameError,
    workspace_id_for_name,
    workspace_ids_for_names,
)

_TOO_LONG_ONCE_PREFIXED = "1" * 64  # gains a leading underscore: 65 characters


@pytest.mark.parametrize(
    ("name", "workspace_id"),
    [("Finance Reports", "finance_reports"), ("2026", "_2026"), ("a" * 64, "a" * 64)],
)
def test_a_name_resolves_to_its_canonical_id(name: str, workspace_id: str) -> None:
    assert workspace_id_for_name(name) == workspace_id


@pytest.mark.parametrize("name", ["", "   ", _TOO_LONG_ONCE_PREFIXED])
def test_a_name_without_a_canonical_id_refuses_without_echoing_it(name: str) -> None:
    with pytest.raises(WorkspaceNameError) as refused:
        workspace_id_for_name(name)

    assert "1-64 letters, digits, or underscores" in str(refused.value)
    if name.strip():
        assert name not in str(refused.value)


def test_a_list_of_names_keeps_order_drops_repeats_and_refuses_a_blank() -> None:
    assert workspace_ids_for_names(["Finance", "legal", "finance"]) == ["finance", "legal"]
    with pytest.raises(WorkspaceNameError):
        workspace_ids_for_names(["finance", "  "])


def test_rest_create_refuses_a_name_whose_id_would_be_too_long() -> None:
    """Formerly a 500: the name validated, but its id did not."""
    with pytest.raises(HTTPException) as refused:
        _normalize_create_body(WorkspaceCreateRequest(workspace=_TOO_LONG_ONCE_PREFIXED))

    assert refused.value.status_code == 400
    assert _TOO_LONG_ONCE_PREFIXED not in str(refused.value.detail)


def test_mcp_create_refuses_the_same_name_as_rest() -> None:
    with pytest.raises(WorkspaceNameError):
        _normalize_workspace_argument(CreateWorkspaceInput(workspace=_TOO_LONG_ONCE_PREFIXED))
