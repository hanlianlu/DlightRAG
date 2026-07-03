# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace lifecycle API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from dlightrag.access_control import AccessAction
from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import (
    WorkspaceCreateRequest,
    WorkspaceCreateResponse,
    WorkspaceDeleteResponse,
    WorkspacesResponse,
)
from dlightrag.utils import normalize_workspace, validate_workspace_name

from .deps import enforce_access, filter_workspace_records, get_manager

router = APIRouter()


def _normalize_create_body(body: WorkspaceCreateRequest) -> tuple[str, str]:
    """Return internal workspace id and display name for a create request."""
    try:
        label = validate_workspace_name(body.workspace)
        display_name = validate_workspace_name(body.display_name or label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return normalize_workspace(label), display_name


@router.get("/workspaces", response_model=WorkspacesResponse)
async def list_workspaces(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """List all registered workspaces."""
    manager = get_manager(request)
    records = await manager.list_workspace_records()
    records = await filter_workspace_records(request, user, AccessAction.WORKSPACE_QUERY, records)
    return {
        "workspaces": [row["workspace"] for row in records],
        "records": records,
    }


@router.post(
    "/workspaces",
    status_code=status.HTTP_201_CREATED,
    response_model=WorkspaceCreateResponse,
)
async def create_workspace(
    body: WorkspaceCreateRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Create an empty workspace in the durable registry."""
    manager = get_manager(request)
    workspace, display_name = _normalize_create_body(body)
    await enforce_access(request, user, AccessAction.WORKSPACE_CREATE, workspace=workspace)
    existing = await manager.list_workspaces()
    if workspace in existing:
        raise HTTPException(status_code=409, detail=f"Workspace '{display_name}' already exists")

    await manager.acreate_workspace(workspace, display_name=display_name)
    return {
        "workspace": workspace,
        "display_name": display_name,
        "created": True,
    }


@router.delete("/workspaces/{workspace}", response_model=WorkspaceDeleteResponse)
async def delete_workspace(
    workspace: str,
    request: Request,
    keep_files: bool = Query(default=False),
    dry_run: bool = Query(default=False),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete/reset one workspace and remove its registry row."""
    manager = get_manager(request)
    try:
        label = validate_workspace_name(workspace)
        normalized = normalize_workspace(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await enforce_access(request, user, AccessAction.WORKSPACE_DELETE, workspace=normalized)
    result = await manager.areset(
        workspace=label,
        keep_files=keep_files,
        dry_run=dry_run,
    )
    return {
        "workspace": normalized,
        "deleted": not dry_run,
        "result": result,
    }
