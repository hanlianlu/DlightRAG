# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace lifecycle API routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from dlightrag.access import AccessAction, UserContext
from dlightrag.api.auth import get_current_user
from dlightrag.api.models import (
    WorkspaceCreateRequest,
    WorkspaceCreateResponse,
    WorkspaceDeleteResponse,
    WorkspacesResponse,
)
from dlightrag.rag.workspaces import normalize_workspace
from dlightrag.services.corpora import validate_workspace_name

from .deps import enforce_access, filter_workspace_records, get_application

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
    application = get_application(request)
    records = await application.corpora.alist_workspace_records()
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
    application = get_application(request)
    workspace, display_name = _normalize_create_body(body)
    await enforce_access(request, user, AccessAction.WORKSPACE_CREATE, workspace=workspace)
    existing = await application.corpora.list_workspaces()
    if workspace in existing:
        raise HTTPException(status_code=409, detail=f"Workspace '{display_name}' already exists")

    await application.corpora.create_workspace(workspace, display_name=display_name)
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
    application = get_application(request)
    try:
        label = validate_workspace_name(workspace)
        normalized = normalize_workspace(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await enforce_access(request, user, AccessAction.WORKSPACE_DELETE, workspace=normalized)
    result = await application.corpora.reset(
        workspace_ids=(normalized,),
        keep_files=keep_files,
        dry_run=dry_run,
    )
    return {
        "workspace": normalized,
        "deleted": not dry_run,
        "result": result,
    }
