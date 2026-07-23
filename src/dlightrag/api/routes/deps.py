# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Common dependencies for API routes."""

from typing import Any

from fastapi import HTTPException, Request

from dlightrag.access_control import AccessDeniedError, access_control_from_config
from dlightrag.app_state import request_config
from dlightrag.config import get_config
from dlightrag.core import access as core_access
from dlightrag.core.request.workspaces import (
    NoQueryableWorkspacesError,
    WorkspaceSelectionConflictError,
)
from dlightrag.core.scope import RequestScope
from dlightrag.core.servicemanager import RAGServiceManager


def get_manager(request: Request) -> RAGServiceManager:
    return request.app.state.manager


def resolve_workspace(ws: str | None, request: Request | None = None) -> str:
    from dlightrag.utils import normalize_workspace

    workspace = request_config(request).workspace if request is not None else get_config().workspace
    return normalize_workspace(ws or workspace)


def request_scope(
    user: object | None, workspaces: list[str] | tuple[str, ...] | None
) -> RequestScope:
    return RequestScope.from_user(user).for_workspaces(workspaces)


def get_access_control(request: Request):
    return getattr(request.app.state, "access_control", None) or access_control_from_config(
        request_config(request)
    )


async def enforce_access(
    request: Request,
    user: object,
    action: str,
    *,
    workspace: str | None = None,
) -> None:
    try:
        await get_access_control(request).check(user, action, workspace=workspace)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None


async def filter_workspace_records(
    request: Request,
    user: object,
    action: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return await core_access.filter_workspace_records(
        get_access_control(request), user, action, records
    )


async def resolve_authorized_query_workspaces(
    request: Request,
    user: object,
    *,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve query targets after applying the caller's existing ACL."""
    try:
        return await core_access.resolve_authorized_query_workspaces(
            get_access_control(request),
            user,
            get_manager(request),
            default_workspace=request_config(request).workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
    except NoQueryableWorkspacesError:
        raise HTTPException(
            status_code=403,
            detail="No workspaces are available for query",
        ) from None
    except WorkspaceSelectionConflictError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
