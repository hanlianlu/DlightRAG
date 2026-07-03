# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Common dependencies for API routes."""

from typing import Any

from fastapi import HTTPException, Request

from dlightrag.access_control import AccessDeniedError, access_control_from_config
from dlightrag.config import get_config
from dlightrag.core.scope import RequestScope
from dlightrag.core.servicemanager import RAGServiceManager


def get_manager(request: Request) -> RAGServiceManager:
    return request.app.state.manager


def resolve_workspace(ws: str | None) -> str:
    from dlightrag.utils import normalize_workspace

    return normalize_workspace(ws or get_config().workspace)


def request_scope(
    user: object | None, workspaces: list[str] | tuple[str, ...] | None
) -> RequestScope:
    return RequestScope.from_user(user).for_workspaces(workspaces)


def get_access_control(request: Request):
    return getattr(request.app.state, "access_control", None) or access_control_from_config(
        get_config()
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


async def enforce_workspaces_access(
    request: Request,
    user: object,
    action: str,
    workspaces: list[str] | tuple[str, ...] | None,
) -> None:
    targets = workspaces or [get_config().workspace]
    for workspace in targets:
        await enforce_access(request, user, action, workspace=resolve_workspace(workspace))


async def filter_workspace_records(
    request: Request,
    user: object,
    action: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    workspaces = [str(row["workspace"]) for row in records]
    allowed = set(await get_access_control(request).filter_workspaces(user, action, workspaces))
    return [row for row in records if str(row["workspace"]) in allowed]
