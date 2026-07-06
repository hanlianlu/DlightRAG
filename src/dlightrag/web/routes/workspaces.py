# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for workspace management."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from dlightrag.access_control import AccessAction
from dlightrag.utils import normalize_workspace
from dlightrag.web.deps import (
    enforce_web_access,
    error_response,
    filter_web_workspace_records,
    get_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _ordered_unique(workspaces: list[str]) -> list[str]:
    result: list[str] = []
    for workspace in workspaces:
        if workspace and workspace not in result:
            result.append(workspace)
    return result


async def _visible_workspace_names(request: Request, manager: Any) -> list[str]:
    records = [{"workspace": workspace} for workspace in await manager.list_workspaces()]
    visible = await filter_web_workspace_records(request, AccessAction.WORKSPACE_QUERY, records)
    return [str(row["workspace"]) for row in visible]


def _default_workspace(workspaces: list[str]) -> str:
    if not workspaces:
        return ""
    return "default" if "default" in workspaces else workspaces[0]


def _cookie_active_workspaces(request: Request, visible_workspaces: list[str]) -> list[str]:
    visible = set(visible_workspaces)
    raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in raw.split(",") if item.strip()]
    return _ordered_unique([workspace for workspace in active if workspace in visible])


def _set_workspace_cookies(
    response: HTMLResponse,
    request: Request,
    visible_workspaces: list[str],
    *,
    active_workspaces: list[str] | None = None,
    primary_workspace: str | None = None,
) -> None:
    """Persist selector state using only canonical workspace names.

    All workspace values are server-trusted (sourced from the DB or
    normalized via ``normalize_workspace`` before reaching this function),
    so cookie values are set directly without runtime sanitization.
    """
    canonical_visible = _ordered_unique(
        [normalize_workspace(w) for w in visible_workspaces if normalize_workspace(w)]
    )
    visible = set(canonical_visible)
    if not visible:
        response.delete_cookie("dlightrag_workspace", path="/")
        response.delete_cookie("dlightrag_workspace_ids", path="/")
        return

    canonical_active = _ordered_unique(
        [
            normalize_workspace(w)
            for w in (active_workspaces or [])
            if normalize_workspace(w) in visible
        ]
    )
    active = canonical_active
    if not active:
        fallback = (
            normalize_workspace(primary_workspace)
            if primary_workspace and normalize_workspace(primary_workspace) in visible
            else _default_workspace(canonical_visible)
        )
        active = [fallback] if fallback else []
    primary = (
        normalize_workspace(primary_workspace)
        if primary_workspace and normalize_workspace(primary_workspace) in active
        else active[0]
    )
    joined = ",".join(active)
    secure = request.url.scheme == "https"
    response.set_cookie(
        key="dlightrag_workspace",
        value=primary,
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )
    response.set_cookie(
        key="dlightrag_workspace_ids",
        value=joined,
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )


@router.post("/workspaces/create", response_class=HTMLResponse)
async def create_workspace(
    request: Request,
    workspace_name: str = Form(default=""),
):
    """Create a new workspace and return updated workspace list."""
    from dlightrag.utils import validate_workspace_name

    manager = get_manager(request)

    try:
        name = validate_workspace_name(workspace_name)
    except ValueError as exc:
        return error_response(str(exc))

    ws = normalize_workspace(name)
    await enforce_web_access(request, AccessAction.WORKSPACE_CREATE, ws)

    # Duplicate check
    existing = await manager.list_workspaces()
    if ws in existing:
        return error_response(f"Workspace '{name}' already exists", status_code=409)

    # Initialize workspace (creates the RAGService)
    try:
        await manager.acreate_workspace(ws, display_name=name)
    except Exception:
        logger.exception("Workspace creation failed")
        return error_response(
            "Failed to create workspace; see server logs for details.",
            status_code=500,
        )

    response = HTMLResponse(
        "",
        headers={
            "HX-Trigger": json.dumps({"workspaceCreated": {"workspace": ws, "display_name": name}})
        },
    )
    visible_workspaces = await _visible_workspace_names(request, manager)
    _set_workspace_cookies(
        response,
        request,
        visible_workspaces,
        active_workspaces=[ws],
        primary_workspace=ws,
    )
    return response


@router.post("/workspaces/delete", response_class=HTMLResponse)
async def delete_workspace(
    request: Request,
    workspace_name: str = Form(default=""),
    confirm_name: str = Form(default=""),
):
    """Delete a workspace after type-to-confirm verification."""
    manager = get_manager(request)
    name = workspace_name.strip()
    confirm = confirm_name.strip()

    if not name:
        return error_response("Workspace name cannot be empty")
    if normalize_workspace(name) != normalize_workspace(confirm):
        return error_response("Confirmation name does not match")

    ws = normalize_workspace(name)
    await enforce_web_access(request, AccessAction.WORKSPACE_DELETE, ws)

    try:
        await manager.areset(workspace=ws)
    except Exception:
        logger.exception("Workspace deletion failed")
        return error_response(
            "Failed to delete workspace; see server logs for details.",
            status_code=500,
        )

    visible_workspaces = await _visible_workspace_names(request, manager)
    active = _cookie_active_workspaces(request, visible_workspaces)
    next_workspace = active[0] if active else _default_workspace(visible_workspaces)

    response = HTMLResponse(
        "",
        headers={
            "HX-Trigger": json.dumps(
                {
                    "workspaceDeleted": {
                        "workspace": ws,
                        "next_workspace": next_workspace,
                    }
                }
            )
        },
    )
    _set_workspace_cookies(
        response,
        request,
        visible_workspaces,
        active_workspaces=active or ([next_workspace] if next_workspace else []),
        primary_workspace=next_workspace,
    )
    return response
