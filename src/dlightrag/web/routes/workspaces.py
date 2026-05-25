# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for workspace management."""

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from dlightrag.web.deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


def _error_response(message: str, status_code: int = 400) -> HTMLResponse:
    return HTMLResponse(
        _render_partial("partials/error.html", message=message),
        status_code=status_code,
    )


@router.get("/workspaces", response_class=HTMLResponse)
async def workspace_list(request: Request, workspace: str = Depends(get_workspace)):
    """Return workspace list fragment."""
    manager = get_manager(request)
    try:
        workspaces = await manager.list_workspaces()
    except Exception:
        workspaces = ["default"]

    return templates.TemplateResponse(
        request,
        "partials/workspace_list.html",
        {"workspaces": workspaces, "current_workspace": workspace},
    )


_WS_FORBIDDEN_RE = re.compile(r'[/\\<>"\']')


@router.post("/workspaces/create", response_class=HTMLResponse)
async def create_workspace(
    request: Request,
    workspace_name: str = Form(default=""),
    workspace: str = Depends(get_workspace),
):
    """Create a new workspace and return updated workspace list."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    name = workspace_name.strip()

    # Validation
    if not name:
        return _error_response("Workspace name cannot be empty")
    if len(name) > 64:
        return _error_response("Workspace name too long (max 64 characters)")
    if _WS_FORBIDDEN_RE.search(name):
        return _error_response("Workspace name contains forbidden characters")

    ws = normalize_workspace(name)

    # Duplicate check
    existing = await manager.list_workspaces()
    if ws in existing:
        return _error_response(f"Workspace '{name}' already exists", status_code=409)

    # Initialize workspace (creates the RAGService)
    try:
        await manager.acreate_workspace(ws)
    except Exception:
        logger.exception("Workspace creation failed")
        return _error_response(
            "Failed to create workspace; see server logs for details.",
            status_code=500,
        )

    # Return updated workspace list
    workspaces = await manager.list_workspaces()
    body = _render_partial(
        "partials/workspace_list.html",
        workspaces=workspaces,
        current_workspace=workspace,
    )
    return HTMLResponse(
        body,
        headers={"HX-Trigger": json.dumps({"workspaceCreated": {"workspace": ws}})},
    )


@router.post("/workspaces/delete", response_class=HTMLResponse)
async def delete_workspace(
    request: Request,
    workspace_name: str = Form(default=""),
    confirm_name: str = Form(default=""),
):
    """Delete a workspace after type-to-confirm verification."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    name = workspace_name.strip()
    confirm = confirm_name.strip()

    if not name:
        return _error_response("Workspace name cannot be empty")
    if normalize_workspace(name) != normalize_workspace(confirm):
        return _error_response("Confirmation name does not match")

    ws = normalize_workspace(name)

    try:
        await manager.areset(workspace=ws)
    except Exception:
        logger.exception("Workspace deletion failed")
        return _error_response(
            "Failed to delete workspace; see server logs for details.",
            status_code=500,
        )

    # Return updated workspace list, falling back to first available
    workspaces = await manager.list_workspaces()
    fallback = workspaces[0] if workspaces else "default"

    # Switch cookie to fallback workspace
    body = _render_partial(
        "partials/workspace_list.html",
        workspaces=workspaces,
        current_workspace=fallback,
    )
    return HTMLResponse(
        body,
        headers={"HX-Trigger": json.dumps({"workspaceDeleted": {"workspace": ws}})},
    )
