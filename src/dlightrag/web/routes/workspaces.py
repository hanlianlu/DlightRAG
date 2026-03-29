# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for workspace management."""

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from dlightrag.web.deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


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


@router.post("/workspaces/switch")
async def switch_workspace(request: Request):
    """Switch workspace via cookie."""
    form = await request.form()
    workspace = str(form.get("workspace", "default"))
    response = RedirectResponse(url="/web/", status_code=303)
    response.set_cookie("dlightrag_workspace", workspace, httponly=True)
    return response


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
        return HTMLResponse("Workspace name cannot be empty", status_code=400)
    if len(name) > 64:
        return HTMLResponse("Workspace name too long (max 64 characters)", status_code=400)
    if _WS_FORBIDDEN_RE.search(name):
        return HTMLResponse("Workspace name contains forbidden characters", status_code=400)

    ws = normalize_workspace(name)

    # Duplicate check
    existing = await manager.list_workspaces()
    if ws in existing:
        return HTMLResponse(f"Workspace '{name}' already exists", status_code=409)

    # Initialize workspace (creates the RAGService)
    try:
        await manager._get_service(ws)
    except Exception as e:
        logger.exception("Workspace creation failed")
        return HTMLResponse(f"Failed to create workspace: {e}", status_code=500)

    # Return updated workspace list
    workspaces = await manager.list_workspaces()
    html = _render_partial(
        "partials/workspace_list.html",
        workspaces=workspaces,
        current_workspace=workspace,
    )
    return HTMLResponse(
        html,
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
        return HTMLResponse("Workspace name cannot be empty", status_code=400)
    if normalize_workspace(name) != normalize_workspace(confirm):
        return HTMLResponse("Confirmation name does not match", status_code=400)

    ws = normalize_workspace(name)

    try:
        await manager.areset(workspace=ws)
    except Exception as e:
        logger.exception("Workspace deletion failed")
        return HTMLResponse(f"Failed to delete workspace: {e}", status_code=500)

    # Return updated workspace list, falling back to first available
    workspaces = await manager.list_workspaces()
    fallback = workspaces[0] if workspaces else "default"

    # Switch cookie to fallback workspace
    html = _render_partial(
        "partials/workspace_list.html",
        workspaces=workspaces,
        current_workspace=fallback,
    )
    return HTMLResponse(
        html,
        headers={"HX-Trigger": json.dumps({"workspaceDeleted": {"workspace": ws}})},
    )
