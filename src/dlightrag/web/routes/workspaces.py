# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for workspace management."""

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from dlightrag.web.deps import get_manager, templates

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


def _sanitize_cookie_value(value: str) -> str:
    """Keep only valid workspace name characters (alphanumeric, dash, underscore, comma)."""
    return re.sub(r"[^A-Za-z0-9,_-]", "", value).strip(",")


def _set_workspace_cookies(response: HTMLResponse, request: Request, workspaces: list[str]) -> None:
    primary = workspaces[0] if workspaces else "default"
    joined = ",".join(workspaces or [primary])
    secure = request.url.scheme == "https"
    response.set_cookie(
        key="dlightrag_workspace",
        value=_sanitize_cookie_value(primary),
        httponly=False,
        samesite="lax",
        secure=secure,
        path="/",
    )
    response.set_cookie(
        key="dlightrag_workspace_ids",
        value=_sanitize_cookie_value(joined),
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
    from dlightrag.utils import normalize_workspace, validate_workspace_name

    manager = get_manager(request)

    try:
        name = validate_workspace_name(workspace_name)
    except ValueError as exc:
        return _error_response(str(exc))

    ws = normalize_workspace(name)

    # Duplicate check
    existing = await manager.list_workspaces()
    if ws in existing:
        return _error_response(f"Workspace '{name}' already exists", status_code=409)

    # Initialize workspace (creates the RAGService)
    try:
        await manager.acreate_workspace(ws, display_name=name)
    except Exception:
        logger.exception("Workspace creation failed")
        return _error_response(
            "Failed to create workspace; see server logs for details.",
            status_code=500,
        )

    response = HTMLResponse(
        "",
        headers={
            "HX-Trigger": json.dumps({"workspaceCreated": {"workspace": ws, "display_name": name}})
        },
    )
    _set_workspace_cookies(response, request, [ws])
    return response


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

    workspaces = await manager.list_workspaces()
    fallback = workspaces[0] if workspaces else "default"

    response = HTMLResponse(
        "",
        headers={
            "HX-Trigger": json.dumps({"workspaceDeleted": {"workspace": ws, "fallback": fallback}})
        },
    )
    _set_workspace_cookies(response, request, [fallback])
    return response
