# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from dlightrag.core.client_contracts import dump_optional_list
from dlightrag.web.answer_events import stream_answer_events
from dlightrag.web.deps import get_manager, get_request_scope, get_workspace, templates
from dlightrag.web.requests import WebAnswerRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/history")
async def get_checkpoint_history(
    request: Request,
    session_id: str = Query(default=""),
    workspace: str = Depends(get_workspace),
):
    """Return conversation history for a session from the checkpoint store.

    Reconstructs the scoped session key using the current request scope
    so that the same session_id returns the same history across page
    refreshes and browser restarts.
    """
    if not session_id:
        return JSONResponse({"history": []})

    manager = get_manager(request)
    scope = get_request_scope(request, [workspace])

    try:
        history = await manager.aget_checkpoint_history(
            session_id=session_id,
            scope=scope,
        )
        return JSONResponse({"history": history})
    except Exception:
        logger.debug("Failed to load checkpoint history for session %s", session_id, exc_info=True)
        return JSONResponse({"history": []})


@router.delete("/history")
async def delete_checkpoint_history(
    request: Request,
    session_id: str = Query(default=""),
    workspace: str = Depends(get_workspace),
):
    """Delete all conversation history checkpoints for a session."""
    if not session_id:
        return JSONResponse({"deleted": 0})

    manager = get_manager(request)
    scope = get_request_scope(request, [workspace])

    try:
        deleted = await manager.adelete_checkpoint_session(
            session_id=session_id,
            scope=scope,
        )
        return JSONResponse({"deleted": deleted})
    except Exception:
        logger.debug(
            "Failed to delete checkpoint history for session %s", session_id, exc_info=True
        )
        return JSONResponse({"deleted": 0})


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    try:
        workspaces = await manager.list_workspace_records()
    except Exception:
        workspaces = [
            {
                "workspace": workspace,
                "display_name": workspace,
                "embedding_model": manager.config.embedding.model,
            }
        ]

    known = {str(row["workspace"]) for row in workspaces}
    active_raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in active_raw.split(",") if item.strip()]
    active = [item for item in active if item in known]

    primary = normalize_workspace(request.cookies.get("dlightrag_workspace", workspace))
    if not active and primary in known:
        active = [primary]
    if not active and "default" in known:
        active = ["default"]

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "workspace": workspace,
            "workspaces": workspaces,
            "active_workspaces": active,
        },
    )


@router.post("/answer")
async def answer_stream(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Stream answer via SSE, then swap in enriched citations."""
    try:
        body = WebAnswerRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    query = body.query
    if not query:
        return HTMLResponse("<span>Please enter a question.</span>")

    conversation_history = dump_optional_list(body.conversation_history)

    # Extract images (base64 from frontend). Raw base64 is passed to the
    # manager; it handles answer budgeting, semantic VLM enhancement, session
    # memory, and direct visual retrieval.
    images_b64 = body.images
    clean_images: list[str] = []
    if images_b64:
        for b64 in images_b64[:3]:  # enforce 3-image limit
            try:
                img_bytes = base64.b64decode(b64)
                if len(img_bytes) > 10 * 1024 * 1024:  # 10MB server-side limit
                    logger.warning("Skipping oversized image (%d bytes)", len(img_bytes))
                    continue
                clean_images.append(b64)
            except Exception:
                logger.warning("Failed to decode uploaded image", exc_info=True)

    # Extract workspaces (multi-select from frontend).
    workspaces = body.workspaces
    session_id = body.session_id
    scope = get_request_scope(request, workspaces or [workspace])

    manager = get_manager(request)
    cfg = manager.config

    return StreamingResponse(
        stream_answer_events(
            manager=manager,
            cfg=cfg,
            query=query,
            conversation_history=conversation_history,
            workspaces=workspaces,
            workspace=workspace,
            query_images=clean_images,
            session_id=session_id,
            scope=scope,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
