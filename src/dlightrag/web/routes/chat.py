# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.access_control import AccessAction
from dlightrag.utils import normalize_workspace
from dlightrag.utils.images import validate_web_images
from dlightrag.web.answer_events import stream_answer_events
from dlightrag.web.attachment_models import MAX_CURRENT_DOCUMENTS, MAX_DOCUMENT_BYTES
from dlightrag.web.attachment_requests import parse_web_answer_request
from dlightrag.web.conversations import WebConversationService
from dlightrag.web.deps import (
    enforce_web_access,
    filter_web_workspace_records,
    get_manager,
    get_request_scope,
    get_web_conversation_service,
    get_workspace,
    templates,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    try:
        workspaces = await manager.alist_workspace_records()
    except Exception:
        workspaces = [
            {
                "workspace": workspace,
                "display_name": workspace,
                "embedding_model": manager.config.embedding.model,
            }
        ]
    workspaces = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_QUERY,
        workspaces,
    )

    authorized = [normalize_workspace(str(row["workspace"])) for row in workspaces]
    known = set(authorized)
    active_raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in active_raw.split(",") if item.strip()]
    active = [item for item in active if item in known]

    primary = normalize_workspace(request.cookies.get("dlightrag_workspace", workspace))
    if not active:
        active = authorized
    if primary not in known:
        primary = "default" if "default" in known else (authorized[0] if authorized else "")

    capability = manager.answer_image_capability
    if capability is None:
        capability_status = "unknown"
        effective_current_upload_limit = 0
    else:
        capability_status = capability.status
        effective_current_upload_limit = min(
            manager.config.query_images.max_current_images,
            capability.effective_max_images,
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "workspace": workspace,
            "workspaces": workspaces,
            "primary_workspace": primary,
            "active_workspaces": active,
            "query_image_max_upload_bytes": manager.config.query_images.max_upload_bytes,
            "answer_image_capability_status": capability_status,
            "effective_current_upload_limit": effective_current_upload_limit,
            "query_document_max_upload_bytes": MAX_DOCUMENT_BYTES,
            "query_document_current_upload_limit": MAX_CURRENT_DOCUMENTS,
        },
    )


@router.post("/answer")
async def answer_stream(
    request: Request,
    workspace: str = Depends(get_workspace),
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
):
    """Stream answer via SSE, then swap in enriched citations."""
    manager = get_manager(request)
    cfg = manager.config
    body = await parse_web_answer_request(
        request,
        max_images=cfg.query_images.max_current_images,
        max_image_upload_bytes=cfg.query_images.max_upload_bytes,
    )

    query = body.query
    if not query:
        return HTMLResponse("<span>Please enter a question.</span>")

    prepared_conversation = await conversation_service.prepare_answer(
        getattr(request.state, "user_context", None),
        str(body.conversation_id),
        str(body.submission_id),
    )
    if prepared_conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Extract workspaces (multi-select from frontend).
    workspaces = body.workspaces
    target_workspaces = workspaces or [workspace]
    for ws in target_workspaces:
        await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, ws)
    downloadable_records = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
        [{"workspace": ws} for ws in target_workspaces],
    )
    downloadable_workspaces = {
        normalize_workspace(str(record["workspace"])) for record in downloadable_records
    }
    scope = get_request_scope(request, target_workspaces)

    try:
        validated_images = validate_web_images(
            body.images,
            max_images=cfg.query_images.max_current_images,
            max_bytes=cfg.query_images.max_upload_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Planning runs lazily inside the stream (under the request-root span), not
    # here: this keeps query_planning nested in the answer_stream_pipeline trace
    # and lets an already-committed (duplicate) submission replay without
    # re-planning. The handler stays synchronous request gating only.
    return StreamingResponse(
        stream_answer_events(
            manager=manager,
            cfg=cfg,
            query=query,
            workspaces=workspaces,
            workspace=workspace,
            scope=scope,
            downloadable_workspaces=downloadable_workspaces,
            conversation_service=conversation_service,
            prepared_conversation=prepared_conversation,
            validated_images=validated_images,
            submission_id=str(body.submission_id),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
