# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.access_control import AccessAction
from dlightrag.core.access import workspace_names
from dlightrag.utils import normalize_workspace
from dlightrag.web.answer_events import stream_answer_events
from dlightrag.web.attachment_models import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from dlightrag.web.attachment_requests import parse_web_answer_request
from dlightrag.web.conversations import WebConversationService
from dlightrag.web.deps import (
    enforce_web_access,
    filter_web_workspace_records,
    get_manager,
    get_web_conversation_service,
    get_workspace,
    templates,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""

    manager = get_manager(request)
    await manager._maybe_reprobe_answer_image_capability()
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
        effective_current_upload_limit = capability.effective_max_images
    document_extensions = sorted(SUPPORTED_DOCUMENT_EXTENSIONS)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "workspace": workspace,
            "workspaces": workspaces,
            "primary_workspace": primary,
            "active_workspaces": active,
            "query_attachment_count_limit": manager.config.answer.max_attachments,
            "query_attachment_image_max_bytes": manager.config.answer.max_attachment_bytes,
            "query_attachment_document_max_bytes": manager.config.answer.max_attachment_bytes,
            "query_attachment_extensions": document_extensions,
            "query_attachment_image_capability": capability_status,
            "query_attachment_image_limit": effective_current_upload_limit,
            "query_attachment_accept": ",".join(
                ["image/*", *(f".{extension}" for extension in document_extensions)]
            ),
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
    # Enforce the probed answer capability at admission (pre-stream 4xx).
    if "multipart/form-data" in request.headers.get("content-type", "").lower():
        await manager._maybe_reprobe_answer_image_capability()
    capability = manager.answer_image_capability
    body = await parse_web_answer_request(
        request,
        max_attachments=cfg.answer.max_attachments,
        max_attachment_bytes=cfg.answer.max_attachment_bytes,
        max_total_attachment_bytes=cfg.answer.max_total_attachment_bytes,
        image_max_pixels=cfg.answer.image_max_pixels,
        answer_image_capability=capability,
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
    projection_workspaces = target_workspaces
    committed = prepared_conversation.committed_submission
    if committed is not None and committed.queried_workspaces:
        projection_workspaces = list(committed.queried_workspaces)
    downloadable_records = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
        [{"workspace": ws} for ws in projection_workspaces],
    )
    downloadable_workspaces = workspace_names(downloadable_records)
    visual_records = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
        [{"workspace": ws} for ws in projection_workspaces],
    )
    visual_workspaces = workspace_names(visual_records)

    # Planning runs lazily inside the stream (under the request-root span), not
    # here: this keeps retrieval_planning nested in the answer_pipeline trace
    # and lets an already-committed (duplicate) submission replay without
    # re-planning. The handler stays synchronous request gating only.
    return StreamingResponse(
        stream_answer_events(
            manager=manager,
            cfg=cfg,
            query=query,
            workspaces=workspaces,
            workspace=workspace,
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            conversation_service=conversation_service,
            prepared_conversation=prepared_conversation,
            validated_attachments=body.attachments,
            submission_id=str(body.submission_id),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
