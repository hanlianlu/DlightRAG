# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for chat interface and answer generation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import ValidationError

from dlightrag.access_control import AccessAction
from dlightrag.utils import normalize_workspace
from dlightrag.utils.images import validate_web_images
from dlightrag.web.answer_events import stream_answer_events
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
from dlightrag.web.requests import WebAnswerRequest

logger = logging.getLogger(__name__)

router = APIRouter()
_WEB_ANSWER_JSON_OVERHEAD_BYTES = 64 * 1024


async def _read_limited_answer_body(
    request: Request, *, max_images: int, max_upload_bytes: int
) -> bytes:
    encoded_image_bytes = ((max_upload_bytes + 2) // 3) * 4
    max_body_bytes = _WEB_ANSWER_JSON_OVERHEAD_BYTES + max(0, max_images) * encoded_image_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_body_bytes:
                raise HTTPException(status_code=413, detail="Web answer request body is too large")
        except ValueError:
            # Malformed Content-Length header: ignore it and rely on the
            # streaming byte cap below to bound the request body size.
            pass
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_body_bytes:
            raise HTTPException(status_code=413, detail="Web answer request body is too large")
        body.extend(chunk)
    return bytes(body)


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
    try:
        raw_body = await _read_limited_answer_body(
            request,
            max_images=cfg.query_images.max_current_images,
            max_upload_bytes=cfg.query_images.max_upload_bytes,
        )
        body = WebAnswerRequest.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

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
    turn = await conversation_service.prepare_answer_turn(
        manager=manager,
        prepared=prepared_conversation,
        query=query,
        current_images=[image.model_block for image in validated_images],
        workspaces=target_workspaces,
    )

    return StreamingResponse(
        stream_answer_events(
            manager=manager,
            cfg=cfg,
            turn=turn,
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
