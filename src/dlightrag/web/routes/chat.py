# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for the chat interface and durable answer runs."""

import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from dlightrag_rag.workspaces import normalize_workspace, normalize_workspace_ids
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.access import AccessAction, WorkspaceRecord, owner_id_from_user
from dlightrag.api.answer_stream import follow_run_frames, resume_cursor
from dlightrag.runtime import IdempotencyKeyConflict
from dlightrag.web.answer_events import browser_frame
from dlightrag.web.attachment_models import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from dlightrag.web.attachment_requests import parse_web_answer_request
from dlightrag.web.conversation_models import AnswerRunDescriptor, ConversationTurn
from dlightrag.web.conversations import (
    ConversationSubmissionConflict,
    WebAnswerSubmission,
    WebConversationService,
    project_conversation_turn,
)
from dlightrag.web.deps import (
    enforce_web_access,
    filter_web_workspace_records,
    get_manager,
    get_web_access_gate,
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
    workspaces: list[WorkspaceRecord]
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

    authorized = [row["workspace"] for row in workspaces]
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


@router.post("/answer", status_code=202, response_model=AnswerRunDescriptor)
async def start_answer_run(
    request: Request,
    workspace: str = Depends(get_workspace),
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AnswerRunDescriptor:
    """Accept one submission as a durable run linked to its conversation entry.

    The run, its uploaded bytes, and the conversation turn are committed in one
    transaction before this descriptor is returned, so the browser follows the
    run's own event stream and a page reload rediscovers it from history.
    """
    manager = get_manager(request)
    cfg = manager.config
    # Enforce the probed answer capability at admission (pre-acceptance 4xx).
    if "multipart/form-data" in request.headers.get("content-type", "").lower():
        await manager._maybe_reprobe_answer_image_capability()
    body = await parse_web_answer_request(
        request,
        max_attachments=cfg.answer.max_attachments,
        max_attachment_bytes=cfg.answer.max_attachment_bytes,
        max_total_attachment_bytes=cfg.answer.max_total_attachment_bytes,
        image_max_pixels=cfg.answer.image_max_pixels,
        answer_image_capability=manager.answer_image_capability,
    )
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="A question is required")

    target_workspaces = normalize_workspace_ids(body.workspaces or [workspace])
    for ws in target_workspaces:
        await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, ws)

    try:
        submission = await conversation_service.start_answer(
            getattr(request.state, "user_context", None),
            conversation_id=str(body.conversation_id),
            submission_id=str(body.submission_id),
            query=query,
            workspaces=target_workspaces,
            attachments=body.attachments,
        )
    except ConversationSubmissionConflict, IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="This submission id was already used for a different request",
        ) from None
    if submission is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return answer_run_descriptor(submission)


@router.get("/answer/{run_id}", response_model=ConversationTurn)
async def answer_run_status(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationTurn:
    """Return one owned run's conversation entry and authoritative state."""
    turn = await conversation_service.turn_for_run(
        getattr(request.state, "user_context", None), run_id
    )
    if turn is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    downloadable, visual = await _projection_workspaces(request, turn.run.request)
    return project_conversation_turn(
        turn, downloadable_workspaces=downloadable, visual_workspaces=visual
    )


@router.delete("/answer/{run_id}", response_model=ConversationTurn)
async def cancel_answer_run(
    run_id: str,
    request: Request,
    response: Response,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationTurn:
    """Request cancellation of one owned run; repeating it is a no-op."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    outcome = await get_manager(request).acancel_answer_run(
        owner_id=owner_id_from_user(user), run_id=run_id
    )
    if outcome.run is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    # 202 only while a running worker still has to observe the request.
    response.status_code = 202 if outcome.outcome == "pending" else 200
    downloadable, visual = await _projection_workspaces(request, outcome.run.request)
    return project_conversation_turn(
        replace(turn, run=outcome.run),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
    )


@router.get("/answer/{run_id}/events")
async def answer_run_events(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> StreamingResponse:
    """Replay this run's durable events from a cursor, then follow it live."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    if turn.run.events_trimmed_at is not None:
        raise HTTPException(
            status_code=410,
            detail="Answer run events expired; read its result from the conversation",
        )
    downloadable, visual = await _projection_workspaces(request, turn.run.request)
    events = await get_manager(request).asubscribe_answer_run(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        after_sequence=resume_cursor(request),
    )
    return StreamingResponse(
        follow_run_frames(
            events,
            partial(
                browser_frame,
                downloadable_workspaces=downloadable,
                visual_workspaces=visual,
            ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def answer_run_descriptor(submission: WebAnswerSubmission) -> AnswerRunDescriptor:
    """Project the owner-scoped links one accepted submission is followed by."""
    run_id = submission.run.run_id
    return AnswerRunDescriptor(
        run_id=run_id,
        status=submission.run.status,
        cancel_requested=submission.run.cancel_requested,
        turn_id=submission.turn_id,
        turn_number=submission.turn_number,
        submission_id=str(submission.run.idempotency_key or ""),
        events_url=f"/web/answer/{run_id}/events",
        status_url=f"/web/answer/{run_id}",
        cancel_url=f"/web/answer/{run_id}",
        conversation=submission.conversation,
    )


async def _projection_workspaces(
    request: Request, run_request: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    """Authorize this reader's source downloads and visuals for one run."""
    workspace_ids = [str(value) for value in run_request.get("workspaces") or ()]
    gate = get_web_access_gate(request)
    return (
        await gate.authorized_workspace_ids(
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            workspace_ids,
        ),
        await gate.authorized_workspace_ids(
            AccessAction.WORKSPACE_READ_VISUAL_ASSET,
            workspace_ids,
        ),
    )
