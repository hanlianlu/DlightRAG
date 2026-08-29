# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for the chat interface and durable answer runs."""

import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.adapters.http.browser.answer_events import browser_frame
from dlightrag.adapters.http.browser.app_shell import app_html_response
from dlightrag.adapters.http.browser.attachment_requests import parse_web_answer_request
from dlightrag.adapters.http.browser.conversation_models import (
    AnswerRunDescriptor,
    ConversationTurn,
)
from dlightrag.adapters.http.browser.conversations import (
    WEB_IMAGE_URL_BASE,
    WEB_SOURCE_DOWNLOAD_BASE,
    project_conversation_summary,
    project_conversation_turn,
)
from dlightrag.adapters.http.browser.deps import (
    enforce_web_access,
    get_application,
    get_web_access_gate,
    get_web_conversation_service,
    get_workspace,
)
from dlightrag.adapters.http.browser.presentation import (
    AnswerPresentation,
    build_answer_presentation,
)
from dlightrag.adapters.http.streaming.answer_stream import follow_run_frames, resume_cursor
from dlightrag.application.access import AccessAction, owner_id_from_user
from dlightrag.application.answer_runs import (
    CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    CHILD_ROSTER_PAGE_MAX_LIMIT,
    ChildRosterCursorError,
    ChildRosterPageRequest,
    IdempotencyKeyConflict,
)
from dlightrag.application.answer_runs.results import project_answer_result, project_report_sources
from dlightrag.application.answer_runs.sources import SourceDownloadLinkBuilder
from dlightrag.application.corpus_admin import normalize_workspace_ids
from dlightrag.application.web_conversations import (
    ConversationSubmissionConflict,
    WebAnswerSubmission,
    WebConversationService,
)

logger = logging.getLogger(__name__)

router = APIRouter()
page_router = APIRouter()
_INERT_SVG_CSP = "sandbox; default-src 'none'; img-src data:"


class _WebAgentControl(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


class _WebContinuation(_WebAgentControl):
    submission_id: UUID


@page_router.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """Serve the Vite-owned application document."""
    return app_html_response("index.html")


@page_router.get("/conversations/{conversation_id}", response_class=FileResponse)
async def conversation_page(
    conversation_id: str,  # noqa: ARG001 - the browser router owns selection
) -> FileResponse:
    """Serve the same application document for one explicit client route."""
    return app_html_response("index.html")


@page_router.get("/design-system", response_class=FileResponse)
async def design_system_page() -> FileResponse:
    """Serve the application document for the shared-control reference page."""
    return app_html_response("index.html")


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
    application = get_application(request)
    cfg = application.config
    # Enforce the probed answer capability at admission (pre-acceptance 4xx).
    capability = (await application.answers.capabilities()).answer
    body = await parse_web_answer_request(
        request,
        max_attachments=cfg.answer.generation.max_attachments,
        max_attachment_bytes=cfg.answer.generation.max_attachment_bytes,
        max_total_attachment_bytes=cfg.answer.generation.max_total_attachment_bytes,
        image_max_pixels=cfg.answer.generation.image_max_pixels,
        answer_image_capability=capability,
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
            conversation_id=(
                str(body.conversation_id) if body.conversation_id is not None else None
            ),
            submission_id=str(body.submission_id),
            query=query,
            workspaces=target_workspaces,
            attachments=body.attachments,
            mode=body.mode,
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
    downloadable, visual = await _projection_workspaces(request, turn.run.request_input())
    return project_conversation_turn(
        turn, downloadable_workspaces=downloadable, visual_workspaces=visual
    )


@router.post("/answer/{run_id}/resume", response_model=ConversationTurn)
async def resume_answer_run(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationTurn:
    """Reattach through the same authoritative Web turn projection."""
    return await answer_run_status(run_id, request, conversation_service)


@router.post("/answer/{run_id}/steer", status_code=202)
async def steer_answer_run(
    run_id: str,
    body: _WebAgentControl,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    receipt = await get_application(request).answers.steer(
        owner_id=owner_id_from_user(user), run_id=run_id, instruction=body.content
    )
    if receipt is None:
        raise HTTPException(status_code=409, detail="Run is not a live Research session")
    return {
        "run_id": receipt.run_id,
        "control_sequence": receipt.control_sequence,
        "kind": receipt.kind,
    }


@router.get("/answer/{run_id}/children")
async def answer_run_children(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
    limit: Annotated[
        int,
        Query(ge=1, le=CHILD_ROSTER_PAGE_MAX_LIMIT),
    ] = CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    answers = get_application(request).answers
    try:
        decoded_cursor = (
            answers.child_roster_cursor_codec.decode(cursor) if cursor is not None else None
        )
        if decoded_cursor is not None and str(decoded_cursor.run_id) != run_id:
            raise ChildRosterCursorError("child-roster cursor belongs to another run")
        page_request = ChildRosterPageRequest(limit=limit, cursor=decoded_cursor)
    except (ChildRosterCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    page = await answers.children(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        page=page_request,
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return {
        "run_id": run_id,
        "children": [dict(child) for child in page.children],
        "next_cursor": (
            answers.child_roster_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


async def _continue_answer_run(
    *,
    kind: str,
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService,
) -> AnswerRunDescriptor:
    user = getattr(request.state, "user_context", None)
    parent = await conversation_service.turn_for_run(user, run_id)
    authorized_workspaces: list[str] | None = None
    if parent is not None and parent.run.terminal:
        authorized_workspaces = [
            str(item) for item in parent.run.request_input().get("workspaces") or ()
        ]
        for workspace_id in authorized_workspaces:
            await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace_id)
    try:
        submission = await conversation_service.continue_answer(
            user,
            parent_run_id=run_id,
            submission_id=str(body.submission_id),
            query=body.content,
            kind=kind,
            authorized_workspaces=authorized_workspaces,
        )
    except ConversationSubmissionConflict, IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="This submission id was already used for a different continuation",
        ) from None
    if submission is None:
        raise HTTPException(status_code=409, detail="Continuation requires a terminal answer")
    return answer_run_descriptor(submission)


@router.post("/answer/{run_id}/follow-up", status_code=202)
async def follow_up_answer_run(
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AnswerRunDescriptor:
    return await _continue_answer_run(
        kind="follow_up",
        run_id=run_id,
        body=body,
        request=request,
        conversation_service=conversation_service,
    )


@router.post("/answer/{run_id}/fork", status_code=202)
async def fork_answer_run(
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AnswerRunDescriptor:
    return await _continue_answer_run(
        kind="fork",
        run_id=run_id,
        body=body,
        request=request,
        conversation_service=conversation_service,
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
    outcome = await get_application(request).answers.cancel(
        owner_id=owner_id_from_user(user), run_id=run_id
    )
    if outcome.run is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    # 202 only while a running worker still has to observe the request.
    response.status_code = 202 if outcome.outcome == "pending" else 200
    downloadable, visual = await _projection_workspaces(request, outcome.run.request_input())
    return project_conversation_turn(
        replace(turn, run=outcome.run),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
    )


def _artifact_descriptor(result: Mapping[str, Any], resource_id: str) -> Mapping[str, Any] | None:
    for item in result.get("artifacts") or ():
        if isinstance(item, Mapping) and item.get("resource_id") == resource_id:
            return item
    return None


def _artifact_range(header: str, total: int) -> tuple[int, int | None, int, str | None]:
    if not header:
        return 0, None, 200, None
    if not header.lower().startswith("bytes=") or "," in header:
        raise HTTPException(
            status_code=416,
            detail="range not satisfiable",
            headers={"Content-Range": f"bytes */{total}"},
        )
    start_s, _, end_s = header.split("=", 1)[1].partition("-")
    try:
        if start_s == "":
            suffix = int(end_s)
            if suffix <= 0 or total == 0:
                raise ValueError
            length = min(suffix, total)
            offset = total - length
        else:
            offset = int(start_s)
            end = int(end_s) if end_s else total - 1
            if offset >= total or end < offset:
                raise ValueError
            length = min(end, total - 1) - offset + 1
    except ValueError as exc:
        raise HTTPException(
            status_code=416,
            detail="range not satisfiable",
            headers={"Content-Range": f"bytes */{total}"},
        ) from exc
    return offset, length, 206, f"bytes {offset}-{offset + length - 1}/{total}"


@router.get("/answer/{run_id}/artifacts/{resource_id}")
async def answer_artifact_data(
    run_id: str,
    resource_id: str,
    request: Request,
    download: bool = False,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> StreamingResponse:
    """Stream one owned Artifact as inert data with Range support."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    result = turn.run.result if turn is not None else None
    descriptor = _artifact_descriptor(result or {}, resource_id)
    if descriptor is None or descriptor.get("status") != "available":
        raise HTTPException(status_code=404, detail="Artifact not found")
    owner = owner_id_from_user(user)
    application = get_application(request)
    total = await application.answers.artifact_size(
        owner_id=owner, run_id=run_id, resource_id=resource_id
    )
    if total is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    offset, length, status_code, content_range = _artifact_range(
        request.headers.get("range", "").strip(), total
    )
    stream = await application.answers.open_artifact(
        owner_id=owner,
        run_id=run_id,
        resource_id=resource_id,
        offset=offset,
        length=length,
    )
    if stream is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = str(descriptor.get("media_type") or "application/octet-stream")
    safe_inline = media_type.startswith("image/") or media_type == "application/pdf"
    filename = str(descriptor.get("filename") or "artifact").replace('"', "_")
    disposition = "attachment" if download or not safe_inline else "inline"
    return StreamingResponse(
        stream,
        media_type=media_type if safe_inline and not download else "application/octet-stream",
        status_code=status_code,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
            **(
                {"Content-Security-Policy": _INERT_SVG_CSP} if media_type == "image/svg+xml" else {}
            ),
            **({"Content-Range": content_range} if content_range else {}),
        },
    )


@router.get(
    "/answer/{run_id}/artifacts/{resource_id}/presentation",
    response_model=AnswerPresentation,
)
async def answer_artifact_presentation(
    run_id: str,
    resource_id: str,
    request: Request,
    response: Response,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AnswerPresentation:
    """Return a safe AnswerPresentation for any Markdown Artifact."""
    user = getattr(request.state, "user_context", None)
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None or turn.run.status != "succeeded" or turn.run.result is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    descriptor = _artifact_descriptor(turn.run.result, resource_id)
    if descriptor is None or descriptor.get("media_type") != "text/markdown":
        raise HTTPException(status_code=404, detail="Artifact presentation not found")
    blob = await get_application(request).answers.read_artifact(
        owner_id=owner_id_from_user(user), run_id=run_id, resource_id=resource_id
    )
    if blob is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        markdown = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Artifact is not UTF-8") from exc
    downloadable, visual = await _projection_workspaces(request, turn.run.request_input())
    projected = project_answer_result(
        turn.run.result,
        source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        image_url_prefix=WEB_IMAGE_URL_BASE,
        run_id=run_id,
        artifact_url_prefix="/web/api/answer",
    )
    sources = (
        project_report_sources(
            turn.run.result,
            source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
            downloadable_workspaces=downloadable,
            visual_workspaces=visual,
            image_url_prefix=WEB_IMAGE_URL_BASE,
        )
        if descriptor.get("role") == "primary_report"
        else []
    )
    response.headers["Cache-Control"] = "private, no-store"
    return build_answer_presentation(
        answer=markdown,
        sources=sources,
        evidence_images=[],
        artifacts=projected["artifacts"],
        artifact_outcome=projected["artifact_outcome"],
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
    downloadable, visual = await _projection_workspaces(request, turn.run.request_input())
    events = get_application(request).answers.subscribe(
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
                live_after=turn.run.next_event_sequence - 1,
                run_id=run_id,
            ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def answer_run_descriptor(submission: WebAnswerSubmission) -> AnswerRunDescriptor:
    """Project the owner-scoped links one accepted submission is followed by."""
    run_id = submission.run.run_id
    accepted = submission.run.request_input()
    return AnswerRunDescriptor(
        run_id=run_id,
        status=submission.run.status,
        cancel_requested=submission.run.cancel_requested,
        turn_id=submission.turn_id,
        turn_number=submission.turn_number,
        submission_id=str(submission.run.idempotency_key or ""),
        events_url=f"/web/api/answer/{run_id}/events",
        status_url=f"/web/api/answer/{run_id}",
        cancel_url=f"/web/api/answer/{run_id}",
        conversation=project_conversation_summary(submission.conversation),
        parent_run_id=accepted.get("parent_run_id"),
        continuation_kind=accepted.get("continuation_kind"),
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
