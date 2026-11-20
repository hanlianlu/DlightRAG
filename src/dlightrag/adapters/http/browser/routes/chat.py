# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for the chat interface and durable answer runs."""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.adapters.http.artifact_delivery import (
    artifact_presentation_available,
    artifact_range,
    artifact_response,
)
from dlightrag.adapters.http.browser.answer_events import ReaderFrameRenderer
from dlightrag.adapters.http.browser.app_shell import app_html_response
from dlightrag.adapters.http.browser.attachment_requests import parse_web_answer_request
from dlightrag.adapters.http.browser.conversation_models import (
    AcceptedAnswer,
    AnswerDeliveryReceipt,
    AnswerSubmissionResponse,
    ConversationTurn,
    WebCommandError,
    WebCommandErrorKind,
)
from dlightrag.adapters.http.browser.conversations import (
    WEB_IMAGE_URL_BASE,
    WEB_SOURCE_DOWNLOAD_BASE,
    project_answer_delivery,
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
from dlightrag.adapters.http.browser.routes.skills import require_known_skill
from dlightrag.adapters.http.browser.run_resources import (
    RUN_RESOURCE_URL_BASE,
    image_rewrites,
)
from dlightrag.adapters.http.streaming.answer_stream import follow_frames, resume_cursor
from dlightrag.application.access.control import AccessAction
from dlightrag.application.access.principal import (
    Principal,
    owner_id_from_user,
)
from dlightrag.application.answer.provenance import (
    ArtifactAttestation,
    attest_projected_artifacts,
    run_attestation,
)
from dlightrag.application.answer_runs.artifacts import (
    artifact_descriptor,
    published_artifact_descriptor,
)
from dlightrag.application.answer_runs.child_roster import (
    CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    CHILD_ROSTER_PAGE_MAX_LIMIT,
    ChildRosterCursorError,
    ChildRosterPageRequest,
    stamp,
)
from dlightrag.application.answer_runs.service import (
    ACTIVITY_PAGE_DEFAULT_LIMIT,
    ACTIVITY_PAGE_MAX_LIMIT,
    ActiveRunLimitError,
    child_control_receipt_payload,
    child_control_succeeded,
)
from dlightrag.application.corpus_admin.service import workspace_ids_for_names
from dlightrag.application.errors import (
    ApplicationConflictError,
    ApplicationInputError,
    ApplicationUnavailableError,
)
from dlightrag.application.policies.service import (
    ANSWER_FEEDBACK,
)
from dlightrag.application.runs import IdempotencyKeyConflict, RunAdmissionLimitExceededError
from dlightrag.application.web_conversations.models import (
    ConversationSessionBusy,
    ConversationSubmissionConflict,
    LinkedTurn,
)
from dlightrag.application.web_conversations.service import (
    WebAnswerDelivery,
    WebAnswerSubmission,
    WebConversationService,
)
from dlightrag.engine.answer.citations.sources import SourceDownloadLinkBuilder
from dlightrag.engine.answer.errors import AnswerInputError
from dlightrag.engine.answer.feedback import (
    FEEDBACK_NOTE_LIMIT,
    FEEDBACK_REASONS,
    AnswerFeedback,
    FeedbackValue,
)
from dlightrag.engine.answer.results import (
    project_answer_result,
    project_artifact_sources,
)

logger = logging.getLogger(__name__)

router = APIRouter()
page_router = APIRouter()


class _WebAgentControl(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


class _WebContinuation(_WebAgentControl):
    submission_id: UUID


class _WebChildControl(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["steer", "continue", "cancel"]
    content: str = Field(default="", max_length=20_000)
    reauthorize_user_cancelled: bool = False


# An agent is `main` or a Child Session's id, which is the 36 characters of a UUID. A message's
# sequence is a PostgreSQL BIGINT, so a larger number would fail the query instead of finding
# no message.
_AGENT_ID_LENGTH = 36
_MAX_SEQUENCE = 2**63 - 1


class _WebFeedback(BaseModel):
    """One reader's judgement of one message. What may be said about it is the Engine's to rule."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    agent: str = Field(min_length=1, max_length=_AGENT_ID_LENGTH)
    sequence: int = Field(ge=1, le=_MAX_SEQUENCE)
    value: FeedbackValue
    reasons: list[str] = Field(default_factory=list, max_length=len(FEEDBACK_REASONS))
    note: str | None = Field(default=None, max_length=FEEDBACK_NOTE_LIMIT)


IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)]


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
    """Serve the isolated Vite design-system entry document."""
    return app_html_response("design-system.html")


@router.post("/answer", status_code=202, response_model=AnswerSubmissionResponse)
async def start_answer_run(
    request: Request,
    workspace: str = Depends(get_workspace),
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer | AnswerDeliveryReceipt:
    """Accept one submission as a durable run linked to its conversation entry.

    The run, its uploaded bytes, and the conversation turn are committed in one
    transaction before this result is returned, so the browser can discard its
    local File and Blob resources at the acceptance seam.
    """
    with _answer_command(
        reuse="request", unavailable="Answer submission is temporarily unavailable"
    ):
        return await _start_answer_run(request, workspace, conversation_service)


async def _start_answer_run(
    request: Request,
    workspace: str,
    conversation_service: WebConversationService,
) -> AcceptedAnswer | AnswerDeliveryReceipt:
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
        raise _command_error(422, "invalid_request", "A question is required")

    requested_skill = body.requested_skill
    if requested_skill is not None:
        try:
            owner_id = owner_id_from_user(getattr(request.state, "user_context", None))
            requested_skill = require_known_skill(application, owner_id, requested_skill)
        except ValueError as exc:
            raise _command_error(422, "invalid_request", str(exc)) from exc
    mode = body.mode
    if requested_skill is not None and mode != "research":
        # Skills (load_skill tool + metadata) exist only in Research runs.
        mode = "research"

    target_workspaces = workspace_ids_for_names(body.workspaces or [workspace])
    for ws in target_workspaces:
        await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, ws)

    submitted = await conversation_service.start_answer(
        getattr(request.state, "user_context", None),
        conversation_id=(str(body.conversation_id) if body.conversation_id is not None else None),
        submission_id=str(body.submission_id),
        query=query,
        workspaces=target_workspaces,
        attachments=body.attachments,
        mode=mode,
        requested_skill=requested_skill,
        effort=body.effort,
        delivery=body.delivery,
    )
    if submitted is None:
        raise _command_error(404, "conversation_missing", "Conversation not found")
    if isinstance(submitted, WebAnswerDelivery):
        return project_answer_delivery(
            submitted,
            mask_reader_text=application.skill_protection.mask,
        )
    return await accepted_answer(request, submitted)


@router.get("/answer-submissions/{submission_id}", response_model=AcceptedAnswer)
async def accepted_answer_submission(
    submission_id: UUID,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer:
    """Recover one owner-scoped accepted command after an ambiguous POST result."""
    with _answer_command(unavailable="Answer submission lookup is temporarily unavailable"):
        submission = await conversation_service.submission(
            getattr(request.state, "user_context", None), str(submission_id)
        )
        if submission is None:
            raise _command_error(404, "invalid_request", "Answer submission not found")
        return await accepted_answer(request, submission)


@router.get("/runs/{run_id}", response_model=ConversationTurn)
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
        turn,
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        image_rewrites=image_rewrites(
            run_id,
            await conversation_service.run_external_sources(
                getattr(request.state, "user_context", None), run_id
            ),
        ),
        attestation=(
            await _answer_attestation(
                request, run_id=run_id, records_pruned=turn.run.events_trimmed_at is not None
            )
            if (turn.run.result or {}).get("artifacts")
            else None
        ),
        mask_reader_text=get_application(request).skill_protection.mask,
    )


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
    turn = await conversation_service.turn_for_run(user, run_id)
    if turn is None:
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
        # The Run's own row, stated the way every child row below is: its status and
        # its clock come from the same record, so a client draws the main agent with
        # what a child's row is drawn from instead of inventing either.
        "run_status": turn.run.status,
        "run_started_at": stamp(turn.run.started_at),
        "run_finished_at": stamp(turn.run.finished_at),
        "children": [dict(child) for child in page.children],
        "next_cursor": (
            answers.child_roster_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


@router.get("/answer/{run_id}/children/{child_session_id}")
async def observe_answer_child(
    run_id: str,
    child_session_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    observation = await get_application(request).answers.observe_child(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        child_session_id=child_session_id,
        limit=limit,
        with_transcript=False,
    )
    if observation is None:
        raise HTTPException(status_code=404, detail="Answer child not found")
    return observation.payload()


@router.get("/answer/{run_id}/transcript/search")
async def answer_run_trace_search(
    run_id: str,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    """Every message of every agent of one Run that holds ``q``, each in its neighbourhood."""
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    answer = await get_application(request).answers.search_run_traces(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        query=q,
    )
    if answer is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    terms, messages, total, matched = answer
    return {
        "run_id": run_id,
        "query": q,
        # The words the search was for, as the search read them, so the dock marks exactly those.
        "terms": list(terms),
        "match": matched,
        "total": total,
        "messages": [dict(message) for message in messages],
    }


@router.get("/answer/{run_id}/transcript")
async def answer_run_activity(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
    child: Annotated[str | None, Query(min_length=1)] = None,
    before: Annotated[int | None, Query(ge=1)] = None,
    since: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=ACTIVITY_PAGE_MAX_LIMIT)] = ACTIVITY_PAGE_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """One page of the Run's main agent transcript, or of the Child named by ``child``.

    ``since`` reads forward from a sequence instead: the messages written after it, oldest first,
    which is how a client holding a transcript appends the ones that landed since.
    """
    if before is not None and since is not None:
        raise HTTPException(
            status_code=422,
            detail="A transcript read walks back from `before` or forward from `since`, not both",
        )
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    page = await get_application(request).answers.activity_page(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        child_session_id=child,
        before=before,
        limit=limit,
        since=since,
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Answer child not found")
    return page.payload()


@router.get("/answer/{run_id}/feedback")
async def answer_run_feedback(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    """Whether the reader is offered feedback, and the judgements they gave the messages of a Run.

    The judgements are listed with the Run's own agent first, then each Child, and within an
    agent the oldest message first. A reader the policy turns feedback off for is told so and
    given none, and nothing is read.
    """
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    application = get_application(request)
    if not await application.policies.value_for(ANSWER_FEEDBACK, user):
        return {"run_id": run_id, "enabled": False, "feedback": []}
    given = await application.answers.answer_feedback(
        owner_id=owner_id_from_user(user), run_id=run_id
    )
    if given is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return {"run_id": run_id, "enabled": True, "feedback": [_feedback_view(item) for item in given]}


@router.put("/answer/{run_id}/feedback")
async def record_answer_feedback(
    run_id: str,
    body: _WebFeedback,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    """Keep the reader's judgement of one message of the Run, replacing an earlier one.

    The Run has to be the reader's own, and the policy has to offer them feedback. Rating one's
    own answer is the reader's own state, like their history, so it asks for no access to the
    workspaces the answer searched. Whether the judgement can be kept at all is the Engine's
    rule, answered as a 422 before the message is looked up.
    """
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    await _require_feedback_offered(request, user)
    kept = await get_application(request).answers.record_feedback(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        agent=body.agent,
        sequence=body.sequence,
        value=body.value,
        reasons=body.reasons,
        note=body.note,
    )
    if kept is None:
        raise HTTPException(status_code=404, detail="Answer message not found")
    return {"run_id": run_id, **_feedback_view(kept)}


@router.delete("/answer/{run_id}/feedback", status_code=204)
async def retract_answer_feedback(
    run_id: str,
    request: Request,
    agent: Annotated[str, Query(min_length=1, max_length=_AGENT_ID_LENGTH)],
    sequence: Annotated[int, Query(ge=1, le=_MAX_SEQUENCE)],
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> None:
    """Take the reader's judgement of one message back. One never given is no error.

    The policy is not asked: taking back removes only the reader's own data, so it works for a
    reader the policy has since turned feedback off for.
    """
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    await get_application(request).answers.retract_feedback(
        owner_id=owner_id_from_user(user), run_id=run_id, agent=agent, sequence=sequence
    )


@router.post("/answer/{run_id}/children/{child_session_id}/control", status_code=202)
async def control_answer_child(
    run_id: str,
    child_session_id: str,
    body: _WebChildControl,
    request: Request,
    idempotency_key: IdempotencyKey,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    try:
        receipt = await get_application(request).answers.control_child(
            owner_id=owner_id_from_user(user),
            run_id=run_id,
            child_session_id=child_session_id,
            action=body.action,
            content=body.content,
            idempotency_key=idempotency_key,
            reauthorize_user_cancelled=body.reauthorize_user_cancelled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if receipt is None:
        raise HTTPException(status_code=404, detail="Answer child not found")
    if not child_control_succeeded(receipt.outcome):
        raise HTTPException(status_code=409, detail=receipt.outcome)
    return child_control_receipt_payload(receipt)


@router.post("/answer/{run_id}/child-guidance/{request_id}/reply", status_code=202)
async def reply_to_answer_child(
    run_id: str,
    request_id: str,
    body: _WebAgentControl,
    request: Request,
    idempotency_key: IdempotencyKey,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> dict[str, Any]:
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    try:
        receipt = await get_application(request).answers.reply_to_child(
            owner_id=owner_id_from_user(user),
            run_id=run_id,
            request_id=request_id,
            content=body.content,
            idempotency_key=idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if receipt is None:
        raise HTTPException(status_code=404, detail="Child guidance request not found")
    if receipt.outcome != "replied":
        raise HTTPException(status_code=409, detail=receipt.outcome)
    return child_control_receipt_payload(receipt)


async def _fork_answer_run(
    *,
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService,
) -> AcceptedAnswer:
    user = getattr(request.state, "user_context", None)
    parent = await conversation_service.turn_for_run(user, run_id)
    authorized_workspaces: list[str] | None = None
    if parent is not None and parent.run.terminal:
        authorized_workspaces = [
            str(item) for item in parent.run.request_input().get("workspaces") or ()
        ]
        for workspace_id in authorized_workspaces:
            await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace_id)
    submission = await conversation_service.fork_answer(
        user,
        parent_run_id=run_id,
        submission_id=str(body.submission_id),
        query=body.content,
        authorized_workspaces=authorized_workspaces,
    )
    if submission is None:
        raise _command_error(
            409,
            "invalid_request",
            "Continuation requires a terminal answer",
        )
    return await accepted_answer(request, submission)


@router.post("/answer/{run_id}/fork", status_code=202)
async def fork_answer_run(
    run_id: str,
    body: _WebContinuation,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> AcceptedAnswer:
    """Open a new conversation branch at the state one terminal Answer settled at.

    The browser offers no per-turn Follow-Up: a conversation continues through the
    composer, which appends at the Lane tip (ADR 0022).
    """
    with _answer_command(
        reuse="continuation", unavailable="Answer submission is temporarily unavailable"
    ):
        return await _fork_answer_run(
            run_id=run_id,
            body=body,
            request=request,
            conversation_service=conversation_service,
        )


@router.delete("/runs/{run_id}", response_model=ConversationTurn)
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
    outcome = await get_application(request).runs.cancel(
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
        mask_reader_text=get_application(request).skill_protection.mask,
    )


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
    descriptor = published_artifact_descriptor(result, resource_id)
    if descriptor is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    owner = owner_id_from_user(user)
    application = get_application(request)
    total = await application.answers.artifact_size(
        owner_id=owner, run_id=run_id, resource_id=resource_id
    )
    if total is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    offset, length, status_code, content_range = artifact_range(
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
    media_type, headers = artifact_response(
        descriptor,
        download=download,
        content_range=content_range,
    )
    return StreamingResponse(
        stream,
        media_type=media_type,
        status_code=status_code,
        headers=headers,
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
    descriptor = artifact_descriptor(turn.run.result, resource_id)
    if not artifact_presentation_available(descriptor):
        raise HTTPException(status_code=404, detail="Artifact presentation not found")
    application = get_application(request)
    blob = await application.answers.read_artifact(
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
        run_resource_prefix=RUN_RESOURCE_URL_BASE,
        mask_reader_text=application.skill_protection.mask,
    )
    projected = await attest_projected_artifacts(
        projected,
        runs=application.runs,
        config=application.config,
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        records_pruned=turn.run.events_trimmed_at is not None,
    )
    sources = project_artifact_sources(
        turn.run.result,
        resource_id=resource_id,
        source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        image_url_prefix=WEB_IMAGE_URL_BASE,
        mask_reader_text=application.skill_protection.mask,
    )
    response.headers["Cache-Control"] = "private, no-store"
    return build_answer_presentation(
        # The Artifact's own bytes are the answer this projection shows, so they pass the
        # content gate like any other answer text (ADR 0061). Publication refused a file that
        # reproduced a protected document; an older Artifact, or one whose passage sits below
        # that 320-character bar, still must not hand the words to a reader here.
        answer=application.skill_protection.mask(markdown),
        sources=sources,
        evidence_images=[],
        link_cards=projected.get("link_cards") or [],
        artifacts=projected["artifacts"],
        artifact_outcome=projected["artifact_outcome"],
        artifact_bindings=dict((descriptor or {}).get("artifact_bindings") or {}),
    )


@router.get("/runs/{run_id}/events")
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
    owner = owner_id_from_user(user)
    application = get_application(request)
    skill_protection = application.skill_protection
    # Resolved once per subscription, beside the other read-time projections: a
    # pinned Connection tool keeps a human name even after the Connection later
    # changes, and the durable event keeps transport-neutral identity.
    tool_labels = await application.connections.pinned_tool_labels(owner_id=owner, run_id=run_id)
    # The finished answer must show the same stored images a reload would, so the
    # frame resolver reads the run's own recorded sources once per subscription.
    image_sources = image_rewrites(
        run_id,
        await conversation_service.run_external_sources(user, run_id),
    )
    events = application.runs.subscribe(
        owner_id=owner,
        run_id=run_id,
        after_sequence=resume_cursor(request),
    )
    pruned = turn.run.events_trimmed_at is not None
    publishes = bool((turn.run.result or {}).get("artifacts"))

    async def _attestation() -> ArtifactAttestation | None:
        """The Artifact record this subscription's terminal frame carries.

        Read at the moment the frame is rendered rather than when the subscription opened: the Run
        is only settled by then, and this is the same entry point the turn's status read and the
        conversation history stamp through. A Run that published nothing is not read at all, which
        is the rule the history route already applies and what keeps a terminal frame as cheap as
        it was before any record existed -- and what keeps a records read from being able to fail a
        frame that used to be a pure function of the event.
        """
        if not publishes:
            return None
        return await _answer_attestation(request, run_id=run_id, records_pruned=pruned)

    return StreamingResponse(
        follow_frames(
            events,
            # One renderer per subscription: its carry is how a protected run that spans
            # token batches stays out of every frame, and it belongs to this reader alone.
            ReaderFrameRenderer(
                skill_protection.index,
                skill_protection.mask,
                downloadable_workspaces=downloadable,
                visual_workspaces=visual,
                live_after=turn.run.next_event_sequence - 1,
                run_id=run_id,
                tool_labels=tool_labels,
                image_rewrites=image_sources,
                attestation=_attestation,
            ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def accepted_answer(
    request: Request,
    submission: WebAnswerSubmission,
) -> AcceptedAnswer:
    """Project one accepted command through the same canonical history model."""
    downloadable, visual = await _projection_workspaces(request, submission.run.request_input())
    linked = LinkedTurn(
        turn_id=submission.turn_id,
        turn_number=submission.turn_number,
        submission_id=submission.submission_id,
        created_at=submission.created_at or submission.run.created_at,
        run=submission.run,
        conversation_id=submission.conversation.conversation_id,
    )
    return AcceptedAnswer(
        conversation=project_conversation_summary(submission.conversation),
        turn=project_conversation_turn(
            linked,
            downloadable_workspaces=downloadable,
            visual_workspaces=visual,
            # A run accepted a moment ago has stored nothing yet; the rewrite is
            # resolved from the same surface so the rule has one home.
            image_rewrites=image_rewrites(
                linked.run.run_id,
                await get_web_conversation_service(request).run_external_sources(
                    getattr(request.state, "user_context", None), linked.run.run_id
                ),
            ),
            mask_reader_text=get_application(request).skill_protection.mask,
        ),
    )


def _command_error(
    status_code: int,
    kind: WebCommandErrorKind,
    message: str,
    *,
    error_kind: str | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=WebCommandError(kind=kind, message=message, error_kind=error_kind).model_dump(
            exclude_none=True
        ),
    )


def _rejected_input(exc: AnswerInputError) -> HTTPException:
    """Admission refused the request's input: say why, with its stable kind."""
    return _command_error(422, "invalid_request", exc.public_message, error_kind=exc.error_kind)


# The command kinds an HTTP refusal maps to; any other status (a 401 that asks
# the browser to sign in, a server error) keeps the general envelope.
_HTTP_COMMAND_KINDS: dict[int, WebCommandErrorKind] = {
    400: "invalid_request",
    403: "scope_forbidden",
    404: "invalid_request",
    409: "submission_conflict",
    413: "attachment_rejected",
    422: "invalid_request",
    503: "service_unavailable",
}


@contextmanager
def _answer_command(*, unavailable: str, reuse: str = "request") -> Iterator[None]:
    """Answer the typed failures of one browser Answer command with its envelope.

    ``unavailable`` is the retryable text for storage and runtime outages, which
    may also follow an accepted submission: the browser then reconciles through
    the submission lookup. ``reuse`` names what a reused submission id collided
    with. Untyped failures stay internal errors.
    """
    try:
        yield
    except HTTPException as exc:
        if isinstance(exc.detail, dict):
            raise  # already a command envelope
        kind = _HTTP_COMMAND_KINDS.get(exc.status_code)
        if kind is None:
            raise
        raise _command_error(exc.status_code, kind, str(exc.detail)) from exc
    except AnswerInputError as exc:
        raise _rejected_input(exc) from None
    except ApplicationInputError as exc:
        raise _command_error(422, "invalid_request", str(exc)) from None
    except ConversationSessionBusy as exc:
        # The Session already has a live Run: nothing was accepted, and the typed kind
        # lets the composer offer the queue instead of a second competing send.
        raise _command_error(
            409,
            "submission_conflict",
            str(exc),
            error_kind=exc.error_kind,
        ) from None
    except ConversationSubmissionConflict, IdempotencyKeyConflict:
        raise _command_error(
            409,
            "submission_conflict",
            f"This submission id was already used for a different {reuse}",
        ) from None
    except ApplicationConflictError as exc:
        raise _command_error(409, "submission_conflict", str(exc)) from None
    except ActiveRunLimitError as exc:
        # The reader's own limit, not an outage: the status and the error kind say so, while the
        # command kind stays the retryable one, so the draft is kept and Retry stays offered.
        raise _command_error(
            429, "service_unavailable", str(exc), error_kind=exc.error_kind
        ) from None
    except ApplicationUnavailableError as exc:
        # As the shared handler does: a translated outage keeps its cause's traceback.
        logger.warning(
            "Browser Answer command is unavailable: %s",
            exc,
            exc_info=exc if exc.__cause__ is not None else None,
        )
        message = str(exc) if isinstance(exc, RunAdmissionLimitExceededError) else unavailable
        raise _command_error(503, "service_unavailable", message) from None


async def _require_feedback_offered(request: Request, user: Principal | None) -> None:
    """Refuse a reader the policy turns feedback off for, when they rate."""
    if not await get_application(request).policies.value_for(ANSWER_FEEDBACK, user):
        raise HTTPException(status_code=403, detail="Answer feedback is turned off for you")


def _feedback_view(feedback: AnswerFeedback) -> dict[str, Any]:
    return {
        "agent": feedback.agent,
        "sequence": feedback.sequence,
        "value": feedback.value,
        "reasons": list(feedback.reasons),
        "note": feedback.note,
        "updated_at": stamp(feedback.updated_at),
    }


async def _answer_attestation(
    request: Request, *, run_id: str, records_pruned: bool
) -> ArtifactAttestation:
    """The one Artifact record every live projection of one Run stamps through.

    Three browser projections render the same finished Answer -- the live ``done`` frame, this
    turn's own status read, and the conversation history -- and all three must tell a reader one
    thing. Each derives here rather than sharing a cached value, because the Run is only settled
    when its terminal record exists and a read that arrives earlier must not freeze a verdict it
    would have to keep.
    """
    application = get_application(request)
    return await run_attestation(
        runs=application.runs,
        config=application.config,
        owner_id=owner_id_from_user(getattr(request.state, "user_context", None)),
        run_id=run_id,
        records_pruned=records_pruned,
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
