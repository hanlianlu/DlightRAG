# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web-only durable conversation lifecycle routes."""

from collections.abc import Sequence
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from dlightrag.adapters.http.artifact_delivery import INERT_SVG_CSP
from dlightrag.adapters.http.browser.conversation_models import (
    AgentspaceFileHistory,
    AgentspaceFileVersionView,
    AgentspaceFileView,
    AgentspaceHeadView,
    AgentspaceSnapshotChangesView,
    AgentspaceSnapshotDetail,
    AgentspaceSnapshotView,
    ConversationAgentspace,
    ConversationCommandRequest,
    ConversationCommandResponse,
    ConversationGoalState,
    ConversationHistory,
    ConversationPage,
    ConversationQueue,
    ConversationSummary,
    GoalCommandRequest,
    QueueMessageRequest,
    QueueMessageResponse,
    RenameConversationRequest,
    ReorderQueueRequest,
    WebCommandError,
)
from dlightrag.adapters.http.browser.conversations import (
    project_conversation_command,
    project_conversation_goal,
    project_conversation_history,
    project_conversation_summary,
    project_queued_message,
)
from dlightrag.adapters.http.browser.deps import (
    enforce_web_access,
    get_application,
    get_web_access_gate,
    get_web_conversation_service,
)
from dlightrag.adapters.http.browser.run_resources import image_rewrites
from dlightrag.application.access.control import AccessAction
from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.answer.provenance import ArtifactAttestation, run_attestation
from dlightrag.application.corpus_admin.service import workspace_ids_for_names
from dlightrag.application.web_conversations.models import (
    CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT,
    CONVERSATION_HISTORY_PAGE_MAX_LIMIT,
    CONVERSATION_PAGE_DEFAULT_LIMIT,
    CONVERSATION_PAGE_MAX_LIMIT,
    ConversationCursorError,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPageRequest,
    ConversationSessionBusy,
)
from dlightrag.application.web_conversations.models import (
    QueuedMessage as ApplicationQueuedMessage,
)
from dlightrag.application.web_conversations.service import WebConversationService
from dlightrag.engine.answer.goal import GoalError

router = APIRouter()


def _user(request: Request):
    return getattr(request.state, "user_context", None)


def _attachment_content_disposition(filename: str) -> str:
    """Build a latin-1-safe ``Content-Disposition`` value.

    Mirrors Starlette's ``FileResponse`` encoding: non-ASCII filenames (e.g.
    ``报告.pdf``) and any name that is not already URL-safe (e.g. one containing
    a ``"``) are emitted as RFC 5987 ``filename*=utf-8''...``; only fully
    URL-safe ASCII names use the plain quoted ``filename="..."`` form. This
    avoids the latin-1 ``UnicodeEncodeError`` and the quote-breakout that a raw
    interpolation would cause.
    """
    quoted = quote(filename)
    if quoted != filename:
        return f"attachment; filename*=utf-8''{quoted}"
    return f'attachment; filename="{filename}"'


@router.get("/conversations", response_model=ConversationPage)
async def list_conversations(
    request: Request,
    limit: Annotated[
        int,
        Query(ge=1, le=CONVERSATION_PAGE_MAX_LIMIT),
    ] = CONVERSATION_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationPage:
    try:
        decoded_cursor = service.cursor_codec.decode(cursor) if cursor is not None else None
    except ConversationCursorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    page = await service.list(
        _user(request),
        page=ConversationPageRequest(limit=limit, cursor=decoded_cursor),
    )
    return ConversationPage(
        items=[project_conversation_summary(summary) for summary in page.items],
        next_cursor=(
            service.cursor_codec.encode(page.next_cursor) if page.next_cursor is not None else None
        ),
    )


@router.post(
    "/conversations",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationSummary:
    return project_conversation_summary(await service.create(_user(request)))


@router.delete(
    "/conversations",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_all_conversations(
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    await service.delete_all(_user(request))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/conversations/{conversation_id}/history",
    response_model=ConversationHistory,
)
async def conversation_history(
    conversation_id: UUID,
    request: Request,
    limit: Annotated[
        int,
        Query(ge=1, le=CONVERSATION_HISTORY_PAGE_MAX_LIMIT),
    ] = CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationHistory:
    records = await get_application(request).corpora.alist_workspace_records()
    gate = get_web_access_gate(request)
    downloadable = await gate.filter_workspace_records(
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
        records,
    )
    visual = await gate.filter_workspace_records(
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
        records,
    )
    try:
        decoded_cursor = service.history_cursor_codec.decode(cursor) if cursor is not None else None
        if decoded_cursor is not None and decoded_cursor.conversation_id != conversation_id:
            raise ConversationCursorError(
                "conversation history cursor belongs to another conversation"
            )
        page_request = ConversationHistoryPageRequest(
            limit=limit,
            cursor=decoded_cursor,
        )
    except (ConversationCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    page = await service.history(
        _user(request),
        str(conversation_id),
        page=page_request,
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return project_conversation_history(
        page,
        image_rewrites=await _run_image_rewrites(service, _user(request), page),
        attestations=await _turn_attestations(request, page),
        mask_reader_text=get_application(request).skill_protection.mask,
        next_cursor=(
            service.history_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
        downloadable_workspaces={record["workspace"] for record in downloadable},
        visual_workspaces={record["workspace"] for record in visual},
    )


async def _turn_attestations(
    request: Request, page: ConversationHistoryPage
) -> dict[str, ArtifactAttestation]:
    """The derived Artifact record for every turn on this page that published one.

    The derivation reads a Run's records asynchronously and the projection that consumes it does not,
    so it happens here, once per Run, and travels as an argument -- the same record the live Answer
    published, stamped by the same helper, so reopening a conversation shows what watching it showed.

    A turn with nothing published is skipped: a page costs one derivation per Artifact-publishing
    Run, not one per turn, and a Run that published nothing has no Artifact to label.
    """
    application = get_application(request)
    owner = owner_id_from_user(_user(request))
    derived: dict[str, ArtifactAttestation] = {}
    for turn in page.turns:
        run = turn.run
        if run.status != "succeeded" or not (run.result or {}).get("artifacts"):
            continue
        derived[run.run_id] = await run_attestation(
            runs=application.runs,
            config=application.config,
            owner_id=owner,
            run_id=run.run_id,
            records_pruned=run.events_trimmed_at is not None,
        )
    return derived


@router.get(
    "/conversations/{conversation_id}/goal",
    response_model=ConversationGoalState | None,
)
async def conversation_goal(
    conversation_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationGoalState | None:
    goal = await service.goal(_user(request), str(conversation_id))
    return (
        None
        if goal is None
        else project_conversation_goal(
            goal, mask_reader_text=get_application(request).skill_protection.mask
        )
    )


async def _authorized_command_workspaces(
    request: Request, names: Sequence[str] | None
) -> list[str] | None:
    """Resolve and authorize the composer's workspace names for a goal command.

    A goal command is a write path that can start a corpus-reading round, so it applies the same
    workspace access check as a normal Answer send. A name this deployment does not grant is a
    403, never a silently unscoped round.
    """
    if not names:
        return None
    canonical = workspace_ids_for_names(names)
    for workspace_id in canonical:
        await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace_id)
    return canonical


@router.post(
    "/conversations/{conversation_id}/goal",
    response_model=ConversationGoalState | None,
)
async def command_conversation_goal(
    conversation_id: UUID,
    body: GoalCommandRequest,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationGoalState | None:
    try:
        goal = await service.goal_command(
            _user(request),
            str(conversation_id),
            action=body.action,
            objective=body.objective,
            max_rounds=body.max_goal_rounds,
            workspaces=await _authorized_command_workspaces(request, body.workspaces),
        )
    except GoalError as error:
        raise HTTPException(
            status_code=409, detail={"kind": error.code, "message": str(error)}
        ) from None
    return (
        None
        if goal is None
        else project_conversation_goal(
            goal, mask_reader_text=get_application(request).skill_protection.mask
        )
    )


@router.post(
    "/conversations/{conversation_id}/commands",
    response_model=ConversationCommandResponse,
)
async def execute_conversation_command(
    conversation_id: UUID,
    body: ConversationCommandRequest,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationCommandResponse:
    """Execute, persist, and return one server-owned command for this conversation.

    The server owns the command's result text and tone, so the browser renders exactly
    what was recorded. A refused command is a recorded result too: it answers 200 with an
    error tone. 404 keeps the same opaque miss as every other owned read; an unregistered
    name raises the Application conflict the transport maps to 409.
    """
    command = await service.command(
        _user(request),
        str(conversation_id),
        body.name,
        body.input,
        workspaces=await _authorized_command_workspaces(request, body.workspaces),
    )
    if command is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationCommandResponse(
        command=project_conversation_command(
            command,
            mask_reader_text=get_application(request).skill_protection.mask,
        )
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationSummary,
)
async def rename_conversation(
    conversation_id: UUID,
    body: RenameConversationRequest,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationSummary:
    summary = await service.rename(_user(request), str(conversation_id), body.title)
    if summary is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return project_conversation_summary(summary)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    deleted = await service.delete(_user(request), str(conversation_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _run_image_rewrites(
    service: WebConversationService,
    user: UserContext | None,
    page: ConversationHistoryPage,
) -> dict[str, dict[str, str]]:
    """Resolve a same-origin address per stored answer image on this page.

    Only a succeeded answer that may name an external image costs a lookup, and a
    run that stored no such bytes contributes nothing, so a page of ordinary
    turns reads exactly what it read before.
    """
    rewrites: dict[str, dict[str, str]] = {}
    for turn in page.turns:
        run = turn.run
        probed = str((run.result or {}).get("answer") or "").casefold()
        if run.status != "succeeded" or ("](http" not in probed and "<img" not in probed):
            continue
        sources = await service.run_external_sources(user, run.run_id)
        if sources:
            rewrites[run.run_id] = image_rewrites(run.run_id, sources)
    return rewrites


@router.get("/runs/{run_id}/resources/{resource_id}")
async def run_resource(
    run_id: str,
    resource_id: str,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    """Serve one owned run's stored bytes under the id its registries share.

    The same address answers an accepted upload, a publication, and an image the
    run fetched, rendered, or adopted, so a conversation keeps one way to reach
    what it showed.
    """
    stored = await service.run_resource(_user(request), run_id, resource_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Resource not found")
    descriptor, content = stored
    headers = {
        "Cache-Control": "private, max-age=3600",
        "X-Content-Type-Options": "nosniff",
    }
    media_type = descriptor.mime_type.lower()
    if not media_type.startswith("image/"):
        headers["Content-Disposition"] = _attachment_content_disposition(descriptor.filename)
    elif media_type.partition(";")[0].strip() == "image/svg+xml":
        # An SVG is a document: opened inline on this origin its script would run as the
        # owner, whether the Run fetched it or a page's download named it.
        headers["Content-Security-Policy"] = INERT_SVG_CSP
    return Response(content=content, media_type=descriptor.mime_type, headers=headers)


@router.get("/runs/{run_id}/resources/{resource_id}/thumbnail")
async def run_resource_thumbnail(
    run_id: str,
    resource_id: str,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    thumbnail = await service.thumbnail(_user(request), run_id, resource_id)
    if thumbnail is None:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    payload, mime_type = thumbnail
    return Response(
        content=payload,
        media_type=mime_type,
        headers={
            "Cache-Control": "private, max-age=86400, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------------------------
# Messages queued behind a live Run
# ---------------------------------------------------------------------------


def _conversation_queue(
    messages: Sequence[ApplicationQueuedMessage],
    request: Request,
) -> ConversationQueue:
    mask = get_application(request).skill_protection.mask
    return ConversationQueue(
        items=[project_queued_message(message, mask_reader_text=mask) for message in messages]
    )


def _queued_message_response(
    message: ApplicationQueuedMessage,
    request: Request,
) -> QueueMessageResponse:
    return QueueMessageResponse(
        message=project_queued_message(
            message, mask_reader_text=get_application(request).skill_protection.mask
        )
    )


@router.get(
    "/conversations/{conversation_id}/queue",
    response_model=ConversationQueue,
)
async def read_conversation_queue(
    conversation_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationQueue:
    """The messages this reader queued behind the conversation's live Run, oldest first."""
    messages = await service.queue(_user(request), str(conversation_id))
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _conversation_queue(messages, request)


@router.post(
    "/conversations/{conversation_id}/queue",
    response_model=QueueMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def enqueue_conversation_message(
    conversation_id: UUID,
    body: QueueMessageRequest,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> QueueMessageResponse:
    """Hold one message behind the live Run; it becomes the next turn when that Run settles."""
    message = await service.enqueue(_user(request), str(conversation_id), body.content)
    if message is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _queued_message_response(message, request)


@router.put(
    "/conversations/{conversation_id}/queue/order",
    response_model=ConversationQueue,
)
async def reorder_conversation_queue(
    conversation_id: UUID,
    body: ReorderQueueRequest,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationQueue:
    """Rewrite the queue order; a stale read conflicts instead of dropping an arrival."""
    messages = await service.reorder_queue(
        _user(request),
        str(conversation_id),
        [str(message_id) for message_id in body.message_ids],
    )
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _conversation_queue(messages, request)


@router.patch(
    "/conversations/{conversation_id}/queue/{message_id}",
    response_model=QueueMessageResponse,
)
async def edit_queued_message(
    conversation_id: UUID,
    message_id: UUID,
    body: QueueMessageRequest,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> QueueMessageResponse:
    """Rewrite one queued message's words before it is delivered."""
    message = await service.edit_queued(
        _user(request), str(conversation_id), str(message_id), body.content
    )
    if message is None:
        raise HTTPException(status_code=404, detail="Queued message not found")
    return _queued_message_response(message, request)


@router.delete(
    "/conversations/{conversation_id}/queue/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_queued_message_route(
    conversation_id: UUID,
    message_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    """Drop one queued message before it is delivered."""
    removed = await service.remove_queued(_user(request), str(conversation_id), str(message_id))
    if not removed:
        raise HTTPException(status_code=404, detail="Queued message not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]


def _agentspace_head_view(head: Any) -> AgentspaceHeadView:
    return AgentspaceHeadView(
        lane_id=head.lane_id,
        head_run_id=head.head_run_id,
        manifest_digest=head.manifest_digest,
        revision=head.revision,
        committed_at=head.committed_at,
    )


def _agentspace_snapshot_view(snapshot: Any) -> AgentspaceSnapshotView:
    return AgentspaceSnapshotView(
        run_id=snapshot.run_id,
        parent_run_id=snapshot.parent_run_id,
        manifest_digest=snapshot.manifest_digest,
        file_count=snapshot.file_count,
        changes=AgentspaceSnapshotChangesView(
            added=snapshot.changes.added,
            changed=snapshot.changes.changed,
            removed=snapshot.changes.removed,
        ),
        created_at=snapshot.created_at,
    )


@router.get(
    "/conversations/{conversation_id}/workspace",
    response_model=ConversationAgentspace,
)
async def read_conversation_agentspace(
    conversation_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationAgentspace:
    """Every branch head and the newest restorable snapshots of this conversation."""
    workspace = await service.agentspace(_user(request), str(conversation_id))
    if workspace is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    heads, snapshots = workspace
    return ConversationAgentspace(
        heads=[_agentspace_head_view(head) for head in heads],
        snapshots=[_agentspace_snapshot_view(snapshot) for snapshot in snapshots],
    )


@router.get(
    "/conversations/{conversation_id}/workspace/snapshots/{run_id}",
    response_model=AgentspaceSnapshotDetail,
)
async def read_conversation_agentspace_snapshot(
    conversation_id: UUID,
    run_id: str,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> AgentspaceSnapshotDetail:
    """The files one settled Run's snapshot carried."""
    workspace = await service.agentspace(_user(request), str(conversation_id))
    files = await service.agentspace_snapshot(_user(request), str(conversation_id), run_id)
    if workspace is None or files is None:
        raise HTTPException(status_code=404, detail="Workspace snapshot not found")
    snapshot = next(
        (item for item in workspace[1] if item.run_id == run_id),
        None,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Workspace snapshot not found")
    return AgentspaceSnapshotDetail(
        snapshot=_agentspace_snapshot_view(snapshot),
        files=[
            AgentspaceFileView(
                relative_path=file.relative_path,
                content_digest=file.content_digest,
                size_bytes=file.size_bytes,
                change=file.change,
            )
            for file in files
        ],
    )


@router.get(
    "/conversations/{conversation_id}/workspace/files",
    response_model=AgentspaceFileHistory,
)
async def read_conversation_agentspace_file(
    conversation_id: UUID,
    request: Request,
    path: Annotated[str, Query(min_length=1, max_length=1024)],
    service: WebConversationService = Depends(get_web_conversation_service),
) -> AgentspaceFileHistory:
    """Every snapshot that carried a path, and which Run changed it last."""
    versions = await service.agentspace_file(_user(request), str(conversation_id), path)
    if versions is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return AgentspaceFileHistory(
        relative_path=path,
        versions=[
            AgentspaceFileVersionView(
                run_id=version.run_id,
                content_digest=version.content_digest,
                size_bytes=version.size_bytes,
                created_at=version.created_at,
                change=version.change,
            )
            for version in versions
        ],
    )


@router.post(
    "/conversations/{conversation_id}/workspace/reset",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def reset_conversation_agentspace(
    conversation_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    """Forget the conversation's workspace branches and snapshots, at rest only."""
    try:
        cleared = await service.reset_agentspace(_user(request), str(conversation_id))
    except ConversationSessionBusy as exc:
        raise HTTPException(
            status_code=409,
            detail=WebCommandError(
                kind="submission_conflict",
                message=str(exc),
                error_kind=exc.error_kind,
            ).model_dump(exclude_none=True),
        ) from None
    if not cleared:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
