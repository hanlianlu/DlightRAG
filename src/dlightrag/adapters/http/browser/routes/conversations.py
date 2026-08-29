# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web-only durable conversation lifecycle routes."""

from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from dlightrag.adapters.http.browser.conversation_models import (
    ConversationHistory,
    ConversationPage,
    ConversationSummary,
    RenameConversationRequest,
)
from dlightrag.adapters.http.browser.conversations import (
    project_conversation_history,
    project_conversation_summary,
)
from dlightrag.adapters.http.browser.deps import (
    get_application,
    get_web_access_gate,
    get_web_conversation_service,
)
from dlightrag.application.access import AccessAction
from dlightrag.application.web_conversations import (
    CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT,
    CONVERSATION_HISTORY_PAGE_MAX_LIMIT,
    CONVERSATION_PAGE_DEFAULT_LIMIT,
    CONVERSATION_PAGE_MAX_LIMIT,
    ConversationCursorError,
    ConversationHistoryPageRequest,
    ConversationPageRequest,
    WebConversationService,
)

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
        next_cursor=(
            service.history_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
        downloadable_workspaces={record["workspace"] for record in downloadable},
        visual_workspaces={record["workspace"] for record in visual},
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


@router.get("/runs/{run_id}/attachments/{ordinal}")
async def run_attachment(
    run_id: str,
    ordinal: int,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    stored = await service.attachment(_user(request), run_id, ordinal)
    if stored is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    headers = {
        "Cache-Control": "private, max-age=3600",
        "X-Content-Type-Options": "nosniff",
    }
    if not stored.mime_type.lower().startswith("image/"):
        headers["Content-Disposition"] = _attachment_content_disposition(stored.filename)
    return Response(content=stored.content, media_type=stored.mime_type, headers=headers)


@router.get("/runs/{run_id}/attachments/{ordinal}/thumbnail")
async def run_attachment_thumbnail(
    run_id: str,
    ordinal: int,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    thumbnail = await service.thumbnail(_user(request), run_id, ordinal)
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


__all__ = ["router"]
