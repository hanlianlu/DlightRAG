# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web-only durable conversation lifecycle routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from dlightrag.web.conversation_models import (
    ConversationHistory,
    ConversationSummary,
    RenameConversationRequest,
)
from dlightrag.web.conversations import WebConversationService
from dlightrag.web.deps import get_web_conversation_service

router = APIRouter()


def _user(request: Request):
    return getattr(request.state, "user_context", None)


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> list[ConversationSummary]:
    return await service.list(_user(request))


@router.post(
    "/conversations",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationSummary:
    return await service.create(_user(request))


@router.get(
    "/conversations/{conversation_id}/history",
    response_model=ConversationHistory,
)
async def conversation_history(
    conversation_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> ConversationHistory:
    history = await service.history(_user(request), str(conversation_id))
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return history


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
    return summary


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


@router.get("/conversations/{conversation_id}/images/{image_id}")
async def conversation_image(
    conversation_id: UUID,
    image_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    image = await service.image(_user(request), str(conversation_id), str(image_id))
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=image.image_bytes,
        media_type=image.mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/conversations/{conversation_id}/images/{image_id}/thumbnail")
async def conversation_image_thumbnail(
    conversation_id: UUID,
    image_id: UUID,
    request: Request,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    thumbnail = await service.thumbnail(_user(request), str(conversation_id), str(image_id))
    if thumbnail is None:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return Response(
        content=thumbnail.image_bytes,
        media_type=thumbnail.mime_type,
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


__all__ = ["router"]
