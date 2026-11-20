# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web-only read of the owner's recent conversation-linked Answer runs."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from dlightrag.adapters.http.browser.conversation_models import WebRunActivityPage
from dlightrag.adapters.http.browser.conversations import project_run_activity
from dlightrag.adapters.http.browser.deps import get_web_conversation_service
from dlightrag.application.web_conversations.models import (
    RUN_ACTIVITY_DEFAULT_LIMIT,
    RUN_ACTIVITY_MAX_LIMIT,
)
from dlightrag.application.web_conversations.service import WebConversationService

router = APIRouter()


def _user(request: Request):
    """The authenticated principal the access middleware attached to this request."""
    return getattr(request.state, "user_context", None)


@router.get("/run-activity", response_model=WebRunActivityPage)
async def run_activity(
    request: Request,
    limit: Annotated[
        int,
        Query(ge=1, le=RUN_ACTIVITY_MAX_LIMIT),
    ] = RUN_ACTIVITY_DEFAULT_LIMIT,
    service: WebConversationService = Depends(get_web_conversation_service),
) -> WebRunActivityPage:
    """Return one bounded page of this principal's recent Answer run activity.

    Only a run a Web conversation turn links is Web run activity, so a run accepted
    through REST or MCP is absent. A malformed ``limit`` is refused before the service
    runs; an unavailable store keeps the Web surface's shared 503 mapping.
    """
    activity = await service.run_activity(_user(request), limit=limit)
    return WebRunActivityPage(items=[project_run_activity(item) for item in activity])


__all__ = ["router"]
