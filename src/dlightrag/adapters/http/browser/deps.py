"""FastAPI dependency injection for authenticated browser routes."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from fastapi import Cookie, HTTPException, Request

from dlightrag.adapters.http.application import get_application
from dlightrag.application.access import (
    AccessControl,
    AccessDeniedError,
    AccessGate,
    WorkspaceRecord,
    access_control_from_settings,
)
from dlightrag.application.settings import access_settings

if TYPE_CHECKING:
    from dlightrag.application.web_conversations import WebConversationService


def get_workspace(
    request: Request,
    dlightrag_workspace: str | None = Cookie(default=None),
) -> str:
    """Read the current workspace from its cookie, else the configured deployment one."""
    from dlightrag.application.corpus_admin import normalize_workspace

    return normalize_workspace(
        dlightrag_workspace or get_application(request).config.deployment.workspace
    )


def get_web_conversation_service(request: Request) -> WebConversationService:
    """Return the typed Web service through the Application lifetime guard."""
    return get_application(request).web_conversations


def _web_access_control(request: Request) -> AccessControl:
    return getattr(request.app.state, "access_control", None) or access_control_from_settings(
        access_settings(get_application(request).config)
    )


def get_web_access_gate(request: Request) -> AccessGate:
    return AccessGate(
        _web_access_control(request),
        getattr(request.state, "user_context", None),
    )


async def enforce_web_access(request: Request, action: str, workspace: str | None) -> None:
    try:
        await get_web_access_gate(request).check(action, workspace=workspace)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None


async def filter_web_workspace_records(
    request: Request,
    action: str,
    records: Sequence[WorkspaceRecord],
) -> list[WorkspaceRecord]:
    return await get_web_access_gate(request).filter_workspace_records(action, records)
