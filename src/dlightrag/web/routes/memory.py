# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web-only Cross-conversation Memory management routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.access import owner_id_from_user
from dlightrag.answer.errors import MemoryUnavailableError
from dlightrag.web.deps import get_application

router = APIRouter()


def _user(request: Request) -> Any:
    return getattr(request.state, "user_context", None)


@router.get("/memory/settings")
async def memory_settings(request: Request) -> dict[str, object]:
    application = get_application(request)
    user = _user(request)
    try:
        settings = await application.memory.settings(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode
        )
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc
    return {"enabled": settings.enabled, "active_count": settings.active_count}


class MemorySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="Whether answer injection may use this owner's memory.")


@router.put("/memory/settings")
async def update_memory_settings(request: Request, body: MemorySettingsInput) -> dict[str, object]:
    application = get_application(request)
    user = _user(request)
    try:
        await application.memory.set_enabled(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode, enabled=body.enabled
        )
        settings = await application.memory.settings(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode
        )
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc
    return {"enabled": settings.enabled, "active_count": settings.active_count}


@router.post("/memory/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(request: Request) -> None:
    """Idempotently delete every Memory Record; enablement is untouched."""
    application = get_application(request)
    user = _user(request)
    try:
        await application.memory.clear(owner_id=owner_id_from_user(user), auth_mode=user.auth_mode)
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc


__all__ = ["router"]
