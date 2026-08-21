# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST list, forget, settings, and clear for owner-scoped Memory Records."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.access import UserContext, owner_id_from_user
from dlightrag.answer.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag.api.auth import get_current_user

from .deps import get_application

router = APIRouter()


@router.get("/memory")
async def list_memories(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
    try:
        rows = await application.memory.list_active(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode
        )
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc
    return {
        "memories": [
            {
                "memory_id": row.memory_id,
                "kind": row.kind,
                "body": row.body,
                "confidence": row.confidence,
            }
            for row in rows
        ]
    }


@router.delete("/memory/{memory_id}", status_code=204)
async def forget_memory(
    memory_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> None:
    application = get_application(request)
    try:
        await application.memory.forget(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            memory_id=memory_id,
        )
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=404, detail=exc.public_message) from exc


@router.get("/memory/settings")
async def memory_settings(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
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
async def update_memory_settings(
    body: MemorySettingsInput, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
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


@router.post("/memory/clear", status_code=204)
async def clear_memory(request: Request, user: UserContext = Depends(get_current_user)) -> None:
    """Idempotently delete every Memory Record; enablement is untouched."""
    application = get_application(request)
    try:
        await application.memory.clear(owner_id=owner_id_from_user(user), auth_mode=user.auth_mode)
    except MemoryUnavailableError as exc:
        raise HTTPException(status_code=403, detail=exc.public_message) from exc


__all__ = ["router"]
