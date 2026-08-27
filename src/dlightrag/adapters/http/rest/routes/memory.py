# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST Adapter for owner Profile Memory."""

from typing import Annotated, Any, Literal

from dlightrag_memory import MemoryOperationReceipt, MemoryProvenance
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.application.access import UserContext, owner_id_from_user
from dlightrag.application.answer_runs.errors import (
    MemoryDisabledError,
    MemoryUnavailableError,
    MemoryWriteRejectedError,
)
from dlightrag.application.memory import MemorySettings

from .deps import get_application

router = APIRouter()
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)]


class RememberMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["preference", "fact"]
    body: str = Field(min_length=1, max_length=500)
    supersedes_id: str | None = None


class MemorySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="Whether this owner's Profile Memory capability is active.")


@router.get("/memory")
async def list_memories(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
    try:
        rows = await application.memory.list_active(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _memory_http_error(exc) from exc
    return {
        "memories": [
            {"memory_id": row.memory_id, "kind": row.kind, "body": row.body} for row in rows
        ]
    }


@router.post("/memory")
async def remember_memory(
    body: RememberMemoryInput,
    request: Request,
    idempotency_key: IdempotencyKey,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application = get_application(request)
    try:
        receipt = await application.memory.remember(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            kind=body.kind,
            body=body.body,
            supersedes_id=body.supersedes_id,
            provenance=_management_provenance(idempotency_key),
            idempotency_key=f"rest:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _memory_http_error(exc) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=409, detail=exc.public_message) from exc
    return _receipt_payload(receipt)


@router.delete("/memory/{memory_id}")
async def forget_memory(
    memory_id: str,
    request: Request,
    idempotency_key: IdempotencyKey,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application = get_application(request)
    try:
        receipt = await application.memory.forget(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            memory_id=memory_id,
            provenance=_management_provenance(idempotency_key),
            idempotency_key=f"rest:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _memory_http_error(exc) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=409, detail=exc.public_message) from exc
    return _receipt_payload(receipt)


@router.post("/memory/changes/{change_id}/undo")
async def undo_memory_change(
    change_id: str,
    request: Request,
    idempotency_key: IdempotencyKey,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application = get_application(request)
    try:
        receipt = await application.memory.undo(
            owner_id=owner_id_from_user(user),
            auth_mode=user.auth_mode,
            change_id=change_id,
            provenance=_undo_provenance(idempotency_key),
            idempotency_key=f"rest:{idempotency_key}",
        )
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _memory_http_error(exc) from exc
    except MemoryWriteRejectedError as exc:
        raise HTTPException(status_code=409, detail=exc.public_message) from exc
    return _receipt_payload(receipt)


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
        raise _memory_http_error(exc) from exc
    return _settings_payload(settings)


@router.put("/memory/settings")
async def update_memory_settings(
    body: MemorySettingsInput,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    application = get_application(request)
    try:
        settings = await application.memory.set_enabled(
            owner_id=owner_id_from_user(user), auth_mode=user.auth_mode, enabled=body.enabled
        )
    except MemoryUnavailableError as exc:
        raise _memory_http_error(exc) from exc
    return _settings_payload(settings)


@router.post("/memory/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(request: Request, user: UserContext = Depends(get_current_user)) -> None:
    application = get_application(request)
    try:
        await application.memory.clear(owner_id=owner_id_from_user(user), auth_mode=user.auth_mode)
    except (MemoryUnavailableError, MemoryDisabledError) as exc:
        raise _memory_http_error(exc) from exc


def _management_provenance(idempotency_key: str) -> MemoryProvenance:
    return MemoryProvenance(origin_kind="management", origin_id=idempotency_key)


def _undo_provenance(idempotency_key: str) -> MemoryProvenance:
    return MemoryProvenance(origin_kind="undo", origin_id=idempotency_key)


def _receipt_payload(receipt: MemoryOperationReceipt) -> dict[str, Any]:
    return {
        "action": receipt.action,
        "body": receipt.body,
        "change_id": receipt.change_id,
        "kind": receipt.kind,
        "memory_ids": list(receipt.memory_ids),
        "outcome": receipt.outcome,
        "supersedes_id": receipt.supersedes_id,
        "target_change_id": receipt.target_change_id,
    }


def _settings_payload(settings: MemorySettings) -> dict[str, Any]:
    return {"enabled": settings.enabled, "active_count": settings.active_count}


def _memory_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MemoryDisabledError):
        return HTTPException(status_code=409, detail=exc.public_message)
    return HTTPException(status_code=403, detail=getattr(exc, "public_message", str(exc)))


__all__ = ["router"]
