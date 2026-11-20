# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web Adapter for owner Profile Memory settings and mutations."""

from typing import Annotated, Any
from uuid import UUID

from dlightrag_memory import MemoryKind, MemoryProvenance
from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.adapters.http.browser.deps import get_application
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.memory.memory_list import (
    MEMORY_LIST_PAGE_DEFAULT_LIMIT,
    MEMORY_LIST_PAGE_MAX_LIMIT,
    MemoryListCursorError,
    MemoryListPageRequest,
)
from dlightrag.application.memory.projections import (
    memory_receipt_payload,
    memory_run_changes_payload,
)
from dlightrag.application.memory.service import MemorySettings

router = APIRouter()
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)]


def _user(request: Request) -> Any:
    return getattr(request.state, "user_context", None)


class MemorySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="Whether this owner's Profile Memory capability is active.")


class RememberMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: MemoryKind
    body: str = Field(min_length=1, max_length=500)
    supersedes_id: UUID | None = None


@router.get("/memory")
async def list_memories(
    request: Request,
    limit: Annotated[
        int,
        Query(ge=1, le=MEMORY_LIST_PAGE_MAX_LIMIT),
    ] = MEMORY_LIST_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    try:
        decoded_cursor = (
            application.memory.memory_list_cursor_codec.decode(cursor)
            if cursor is not None
            else None
        )
        page_request = MemoryListPageRequest(limit=limit, cursor=decoded_cursor)
    except (MemoryListCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    page = await application.memory.list_active_page(
        owner_id=owner_id_from_user(user),
        page=page_request,
    )
    mask_reader_text = application.skill_protection.mask
    return {
        "memories": [
            {
                "memory_id": row.memory_id,
                "kind": row.kind,
                "body": mask_reader_text(row.body),
                "origin_kind": row.provenance.origin_kind,
            }
            for row in page.records
        ],
        "next_cursor": (
            application.memory.memory_list_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


@router.get("/memory/runs/{run_id}/changes")
async def run_memory_changes(run_id: str, request: Request) -> dict[str, Any]:
    """What this Run's writers left in the owner's Profile Memory, still active; empty while off."""
    application = get_application(request)
    user = _user(request)
    changes = await application.memory.run_changes(owner_id=owner_id_from_user(user), run_id=run_id)
    return memory_run_changes_payload(changes, mask_reader_text=application.skill_protection.mask)


@router.get("/memory/settings")
async def memory_settings(request: Request) -> dict[str, object]:
    application = get_application(request)
    user = _user(request)
    settings = await application.memory.settings(owner_id=owner_id_from_user(user))
    return _settings(settings)


@router.put("/memory/settings")
async def update_memory_settings(request: Request, body: MemorySettingsInput) -> dict[str, object]:
    application = get_application(request)
    user = _user(request)
    settings = await application.memory.set_enabled(
        owner_id=owner_id_from_user(user), enabled=body.enabled
    )
    return _settings(settings)


@router.post("/memory")
async def remember_memory(
    request: Request, body: RememberMemoryInput, idempotency_key: IdempotencyKey
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    receipt = await application.memory.remember(
        owner_id=owner_id_from_user(user),
        kind=body.kind,
        body=body.body,
        supersedes_id=None if body.supersedes_id is None else str(body.supersedes_id),
        provenance=MemoryProvenance(origin_kind="management", origin_id=idempotency_key),
        idempotency_key=f"web:{idempotency_key}",
    )
    return memory_receipt_payload(receipt, mask_reader_text=application.skill_protection.mask)


@router.delete("/memory/{memory_id}")
async def forget_memory(
    memory_id: UUID, request: Request, idempotency_key: IdempotencyKey
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    receipt = await application.memory.forget(
        owner_id=owner_id_from_user(user),
        memory_id=str(memory_id),
        provenance=MemoryProvenance(origin_kind="management", origin_id=idempotency_key),
        idempotency_key=f"web:{idempotency_key}",
    )
    return memory_receipt_payload(receipt, mask_reader_text=application.skill_protection.mask)


@router.post("/memory/changes/{change_id}/undo")
async def undo_memory_change(
    change_id: UUID, request: Request, idempotency_key: IdempotencyKey
) -> dict[str, Any]:
    application = get_application(request)
    user = _user(request)
    receipt = await application.memory.undo(
        owner_id=owner_id_from_user(user),
        change_id=str(change_id),
        provenance=MemoryProvenance(origin_kind="undo", origin_id=idempotency_key),
        idempotency_key=f"web:{idempotency_key}",
    )
    return memory_receipt_payload(receipt, mask_reader_text=application.skill_protection.mask)


@router.post("/memory/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(request: Request) -> None:
    application = get_application(request)
    user = _user(request)
    await application.memory.clear(owner_id=owner_id_from_user(user))


def _settings(settings: MemorySettings) -> dict[str, object]:
    return {"enabled": settings.enabled, "active_count": settings.active_count}


__all__ = ["router"]
