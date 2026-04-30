# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata operations API routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import MetadataUpdateRequest

from .deps import get_manager, resolve_workspace

router = APIRouter()


@router.get("/metadata/{doc_id}")
async def get_metadata(
    doc_id: str,
    request: Request,
    workspace: str | None = None,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve metadata of a specific document incrementally."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    data = await manager.aget_metadata(ws, doc_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"doc_id": doc_id, "metadata": data}


@router.post("/metadata/{doc_id}")
async def update_metadata(
    doc_id: str,
    body: MetadataUpdateRequest,
    request: Request,
    workspace: str | None = None,
    user: UserContext = Depends(get_current_user),
) -> dict[str, str]:
    """Merge custom metadata dict into existing document's metadata JSONB."""
    if not body.metadata:
        raise HTTPException(status_code=400, detail="Empty 'metadata' dictionary")

    for k in body.metadata:
        if k in ("id", "_id", "doc_id", "content", "source"):
            raise HTTPException(status_code=400, detail=f"Cannot overwrite reserved key '{k}'")

    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    await manager.aupdate_metadata(ws, doc_id, body.metadata)
    return {"status": "success", "doc_id": doc_id}


@router.post("/metadata/search")
async def search_metadata(
    filters: dict[str, Any],
    request: Request,
    workspace: str | None = None,
    user: UserContext = Depends(get_current_user),
) -> list[str]:
    """Return a list of document IDs matching all key-value pairs in 'filters'."""
    from pydantic import ValidationError

    from dlightrag.core.retrieval.models import MetadataFilter

    # Validate the user-supplied dict against the MetadataFilter schema.
    # The storage backend's query() takes a Pydantic model, not a raw dict,
    # so this also rejects unknown keys before they reach the SQL layer.
    try:
        validated = MetadataFilter.model_validate(filters)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    return await manager.asearch_metadata(ws, validated)
