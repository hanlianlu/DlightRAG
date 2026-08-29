# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata operations API routes."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import (
    MetadataResponse,
    MetadataUpdateRequest,
    MetadataUpdateResponse,
    SearchMetadataResponse,
)
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.corpus_admin import (
    METADATA_SEARCH_PAGE_DEFAULT_LIMIT,
    METADATA_SEARCH_PAGE_MAX_LIMIT,
    MetadataSearchCursorError,
    MetadataSearchPageRequest,
)

from .deps import enforce_access, get_application, resolve_workspace

router = APIRouter()


@router.post("/metadata/search", response_model=SearchMetadataResponse)
async def search_metadata(
    filters: dict[str, Any],
    request: Request,
    workspace: str | None = None,
    user: UserContext = Depends(get_current_user),
    limit: Annotated[
        int,
        Query(ge=1, le=METADATA_SEARCH_PAGE_MAX_LIMIT),
    ] = METADATA_SEARCH_PAGE_DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
) -> dict[str, Any]:
    """Return one bounded page of document IDs matching 'filters'."""
    from pydantic import ValidationError

    from dlightrag.application.retrieval import MetadataFilter

    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_READ_METADATA, workspace=ws)

    # Validate the user-supplied dict against the MetadataFilter schema.
    # The storage backend takes a Pydantic model, not a raw dict, so this also
    # rejects unknown keys before they reach the SQL layer.
    try:
        validated = MetadataFilter.model_validate(filters)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata filter: {exc}") from exc

    try:
        decoded_cursor = (
            application.corpora.metadata_search_cursor_codec.decode(cursor)
            if cursor is not None
            else None
        )
        if decoded_cursor is not None and decoded_cursor.workspace != ws:
            raise MetadataSearchCursorError("metadata-search cursor belongs to another workspace")
        page_request = MetadataSearchPageRequest(limit=limit, cursor=decoded_cursor)
    except (MetadataSearchCursorError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    page = await application.corpora.search_metadata(ws, validated, page=page_request)
    return {
        "document_ids": list(page.document_ids),
        "count": len(page.document_ids),
        "workspace": ws,
        "next_cursor": (
            application.corpora.metadata_search_cursor_codec.encode(page.next_cursor)
            if page.next_cursor is not None
            else None
        ),
    }


@router.get("/metadata/{doc_id}", response_model=MetadataResponse)
async def get_metadata(
    doc_id: str,
    request: Request,
    workspace: str | None = None,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve metadata of a specific document incrementally."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_READ_METADATA, workspace=ws)
    data = await application.corpora.get_metadata(ws, doc_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"doc_id": doc_id, "metadata": data}


@router.post("/metadata/{doc_id}", response_model=MetadataUpdateResponse)
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

    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_UPDATE_METADATA, workspace=ws)
    try:
        await application.corpora.update_metadata(ws, doc_id, body.metadata)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found") from None
    return {"status": "success", "doc_id": doc_id}
