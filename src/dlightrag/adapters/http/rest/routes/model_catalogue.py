# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST adapter for the runtime model catalogue application interface."""

from fastapi import APIRouter, Depends, Header, Query, Request, Response

from dlightrag.adapters.http.model_catalogue import (
    ModelCatalogueEntryPayload,
    ModelCatalogueResponse,
    mutate_catalogue,
    normalize_if_match,
    read_catalogue,
)
from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.routes.deps import enforce_access, get_application
from dlightrag.application.access import AccessAction, UserContext

router = APIRouter()


@router.get("/models/catalogue", response_model=ModelCatalogueResponse)
async def read_model_catalogue(
    request: Request,
    response: Response,
    _user: UserContext = Depends(get_current_user),
) -> ModelCatalogueResponse:
    return read_catalogue(get_application(request).model_catalogue, response)


@router.put("/models/catalogue", response_model=ModelCatalogueResponse)
async def upsert_model_catalogue_entry(
    payload: ModelCatalogueEntryPayload,
    request: Request,
    response: Response,
    if_match: str = Header(alias="If-Match"),
    user: UserContext = Depends(get_current_user),
) -> ModelCatalogueResponse:
    await enforce_access(request, user, AccessAction.MODEL_CATALOGUE_WRITE)
    return await mutate_catalogue(
        get_application(request).model_catalogue.upsert(
            payload.model_dump(),
            expected_revision=normalize_if_match(if_match),
            actor=user.user_id,
        ),
        response,
    )


@router.delete("/models/catalogue", response_model=ModelCatalogueResponse)
async def remove_model_catalogue_entry(
    request: Request,
    response: Response,
    provider: str = Query(),
    model: str = Query(),
    base_url: str | None = Query(default=None),
    if_match: str = Header(alias="If-Match"),
    user: UserContext = Depends(get_current_user),
) -> ModelCatalogueResponse:
    await enforce_access(request, user, AccessAction.MODEL_CATALOGUE_WRITE)
    return await mutate_catalogue(
        get_application(request).model_catalogue.remove(
            provider=provider,
            model=model,
            base_url=base_url,
            expected_revision=normalize_if_match(if_match),
            actor=user.user_id,
        ),
        response,
    )


__all__ = [
    "ModelCatalogueEntryPayload",
    "ModelCatalogueResponse",
    "router",
]
