# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-session adapter for runtime model catalogue administration."""

from fastapi import APIRouter, Header, Query, Request, Response

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
from dlightrag.adapters.http.model_catalogue import (
    ModelCatalogueEntryPayload,
    ModelCatalogueResponse,
    mutate_catalogue,
    normalize_if_match,
    read_catalogue,
)
from dlightrag.application.access import AccessAction

router = APIRouter()


def _actor(request: Request) -> str:
    user = getattr(request.state, "user_context", None)
    return str(getattr(user, "user_id", None) or "anonymous")


@router.get("/models/catalogue", response_model=ModelCatalogueResponse)
async def read_model_catalogue(request: Request, response: Response) -> ModelCatalogueResponse:
    return read_catalogue(get_application(request).model_catalogue, response)


@router.put("/models/catalogue", response_model=ModelCatalogueResponse)
async def upsert_model_catalogue_entry(
    payload: ModelCatalogueEntryPayload,
    request: Request,
    response: Response,
    if_match: str = Header(alias="If-Match"),
) -> ModelCatalogueResponse:
    await enforce_web_access(request, AccessAction.MODEL_CATALOGUE_WRITE, None)
    return await mutate_catalogue(
        get_application(request).model_catalogue.upsert(
            payload.model_dump(),
            expected_revision=normalize_if_match(if_match),
            actor=_actor(request),
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
) -> ModelCatalogueResponse:
    await enforce_web_access(request, AccessAction.MODEL_CATALOGUE_WRITE, None)
    return await mutate_catalogue(
        get_application(request).model_catalogue.remove(
            provider=provider,
            model=model,
            base_url=base_url,
            expected_revision=normalize_if_match(if_match),
            actor=_actor(request),
        ),
        response,
    )


__all__ = ["router"]
