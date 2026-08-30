# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST adapter for the runtime model catalogue application interface."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.routes.deps import enforce_access, get_application
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.model_catalogue import (
    ModelCatalogueEntryNotFoundError,
    ModelCatalogueReadOnlyError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueUnavailableError,
    ModelCatalogueValidationError,
    ModelCatalogueView,
)

router = APIRouter()


class ReasoningLevelsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    off: str | None
    minimal: str | None
    low: str | None
    medium: str | None
    high: str | None
    xhigh: str | None
    max: str | None


class ReasoningProfilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str
    levels: ReasoningLevelsPayload


class ModelProfilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_window_tokens: int
    max_input_tokens: int | None
    max_output_tokens: int | None
    supports_images: bool
    reasoning: ReasoningProfilePayload | None


class ModelCatalogueEntryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    base_url: str | None
    profile: ModelProfilePayload


class ModelCatalogueEntryResponse(ModelCatalogueEntryPayload):
    source: str


class ModelCatalogueResponse(BaseModel):
    revision: str
    models: list[ModelCatalogueEntryResponse]


def _etag(revision: str) -> str:
    return f'"{revision}"'


def _if_match(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1]
    return normalized


def _response(view: ModelCatalogueView) -> ModelCatalogueResponse:
    return ModelCatalogueResponse(
        revision=view.revision,
        models=[
            ModelCatalogueEntryResponse(
                provider=item.provider,
                model=item.model,
                base_url=item.base_url,
                profile=ModelProfilePayload.model_validate(item.profile),
                source=item.source,
            )
            for item in view.models
        ],
    )


@router.get("/models/catalogue", response_model=ModelCatalogueResponse)
async def read_model_catalogue(
    request: Request,
    response: Response,
    _user: UserContext = Depends(get_current_user),
) -> ModelCatalogueResponse:
    try:
        view = get_application(request).model_catalogue.read()
    except ModelCatalogueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    response.headers["ETag"] = _etag(view.revision)
    return _response(view)


@router.put("/models/catalogue", response_model=ModelCatalogueResponse)
async def upsert_model_catalogue_entry(
    payload: ModelCatalogueEntryPayload,
    request: Request,
    response: Response,
    if_match: str = Header(alias="If-Match"),
    user: UserContext = Depends(get_current_user),
) -> ModelCatalogueResponse:
    await enforce_access(request, user, AccessAction.MODEL_CATALOGUE_WRITE)
    try:
        view = await get_application(request).model_catalogue.upsert(
            payload.model_dump(),
            expected_revision=_if_match(if_match),
            actor=user.user_id,
        )
    except ModelCatalogueRevisionConflict as exc:
        raise HTTPException(
            status_code=412,
            detail=str(exc),
            headers={"ETag": _etag(exc.current_revision)},
        ) from None
    except ModelCatalogueReadOnlyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ModelCatalogueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ModelCatalogueValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    response.headers["ETag"] = _etag(view.revision)
    return _response(view)


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
    try:
        view = await get_application(request).model_catalogue.remove(
            provider=provider,
            model=model,
            base_url=base_url,
            expected_revision=_if_match(if_match),
            actor=user.user_id,
        )
    except ModelCatalogueRevisionConflict as exc:
        raise HTTPException(
            status_code=412,
            detail=str(exc),
            headers={"ETag": _etag(exc.current_revision)},
        ) from None
    except ModelCatalogueEntryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from None
    except ModelCatalogueReadOnlyError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except ModelCatalogueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ModelCatalogueValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    response.headers["ETag"] = _etag(view.revision)
    return _response(view)


__all__ = [
    "ModelCatalogueEntryPayload",
    "ModelCatalogueResponse",
    "router",
]
