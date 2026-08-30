# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-session adapter for runtime model catalogue administration."""

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
from dlightrag.adapters.http.rest.routes.model_catalogue import (
    ModelCatalogueEntryPayload,
    ModelCatalogueResponse,
    _etag,
    _if_match,
    _response,
)
from dlightrag.application.access import AccessAction
from dlightrag.application.model_catalogue import (
    ModelCatalogueEntryNotFoundError,
    ModelCatalogueReadOnlyError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueUnavailableError,
    ModelCatalogueValidationError,
)

router = APIRouter()


def _actor(request: Request) -> str:
    user = getattr(request.state, "user_context", None)
    return str(getattr(user, "user_id", None) or "anonymous")


@router.get("/models/catalogue", response_model=ModelCatalogueResponse)
async def read_model_catalogue(request: Request, response: Response) -> ModelCatalogueResponse:
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
) -> ModelCatalogueResponse:
    await enforce_web_access(request, AccessAction.MODEL_CATALOGUE_WRITE, None)
    try:
        view = await get_application(request).model_catalogue.upsert(
            payload.model_dump(),
            expected_revision=_if_match(if_match),
            actor=_actor(request),
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
) -> ModelCatalogueResponse:
    await enforce_web_access(request, AccessAction.MODEL_CATALOGUE_WRITE, None)
    try:
        view = await get_application(request).model_catalogue.remove(
            provider=provider,
            model=model,
            base_url=base_url,
            expected_revision=_if_match(if_match),
            actor=_actor(request),
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


__all__ = ["router"]
