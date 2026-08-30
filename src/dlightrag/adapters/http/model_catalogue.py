# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared HTTP contract and error projection for runtime model catalogue routes."""

from collections.abc import Awaitable

from fastapi import HTTPException, Response
from pydantic import BaseModel, ConfigDict

from dlightrag.application.model_catalogue import (
    ModelCatalogueAdmin,
    ModelCatalogueEntryNotFoundError,
    ModelCatalogueReadOnlyError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueUnavailableError,
    ModelCatalogueValidationError,
    ModelCatalogueView,
)


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


def read_catalogue(catalogue: ModelCatalogueAdmin, response: Response) -> ModelCatalogueResponse:
    """Read and project the effective catalogue."""
    try:
        view = catalogue.read()
    except ModelCatalogueUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return _response(response, view)


async def mutate_catalogue(
    operation: Awaitable[ModelCatalogueView],
    response: Response,
) -> ModelCatalogueResponse:
    """Await one mutation and project the catalogue error contract once."""
    try:
        view = await operation
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
    return _response(response, view)


def normalize_if_match(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1]
    return normalized


def _response(response: Response, view: ModelCatalogueView) -> ModelCatalogueResponse:
    response.headers["ETag"] = _etag(view.revision)
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


def _etag(revision: str) -> str:
    return f'"{revision}"'


__all__ = [
    "ModelCatalogueEntryPayload",
    "ModelCatalogueResponse",
    "mutate_catalogue",
    "normalize_if_match",
    "read_catalogue",
]
