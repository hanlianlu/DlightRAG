# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System status and health API routes.

``/health`` is liveness only: it answers from in-process state and never touches
PostgreSQL, so an unauthenticated poll loop cannot turn it into database load.
``/ready`` projects the application-owned readiness verdict. Composition injects
the database/corpus probe; this transport imports no storage implementation.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dlightrag.api.models import HealthResponse, ReadinessResponse
from dlightrag.app_state import request_config
from dlightrag.application import ApplicationHealth
from dlightrag.contracts import ServiceRole

router = APIRouter()


def _application_health(request: Request) -> ApplicationHealth:
    return request.app.state.health


def _not_ready(*, service_role: ServiceRole, detail: str) -> JSONResponse:
    payload = ReadinessResponse(
        status="not_ready",
        service_role=service_role,
        detail=detail,
    )
    return JSONResponse(status_code=503, content=payload.model_dump(exclude_none=True))


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health(request: Request) -> dict[str, object]:
    """Report process liveness and the capabilities this build exposes."""
    config = request_config(request)
    application_health = _application_health(request)

    warnings = application_health.warnings
    status: dict[str, object] = {
        "status": "degraded" if application_health.is_degraded else "healthy",
        "rag_initialized": application_health.is_ready,
        "service_role": config.service_role,
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
        "answer_image_capability": application_health.answer_image_capability,
    }
    if warnings:
        status["warnings"] = warnings
    return status


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    response_model_exclude_none=True,
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    """Return whether this process can accept query traffic."""
    config = request_config(request)
    detail = await _application_health(request).readiness_detail()
    if detail is not None:
        return _not_ready(service_role=config.service_role, detail=detail)

    return ReadinessResponse(status="ready", service_role=config.service_role)
