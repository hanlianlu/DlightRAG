# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System status and health API routes."""

import logging
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dlightrag.api.models import HealthResponse, ReadinessResponse
from dlightrag.app_state import request_config
from dlightrag.core.answer.capability import answer_image_capability_summary

from .deps import get_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _not_ready(*, service_role: Literal["writer", "reader"], detail: str) -> JSONResponse:
    payload = ReadinessResponse(
        status="not_ready",
        service_role=service_role,
        detail=detail,
    )
    return JSONResponse(status_code=503, content=payload.model_dump(exclude_none=True))


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health(request: Request) -> dict[str, object]:
    """Health check including RAG service status."""
    config = request_config(request)
    manager = get_manager(request)

    is_degraded = manager.is_degraded()
    warnings = manager.get_warnings()

    status: dict[str, object] = {
        "status": "degraded" if is_degraded else "healthy",
        "rag_initialized": manager.is_ready(),
        "service_role": config.service_role,
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
        "answer_image_capability": answer_image_capability_summary(manager.answer_image_capability),
    }
    if warnings:
        status["warnings"] = warnings

    # Check PostgreSQL connectivity through the shared pool rather than opening a
    # fresh connection per request -- the latter is wasteful and lets unauthenticated
    # health traffic exhaust the server's connection slots. Readers additionally
    # confirm the domain pool holds the read-only session invariant.
    try:
        from dlightrag.storage.pool import pg_pool

        if config.is_reader:
            read_only = await pg_pool.run_once(
                lambda conn: conn.fetchval("SHOW transaction_read_only")
            )
            if str(read_only).lower() != "on":
                raise RuntimeError("reader domain pool is not read-only")
        else:
            await pg_pool.run_once(lambda conn: conn.fetchval("SELECT 1"))
        status["postgres"] = "connected"
    except Exception:
        logger.warning("Health check: PostgreSQL probe failed", exc_info=True)
        status["postgres"] = "error"
        status["status"] = "degraded"

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
    manager = get_manager(request)
    if not manager.is_ready():
        return _not_ready(
            service_role=config.service_role,
            detail="RAG service is not ready",
        )

    try:
        from dlightrag.storage.pool import pg_pool

        if config.is_reader:
            read_only = await pg_pool.run_once(
                lambda conn: conn.fetchval("SHOW transaction_read_only")
            )
            if str(read_only).lower() != "on":
                return _not_ready(
                    service_role=config.service_role,
                    detail="Reader database session is not read-only",
                )
        else:
            await pg_pool.run_once(lambda conn: conn.fetchval("SELECT 1"))
    except Exception:
        logger.warning("Readiness check: PostgreSQL probe failed", exc_info=True)
        return _not_ready(
            service_role=config.service_role,
            detail="PostgreSQL readiness check failed",
        )

    return ReadinessResponse(status="ready", service_role=config.service_role)
