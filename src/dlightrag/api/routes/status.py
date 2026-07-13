# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System status and health API routes."""

import logging

from fastapi import APIRouter, Request

from dlightrag.api.models import HealthResponse
from dlightrag.app_state import request_config
from dlightrag.core.answer_capability import answer_image_capability_summary

from .deps import get_manager

logger = logging.getLogger(__name__)

router = APIRouter()


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

    # Check PostgreSQL connectivity.
    try:
        import asyncpg

        conn = await asyncpg.connect(**config.pg_connection_kwargs())
        await conn.fetchval("SELECT 1")
        await conn.close()
        status["postgres"] = "connected"
    except Exception:
        logger.warning("Health check: PostgreSQL probe failed", exc_info=True)
        status["postgres"] = "error"
        status["status"] = "degraded"

    return status
