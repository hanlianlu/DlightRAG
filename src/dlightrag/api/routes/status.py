# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System status and health API routes."""

from typing import Any

from fastapi import APIRouter, Depends, Request

from dlightrag.api.auth import UserContext, get_current_user

from .deps import get_manager

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Health check including RAG service status."""
    from dlightrag.config import get_config

    config = get_config()
    manager = get_manager(request)

    is_degraded = manager.is_degraded()
    warnings = manager.get_warnings()

    status: dict[str, Any] = {
        "status": "degraded" if is_degraded else "healthy",
        "rag_initialized": manager.is_ready(),
        "runtime_role": config.runtime_role,
        "postgres_target": config.pg_target_for_runtime(),
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
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
    except Exception as e:
        status["postgres"] = f"error: {e}"
        status["status"] = "degraded"

    return status


@router.get("/workspaces")
async def workspaces(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """List all available workspaces."""
    manager = get_manager(request)
    ws_list = await manager.list_workspaces()
    return {"workspaces": ws_list}
