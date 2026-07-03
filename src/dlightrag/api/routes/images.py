# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authenticated visual chunk image routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from dlightrag.access_control import AccessAction
from dlightrag.api.auth import UserContext, get_current_user

from .deps import enforce_access, get_manager, resolve_workspace

router = APIRouter()


@router.get("/images/{workspace}/{chunk_id}")
async def image(
    workspace: str,
    chunk_id: str,
    request: Request,
    size: Literal["full", "thumb"] = "thumb",
    user: UserContext = Depends(get_current_user),
) -> Response:
    """Serve a LightRAG sidecar-backed visual chunk asset."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    await enforce_access(request, user, AccessAction.WORKSPACE_READ_VISUAL_ASSET, workspace=ws)
    asset = await manager.aget_visual_asset(ws, chunk_id, size=size)
    if asset is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=asset.data,
        media_type=asset.media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


__all__ = ["router"]
