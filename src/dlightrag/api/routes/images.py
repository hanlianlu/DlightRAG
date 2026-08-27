# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authenticated visual chunk image routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from dlightrag.api.auth import get_current_user
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.engine.rag.corpus.contracts import VisualAssetSize

from .deps import enforce_access, get_application, resolve_workspace

router = APIRouter()


@router.get("/images/{workspace}/{chunk_id}")
async def image(
    workspace: str,
    chunk_id: str,
    request: Request,
    size: VisualAssetSize = "thumb",
    user: UserContext = Depends(get_current_user),
) -> Response:
    """Serve a LightRAG sidecar-backed visual chunk asset."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_READ_VISUAL_ASSET, workspace=ws)
    asset = await application.corpora.get_visual_asset(ws, chunk_id, size=size)
    if asset is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=asset.data,
        media_type=asset.media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


__all__ = ["router"]
