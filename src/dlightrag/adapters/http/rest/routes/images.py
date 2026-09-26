# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authenticated visual chunk image routes."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.visual_asset_delivery import visual_asset_response
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.corpus_admin import VisualAssetSize

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
    return await visual_asset_response(
        application.corpora, workspace=ws, chunk_id=chunk_id, size=size
    )


__all__ = ["router"]
