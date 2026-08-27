# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web visual chunk image routes."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
from dlightrag.application.access import AccessAction
from dlightrag.application.corpus_admin import VisualAssetSize, normalize_workspace

router = APIRouter()


@router.get("/images/{workspace}/{chunk_id}")
async def image(
    workspace: str,
    chunk_id: str,
    request: Request,
    size: VisualAssetSize = "thumb",
) -> Response:
    """Serve a same-origin source panel image."""
    application = get_application(request)
    workspace_id = normalize_workspace(workspace)
    await enforce_web_access(request, AccessAction.WORKSPACE_READ_VISUAL_ASSET, workspace_id)
    asset = await application.corpora.get_visual_asset(workspace_id, chunk_id, size=size)
    if asset is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=asset.data,
        media_type=asset.media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


__all__ = ["router"]
