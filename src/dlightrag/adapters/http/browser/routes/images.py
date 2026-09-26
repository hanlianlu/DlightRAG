# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web visual chunk image routes."""

from fastapi import APIRouter, Request
from fastapi.responses import Response

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
from dlightrag.adapters.http.visual_asset_delivery import visual_asset_response
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
    return await visual_asset_response(
        application.corpora, workspace=workspace_id, chunk_id=chunk_id, size=size
    )


__all__ = ["router"]
