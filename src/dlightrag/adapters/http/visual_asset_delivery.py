# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared HTTP response projection for authorized visual chunk assets."""

from fastapi import HTTPException
from fastapi.responses import Response

from dlightrag.application.corpus_admin import CorpusAdmin, VisualAssetSize


async def visual_asset_response(
    corpora: CorpusAdmin,
    *,
    workspace: str,
    chunk_id: str,
    size: VisualAssetSize,
) -> Response:
    """Serve one visual asset the caller was just authorized to read.

    The bytes are per-authorization: a shared cache must never replay them to
    another caller, and a hidden document must stop resolving on the next read.
    """
    asset = await corpora.get_visual_asset(workspace, chunk_id, size=size)
    if asset is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=asset.data,
        media_type=asset.media_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = ["visual_asset_response"]
