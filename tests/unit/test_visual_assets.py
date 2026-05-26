# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG sidecar visual asset resolution."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.visual_assets import ThumbnailCache, VisualAssetResolver

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


async def test_visual_asset_resolver_returns_full_asset() -> None:
    chunk = {"chunk_id": "chunk_1", "image_data": _PNG_B64, "image_mime_type": "image/png"}
    rag = MagicMock(text_chunks=object())
    resolver = VisualAssetResolver(lightrag=rag)

    with (
        patch(
            "dlightrag.core.visual_assets.fetch_chunks_by_ids",
            new=AsyncMock(return_value=[chunk]),
        ) as fetch,
        patch(
            "dlightrag.core.visual_assets.hydrate_lightrag_chunk_provenance",
            new=AsyncMock(),
        ) as hydrate,
    ):
        asset = await resolver.resolve("chunk_1")

    assert asset is not None
    assert asset.chunk_id == "chunk_1"
    assert asset.media_type == "image/png"
    assert asset.data == base64.b64decode(_PNG_B64)
    fetch.assert_awaited_once()
    hydrate.assert_awaited_once()


async def test_visual_asset_resolver_generates_and_caches_thumbnail() -> None:
    chunk = {"chunk_id": "chunk_1", "image_data": _PNG_B64, "image_mime_type": "image/png"}
    rag = MagicMock(text_chunks=object())
    resolver = VisualAssetResolver(lightrag=rag, thumb_cache=ThumbnailCache(max_size=2))

    with (
        patch(
            "dlightrag.core.visual_assets.fetch_chunks_by_ids",
            new=AsyncMock(return_value=[chunk]),
        ) as fetch,
        patch(
            "dlightrag.core.visual_assets.hydrate_lightrag_chunk_provenance",
            new=AsyncMock(),
        ),
    ):
        first = await resolver.resolve_thumbnail("chunk_1", max_px=16)
        second = await resolver.resolve_thumbnail("chunk_1", max_px=16)

    assert first is not None
    assert second is first
    assert first.media_type == "image/png"
    fetch.assert_awaited_once()


async def test_visual_asset_resolver_returns_none_for_missing_image_data() -> None:
    rag = MagicMock(text_chunks=object())
    resolver = VisualAssetResolver(lightrag=rag)

    with (
        patch("dlightrag.core.visual_assets.fetch_chunks_by_ids", new=AsyncMock(return_value=[{"chunk_id": "c1"}])),
        patch("dlightrag.core.visual_assets.hydrate_lightrag_chunk_provenance", new=AsyncMock()),
    ):
        assert await resolver.resolve("chunk_no_image") is None
