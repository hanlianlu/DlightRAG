# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG sidecar visual asset resolution."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.visual_assets import ThumbnailCache, VisualAssetResolver

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


async def test_visual_asset_resolver_returns_full_asset() -> None:
    chunk = {"chunk_id": "chunk_1", "image_data": _PNG_B64, "image_mime_type": "image/png"}
    stores = MagicMock()
    stores.context_chunks_by_ids = AsyncMock(return_value=[chunk])
    resolver = VisualAssetResolver(stores=stores)

    with (
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
    stores.context_chunks_by_ids.assert_awaited_once_with(["chunk_1"])
    hydrate.assert_awaited_once()


async def test_visual_asset_resolver_generates_and_caches_thumbnail() -> None:
    chunk = {"chunk_id": "chunk_1", "image_data": _PNG_B64, "image_mime_type": "image/png"}
    stores = MagicMock()
    stores.context_chunks_by_ids = AsyncMock(return_value=[chunk])
    resolver = VisualAssetResolver(stores=stores, thumb_cache=ThumbnailCache(max_size=2))

    with (
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
    stores.context_chunks_by_ids.assert_awaited_once_with(["chunk_1"])


async def test_visual_asset_resolver_returns_none_for_missing_image_data() -> None:
    stores = MagicMock()
    stores.context_chunks_by_ids = AsyncMock(return_value=[{"chunk_id": "c1"}])
    resolver = VisualAssetResolver(stores=stores)

    with (
        patch("dlightrag.core.visual_assets.hydrate_lightrag_chunk_provenance", new=AsyncMock()),
    ):
        assert await resolver.resolve("chunk_no_image") is None
