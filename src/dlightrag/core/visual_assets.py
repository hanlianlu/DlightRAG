# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resolve and serve LightRAG sidecar-backed visual chunk assets."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from dlightrag.core.retrieval.filtered_vdb import fetch_chunks_by_ids
from dlightrag.core.retrieval.provenance import hydrate_lightrag_chunk_provenance
from dlightrag.utils.images import detect_image_mime, thumbnail_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualAsset:
    """Resolved visual asset bytes for one chunk."""

    chunk_id: str
    data: bytes
    media_type: str


class ThumbnailCache:
    """Small process-local LRU cache for generated thumbnails."""

    def __init__(self, *, max_size: int = 256) -> None:
        self._max_size = max(1, int(max_size))
        self._items: OrderedDict[tuple[str, int], VisualAsset] = OrderedDict()

    def get(self, key: tuple[str, int]) -> VisualAsset | None:
        asset = self._items.get(key)
        if asset is not None:
            self._items.move_to_end(key)
        return asset

    def set(self, key: tuple[str, int], value: VisualAsset) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)


class VisualAssetResolver:
    """Resolve images from LightRAG text chunk sidecar metadata."""

    def __init__(self, *, lightrag: Any, thumb_cache: ThumbnailCache | None = None) -> None:
        self._lightrag = lightrag
        self._thumb_cache = thumb_cache or ThumbnailCache()

    async def resolve(self, chunk_id: str) -> VisualAsset | None:
        """Resolve the full image payload for a chunk id."""
        text_chunks = getattr(self._lightrag, "text_chunks", None)
        if text_chunks is None:
            logger.warning("VisualAssetResolver: text_chunks is None for chunk_id=%s", chunk_id)
            return None

        chunks = await fetch_chunks_by_ids(text_chunks, [chunk_id])
        if not chunks:
            logger.warning(
                "VisualAssetResolver: no chunks from fetch_chunks_by_ids for chunk_id=%s", chunk_id
            )
            return None
        await hydrate_lightrag_chunk_provenance(self._lightrag, chunks)
        chunk = chunks[0]
        image_data = chunk.get("image_data")
        if not isinstance(image_data, str) or not image_data:
            logger.warning(
                "VisualAssetResolver: no image_data in chunk_id=%s keys=%s",
                chunk_id,
                list(chunk.keys()),
            )
            return None
        try:
            raw = base64.b64decode(image_data)
        except Exception as exc:
            logger.warning(
                "VisualAssetResolver: base64 decode failed for chunk_id=%s: %s", chunk_id, exc
            )
            return None
        media_type = str(chunk.get("image_mime_type") or detect_image_mime(raw))
        return VisualAsset(chunk_id=chunk_id, data=raw, media_type=media_type)

    async def resolve_thumbnail(self, chunk_id: str, *, max_px: int) -> VisualAsset | None:
        """Resolve a thumbnail for one chunk id."""
        max_px = max(1, int(max_px))
        cache_key = (chunk_id, max_px)
        cached = self._thumb_cache.get(cache_key)
        if cached is not None:
            return cached
        full = await self.resolve(chunk_id)
        if full is None:
            return None
        data, media_type = await asyncio.to_thread(
            thumbnail_bytes,
            full.data,
            max_px=max_px,
            output_mime=full.media_type,
        )
        thumb = VisualAsset(chunk_id=chunk_id, data=data, media_type=media_type)
        self._thumb_cache.set(cache_key, thumb)
        return thumb


__all__ = ["ThumbnailCache", "VisualAsset", "VisualAssetResolver"]
