# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Direct image->image retrieval leg over the fused visual chunk vectors."""

import asyncio
import io
import logging
from typing import Any

from PIL import Image

from dlightrag.engine.ai.concurrency import bounded_map
from dlightrag.engine.ai.media import (
    decode_image_base64,
    image_url_block,
    verify_web_image_bytes,
)
from dlightrag.engine.rag.retrieval import ContextRow

logger = logging.getLogger(__name__)


class DirectVisualRetriever:
    """Rank chunks by raw query-image similarity, beside the semantic and BM25 legs.

    Embedding the query image preserves the visual signal the VLM-description
    text path loses, so this leg only pays off while its ranks stay independent:
    it is fused once with the other legs and never pre-merged into one of them.
    Every failure degrades to an empty ranking — a query never fails here.
    """

    def __init__(self, *, embedder: Any, stores: Any, top_k: int) -> None:
        self._embedder = embedder
        self._stores = stores
        self._top_k = max(0, int(top_k))

    async def search(self, query_image_blocks: list[dict[str, Any]] | None) -> list[ContextRow]:
        """Embed query images (image-only, one request) and search chunk vectors."""
        if self._top_k <= 0 or not query_image_blocks:
            return []
        images = await asyncio.to_thread(_extract_images, query_image_blocks)
        if not images:
            return []
        try:
            vectors = await self._embedder.embed_query_images(images)
        except Exception:
            logger.warning("Direct visual query embedding failed", exc_info=True)
            return []
        finally:
            for image in images:
                image.close()

        async def _search(vector: list[float]) -> list[ContextRow]:
            return (
                await self._stores.chunks_vdb.query(
                    query="", top_k=self._top_k, query_embedding=vector
                )
                or []
            )

        results = await bounded_map(
            list(vectors),
            _search,
            max_concurrent=min(8, max(1, len(vectors))),
            task_name="direct-visual-query",
        )
        merged: dict[str, ContextRow] = {}
        for raw_chunks in results:
            if isinstance(raw_chunks, Exception):
                continue
            for c in raw_chunks:
                cid = c.get("id")
                if not cid:
                    continue
                dist = c.get("distance")
                existing = merged.get(cid)
                existing_dist = existing.get("relevance_score") if existing else None
                if existing is None or (
                    dist is not None and (existing_dist is None or dist < existing_dist)
                ):
                    merged[cid] = {
                        "chunk_id": cid,
                        "content": c.get("content", ""),
                        "file_path": c.get("file_path", ""),
                        "reference_id": "",
                        "relevance_score": dist,
                    }
                    if c.get("full_doc_id"):
                        merged[cid]["full_doc_id"] = c["full_doc_id"]
        return sorted(
            merged.values(),
            key=lambda c: (
                c["relevance_score"] if c.get("relevance_score") is not None else float("inf")
            ),
        )[: self._top_k]


def _extract_images(blocks: list[dict[str, Any]] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in blocks or []:
        if item.get("type") != "image_url":
            continue
        block = image_url_block(item)
        if block is None:
            continue
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            continue
        url = image_url.get("url")
        if not isinstance(url, str) or not url.strip().startswith("data:"):
            continue
        try:
            raw, _ = decode_image_base64(url)
            verify_web_image_bytes(raw)
            images.append(Image.open(io.BytesIO(raw)))
        except Exception:
            logger.warning("Failed to decode direct visual query image", exc_info=True)
    return images


__all__ = ["DirectVisualRetriever"]
