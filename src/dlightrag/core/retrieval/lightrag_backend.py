# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG mix retrieval backend with DlightRAG visual enrichment."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, cast

from lightrag import QueryParam
from PIL import Image

from dlightrag.core.retrieval import canonicalize_reference_ids
from dlightrag.core.retrieval.filtered_vdb import fetch_missing_chunks
from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.core.retrieval.protocols import RetrievalResult

logger = logging.getLogger(__name__)


class LightRAGMixBackend:
    """RetrievalBackend over a LightRAG instance, always using mix mode."""

    def __init__(
        self,
        *,
        lightrag: Any,
        embedder: Any | None = None,
        rerank_func: Any | None = None,
    ) -> None:
        self._lightrag = lightrag
        self._embedder = embedder
        self._rerank_func = rerank_func

    async def aretrieve(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        multimodal_content: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        del mode, kwargs
        limit = chunk_top_k or top_k or 30
        param = QueryParam(
            mode="mix",
            only_need_context=True,
            top_k=top_k or 60,
            enable_rerank=False,
        )
        raw = await self._lightrag.aquery_data(query, param=param)
        data = raw.get("data", {})

        chunks = self._chunks_from_lightrag(data.get("chunks", []))
        seen_ids = {chunk["chunk_id"] for chunk in chunks}
        injected = await fetch_missing_chunks(self._lightrag.text_chunks, seen_ids, limit)
        chunks.extend(injected)

        image_chunks = await self._retrieve_query_images(multimodal_content, top_k=limit)
        if image_chunks:
            chunks = rrf_fuse([chunks, image_chunks])[:limit]

        await self._hydrate_image_chunks(chunks)
        chunks = await self._rerank(query, chunks, top_k=limit)
        chunks = canonicalize_reference_ids(chunks, references=data.get("references", []))

        return RetrievalResult(
            contexts={
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "reference_id": c.get("reference_id", ""),
                        "file_path": c.get("file_path", ""),
                        "content": c.get("content", ""),
                        "image_data": c.get("image_data"),
                        "page_idx": c.get("page_idx", 1),
                        "relevance_score": c.get("relevance_score"),
                    }
                    for c in chunks[:limit]
                ],
            }
        )

    @staticmethod
    def _chunks_from_lightrag(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in rows:
            cid = raw.get("chunk_id") or raw.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            chunks.append(
                {
                    "chunk_id": cid,
                    "content": raw.get("content", ""),
                    "reference_id": str(raw.get("reference_id", "")),
                    "file_path": raw.get("file_path", ""),
                    "relevance_score": raw.get("score") or raw.get("distance"),
                }
            )
        return chunks

    async def _retrieve_query_images(
        self,
        multimodal_content: list[dict[str, Any]] | None,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self._embedder is None:
            return []
        images = _extract_images(multimodal_content)
        if not images:
            return []

        rankings: list[list[dict[str, Any]]] = []
        for image in images:
            try:
                vectors = await self._embedder.embed_query_images([image])
                raw_chunks = await self._lightrag.chunks_vdb.query(
                    query="",
                    top_k=top_k,
                    query_embedding=vectors[0],
                )
                ranking = [
                    {
                        "chunk_id": c.get("id", ""),
                        "content": c.get("content", ""),
                        "file_path": c.get("file_path", ""),
                        "reference_id": "",
                        "page_idx": (c.get("chunk_order_index") or 0) + 1,
                        "relevance_score": c.get("distance"),
                    }
                    for c in (raw_chunks or [])
                    if c.get("id")
                ]
                rankings.append(ranking)
            except Exception:
                logger.warning("Direct visual query failed", exc_info=True)
        return rrf_fuse(rankings)[:top_k] if rankings else []

    async def _hydrate_image_chunks(self, chunks: list[dict[str, Any]]) -> None:
        await hydrate_image_chunks(self._lightrag, chunks)

    async def _rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self._rerank_func is None or not chunks:
            return chunks[:top_k]
        try:
            return await self._rerank_func(query=query, chunks=chunks, top_k=top_k)
        except Exception:
            logger.warning("Rerank failed; returning unfused chunk order", exc_info=True)
            return chunks[:top_k]


def _extract_images(multimodal_content: list[dict[str, Any]] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in multimodal_content or []:
        if item.get("type") != "image":
            continue
        raw: bytes | None = None
        if item.get("img_path"):
            raw = Path(item["img_path"]).read_bytes()
        elif item.get("data"):
            raw = base64.b64decode(cast(str, item["data"]))
        if raw:
            images.append(Image.open(io.BytesIO(raw)))
    return images


async def hydrate_image_chunks(lightrag: Any, chunks: list[dict[str, Any]]) -> None:
    """Hydrate image bytes/page metadata from LightRAG text_chunks sidecars."""
    if not chunks:
        return
    chunk_ids = [c["chunk_id"] for c in chunks]
    try:
        raw_chunks = await lightrag.text_chunks.get_by_ids(chunk_ids)
    except Exception:
        logger.debug("LightRAG text chunk hydration failed", exc_info=True)
        raw_chunks = [None for _ in chunk_ids]
    for chunk, raw in zip(chunks, raw_chunks, strict=False):
        sidecar: dict[str, Any] = {}
        if isinstance(raw, dict):
            if not chunk.get("file_path"):
                chunk["file_path"] = raw.get("file_path", "")
            raw_sidecar = raw.get("sidecar")
            if isinstance(raw_sidecar, dict):
                sidecar = raw_sidecar
        page_index = sidecar.get("page_index")
        if isinstance(page_index, int):
            chunk["page_idx"] = page_index + 1
        image_path = sidecar.get("path") or chunk.get("file_path")
        if not isinstance(image_path, str):
            continue
        path = Path(image_path)
        if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.exists():
            continue
        chunk["image_data"] = base64.b64encode(path.read_bytes()).decode("ascii")


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
