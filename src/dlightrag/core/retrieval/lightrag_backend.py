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
from dlightrag.core.retrieval.provenance import hydrate_lightrag_chunk_provenance

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
            chunk_top_k=limit,
            enable_rerank=False,
            include_references=True,
        )
        raw = await self._lightrag.aquery_data(query, param=param)
        data = raw.get("data", {})

        chunks = self._chunks_from_lightrag(data.get("chunks", []))
        trace: dict[str, Any] = {
            "lightrag_status": raw.get("status"),
            "lightrag_chunk_count": len(chunks),
            "lightrag_entity_count": len(data.get("entities", [])),
            "lightrag_relationship_count": len(data.get("relationships", [])),
            "direct_visual_chunk_count": 0,
            "hydrated_chunk_count": 0,
            "reranked_chunk_count": 0,
        }
        seen_ids = {chunk["chunk_id"] for chunk in chunks}
        injected = await fetch_missing_chunks(self._lightrag.text_chunks, seen_ids, limit)
        chunks.extend(injected)
        trace["metadata_injected_chunk_count"] = len(injected)

        image_chunks = await self._retrieve_query_images(multimodal_content, top_k=limit)
        trace["direct_visual_chunk_count"] = len(image_chunks)
        if image_chunks:
            chunks = rrf_fuse([chunks, image_chunks])[:limit]

        await self._hydrate_chunk_provenance(chunks)
        trace["hydrated_chunk_count"] = len(chunks)
        chunks = await self._rerank(query, chunks, top_k=limit)
        trace["reranked_chunk_count"] = len(chunks)
        chunks = canonicalize_reference_ids(chunks, references=data.get("references", []))

        context_chunks = []
        for c in chunks[:limit]:
            context_chunk = {
                "chunk_id": c["chunk_id"],
                "reference_id": c.get("reference_id", ""),
                "file_path": c.get("file_path", ""),
                "content": c.get("content", ""),
                "image_data": c.get("image_data"),
                "image_mime_type": c.get("image_mime_type"),
                "relevance_score": c.get("relevance_score"),
            }
            if c.get("page_idx") is not None:
                context_chunk["page_idx"] = c["page_idx"]
            if c.get("bbox") is not None:
                context_chunk["bbox"] = c["bbox"]
            context_chunks.append(context_chunk)

        return RetrievalResult(
            contexts={
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": context_chunks,
            },
            trace=trace,
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
            chunk = {
                "chunk_id": cid,
                "content": raw.get("content", ""),
                "reference_id": str(raw.get("reference_id", "")),
                "file_path": raw.get("file_path", ""),
                "relevance_score": raw.get("score") or raw.get("distance"),
            }
            for key in ("full_doc_id", "sidecar", "sidecar_location", "page_idx"):
                if raw.get(key) is not None:
                    chunk[key] = raw[key]
            chunks.append(chunk)
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

        merged: dict[str, dict[str, Any]] = {}
        for image in images:
            try:
                vectors = await self._embedder.embed_query_images([image])
                raw_chunks = await self._lightrag.chunks_vdb.query(
                    query="",
                    top_k=top_k,
                    query_embedding=vectors[0],
                )
                for c in raw_chunks or []:
                    cid = c.get("id")
                    if not cid:
                        continue
                    dist = c.get("distance")
                    if cid not in merged or (dist is not None and dist < merged[cid].get("distance", float("inf"))):
                        merged[cid] = {
                            "chunk_id": cid,
                            "content": c.get("content", ""),
                            "file_path": c.get("file_path", ""),
                            "reference_id": "",
                            "relevance_score": dist,
                        }
            except Exception:
                logger.warning("Direct visual query failed", exc_info=True)
        return sorted(merged.values(), key=lambda c: c.get("relevance_score") or float("inf"))[:top_k]

    async def _hydrate_chunk_provenance(self, chunks: list[dict[str, Any]]) -> None:
        await hydrate_lightrag_chunk_provenance(self._lightrag, chunks)

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
