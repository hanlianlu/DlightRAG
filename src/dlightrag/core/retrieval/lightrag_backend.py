# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG mix retrieval backend with DlightRAG visual enrichment."""

import asyncio
import io
import logging
from typing import Any

from lightrag import QueryParam
from PIL import Image

from dlightrag.core.retrieval import canonicalize_reference_ids
from dlightrag.core.retrieval.filtered_vdb import fetch_missing_chunks
from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.core.retrieval.protocols import ContextRow, RetrievalResult
from dlightrag.core.retrieval.provenance import hydrate_lightrag_chunk_provenance
from dlightrag.utils.concurrency import bounded_map
from dlightrag.utils.images import decode_image_base64, image_url_block

logger = logging.getLogger(__name__)


class LightRAGMixBackend:
    """RetrievalBackend over a LightRAG instance, always using mix mode."""

    def __init__(
        self,
        *,
        lightrag: Any,
        stores: Any,
        embedder: Any | None = None,
        direct_visual_top_k: int = 20,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000,
        max_total_tokens: int = 40000,
    ) -> None:
        self._lightrag = lightrag
        self._stores = stores
        self._embedder = embedder
        self._direct_visual_top_k = max(0, int(direct_visual_top_k))
        self._max_entity_tokens = max_entity_tokens
        self._max_relation_tokens = max_relation_tokens
        self._max_total_tokens = max_total_tokens

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
            max_entity_tokens=self._max_entity_tokens,
            max_relation_tokens=self._max_relation_tokens,
            max_total_tokens=self._max_total_tokens,
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
        }
        seen_ids = {chunk["chunk_id"] for chunk in chunks}
        injected = await fetch_missing_chunks(self._stores, seen_ids, limit)
        chunks.extend(injected)
        trace["metadata_injected_chunk_count"] = len(injected)

        image_chunks = await self._retrieve_query_images(
            multimodal_content,
            top_k=self._direct_visual_top_k,
        )
        trace["direct_visual_chunk_count"] = len(image_chunks)
        if image_chunks:
            chunks = rrf_fuse([chunks, image_chunks])[:limit]

        await self._hydrate_chunk_provenance(chunks)
        trace["hydrated_chunk_count"] = len(chunks)
        trace["provenance_hydrated_chunk_ids"] = [
            str(chunk["chunk_id"]) for chunk in chunks if chunk.get("chunk_id")
        ]
        chunks = canonicalize_reference_ids(chunks, references=data.get("references", []))

        context_chunks: list[ContextRow] = []
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
            if c.get("full_doc_id"):
                context_chunk["full_doc_id"] = c["full_doc_id"]
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
    def _chunks_from_lightrag(rows: list[ContextRow]) -> list[ContextRow]:
        chunks: list[ContextRow] = []
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
    ) -> list[ContextRow]:
        if self._embedder is None:
            return []
        if top_k <= 0:
            return []
        images = await asyncio.to_thread(_extract_images, multimodal_content)
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

        async def _query_vector(vector: list[float]) -> list[ContextRow]:
            return (
                await self._stores.chunks_vdb.query(
                    query="",
                    top_k=top_k,
                    query_embedding=vector,
                )
                or []
            )

        query_results = await bounded_map(
            list(vectors),
            _query_vector,
            max_concurrent=min(8, max(1, len(vectors))),
            task_name="direct-visual-query",
        )

        merged: dict[str, ContextRow] = {}
        for raw_chunks in query_results:
            if isinstance(raw_chunks, Exception):
                continue
            for c in raw_chunks:
                cid = c.get("id")
                if not cid:
                    continue
                dist = c.get("distance")
                if cid not in merged or (
                    dist is not None and dist < merged[cid].get("distance", float("inf"))
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
        return sorted(merged.values(), key=lambda c: c.get("relevance_score") or float("inf"))[
            :top_k
        ]

    async def _hydrate_chunk_provenance(self, chunks: list[ContextRow]) -> None:
        await hydrate_lightrag_chunk_provenance(self._stores, chunks)


def _extract_images(multimodal_content: list[dict[str, Any]] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in multimodal_content or []:
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
            images.append(Image.open(io.BytesIO(raw)))
        except Exception:
            logger.warning("Failed to decode direct visual query image", exc_info=True)
    return images
