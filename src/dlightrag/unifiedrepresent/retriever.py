# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual retrieval pipeline for unified representational RAG.

Handles query processing: LightRAG KG retrieval -> visual chunk resolution ->
multimodal reranking.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from lightrag import LightRAG, QueryParam
from PIL import Image

from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.unifiedrepresent.multimodal_query import enhance_query_with_images

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]

logger = logging.getLogger(__name__)


def _build_rerank_content(chunk: dict[str, Any]) -> str:
    """Prepend file-level metadata header to content for reranker context.

    Reads Stage 1 fields directly from the chunk dict (populated by
    ``_resolve_visual_chunks``).  Stores original content in
    ``_raw_content`` so it can be restored after reranking.
    """
    parts: list[str] = []
    file_path = chunk.get("file_path", "")
    if file_path:
        parts.append(f"Source: {Path(file_path).name}")
    if title := chunk.get("doc_title", ""):
        parts.append(f"Title: {title}")
    if author := chunk.get("doc_author", ""):
        parts.append(f"Author: {author}")
    if page := chunk.get("page_idx"):
        parts.append(f"Page: {page}")

    raw = chunk.get("content", "")
    chunk["_raw_content"] = raw
    if parts:
        return f"[{' | '.join(parts)}]\n{raw}"
    return raw


class VisualRetriever:
    """Query pipeline for unified representational RAG (Phases 2-4)."""

    def __init__(
        self,
        lightrag: LightRAG,
        visual_chunks: Any,  # BaseKVStorage instance
        config: Any,  # DlightragConfig
        vision_model_func: Callable | None = None,
        rerank_func: Callable | None = None,
        path_resolver: PathResolver | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.lightrag = lightrag
        self.visual_chunks = visual_chunks
        self.config = config
        self.vision_model_func = vision_model_func
        self._rerank_func = rerank_func
        self.path_resolver = path_resolver
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
    ) -> dict:
        """Run Phase 1-3 with optional dual-path for multimodal queries.

        The returned dict always contains an ``enhanced_query`` key holding the
        query string that was actually used for text-path retrieval.  For pure
        text queries this equals *query*; when *images* are supplied it includes
        VLM-generated image descriptions.
        """
        if images:
            if self.vision_model_func:
                enhanced_query = await enhance_query_with_images(
                    query=query,
                    images=images,
                    vision_model_func=self.vision_model_func,
                )
            else:
                enhanced_query = query
                logger.warning("No vision_model_func; using original query for text path")

            text_result = await self._retrieve(enhanced_query, mode, top_k, chunk_top_k)
            visual_chunks = await self.query_by_visual_embedding(images, top_k=chunk_top_k)

            # Round-robin merge text + visual chunks
            text_chunks = text_result.get("contexts", {}).get("chunks", [])
            merged_chunks: list[dict[str, Any]] = []
            max_len = max(len(text_chunks), len(visual_chunks))
            for i in range(max_len):
                if i < len(text_chunks):
                    merged_chunks.append(text_chunks[i])
                if i < len(visual_chunks):
                    merged_chunks.append(visual_chunks[i])

            text_result["contexts"]["chunks"] = merged_chunks[:chunk_top_k]
            text_result["enhanced_query"] = enhanced_query
            return text_result

        result = await self._retrieve(query, mode, top_k, chunk_top_k)
        result["enhanced_query"] = query
        return result

    async def _retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
    ) -> dict:
        """Run Phase 1-3: LightRAG retrieval -> visual resolution -> reranking.

        Uses a single chunk list as the pipeline data structure. Each chunk
        is a dict carrying all fields; the list itself represents pipeline state.
        """
        # Phase 1: LightRAG retrieval
        param = QueryParam(
            mode=cast(QueryMode, mode),
            only_need_context=True,
            top_k=top_k,
            enable_rerank=False,
        )
        result = await self.lightrag.aquery_data(query, param=param)
        data = result.get("data", {})

        # Build chunk list from LightRAG's merged output
        chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw in data.get("chunks", []):
            cid = raw.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(
                    {
                        "chunk_id": cid,
                        "content": raw.get("content", ""),
                        "reference_id": str(raw.get("reference_id", "")),
                        "file_path": raw.get("file_path", ""),
                    }
                )

        # Force-inject metadata-resolved chunks before visual resolution
        from dlightrag.core.retrieval.filtered_vdb import _active_filter

        active_ids = _active_filter.get()
        if active_ids:
            missing = active_ids - seen_ids
            if missing:
                inject_ids = sorted(missing)[:chunk_top_k]
                raw_contents = await self.lightrag.text_chunks.get_by_ids(inject_ids)
                for cid, content_raw in zip(inject_ids, raw_contents, strict=False):
                    if content_raw is None:
                        continue
                    content = (
                        content_raw
                        if isinstance(content_raw, str)
                        else content_raw.get("content", "")
                    )
                    seen_ids.add(cid)
                    chunks.append(
                        {
                            "chunk_id": cid,
                            "content": content,
                            "reference_id": "",
                            "file_path": "",
                        }
                    )
                logger.info("Force-injected %d metadata-resolved chunks", len(missing))

        # Phase 2: Visual resolution (mutates chunks in-place)
        await self._resolve_visual_chunks(chunks)

        # Phase 3: Multimodal reranking
        if self._rerank_func and chunks:
            for chunk in chunks:
                chunk["content"] = _build_rerank_content(chunk)

            pre_rerank_count = len(chunks)
            scored = await self._rerank_func(query=query, chunks=chunks, top_k=chunk_top_k)
            logger.info("Rerank: %d -> %d chunks", pre_rerank_count, len(scored))

            # Restore raw content (strip Stage 1 header)
            for chunk in scored:
                raw = chunk.pop("_raw_content", None)
                if raw is not None:
                    chunk["content"] = raw

            chunks = scored
        elif len(chunks) > chunk_top_k:
            chunks = chunks[:chunk_top_k]

        return {
            "contexts": {
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
                    for c in chunks
                ],
            },
        }

    async def query_by_visual_embedding(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query chunks_vdb using visual embeddings of uploaded images.

        Each image is embedded via VisualEmbedder and queried against the
        chunks vector database directly. Results from multiple images are
        merged via round-robin interleaving.
        """
        if not images or self.embedder is None:
            return []

        per_image_results: list[list[dict[str, Any]]] = []
        for img_bytes in images:
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                vec = await self.embedder.embed_pages([pil_img])
                embedding = vec[0]

                raw_chunks = await self.lightrag.chunks_vdb.query(
                    query="",
                    top_k=top_k,
                    query_embedding=embedding,
                )

                # Build chunk list from VDB results
                vdb_chunks: list[dict[str, Any]] = [
                    {
                        "chunk_id": c.get("id", ""),
                        "content": c.get("content", ""),
                        "file_path": c.get("file_path", ""),
                        "reference_id": "",
                        "page_idx": (c.get("chunk_order_index") or 0) + 1,
                        "relevance_score": c.get("distance", 0.0),
                    }
                    for c in (raw_chunks or [])
                ]

                # Visual resolution adds image_data + Stage 1 metadata
                await self._resolve_visual_chunks(vdb_chunks)
                per_image_results.append(vdb_chunks)
            except Exception:
                logger.warning("Visual embedding query failed for image", exc_info=True)
                per_image_results.append([])

        # Round-robin merge across images
        merged: list[dict[str, Any]] = []
        max_len = max((len(r) for r in per_image_results), default=0)
        for i in range(max_len):
            for img_results in per_image_results:
                if i < len(img_results):
                    merged.append(img_results[i])
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_visual_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Resolve visual_chunks KV data into each chunk dict in-place.

        Adds ``image_data``, ``page_idx``, ``doc_title``, ``doc_author``,
        and backfills ``file_path`` from visual_chunks when missing.
        """
        if not chunks:
            return
        chunk_ids = [c["chunk_id"] for c in chunks]
        raw = await self.visual_chunks.get_by_ids(chunk_ids)
        resolved_count = 0
        for chunk, vd in zip(chunks, raw, strict=False):
            if vd is None:
                continue
            if isinstance(vd, str):
                try:
                    vd = json.loads(vd)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping unparseable visual_chunk %s", chunk["chunk_id"])
                    continue
            if not isinstance(vd, dict):
                continue
            resolved_count += 1
            chunk["image_data"] = vd.get("image_data")
            chunk["page_idx"] = (vd.get("page_index", 0) or 0) + 1
            chunk["doc_title"] = vd.get("doc_title", "")
            chunk["doc_author"] = vd.get("doc_author", "")
            if not chunk.get("file_path"):
                chunk["file_path"] = vd.get("file_path", "")
        logger.info("[Visual Resolve] Resolved %d/%d chunks", resolved_count, len(chunks))
