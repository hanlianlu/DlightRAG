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
        # Dual-path: if images provided, run VLM-enhanced text query + visual embedding
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

            # Text path: standard retrieval with enhanced query
            text_result = await self._retrieve(
                enhanced_query,
                mode,
                top_k,
                chunk_top_k,
            )

            # Image path: visual embedding direct query
            visual_chunks = await self.query_by_visual_embedding(images, top_k=chunk_top_k)

            # Round-robin merge text chunks + visual chunks
            text_chunks = text_result.get("contexts", {}).get("chunks", [])
            merged_chunks: list[dict[str, Any]] = []
            max_len = max(len(text_chunks), len(visual_chunks))
            for i in range(max_len):
                if i < len(text_chunks):
                    merged_chunks.append(text_chunks[i])
                if i < len(visual_chunks):
                    merged_chunks.append(visual_chunks[i])

            # Cap at chunk_top_k to maintain consistent result count with text-only queries
            text_result["contexts"]["chunks"] = merged_chunks[:chunk_top_k]
            text_result["enhanced_query"] = enhanced_query
            return text_result

        # Text-only: unchanged existing path
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

        Returns dict with keys:
            contexts: {entities, relationships, chunks}
        """
        # Phase 1: LightRAG retrieval
        param = QueryParam(
            mode=cast(QueryMode, mode),
            only_need_context=True,
            top_k=top_k,
            enable_rerank=False,  # We handle reranking ourselves
        )
        result = await self.lightrag.aquery_data(query, param=param)

        # Extract chunks from LightRAG result. aquery_data already collects
        # chunks from all sources (vector search + entity source_ids +
        # relationship source_ids) via internal round-robin merge.
        data = result.get("data", {})
        chunk_text: dict[str, str] = {}
        chunk_meta: dict[str, dict[str, str]] = {}  # chunk_id -> {reference_id, file_path}
        chunk_ids: set[str] = set()

        for chunk in data.get("chunks", []):
            cid = chunk.get("chunk_id")
            if cid:
                chunk_ids.add(cid)
                chunk_text[cid] = chunk.get("content", "")
                chunk_meta[cid] = {
                    "reference_id": str(chunk.get("reference_id", "")),
                    "file_path": chunk.get("file_path", ""),
                }

        # Force-inject metadata-resolved chunks BEFORE visual resolution
        # so they participate in dedup (dict keying) and rerank naturally.
        # Cap to chunk_top_k to bound input for large filter results.
        from dlightrag.core.retrieval.filtered_vdb import _active_filter

        active_ids = _active_filter.get()
        logger.info("Force-inject: active_ids=%s, chunk_ids_count=%d", active_ids, len(chunk_ids))
        if active_ids:
            missing = active_ids - chunk_ids
            logger.info(
                "In-filter check: active=%d, in_chunk_ids=%d, missing=%d",
                len(active_ids),
                len(active_ids & chunk_ids),
                len(missing),
            )
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
                    chunk_ids.add(cid)
                    chunk_text[cid] = content
                logger.info("Force-injected %d metadata-resolved chunks", len(missing))

        # Phase 2: Visual resolution
        chunk_id_list = list(chunk_ids)
        resolved = await self._resolve_visual_chunks(chunk_id_list)

        # Phase 3: Multimodal reranking (optional)
        # Build chunk dicts for reranker (content + image_data)
        all_candidates: dict[str, dict] = {}
        for cid, vd in resolved.items():
            all_candidates[cid] = vd
        # Add text-only chunks (not in visual_chunks but have content)
        for cid in chunk_ids:
            if cid not in all_candidates and cid in chunk_text:
                all_candidates[cid] = {}  # No visual data

        if self._rerank_func and all_candidates:
            rerank_chunks = []
            for cid, vd in all_candidates.items():
                # Prepend source filename so reranker can associate
                # content with filename-based queries (e.g. "IMG 9551").
                file_path = chunk_meta.get(cid, {}).get("file_path", vd.get("file_path", ""))
                filename = Path(file_path).name if file_path else ""
                raw_content = chunk_text.get(cid, "")
                content = f"[Source: {filename}]\n{raw_content}" if filename else raw_content
                rerank_chunks.append(
                    {
                        "chunk_id": cid,
                        "content": content,
                        "image_data": vd.get("image_data"),
                    }
                )
            pre_rerank_count = len(rerank_chunks)
            scored = await self._rerank_func(query=query, chunks=rerank_chunks, top_k=chunk_top_k)
            logger.info("Rerank: %d -> %d chunks", pre_rerank_count, len(scored))

            # Rebuild all_candidates from scored results
            scored_ids = [c["chunk_id"] for c in scored]
            all_candidates = {
                cid: all_candidates[cid] for cid in scored_ids if cid in all_candidates
            }
        else:
            # No reranker — cap total candidates
            if len(all_candidates) > chunk_top_k:
                ordered_ids: list[str] = list(resolved.keys())
                for cid in chunk_ids:
                    if cid not in resolved and cid in all_candidates:
                        ordered_ids.append(cid)
                ordered_ids = ordered_ids[:chunk_top_k]
                all_candidates = {
                    cid: all_candidates[cid] for cid in ordered_ids if cid in all_candidates
                }

        # Debug: log final chunk mapping
        for cid in all_candidates:
            meta = chunk_meta.get(cid, {})
            logger.info(
                "Final chunk: %s ref=%s file=%s",
                cid[:20],
                meta.get("reference_id", "?"),
                Path(meta.get("file_path", "")).name if meta.get("file_path") else "?",
            )

        # Build return dict
        return {
            "contexts": {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": [
                    {
                        "chunk_id": cid,
                        "reference_id": chunk_meta.get(cid, {}).get("reference_id", ""),
                        "file_path": chunk_meta.get(cid, {}).get(
                            "file_path", vd.get("file_path", "")
                        ),
                        "content": chunk_text.get(cid, ""),
                        "image_data": vd.get("image_data"),
                        "page_idx": (vd.get("page_index", 0) or 0) + 1,
                        "relevance_score": vd.get("relevance_score"),
                    }
                    for cid, vd in all_candidates.items()
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

        Args:
            images: List of raw image bytes.
            top_k: Number of results per image.

        Returns:
            Round-robin merged list of chunk dicts from all images.
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
                # Resolve visual_chunks to get image_data
                raw_ids = [c.get("id", "") for c in (raw_chunks or []) if c.get("id")]
                visual_data = await self._resolve_visual_chunks(raw_ids)

                # Normalize VDB results to match text chunk schema
                normalized = [
                    {
                        "chunk_id": c.get("id", ""),
                        "reference_id": "",  # Not available from VDB; set by merge
                        "file_path": c.get("file_path", ""),
                        "content": c.get("content", ""),
                        "image_data": visual_data.get(c.get("id", ""), {}).get("image_data"),
                        "page_idx": (c.get("chunk_order_index") or 0) + 1,  # 0→1-based
                        "relevance_score": c.get("distance", 0.0),
                    }
                    for c in (raw_chunks or [])
                ]
                per_image_results.append(normalized)
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

    async def _resolve_visual_chunks(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Look up chunk_ids in visual_chunks KV store, return resolved dict.

        Handles None results, JSONB-as-string parsing, and logs progress.
        """
        if not chunk_ids:
            return {}
        raw = await self.visual_chunks.get_by_ids(chunk_ids)
        resolved: dict[str, dict] = {}
        for cid, vd in zip(chunk_ids, raw, strict=False):
            if vd is None:
                continue
            if isinstance(vd, str):
                try:
                    vd = json.loads(vd)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping unparseable visual_chunk %s", cid)
                    continue
            if isinstance(vd, dict):
                resolved[cid] = vd
        logger.info("[Visual Resolve] Resolved %d/%d chunks", len(resolved), len(chunk_ids))
        return resolved
