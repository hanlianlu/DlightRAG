# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified representational RAG engine.

Top-level orchestrator that composes page rendering, visual embedding,
VLM entity extraction, and visual retrieval into a complete pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.unifiedrepresent.embedder import VisualEmbedder
from dlightrag.unifiedrepresent.extractor import EntityExtractor
from dlightrag.unifiedrepresent.renderer import PageRenderer
from dlightrag.unifiedrepresent.retriever import VisualRetriever

logger = logging.getLogger(__name__)


class UnifiedRepresentEngine:
    """Orchestrates unified representational RAG pipeline.

    Holds a LightRAG instance, a visual_chunks KV store, and all
    sub-components (renderer, embedder, extractor, retriever). Exposes
    ``aingest()``, ``aretrieve()``, ``aclose()``.
    """

    def __init__(
        self,
        lightrag: Any,  # LightRAG instance
        visual_chunks: Any,  # BaseKVStorage instance
        config: Any,  # DlightragConfig instance
        vision_model_func: Callable | None = None,
        visual_embedder: VisualEmbedder | None = None,
        path_resolver: PathResolver | None = None,
        context_model_func: Callable | None = None,
    ) -> None:
        self.lightrag = lightrag
        self.visual_chunks = visual_chunks
        self.config = config
        self.vision_model_func = vision_model_func

        # Sub-components
        from dlightrag.converters.office import LibreOfficeConverter

        converter = LibreOfficeConverter(config)
        self.renderer = PageRenderer(dpi=config.page_render_dpi, converter=converter)

        # Use pre-built embedder (shared with LightRAG's embedding_func)
        # or create one from config as fallback.
        if visual_embedder is not None:
            self.embedder = visual_embedder
        else:
            self.embedder = VisualEmbedder(
                model=config.embedding.model,
                base_url=config.embedding.base_url or "",
                api_key=config.embedding.api_key or "",
                dim=config.embedding.dim,
                batch_size=config.embedding_func_max_async,
            )

        self.extractor = EntityExtractor(
            lightrag=lightrag,
            entity_types=config.kg_entity_types,
            vision_model_func=vision_model_func,
            context_model_func=context_model_func,
        )

        self.retriever = VisualRetriever(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
            vision_model_func=vision_model_func,
            rerank_model=(config.rerank.model if config.rerank.enabled else None),
            rerank_base_url=(config.rerank.base_url if config.rerank.enabled else None),
            rerank_api_key=(config.rerank.api_key if config.rerank.enabled else None),
            rerank_backend=(config.rerank.backend if config.rerank.enabled else None),
            path_resolver=path_resolver,
            embedder=self.embedder,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def aingest(
        self,
        file_path: str,
        doc_id: str | None = None,
    ) -> dict:
        """Ingest a document into the unified representational pipeline.

        Steps:
            1. Render pages to images.
            2. Generate ``doc_id``.
            3. Write ``full_docs`` entry (LightRAG compat).
            4. Run visual embedding and VLM entity extraction in parallel.
            5. Write to ``chunks_vdb``, ``text_chunks``, and ``visual_chunks``.

        Returns
        -------
        dict
            Keys: ``doc_id``, ``page_count``, ``file_path``.
        """
        from datetime import UTC, datetime

        from lightrag.base import DocStatus

        path = Path(file_path)

        # Step 1: Render pages
        render_result = await self.renderer.render_file(path)
        images = [img for _, img in render_result.pages]
        page_count = len(images)
        logger.info("Rendered %d pages from %s", page_count, path.name)

        if not images:
            raise ValueError(f"No pages rendered from {file_path}")

        # Step 2: Generate doc_id
        if doc_id is None:
            doc_id = compute_mdhash_id(file_path, prefix="doc-")

        # Early checkpoint: mark PROCESSING so crashes leave a recoverable state.
        # LightRAG's _validate_and_fix_document_consistency() will find this on
        # restart and reset to PENDING. The hash index won't have this file
        # registered (registration happens in lifecycle AFTER aingest succeeds),
        # so re-ingestion will proceed naturally.
        now = datetime.now(UTC).isoformat()
        await self.lightrag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": f"[unified] {path.name} ({page_count} pages)",
                    "content_length": 0,
                    "created_at": now,
                    "updated_at": now,
                    "file_path": str(path),
                    "chunks_count": 0,
                    "chunks_list": [],
                }
            }
        )

        try:
            # Step 3: Write full_docs entry (minimal, for LightRAG compat)
            await self.lightrag.full_docs.upsert(
                {doc_id: {"content": "", "file_path": str(path), "page_count": page_count}}
            )

            # Step 4: Parallel embedding + entity extraction
            embed_task = self.embedder.embed_pages(images)
            extract_task = self.extractor.extract_from_pages(
                images=images, doc_id=doc_id, file_path=str(path)
            )
            visual_vectors, page_infos = await asyncio.gather(embed_task, extract_task)

            # Step 5a: Write to chunks_vdb (visual vectors via embedding func swap)
            chunks_data: dict[str, dict] = {}
            for info in page_infos:
                chunk_id = info["chunk_id"]
                chunks_data[chunk_id] = {
                    "content": info["content"],  # VLM text description
                    "full_doc_id": doc_id,
                    "file_path": str(path),
                    "tokens": len(info["content"].split()),
                    "chunk_order_index": info["page_index"],
                }
            await self._upsert_with_visual_vectors(chunks_data, visual_vectors)

            # Step 5b: Write to text_chunks (page summaries)
            text_chunks_data: dict[str, dict] = {}
            for info in page_infos:
                chunk_id = info["chunk_id"]
                text_chunks_data[chunk_id] = {
                    "content": info["content"],
                    "full_doc_id": doc_id,
                    "file_path": str(path),
                    "tokens": len(info["content"].split()),
                    "chunk_order_index": info["page_index"],
                    "page_idx": info["page_index"],
                    "source_type": "unified_represent",
                }
            await self.lightrag.text_chunks.upsert(text_chunks_data)

            # Step 5c: Write to visual_chunks (page images + metadata)
            visual_data: dict[str, dict] = {}
            for info in page_infos:
                chunk_id = info["chunk_id"]
                page_idx = info["page_index"]
                img = images[page_idx]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                visual_data[chunk_id] = {
                    "image_data": image_b64,
                    "page_index": page_idx,
                    "full_doc_id": doc_id,
                    "file_path": str(path),
                    "doc_title": render_result.metadata.get("title", ""),
                    "doc_author": render_result.metadata.get("author", ""),
                    "creation_date": render_result.metadata.get("creation_date", ""),
                    "page_count": render_result.metadata.get("page_count", page_count),
                    "original_format": render_result.metadata.get("original_format", ""),
                }
            await self.visual_chunks.upsert(visual_data)

            # Final checkpoint: mark PROCESSED with full chunk list
            chunk_ids = [info["chunk_id"] for info in page_infos]
            await self.lightrag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PROCESSED,
                        "content_summary": f"[unified] {path.name} ({page_count} pages)",
                        "content_length": 0,
                        "created_at": now,
                        "updated_at": datetime.now(UTC).isoformat(),
                        "file_path": str(path),
                        "chunks_count": len(chunk_ids),
                        "chunks_list": chunk_ids,
                    }
                }
            )

        except Exception as exc:
            # Mark FAILED so the document is visible in health/status and
            # LightRAG's consistency checker can clean up on restart.
            try:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.FAILED,
                            "error_msg": str(exc),
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    }
                )
            except Exception:
                logger.warning("Failed to write FAILED doc_status for %s", doc_id)
            raise

        logger.info("Ingested %s: %d pages, doc_id=%s", path.name, page_count, doc_id)
        return {
            "doc_id": doc_id,
            "page_count": page_count,
            "file_path": str(path),
        }

    @staticmethod
    def _extract_image_bytes(
        multimodal_content: list[dict[str, Any]] | None,
    ) -> list[bytes] | None:
        """Convert multimodal_content dicts to raw image bytes."""
        if not multimodal_content:
            return None
        images: list[bytes] = []
        for item in multimodal_content:
            if item.get("type") != "image":
                continue
            if item.get("img_path"):
                images.append(Path(item["img_path"]).read_bytes())
            elif item.get("data"):
                images.append(base64.b64decode(item["data"]))
        return images or None

    async def aretrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        multimodal_content: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant visual chunks (Phases 1-3)."""
        images = self._extract_image_bytes(multimodal_content)
        result = await self.retriever.retrieve(
            query=query,
            mode=mode or self.config.default_mode,
            top_k=top_k or self.config.top_k,
            chunk_top_k=chunk_top_k or self.config.chunk_top_k,
            images=images,
        )
        return RetrievalResult(
            answer=None,
            contexts=result.get("contexts", {}),
        )

    async def aclose(self) -> None:
        """Release resources held by sub-components (HTTP clients, etc.)."""
        await self.embedder.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _upsert_with_visual_vectors(
        self,
        chunks_data: dict[str, dict],
        visual_vectors: np.ndarray,
    ) -> None:
        """Upsert chunks with pre-computed visual vectors to chunks_vdb.

        LightRAG's ``chunks_vdb.upsert()`` always recomputes embeddings
        from the ``content`` field text. There is no way to pass
        pre-computed vectors through the standard API. We work around
        this by temporarily swapping the ``embedding_func`` on
        ``chunks_vdb`` with a cache-backed function that returns our
        pre-computed visual vectors instead of computing new ones.
        """
        if not chunks_data:
            return

        # Build cache: content text -> pre-computed vector
        vector_cache: dict[str, np.ndarray] = {}
        for (_, chunk_dict), vector in zip(chunks_data.items(), visual_vectors, strict=True):
            vector_cache[chunk_dict["content"]] = vector

        # Temporary embedding function that looks up cached vectors
        async def cached_embed(texts: list[str]) -> np.ndarray:
            results = []
            for text in texts:
                if text in vector_cache:
                    results.append(vector_cache[text])
                else:
                    raise ValueError(f"No pre-computed vector for: {text[:80]}...")
            return np.array(results, dtype=np.float32)

        # Swap embedding func on chunks_vdb
        original_func = self.lightrag.chunks_vdb.embedding_func
        self.lightrag.chunks_vdb.embedding_func = EmbeddingFunc(
            embedding_dim=self.config.embedding.dim,
            max_token_size=8192,
            func=cached_embed,
        )
        try:
            await self.lightrag.chunks_vdb.upsert(chunks_data)
        finally:
            self.lightrag.chunks_vdb.embedding_func = original_func
