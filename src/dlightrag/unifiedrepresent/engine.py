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

from lightrag.utils import EmbeddingFunc, compute_mdhash_id
from PIL import Image

from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.unifiedrepresent.embedder import VisualEmbedder
from dlightrag.unifiedrepresent.extractor import EntityExtractor
from dlightrag.unifiedrepresent.renderer import PageRenderer
from dlightrag.unifiedrepresent.retriever import VisualRetriever

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Encoding utilities
# ------------------------------------------------------------------


def encode_png_b64(image: Image.Image) -> str:
    """Encode PIL Image to raw PNG base64 (no data URI prefix) for visual_chunks."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------
# Prefetch helper
# ------------------------------------------------------------------


async def _safe_anext(aiter: Any) -> Any:
    """Return next item from async iterator, or None if exhausted."""
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return None


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

        Processes pages in streaming batches to bound peak memory. Each batch:
            1. Render a batch of pages via ``render_file_batched``.
            2. Run visual embedding + VLM entity extraction in parallel.
            3. Encode PNG per page for visual_chunks, release PIL images.
            4. Batch-upsert visual_chunks, chunks_vdb, text_chunks.
            5. Accumulate page_infos across batches.
            6. Prefetch next batch (after PIL release).

        After all batches:
            7. Write full_docs, doc_status (PROCESSED).

        Returns
        -------
        dict
            Keys: ``doc_id``, ``page_count``, ``file_path``.
        """
        from datetime import UTC, datetime

        from lightrag.base import DocStatus

        path = Path(file_path)

        # Generate doc_id up front
        if doc_id is None:
            doc_id = compute_mdhash_id(file_path, prefix="doc-")

        batch_size = getattr(self.config, "ingestion_batch_pages", 20)

        # Accumulators across batches
        all_page_infos: list[dict] = []
        total_page_count = 0
        last_metadata: dict[str, str | int] = {}

        # Early checkpoint: mark PROCESSING so crashes leave a recoverable state.
        created_at = datetime.now(UTC).isoformat()
        await self.lightrag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": f"[unified] {path.name}",
                    "content_length": 0,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "file_path": str(path),
                    "chunks_count": 0,
                    "chunks_list": [],
                }
            }
        )

        # Set up double-buffered prefetch
        batch_aiter = self.renderer.render_file_batched(path, batch_size=batch_size).__aiter__()
        current_batch = await _safe_anext(batch_aiter)
        prefetch_task: asyncio.Task[Any] | None = None

        try:
            while current_batch is not None:
                render_result = current_batch
                pages = render_result.pages  # list[(page_index, PIL.Image)]
                images = [img for _, img in pages]
                batch_page_count = len(images)

                if batch_page_count == 0:
                    # Empty batch (defensive)
                    prefetch_task = asyncio.create_task(_safe_anext(batch_aiter))
                    current_batch = await prefetch_task
                    prefetch_task = None
                    continue

                # Accumulate metadata from first non-empty batch
                if not last_metadata:
                    last_metadata = dict(render_result.metadata)

                logger.info("Processing batch of %d pages from %s", batch_page_count, path.name)

                # Step 2: Parallel embedding + extraction
                # DlightRAG's embedder and extractor accept PIL Images directly
                embed_task = self.embedder.embed_pages(images)
                extract_task = self.extractor.extract_from_pages(
                    images=images, doc_id=doc_id, file_path=str(path)
                )
                visual_vectors, page_infos = await asyncio.gather(embed_task, extract_task)

                # Step 3: Encode PNG per page using chunk_ids from page_infos,
                # release PIL images eagerly
                visual_data: dict[str, dict] = {}
                for info, img in zip(page_infos, images, strict=True):
                    chunk_id = info["chunk_id"]
                    page_idx = info["page_index"]
                    png_b64 = encode_png_b64(img)
                    img.close()
                    visual_data[chunk_id] = {
                        "image_data": png_b64,
                        "page_index": page_idx,
                        "full_doc_id": doc_id,
                        "file_path": str(path),
                        "doc_title": render_result.metadata.get("title", ""),
                        "doc_author": render_result.metadata.get("author", ""),
                        "creation_date": render_result.metadata.get("creation_date", ""),
                        "page_count": render_result.metadata.get("page_count", 0),
                        "original_format": render_result.metadata.get("original_format", ""),
                    }
                del images  # all PIL images closed

                # Start prefetch of next batch after PIL release
                prefetch_task = asyncio.create_task(_safe_anext(batch_aiter))

                # Step 4a: Upsert visual_chunks
                await self.visual_chunks.upsert(visual_data)

                # Step 4b: Write chunks_vdb (visual vectors via cache-swap)
                chunks_data: dict[str, dict] = {}
                for info in page_infos:
                    chunk_id = info["chunk_id"]
                    chunks_data[chunk_id] = {
                        "content": info["content"],
                        "full_doc_id": doc_id,
                        "file_path": str(path),
                        "tokens": len(info["content"].split()),
                        "chunk_order_index": info["page_index"],
                    }
                await self._upsert_with_visual_vectors(chunks_data, visual_vectors)

                # Step 4c: Write text_chunks (page summaries)
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

                # Step 5: Accumulate metadata across batches
                all_page_infos.extend(page_infos)
                total_page_count += batch_page_count

                # Get next batch (overlaps with stores writes via prefetch)
                current_batch = await prefetch_task
                prefetch_task = None

            # -- After all batches --

            if total_page_count == 0:
                raise ValueError(f"No pages rendered from {file_path}")

            # Step 7a: Write full_docs
            await self.lightrag.full_docs.upsert(
                {
                    doc_id: {
                        "content": "",
                        "file_path": str(path),
                        "page_count": total_page_count,
                    }
                }
            )

            # Step 7b: Final checkpoint — mark PROCESSED with full chunk list
            chunk_ids = [info["chunk_id"] for info in all_page_infos]
            await self.lightrag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PROCESSED,
                        "content_summary": (f"[unified] {path.name} ({total_page_count} pages)"),
                        "content_length": 0,
                        "created_at": created_at,
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
                            "content_summary": f"[unified] {path.name} (FAILED)",
                            "content_length": 0,
                            "error_msg": str(exc),
                            "chunks_list": [info["chunk_id"] for info in all_page_infos],
                            "chunks_count": total_page_count,
                            "file_path": str(path),
                            "created_at": created_at,
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    }
                )
            except Exception:
                logger.warning("Failed to write FAILED doc_status for %s", doc_id)
            raise

        logger.info("Ingested %s: %d pages, doc_id=%s", path.name, total_page_count, doc_id)
        return {
            "doc_id": doc_id,
            "page_count": total_page_count,
            "file_path": str(path),
            "render_metadata": last_metadata,
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
        visual_vectors: list[list[float]],
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
        vector_cache: dict[str, list[float]] = {}
        for (_, chunk_dict), vector in zip(chunks_data.items(), visual_vectors, strict=True):
            vector_cache[chunk_dict["content"]] = vector

        # Temporary embedding function that looks up cached vectors
        async def cached_embed(texts: list[str]) -> list[list[float]]:
            results: list[list[float]] = []
            for text in texts:
                if text in vector_cache:
                    results.append(vector_cache[text])
                else:
                    raise ValueError(f"No pre-computed vector for: {text[:80]}...")
            return results

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
