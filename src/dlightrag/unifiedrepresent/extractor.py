# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Entity extraction for unified representational RAG.

Uses VLM to generate text descriptions from page images, then feeds
them into LightRAG's entity extraction pipeline to build the knowledge graph.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from lightrag.operate import extract_entities, merge_nodes_and_edges
from lightrag.utils import compute_mdhash_id

# LightRAG v1.4.10 merge_nodes_and_edges unconditionally does
# ``async with pipeline_status_lock`` — crashes when lock is None.
# We provide a real lock + status dict to satisfy the contract.
_NOOP_PIPELINE_STATUS: dict = {"latest_message": "", "history_messages": []}
_PIPELINE_STATUS_LOCK = asyncio.Lock()

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities from page images via VLM + LightRAG pipeline."""

    def __init__(
        self,
        lightrag: Any,  # LightRAG instance
        entity_types: list[str],
        vision_model_func: Callable | None = None,
        max_concurrent_vlm: int = 4,
    ) -> None:
        self.lightrag = lightrag
        self.entity_types = entity_types
        self.vision_model_func = vision_model_func
        self._vlm_semaphore = asyncio.Semaphore(max_concurrent_vlm)

    async def extract_from_pages(
        self,
        images: list[Any],  # list[PIL.Image.Image]
        doc_id: str,
        file_path: str,
    ) -> list[dict]:
        """Extract entities from page images and build KG.

        Returns list of dicts with keys: chunk_id, page_index, content
        (VLM description). These are used to populate text_chunks and
        visual_chunks.
        """
        if self.vision_model_func is None:
            raise ValueError("vision_model_func is required for entity extraction")

        # 1. Generate VLM descriptions for all pages (semaphore-controlled)
        description_tasks = [
            self._describe_page(image, page_index) for page_index, image in enumerate(images)
        ]
        descriptions = await asyncio.gather(*description_tasks)

        # 2. Build chunks dict for LightRAG
        chunk_ids: list[str] = []
        chunks_dict: dict[str, dict[str, Any]] = {}
        for page_index, description in enumerate(descriptions):
            chunk_id = compute_mdhash_id(f"{doc_id}:page:{page_index}", prefix="chunk-")
            chunk_ids.append(chunk_id)
            chunks_dict[chunk_id] = {
                "content": description,
                "full_doc_id": doc_id,
                "tokens": len(description.split()),
                "chunk_order_index": page_index,
                "file_path": file_path,
            }

        # 3. Call extract_entities
        chunk_results = await extract_entities(
            chunks=chunks_dict,  # type: ignore[arg-type]
            global_config=self.lightrag.__dict__,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        # 4. Call merge_nodes_and_edges
        await merge_nodes_and_edges(
            chunk_results=chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            llm_response_cache=self.lightrag.llm_response_cache,
            entity_chunks_storage=self.lightrag.entity_chunks,
            relation_chunks_storage=self.lightrag.relation_chunks,
            pipeline_status=_NOOP_PIPELINE_STATUS,
            pipeline_status_lock=_PIPELINE_STATUS_LOCK,
            file_path=file_path,
        )

        # 5. Return page info list
        return [
            {
                "chunk_id": chunk_id,
                "page_index": page_index,
                "content": description,
            }
            for page_index, (chunk_id, description) in enumerate(
                zip(chunk_ids, descriptions, strict=True)
            )
        ]

    async def _describe_page(self, image: Any, page_index: int) -> str:
        """Call VLM to extract structured content and convert to text."""
        from dlightrag.core.vlm_ocr import (
            OCR_SYSTEM_PROMPT,
            OCR_USER_PROMPT,
            blocks_to_text,
            image_to_png_bytes,
            parse_vlm_response,
        )

        if self.vision_model_func is None:
            raise RuntimeError("vision_model_func is required but was not set")

        image_bytes = image_to_png_bytes(image)

        async with self._vlm_semaphore:
            raw = await self.vision_model_func(
                OCR_USER_PROMPT,
                image_data=image_bytes,
                system_prompt=OCR_SYSTEM_PROMPT,
            )

        if not raw or not raw.strip():
            logger.warning("VLM returned empty response for page %d", page_index)
            return f"[Page {page_index + 1}: no content extracted]"

        blocks = parse_vlm_response(raw)
        if blocks is None:
            # Could not parse structured JSON — use raw text as fallback
            return raw.strip()

        text = blocks_to_text(blocks)
        return text or f"[Page {page_index + 1}: no content extracted]"
