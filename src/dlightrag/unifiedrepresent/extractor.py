# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Entity extraction for unified representational RAG.

Uses VLM to generate text descriptions from page images, then feeds
them into LightRAG's entity extraction pipeline to build the knowledge graph.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
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
        context_model_func: Callable | None = None,
    ) -> None:
        self.lightrag = lightrag
        self.entity_types = entity_types
        self.vision_model_func = vision_model_func
        self.context_model_func = context_model_func
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

    async def _describe_page(
        self,
        image: Any,
        page_index: int,
        structural_ctx: str | None = None,
    ) -> str:
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

        b64 = base64.b64encode(image_bytes).decode()

        # Build user content parts
        user_content: list[dict[str, Any]] = []

        # Inject structural context before image if available
        if structural_ctx:
            user_content.append(
                {"type": "text", "text": f"Context from previous pages:\n{structural_ctx}"}
            )

        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
        user_content.append({"type": "text", "text": OCR_USER_PROMPT})

        messages = [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        async with self._vlm_semaphore:
            raw = await self.vision_model_func(messages=messages)

        if not raw or not raw.strip():
            logger.warning("VLM returned empty response for page %d", page_index)
            return f"[Page {page_index + 1}: no content extracted]"

        blocks = parse_vlm_response(raw)
        if blocks is None:
            # Could not parse structured JSON — use raw text as fallback
            return raw.strip()

        text = blocks_to_text(blocks)
        return text or f"[Page {page_index + 1}: no content extracted]"

    async def _update_structural_context(
        self,
        structural_ctx: str | None,
        page_text: str,
    ) -> str | None:
        """Decide whether to update the running structural context.

        Uses a lightweight text-only LLM call to analyze the current page's
        VLM output and decide if the structural context needs updating.

        Returns the (possibly updated) structural context string, or None.
        """
        if self.context_model_func is None:
            return structural_ctx

        from dlightrag.unifiedrepresent.prompts import STRUCTURAL_CONTEXT_PROMPT

        ctx_display = structural_ctx if structural_ctx else "(empty — first page)"
        user_content = (
            f"Current structural context:\n{ctx_display}\n\n"
            f"Current page content:\n{page_text}"
        )
        messages = [
            {"role": "system", "content": STRUCTURAL_CONTEXT_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = await self.context_model_func(messages=messages)
        except Exception:
            logger.warning("Structural context LLM call failed, keeping existing context")
            return structural_ctx

        # Defensive JSON parsing: try direct, then regex extraction
        parsed = self._parse_context_response(raw)
        if parsed is None:
            return structural_ctx

        action = parsed.get("action", "").lower()
        if action == "update":
            new_ctx = parsed.get("context")
            if new_ctx and isinstance(new_ctx, str):
                return new_ctx
        # "keep" or unrecognized action → return existing
        return structural_ctx

    @staticmethod
    def _parse_context_response(raw: str | None) -> dict | None:
        """Parse JSON from LLM response with fallback to regex extraction."""
        if not raw or not raw.strip():
            return None

        # Attempt 1: direct json.loads
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Attempt 2: regex extract first {...} blob
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        return None
