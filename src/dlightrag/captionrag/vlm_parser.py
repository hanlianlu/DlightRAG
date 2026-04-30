# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-based document OCR parser.

Renders document pages via :class:`PageRenderer`, sends each page image to a
vision language model for structured JSON extraction, and produces a
RAGAnything-compatible ``content_list``.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from pathlib import Path

from lightrag.utils import compute_mdhash_id

from dlightrag.converters.page_renderer import PageRenderer
from dlightrag.core.vlm_ocr import (
    image_to_png_bytes,
    parse_vlm_response,
)
from dlightrag.prompts import (
    OCR_SYSTEM_PROMPT,
    OCR_USER_PROMPT,
)

logger = logging.getLogger(__name__)


def _blocks_to_content_list(blocks: list[dict], page_idx: int) -> list[dict]:
    """Convert VLM-extracted blocks into RAGAnything content_list entries."""
    entries: list[dict] = []
    for block in blocks:
        btype = block.get("type", "")

        if btype == "heading":
            level = block.get("level", 1)
            prefix = "#" * level
            entries.append(
                {
                    "type": "text",
                    "text": f"{prefix} {block.get('text', '')}",
                    "page_idx": page_idx,
                    "text_level": level,
                }
            )

        elif btype == "text":
            entries.append(
                {
                    "type": "text",
                    "text": block.get("text", ""),
                    "page_idx": page_idx,
                }
            )

        elif btype == "table":
            entries.append(
                {
                    "type": "table",
                    "table_body": block.get("html", ""),
                    "page_idx": page_idx,
                }
            )

        elif btype == "formula":
            entries.append(
                {
                    "type": "text",
                    "text": block.get("latex", ""),
                    "page_idx": page_idx,
                }
            )

        elif btype == "figure":
            entries.append(
                {
                    "type": "image",
                    "img_path": None,
                    "image_caption": [block.get("description", "")],
                    "page_idx": page_idx,
                }
            )

    return entries


# ---------------------------------------------------------------------------
# VlmOcrParser
# ---------------------------------------------------------------------------


class VlmOcrParser:
    """VLM-based document parser.

    Renders pages via :class:`PageRenderer`, calls a vision model for
    structured JSON extraction per page, and produces a RAGAnything-compatible
    ``content_list``.

    Parameters
    ----------
    vision_model_func:
        Async callable accepting ``messages=`` (OpenAI chat format).
    dpi:
        Rendering resolution passed to :class:`PageRenderer`.
    """

    def __init__(
        self,
        vision_model_func: Callable,
        dpi: int = 250,
        max_concurrent: int = 4,
    ) -> None:
        self.vision_model_func = vision_model_func
        self.renderer = PageRenderer(dpi=dpi)
        self.max_concurrent = max_concurrent

    async def parse(self, file_path: str, output_dir: str) -> tuple[list[dict], str]:
        """Parse a document file into a RAGAnything content_list.

        Parameters
        ----------
        file_path:
            Path to the input file (PDF, image, or Office document).
        output_dir:
            Directory for saving page images.

        Returns
        -------
        tuple[list[dict], str]
            ``(content_list, doc_id)``
        """
        import asyncio

        from dlightrag.utils.concurrency import bounded_gather

        # Step 1: Render pages
        result = await self.renderer.render_file(file_path)

        # Step 2: Ensure output directory for page images
        pages_dir = Path(output_dir) / "pages"
        await asyncio.to_thread(pages_dir.mkdir, parents=True, exist_ok=True)

        # Step 3: Per-page work — save image (sync PIL I/O via thread) +
        # VLM extraction. Pages are independent; cap concurrency so we
        # don't blow past the chat model's rate limit.
        async def _process_page(page_idx: int, image) -> tuple[int, str, list[dict]]:  # noqa: ANN001
            page_image_path = pages_dir / f"page_{page_idx}.png"
            await asyncio.to_thread(image.save, str(page_image_path), format="PNG")
            page_entries = await self._extract_page(image, page_idx)
            return page_idx, str(page_image_path), page_entries

        coros = [_process_page(idx, img) for idx, img in result.pages]
        page_results = await bounded_gather(
            coros, max_concurrent=self.max_concurrent, task_name="vlm-parse"
        )

        # Step 4: Assemble content_list in page order. Per-page entries
        # come first, then the page-image entry, matching the original
        # interleaving used by the linear loop.
        content_list: list[dict] = []
        for outcome in page_results:
            if isinstance(outcome, BaseException):
                # Surface the first failure rather than silently dropping pages.
                raise outcome
            page_idx, image_path, page_entries = outcome
            content_list.extend(page_entries)
            content_list.append(
                {
                    "type": "image",
                    "img_path": image_path,
                    "image_caption": [],
                    "page_idx": page_idx,
                }
            )

        # Step 5: Generate doc_id
        doc_id = compute_mdhash_id(file_path, prefix="doc-")

        return content_list, doc_id

    async def _extract_page(self, image, page_idx: int) -> list[dict]:  # noqa: ANN001
        """Extract structured content from a single page image.

        Parameters
        ----------
        image:
            PIL Image of the page.
        page_idx:
            Zero-based page index.

        Returns
        -------
        list[dict]
            Content-list entries extracted from the page.
        """
        # Convert PIL Image to PNG bytes
        image_bytes = image_to_png_bytes(image)

        # Call vision model
        b64 = base64.b64encode(image_bytes).decode()
        messages = [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": OCR_USER_PROMPT},
                ],
            },
        ]
        raw = await self.vision_model_func(messages=messages)

        # Parse and convert
        blocks = parse_vlm_response(raw)
        if blocks is None and raw and raw.strip():
            # Fallback: use raw response as single text block
            logger.warning(
                "VLM OCR: could not parse structured output for page %d, falling back to raw text",
                page_idx,
            )
            return [{"type": "text", "text": raw.strip(), "page_idx": page_idx}]
        return _blocks_to_content_list(blocks or [], page_idx)
