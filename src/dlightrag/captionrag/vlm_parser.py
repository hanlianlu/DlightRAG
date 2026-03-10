# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-based document OCR parser.

Renders document pages via :class:`PageRenderer`, sends each page image to a
vision language model for structured JSON extraction, and produces a
RAGAnything-compatible ``content_list``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from lightrag.utils import compute_mdhash_id

from dlightrag.core.vlm_ocr import (
    OCR_SYSTEM_PROMPT,
    OCR_USER_PROMPT,
    image_to_png_bytes,
    parse_vlm_response,
)
from dlightrag.unifiedrepresent.renderer import PageRenderer

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
        Async callable with signature
        ``async def(prompt, *, image_data, system_prompt=None, **kw) -> str``.
    dpi:
        Rendering resolution passed to :class:`PageRenderer`.
    """

    def __init__(self, vision_model_func: Callable, dpi: int = 250) -> None:
        self.vision_model_func = vision_model_func
        self.renderer = PageRenderer(dpi=dpi)

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
        # Step 1: Render pages
        result = await self.renderer.render_file(file_path)

        # Step 2: Ensure output directory for page images
        pages_dir = Path(output_dir) / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        content_list: list[dict] = []

        for page_idx, image in result.pages:
            # Save page image
            page_image_path = pages_dir / f"page_{page_idx}.png"
            image.save(str(page_image_path), format="PNG")

            # Step 3: Extract structured content via VLM
            page_entries = await self._extract_page(image, page_idx)
            content_list.extend(page_entries)

            # Step 4: Add page image entry (for media/visual embedding)
            content_list.append(
                {
                    "type": "image",
                    "img_path": str(page_image_path),
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
        raw = await self.vision_model_func(
            OCR_USER_PROMPT,
            image_data=image_bytes,
            system_prompt=OCR_SYSTEM_PROMPT,
        )

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
