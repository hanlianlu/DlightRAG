# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared VLM OCR extraction utilities.

Used by both caption mode (:mod:`dlightrag.core.ingestion.vlm_parser`) and
unified mode (:mod:`dlightrag.unifiedrepresent.extractor`) for consistent,
high-quality page content extraction via a vision language model.
"""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def image_to_png_bytes(image: Any) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def parse_vlm_response(raw: str) -> list[dict] | None:
    """Parse VLM response into a list of block dicts.

    Strategy:
    1. Try ``json.loads(raw)`` directly — expect ``{"blocks": [...]}``.
    2. Fallback: regex-extract the first ``{...}`` blob and parse it.
    3. If all parsing fails, return ``None`` (distinct from empty ``[]``).
    """
    # Attempt 1: direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "blocks" in data:
            return data["blocks"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: regex extract outermost { ... }
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "blocks" in data:
                return data["blocks"]
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def blocks_to_text(blocks: list[dict]) -> str:
    """Convert structured OCR blocks to plain text.

    Used by unified mode to feed VLM-extracted content into LightRAG's
    entity extraction pipeline.
    """
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")

        if btype == "heading":
            level = block.get("level", 1)
            prefix = "#" * level
            parts.append(f"{prefix} {block.get('text', '')}")

        elif btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)

        elif btype == "table":
            html = block.get("html", "")
            if html:
                parts.append(html)

        elif btype == "formula":
            latex = block.get("latex", "")
            if latex:
                parts.append(latex)

        elif btype == "figure":
            desc = block.get("description", "")
            if desc:
                parts.append(f"[Figure: {desc}]")

    return "\n\n".join(parts)
