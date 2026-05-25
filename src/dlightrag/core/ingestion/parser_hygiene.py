# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG parser hygiene for MinerU page furniture.

DlightRAG delegates document parsing to LightRAG. This module does not create a
second MinerU ingestion path; it narrows LightRAG's MinerU IR builder at the
parser boundary when current upstream would otherwise index page furniture as
body text.
"""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MINERU_AUXILIARY_BLOCK_TYPES = frozenset(
    {
        "discarded",
        "discarded_block",
        "discarded_blocks",
        "header",
        "footer",
        "page_header",
        "page_footer",
        "page_number",
        "aside_text",
        "page_aside_text",
        "margin_note",
        "page_margin_note",
        "page_footnote",
    }
)

_PATCH_ATTR = "_dlightrag_filters_mineru_auxiliary_blocks"


def filter_mineru_auxiliary_blocks(content_list: list[Any]) -> list[Any]:
    """Drop MinerU page furniture while preserving semantic/multimodal items."""
    return [item for item in content_list if not _is_mineru_auxiliary_block(item)]


def mineru_ir_builder_needs_auxiliary_filter(builder_cls: Any | None = None) -> bool:
    """Return whether the LightRAG MinerU builder indexes page furniture."""
    if builder_cls is None:
        from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

        builder_cls = MinerUIRBuilder

    method = builder_cls._normalize_content_list
    if getattr(method, _PATCH_ATTR, False):
        return False

    try:
        doc = builder_cls()._normalize_content_list(
            [
                {"type": "header", "text": "DLIGHTRAG_AUXILIARY_PROBE"},
                {"type": "text", "text": "DLIGHTRAG_BODY_PROBE"},
            ],
            Path.cwd(),
            document_name="dlightrag-probe.pdf",
        )
    except Exception as exc:  # pragma: no cover - defensive upstream guard
        logger.debug("Could not probe MinerU IR builder hygiene: %r", exc)
        return True

    content = "\n".join(str(getattr(block, "content_template", "")) for block in doc.blocks)
    return "DLIGHTRAG_AUXILIARY_PROBE" in content


def apply_mineru_auxiliary_block_filter() -> bool:
    """Patch LightRAG's MinerU IR builder only when upstream needs it."""
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    original = MinerUIRBuilder._normalize_content_list
    if getattr(original, _PATCH_ATTR, False):
        return False
    if not mineru_ir_builder_needs_auxiliary_filter(MinerUIRBuilder):
        logger.debug("LightRAG MinerU IR builder already skips auxiliary blocks")
        return False

    @wraps(original)
    def patched_normalize_content_list(
        self: Any,
        content_list: list[Any],
        raw_dir: Path,
        *,
        document_name: str,
    ) -> Any:
        return original(
            self,
            filter_mineru_auxiliary_blocks(content_list),
            raw_dir,
            document_name=document_name,
        )

    setattr(patched_normalize_content_list, _PATCH_ATTR, True)
    patched_normalize_content_list._dlightrag_original = original  # type: ignore[attr-defined]
    MinerUIRBuilder._normalize_content_list = patched_normalize_content_list
    logger.info("Applied LightRAG MinerU auxiliary block hygiene patch")
    return True


def _is_mineru_auxiliary_block(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    block_type = _block_type(item)
    return block_type in MINERU_AUXILIARY_BLOCK_TYPES


def _block_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("label") or "").strip().lower()


__all__ = [
    "MINERU_AUXILIARY_BLOCK_TYPES",
    "apply_mineru_auxiliary_block_filter",
    "filter_mineru_auxiliary_blocks",
    "mineru_ir_builder_needs_auxiliary_filter",
]
