# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG MinerU content-list hygiene at the parser boundary.

DlightRAG delegates document parsing to LightRAG. This module does not create a
second MinerU ingestion path; it narrows LightRAG's MinerU IR builder at the
``_normalize_content_list`` boundary for two gaps current upstream leaves:

1. Drawing aliases — MinerU emits figure blocks (notably ``chart``) that are
   structurally identical to ``image`` (an ``img_path`` plus ``<type>_caption``
   / ``<type>_footnote``) but under type names the IR builder does not map to a
   drawing. Upstream serializes them through the text fallback, and because
   their ``content`` is empty the image *and* its caption are dropped. We alias
   them to ``image`` so they flow through the existing IRDrawing / VLM sidecar
   path and stay retrievable.

2. Auxiliary furniture — MinerU emits running headers, footers, and printed
   page numbers in ``content_list``; indexing those as body text pollutes
   chunks, KG extraction, BM25, and citations. Conservative mode drops only
   discarded blocks, headers, footers, and printed page numbers; extended mode
   also drops aside / margin / page-footnote blocks
   (``DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY``).

Each transform is applied only when a runtime behavior probe shows current
upstream still needs it, so the patch self-disables as LightRAG catches up.
Keep this module small and delete transforms as upstream covers them.
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# MinerU figure types that carry an image but that LightRAG's IR builder does
# not map to IRDrawing (it maps only ``image`` / ``picture`` / ``drawing``).
# They share the ``image`` shape — an ``img_path`` plus ``<type>_caption`` /
# ``<type>_footnote`` — so aliasing them to ``image`` is loss-free. Extend this
# set if MinerU introduces further figure-like types (e.g. ``figure``).
MINERU_DRAWING_ALIAS_TYPES = frozenset({"chart"})

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
    }
)

MINERU_EXTENDED_AUXILIARY_BLOCK_TYPES = frozenset(
    {
        "aside_text",
        "page_aside_text",
        "margin_note",
        "page_margin_note",
        "page_footnote",
    }
)

_PATCH_ATTR = "_dlightrag_normalizes_mineru_content_list"


# ---------------------------------------------------------------------------
# Pure content_list transforms
# ---------------------------------------------------------------------------


def normalize_mineru_drawing_aliases(content_list: list[Any]) -> list[Any]:
    """Rewrite figure-like items (``chart`` …) to ``image`` items.

    The IR builder routes ``image`` through IRDrawing; alias types would
    otherwise fall through to the text fallback and, with empty ``content``, be
    dropped entirely. Caption / footnote fields are renamed to the ``image_*``
    names the drawing builder reads so the figure's caption survives.
    """
    return [_alias_drawing_item(item) if _is_drawing_alias(item) else item for item in content_list]


def filter_mineru_auxiliary_blocks(content_list: list[Any]) -> list[Any]:
    """Drop MinerU page furniture while preserving semantic/multimodal items."""
    return [item for item in content_list if not _is_mineru_auxiliary_block(item)]


# ---------------------------------------------------------------------------
# Upstream behavior probes
# ---------------------------------------------------------------------------


def mineru_ir_builder_needs_drawing_alias_normalization(builder_cls: Any | None = None) -> bool:
    """Return whether upstream drops MinerU drawing-alias figure blocks."""
    builder_cls = _resolve_builder_cls(builder_cls)
    if builder_cls is None or _already_patched(builder_cls):
        return False

    probe_type = next(iter(MINERU_DRAWING_ALIAS_TYPES), "chart")
    try:
        doc = builder_cls()._normalize_content_list(
            [
                {
                    "type": probe_type,
                    "img_path": "dlightrag-probe.jpg",
                    f"{probe_type}_caption": ["DLIGHTRAG_DRAWING_PROBE"],
                }
            ],
            Path.cwd(),
            document_name="dlightrag-probe.pdf",
        )
    except Exception as exc:  # pragma: no cover - defensive upstream guard
        logger.debug("Could not probe MinerU IR builder drawing aliases: %r", exc)
        return True

    content = "\n".join(str(getattr(block, "content_template", "")) for block in doc.blocks)
    # A recognized drawing yields an ``{{IMG:...}}`` placeholder; its absence
    # means upstream dropped the alias figure and we must normalize it.
    return "{{IMG:" not in content


def mineru_ir_builder_needs_auxiliary_filter(builder_cls: Any | None = None) -> bool:
    """Return whether the LightRAG MinerU builder indexes page furniture."""
    builder_cls = _resolve_builder_cls(builder_cls)
    if builder_cls is None or _already_patched(builder_cls):
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


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def apply_mineru_content_list_hygiene() -> bool:
    """Patch LightRAG's MinerU ``_normalize_content_list`` with the transforms
    current upstream still needs. Idempotent; returns True when it installs.
    """
    builder_cls = _resolve_builder_cls(None)
    if builder_cls is None:  # pragma: no cover - MinerU parser always present
        return False

    original = builder_cls._normalize_content_list
    if getattr(original, _PATCH_ATTR, False):
        return False

    transforms: list[Callable[[list[Any]], list[Any]]] = []
    if mineru_ir_builder_needs_drawing_alias_normalization(builder_cls):
        transforms.append(normalize_mineru_drawing_aliases)
    if mineru_ir_builder_needs_auxiliary_filter(builder_cls):
        transforms.append(filter_mineru_auxiliary_blocks)
    if not transforms:
        logger.debug("LightRAG MinerU IR builder already normalizes content_list")
        return False

    @wraps(original)
    def patched_normalize_content_list(
        self: Any,
        content_list: list[Any],
        raw_dir: Path,
        *,
        document_name: str,
    ) -> Any:
        for transform in transforms:
            content_list = transform(content_list)
        return original(self, content_list, raw_dir, document_name=document_name)

    setattr(patched_normalize_content_list, _PATCH_ATTR, True)
    patched_normalize_content_list._dlightrag_original = original  # type: ignore[attr-defined]
    builder_cls._normalize_content_list = patched_normalize_content_list
    logger.info(
        "Applied LightRAG MinerU content_list hygiene patch: %s",
        ", ".join(transform.__name__ for transform in transforms),
    )
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_builder_cls(builder_cls: Any | None) -> Any | None:
    if builder_cls is not None:
        return builder_cls
    try:
        from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder
    except Exception:  # pragma: no cover - defensive import guard
        return None
    return MinerUIRBuilder


def _already_patched(builder_cls: Any) -> bool:
    return bool(getattr(builder_cls._normalize_content_list, _PATCH_ATTR, False))


def _is_drawing_alias(item: Any) -> bool:
    return isinstance(item, dict) and _block_type(item) in MINERU_DRAWING_ALIAS_TYPES


def _alias_drawing_item(item: dict[str, Any]) -> dict[str, Any]:
    alias_type = _block_type(item)
    aliased = dict(item)
    aliased["type"] = "image"
    for source, target in (
        (f"{alias_type}_caption", "image_caption"),
        (f"{alias_type}_footnote", "image_footnote"),
    ):
        value = aliased.pop(source, None)
        if value is not None and target not in aliased:
            aliased[target] = value
    return aliased


def _is_mineru_auxiliary_block(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    block_type = _block_type(item)
    if block_type in MINERU_AUXILIARY_BLOCK_TYPES:
        return True
    if _auxiliary_block_policy() == "extended":
        return block_type in MINERU_EXTENDED_AUXILIARY_BLOCK_TYPES
    return False


def _auxiliary_block_policy() -> str:
    policy = os.getenv("DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY", "conservative")
    return policy.strip().lower()


def _block_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("label") or "").strip().lower()


__all__ = [
    "MINERU_AUXILIARY_BLOCK_TYPES",
    "MINERU_DRAWING_ALIAS_TYPES",
    "MINERU_EXTENDED_AUXILIARY_BLOCK_TYPES",
    "apply_mineru_content_list_hygiene",
    "filter_mineru_auxiliary_blocks",
    "mineru_ir_builder_needs_auxiliary_filter",
    "mineru_ir_builder_needs_drawing_alias_normalization",
    "normalize_mineru_drawing_aliases",
]
