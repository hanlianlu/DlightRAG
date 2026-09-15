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

1b. Media with an image but no payload — MinerU also misclassifies whole image
   pages as ``table`` (and emits empty ``equation`` items). Upstream decides
   "does this table have content?" from whether the HTML *string* is non-blank,
   so ``<table><tr><td></td><td></td></tr></table>`` counts as content and lands
   in the sidecar as an empty table chunk while the source image in ``img_path``
   is discarded; an empty equation is dropped with no log at all. We rewrite
   such items to ``image`` so the pixels reach the drawing / VLM path instead of
   disappearing behind an empty table.

2. Auxiliary furniture — MinerU emits running headers and footers in
   ``content_list``; indexing those as body text pollutes chunks, KG extraction,
   BM25, and citations. We drop discarded blocks, headers, and footers;
   LightRAG already drops printed page numbers. Footnotes are deliberately
   kept: in papers they carry derivations and definitions, not furniture.

Keep this module small and delete transforms as upstream covers them.
"""

import html
import logging
import re
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]*>")

# MinerU figure types that carry an image but that LightRAG's IR builder does
# not map to IRDrawing (it maps only ``image`` / ``picture`` / ``drawing``).
# They share the ``image`` shape — an ``img_path`` plus ``<type>_caption`` /
# ``<type>_footnote`` — so aliasing them to ``image`` is loss-free. Extend this
# set if MinerU introduces further figure-like types (e.g. ``figure``).
MINERU_DRAWING_ALIAS_TYPES = frozenset({"chart"})

# Content-bearing types that can carry a rendered ``img_path``. When such an
# item's payload is empty (no visible cell text, no LaTeX), the image is the
# only surviving content and must reach the drawing path. ``table`` is the
# evidence-backed case (a full-page line-drawing page classified as a table);
# ``equation`` / ``formula`` are the same class — upstream drops them silently
# when the text is missing.
MINERU_PAYLOADLESS_MEDIA_TYPES = frozenset({"table", "equation", "formula"})

# Payload fields per type whose visible text decides "does this item carry
# content?". Table bodies are HTML, so their tags are stripped before testing.
MINERU_MEDIA_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "table": ("table_body", "rows", "grid"),
    "equation": ("latex", "text", "content"),
    "formula": ("latex", "text", "content"),
}
MINERU_AUXILIARY_BLOCK_TYPES = frozenset(
    {
        "discarded",
        "discarded_block",
        "discarded_blocks",
        "header",
        "footer",
        "page_header",
        "page_footer",
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


def normalize_mineru_payloadless_media(content_list: list[Any]) -> list[Any]:
    """Route media items that have an image but no payload to the drawing path.

    A ``table`` whose every cell is empty is not a table — it is a picture the
    upstream table model claimed (a whole-page line drawing, a chart). Upstream
    keeps it because the HTML *string* is non-blank, which produces an empty
    table chunk and discards ``img_path``. Re-tagging the item to ``image``
    keeps the pixels: the drawing builder materializes ``img_path`` as an asset
    and the VLM sidecar describes it. Items without an ``img_path`` are left
    alone — there would be nothing to draw.
    """
    return [
        _alias_to_image(item, _block_type(item)) if _is_payloadless_media(item) else item
        for item in content_list
    ]


def filter_mineru_auxiliary_blocks(content_list: list[Any]) -> list[Any]:
    """Drop MinerU page furniture while preserving semantic/multimodal items."""
    return [item for item in content_list if not _is_mineru_auxiliary_block(item)]


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def apply_mineru_content_list_hygiene() -> bool:
    """Patch LightRAG's MinerU ``_normalize_content_list`` with the transforms
    current upstream still needs. Idempotent; returns True when it installs.
    """
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    original = MinerUIRBuilder._normalize_content_list
    if getattr(original, _PATCH_ATTR, False):
        return False

    transforms: tuple[Callable[[list[Any]], list[Any]], ...] = (
        normalize_mineru_drawing_aliases,
        normalize_mineru_payloadless_media,
        filter_mineru_auxiliary_blocks,
    )

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
    MinerUIRBuilder._normalize_content_list = patched_normalize_content_list
    logger.info(
        "Applied LightRAG MinerU content_list hygiene patch: %s",
        ", ".join(transform.__name__ for transform in transforms),
    )
    return True


def _is_drawing_alias(item: Any) -> bool:
    return isinstance(item, dict) and _block_type(item) in MINERU_DRAWING_ALIAS_TYPES


def _alias_drawing_item(item: dict[str, Any]) -> dict[str, Any]:
    return _alias_to_image(item, _block_type(item))


def _alias_to_image(item: dict[str, Any], source_type: str) -> dict[str, Any]:
    """Re-tag a media item to ``image`` and rename its caption/footnote fields."""
    aliased = dict(item)
    aliased["type"] = "image"
    for source, target in (
        (f"{source_type}_caption", "image_caption"),
        (f"{source_type}_footnote", "image_footnote"),
    ):
        value = aliased.pop(source, None)
        if value is not None and target not in aliased:
            aliased[target] = value
    return aliased


def _is_payloadless_media(item: Any) -> bool:
    """True when ``item`` carries an image but no visible payload of its type."""
    if not isinstance(item, dict):
        return False
    block_type = _block_type(item)
    if block_type not in MINERU_PAYLOADLESS_MEDIA_TYPES:
        return False
    if not str(item.get("img_path") or "").strip():
        return False
    return not _has_visible_payload(item, block_type)


def _has_visible_payload(item: dict[str, Any], block_type: str) -> bool:
    """Whether the item's own content fields carry any non-whitespace text."""
    payload_fields = MINERU_MEDIA_PAYLOAD_FIELDS.get(block_type, ())
    for field_name in payload_fields:
        for text in _iter_payload_text(item.get(field_name)):
            visible = _strip_markup(text) if field_name != "latex" else text
            if visible.strip():
                return True
    return False


def _iter_payload_text(value: Any) -> list[str]:
    """Flatten a payload field (string / grid rows / nested list) into strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for nested in value.values():
            texts.extend(_iter_payload_text(nested))
        return texts
    if isinstance(value, (list, tuple)):
        texts = []
        for nested in value:
            texts.extend(_iter_payload_text(nested))
        return texts
    return []


def _strip_markup(text: str) -> str:
    """Drop HTML tags and decode entities so ``<td></td>`` counts as empty."""
    without_tags = _TAG_RE.sub(" ", text)
    return html.unescape(without_tags)


def _is_mineru_auxiliary_block(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return _block_type(item) in MINERU_AUXILIARY_BLOCK_TYPES


def _block_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("label") or "").strip().lower()


__all__ = [
    "MINERU_AUXILIARY_BLOCK_TYPES",
    "MINERU_DRAWING_ALIAS_TYPES",
    "MINERU_PAYLOADLESS_MEDIA_TYPES",
    "apply_mineru_content_list_hygiene",
    "filter_mineru_auxiliary_blocks",
    "normalize_mineru_drawing_aliases",
    "normalize_mineru_payloadless_media",
]
