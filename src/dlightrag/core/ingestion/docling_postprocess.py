# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Post-process Docling JSON to restore heading markers and per-block page_idx.

RAGAnything's default parser discards heading labels and estimates page_idx
via ``cnt // 10``.  This module reads the raw Docling JSON export and builds
a proper ``content_list`` with markdown-style headings and accurate page_idx
sourced from ``prov[0].page_no``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def rebuild_text_items_from_docling_json(json_path: Path) -> list[dict[str, Any]] | None:
    """Read a Docling JSON file and produce annotated text items.

    Each returned dict represents a single text block with:
    - ``text``: the block text (with ``#`` heading prefix for section_header/title)
    - ``type``: ``"text"`` or ``"equation"`` (for formula blocks)
    - ``page_idx``: page number from ``prov[0].page_no`` (default 0)

    Returns ``None`` on any failure (missing file, bad JSON, missing ``body``
    key) so the caller can fall back to the original content_list.
    """
    # --- Read and parse -------------------------------------------------------
    try:
        raw = json_path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        logger.debug("Docling JSON not found: %s", json_path)
        return None

    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Invalid JSON in %s", json_path)
        return None

    if not isinstance(doc, dict) or "body" not in doc:
        logger.debug("Docling JSON missing 'body' key: %s", json_path)
        return None

    # --- Walk body children ---------------------------------------------------
    texts: list[dict[str, Any]] = doc.get("texts", [])
    groups: list[dict[str, Any]] = doc.get("groups", [])
    body_children: list[dict[str, Any]] = doc.get("body", {}).get("children", [])

    items: list[dict[str, Any]] = []
    _resolve_children(body_children, texts, groups, items)
    return items


def _resolve_children(
    children: list[dict[str, Any]],
    texts: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    out: list[dict[str, Any]],
    *,
    _seen: set[str] | None = None,
) -> None:
    """Recursively resolve ``$ref`` pointers and append items to *out*.

    Handles ``texts/<idx>`` and ``groups/<idx>`` references.  Groups are
    expanded recursively so their children appear in document order.
    A *_seen* set prevents infinite recursion on circular refs.
    """
    if _seen is None:
        _seen = set()

    for child in children:
        ref = child.get("$ref", "")
        if not ref:
            continue

        # Guard against circular references
        if ref in _seen:
            continue
        _seen.add(ref)

        parts = ref.split("/", 1)
        if len(parts) != 2:
            continue
        collection, idx_str = parts

        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue

        if collection == "texts":
            if 0 <= idx < len(texts):
                out.append(_text_block_to_item(texts[idx]))
        elif collection == "groups":
            if 0 <= idx < len(groups):
                group = groups[idx]
                group_children = group.get("children", [])
                _resolve_children(group_children, texts, groups, out, _seen=_seen)


def _text_block_to_item(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a single Docling text block to a content_list item."""
    orig = block.get("orig", "")
    label = block.get("label", "")

    # Page index from provenance
    prov = block.get("prov")
    if prov and isinstance(prov, list) and len(prov) > 0:
        page_idx = prov[0].get("page_no", 0)
    else:
        page_idx = 0

    # Label-dependent transformation
    if label == "formula":
        return {"type": "equation", "text": orig, "page_idx": page_idx}

    if label == "section_header":
        level = block.get("level", 1)
        text = "#" * level + " " + orig
    elif label == "title":
        text = "# " + orig
    else:
        text = orig

    return {"type": "text", "text": text, "page_idx": page_idx}


__all__ = [
    "rebuild_text_items_from_docling_json",
]
