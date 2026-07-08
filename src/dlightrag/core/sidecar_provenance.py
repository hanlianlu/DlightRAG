# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG sidecar provenance helpers shared by ingestion and retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class BlockProvenance:
    """Page-level provenance for one LightRAG sidecar block."""

    page_index: int | None = None
    bbox: dict[str, Any] | None = None


def sidecar_dir_from_location(location: str | None) -> Path | None:
    """Resolve a LightRAG full_doc sidecar_location into a local parsed dir."""
    if not location or not str(location).strip():
        return None
    raw = str(location).strip()
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(raw)


def resolve_sidecar_asset_path(artifact_dir: Path, raw_path: str) -> Path | None:
    """Resolve a sidecar asset path only when it stays inside artifact_dir."""
    raw = raw_path.strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return None
    root = artifact_dir.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def load_block_provenance_index(artifact_dir: Path) -> dict[str, BlockProvenance]:
    """Load ``blockid -> BlockProvenance`` from LightRAG ``*.blocks.jsonl`` files."""
    index: dict[str, BlockProvenance] = {}
    for blocks_path in sorted(artifact_dir.glob("*.blocks.jsonl")):
        for line in blocks_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            block_id = row.get("blockid")
            if not isinstance(block_id, str) or not block_id:
                continue
            provenance = _provenance_from_positions(row.get("positions"))
            if provenance.page_index is not None:
                index[block_id] = provenance
    return index


def block_ids_from_sidecar(sidecar: dict[str, Any]) -> list[str]:
    """Return source block ids from a LightRAG chunk sidecar payload."""
    block_ids: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value and value not in block_ids:
            block_ids.append(value)

    _add(sidecar.get("block_id"))
    if sidecar.get("type") == "block":
        _add(sidecar.get("id"))

    refs = sidecar.get("refs")
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and ref.get("type") == "block":
                _add(ref.get("id"))
    return block_ids


# Multimodal chunks (table / drawing / equation) carry a ``{type, id, refs}``
# sidecar that references their own modality item id (``tb-``/``im-``/``eq-``)
# rather than a source block. Each modality file records the originating
# ``blockid``, so page/bbox provenance is recoverable by an id lookup.
_MULTIMODAL_ITEM_FILES: dict[str, tuple[str, str]] = {
    "table": ("*.tables.json", "tables"),
    "drawing": ("*.drawings.json", "drawings"),
    "equation": ("*.equations.json", "equations"),
}


def is_multimodal_sidecar(sidecar: dict[str, Any]) -> bool:
    """Return whether a chunk sidecar is a table / drawing / equation item."""
    return sidecar.get("type") in _MULTIMODAL_ITEM_FILES


def block_ids_from_multimodal_item(artifact_dir: Path, sidecar: dict[str, Any]) -> list[str]:
    """Resolve a multimodal chunk's source block id from its sidecar item file.

    The sidecar ``id`` keys the matching ``*.tables.json`` / ``*.drawings.json``
    / ``*.equations.json`` item, which records the ``blockid`` it was lifted
    from. Returns an empty list when the sidecar is not multimodal, the item is
    absent, or no modality file is readable.
    """
    kind = sidecar.get("type")
    spec = _MULTIMODAL_ITEM_FILES.get(kind) if isinstance(kind, str) else None
    item_id = sidecar.get("id")
    if spec is None or not isinstance(item_id, str) or not item_id:
        return []

    glob_pattern, root_key = spec
    for path in sorted(artifact_dir.glob(glob_pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            continue
        items = payload.get(root_key) if isinstance(payload, dict) else None
        item = items.get(item_id) if isinstance(items, dict) else None
        block_id = item.get("blockid") if isinstance(item, dict) else None
        if isinstance(block_id, str) and block_id:
            return [block_id]
    return []


def first_provenance_for_blocks(
    block_ids: list[str],
    index: dict[str, BlockProvenance],
) -> BlockProvenance | None:
    """Return the first available provenance for ordered block refs."""
    for block_id in block_ids:
        provenance = index.get(block_id)
        if provenance and (provenance.page_index is not None or provenance.bbox is not None):
            return provenance
    return None


def explicit_item_page_index(item: dict[str, Any]) -> int | None:
    """Read explicit page fields from a sidecar item into zero-based page_index."""
    for key in ("page_index", "page_idx"):
        page_index = coerce_non_negative_int(item.get(key))
        if page_index is not None:
            return page_index

    for key in ("page", "page_number"):
        page_number = coerce_non_negative_int(item.get(key))
        if page_number is not None:
            return max(page_number - 1, 0)
    return None


def explicit_item_bbox(item: dict[str, Any]) -> dict[str, Any] | None:
    """Read an explicit bbox-like payload from a LightRAG sidecar item."""
    raw = item.get("bbox") or item.get("bounding_box")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        page_index = explicit_item_page_index(item)
        payload: dict[str, Any] = {"range": list(raw)}
        if page_index is not None:
            payload["page_index"] = page_index
        return payload
    return None


def coerce_non_negative_int(value: Any) -> int | None:
    """Coerce parser numeric values while rejecting bools and negatives."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _provenance_from_positions(raw_positions: Any) -> BlockProvenance:
    if not isinstance(raw_positions, list):
        return BlockProvenance()
    for position in raw_positions:
        if not isinstance(position, dict):
            continue
        page_index = coerce_non_negative_int(position.get("anchor"))
        if page_index is None:
            continue
        bbox = None
        raw_range = position.get("range")
        if isinstance(raw_range, list):
            bbox = {"page_index": page_index, "range": list(raw_range)}
            origin = position.get("origin")
            if isinstance(origin, str) and origin:
                bbox["origin"] = origin
        return BlockProvenance(page_index=page_index, bbox=bbox)
    return BlockProvenance()
