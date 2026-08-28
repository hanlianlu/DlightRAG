# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG sidecar provenance helpers shared by ingestion and retrieval."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BlockProvenance:
    """Page-level provenance for one LightRAG sidecar block."""

    page_number: int | None = None


@dataclass(frozen=True)
class _DrawingCandidate:
    path: Path
    page_number: int | None


@dataclass(slots=True)
class SidecarArtifactIndex:
    """One parsed view of the provenance sidecars in an artifact directory."""

    block_provenance: dict[str, BlockProvenance] = field(default_factory=dict)
    multimodal_block_ids: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    drawing_candidates: dict[str, list[_DrawingCandidate]] = field(default_factory=dict)

    @classmethod
    def load(cls, artifact_dir: Path) -> SidecarArtifactIndex:
        """Parse each supported sidecar file at most once."""
        root = artifact_dir.resolve()
        index = cls()
        index._load_blocks(root)
        index._load_multimodal_items(root)
        return index

    def block_ids_for_multimodal_item(self, sidecar: dict[str, Any]) -> list[str]:
        """Return the source block ids for one table, drawing, or equation item."""
        kind = sidecar.get("type")
        item_id = sidecar.get("id")
        if not isinstance(kind, str) or not isinstance(item_id, str) or not item_id:
            return []
        return list(self.multimodal_block_ids.get((kind, item_id), ()))

    def drawing_asset_path(
        self,
        drawing_id: str,
        *,
        page_number: int | None = None,
    ) -> Path | None:
        """Return the deterministic, artifact-contained drawing candidate."""
        candidates = self.drawing_candidates.get(drawing_id, ())
        if page_number is not None:
            for candidate in candidates:
                if candidate.page_number == page_number:
                    return candidate.path
        return candidates[0].path if candidates else None

    def _load_blocks(self, artifact_dir: Path) -> None:
        for blocks_path in sorted(artifact_dir.glob("*.blocks.jsonl")):
            try:
                with blocks_path.open(encoding="utf-8") as blocks_file:
                    for line in blocks_file:
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
                        if provenance.page_number is not None:
                            self.block_provenance[block_id] = provenance
            except OSError, UnicodeError:
                continue

    def _load_multimodal_items(self, artifact_dir: Path) -> None:
        files: dict[Path, tuple[str, str]] = {}
        for kind, (glob_pattern, root_key) in _MULTIMODAL_ITEM_FILES.items():
            for path in artifact_dir.glob(glob_pattern):
                files[path] = (kind, root_key)

        for path in sorted(files):
            kind, root_key = files[path]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError, OSError, UnicodeError:
                continue
            items = payload.get(root_key) if isinstance(payload, dict) else None
            if not isinstance(items, dict):
                continue
            for item_id, raw_item in items.items():
                if not isinstance(item_id, str) or not isinstance(raw_item, dict):
                    continue
                block_id = raw_item.get("blockid")
                if isinstance(block_id, str) and block_id:
                    self.multimodal_block_ids.setdefault((kind, item_id), (block_id,))
                if kind == "drawing":
                    self._add_drawing_candidate(artifact_dir, path, item_id, raw_item)

    def _add_drawing_candidate(
        self,
        artifact_dir: Path,
        drawings_path: Path,
        item_id: str,
        item: dict[str, Any],
    ) -> None:
        raw_path = _drawing_asset_path(item)
        if raw_path is None:
            return
        candidate = resolve_sidecar_asset_path(artifact_dir, raw_path)
        if candidate is None:
            return
        page_number = explicit_item_page_number(item)
        if page_number is None:
            page_number = _page_number_from_filename(drawings_path.stem)
        self.drawing_candidates.setdefault(item_id, []).append(
            _DrawingCandidate(path=candidate, page_number=page_number)
        )


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
    return SidecarArtifactIndex.load(artifact_dir).block_provenance


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
# ``blockid``, so page provenance is recoverable by an id lookup.
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

    The compatibility helper keeps the ingestion-facing behavior while retrieval
    caches the complete :class:`SidecarArtifactIndex` across hydration passes.
    """
    return SidecarArtifactIndex.load(artifact_dir).block_ids_for_multimodal_item(sidecar)


def first_provenance_for_blocks(
    block_ids: list[str],
    index: dict[str, BlockProvenance],
) -> BlockProvenance | None:
    """Return the first available provenance for ordered block refs."""
    for block_id in block_ids:
        provenance = index.get(block_id)
        if provenance and provenance.page_number is not None:
            return provenance
    return None


def explicit_item_page_number(item: dict[str, Any]) -> int | None:
    """Read an explicit 1-based page number from a sidecar item."""
    for key in ("page", "page_number"):
        page_number = coerce_positive_int(item.get(key))
        if page_number is not None:
            return page_number
    return None


def coerce_positive_int(value: Any) -> int | None:
    """Coerce page numbers while rejecting bools and non-positive values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 1 else None
    return None


def _provenance_from_positions(raw_positions: Any) -> BlockProvenance:
    if not isinstance(raw_positions, list):
        return BlockProvenance()
    for position in raw_positions:
        if not isinstance(position, dict):
            continue
        page_number = coerce_positive_int(position.get("anchor"))
        if page_number is None:
            continue
        return BlockProvenance(page_number=page_number)
    return BlockProvenance()


def _drawing_asset_path(item: dict[str, Any]) -> str | None:
    raw = item.get("path") or item.get("img_path") or item.get("image_path")
    return raw if isinstance(raw, str) and raw.strip() else None


def _page_number_from_filename(stem: str) -> int | None:
    match = re.search(r"(?:^|[_-])p(?:age)?[_-]?(\d+)", stem, re.IGNORECASE)
    if match is None:
        return None
    page_number = int(match.group(1))
    return page_number if page_number >= 1 else None
