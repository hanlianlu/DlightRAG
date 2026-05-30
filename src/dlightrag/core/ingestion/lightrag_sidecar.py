# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Read canonical LightRAG parser sidecars into typed references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlightrag.core.sidecar_provenance import (
    explicit_item_page_index,
    load_block_provenance_index,
)


@dataclass(frozen=True)
class LightRAGSidecarRef:
    sidecar_type: str
    sidecar_id: str
    asset_path: Path | None = None
    page_index: int | None = None
    bbox: dict[str, Any] | None = None
    block_id: str | None = None
    payload: dict[str, Any] | None = None


def collect_sidecar_refs(artifact_dir: Path) -> list[LightRAGSidecarRef]:
    """Collect drawing/table/equation refs from LightRAG sidecar JSON files."""
    block_index = load_block_provenance_index(artifact_dir)
    refs: list[LightRAGSidecarRef] = []
    for sidecar_type, pattern, item_key in (
        ("drawing", "*.drawings.json", "drawings"),
        ("table", "*.tables.json", "tables"),
        ("equation", "*.equations.json", "equations"),
    ):
        for path in sorted(artifact_dir.glob(pattern)):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                raw_items = data.get(item_key) or data.get("items") or []
            else:
                raw_items = data
            items = raw_items.values() if isinstance(raw_items, dict) else raw_items
            for index, item in enumerate(items):
                item_id = item.get("id")
                if not isinstance(item_id, str):
                    item_id = None
                item_uid = item.get("uid")
                if not isinstance(item_uid, str):
                    item_uid = None
                raw_id = item_id or item_uid
                if raw_id:
                    # Qualify with source file stem so page-local IDs
                    # (e.g. "im-0" in page_1.drawings.json and page_2.drawings.json)
                    # produce distinct chunk IDs.
                    sidecar_id = f"{path.stem}:{raw_id}"
                else:
                    sidecar_id = str(
                        item.get("id") or item.get("uid") or f"{path.stem}:{sidecar_type}-{index}"
                    )
                raw_asset = item.get("path") or item.get("asset_path") or item.get("image_path")
                block_id = item.get("blockid")
                block_provenance = block_index.get(block_id) if isinstance(block_id, str) else None
                explicit_page_index = explicit_item_page_index(item)
                refs.append(
                    LightRAGSidecarRef(
                        sidecar_type=sidecar_type,
                        sidecar_id=sidecar_id,
                        asset_path=(artifact_dir / raw_asset).resolve() if raw_asset else None,
                        page_index=explicit_page_index
                        if explicit_page_index is not None
                        else (
                            block_provenance.page_index if block_provenance is not None else None
                        ),
                        bbox=item.get("bbox")
                        or (block_provenance.bbox if block_provenance is not None else None),
                        block_id=block_id if isinstance(block_id, str) else None,
                        payload=item,
                    )
                )
    return refs
