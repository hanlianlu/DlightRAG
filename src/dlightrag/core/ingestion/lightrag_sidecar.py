# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Read canonical LightRAG parser sidecars into typed references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LightRAGSidecarRef:
    sidecar_type: str
    sidecar_id: str
    asset_path: Path | None = None
    page_number: int | None = None
    bbox: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None


def collect_sidecar_refs(artifact_dir: Path) -> list[LightRAGSidecarRef]:
    """Collect drawing/table/equation refs from LightRAG sidecar JSON files."""
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
                sidecar_id = str(item.get("id") or item.get("uid") or f"{sidecar_type}-{index}")
                raw_asset = item.get("path") or item.get("asset_path") or item.get("image_path")
                refs.append(
                    LightRAGSidecarRef(
                        sidecar_type=sidecar_type,
                        sidecar_id=sidecar_id,
                        asset_path=(artifact_dir / raw_asset).resolve() if raw_asset else None,
                        page_number=item.get("page") or item.get("page_number"),
                        bbox=item.get("bbox"),
                        payload=item,
                    )
                )
    return refs
