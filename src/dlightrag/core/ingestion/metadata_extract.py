# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""System metadata extraction from file paths.

Pure Path-based extraction — no I/O, no storage dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_system_metadata(
    file_path: str | Path,
    rag_mode: str = "caption",
    page_count: int | None = None,
) -> dict[str, Any]:
    """Extract system metadata from a file path.

    Returns a dict suitable for PGMetadataIndex.upsert().
    """
    p = Path(file_path)
    meta: dict[str, Any] = {
        "filename": p.name,
        "filename_stem": p.stem,
        "file_path": str(p),
        "file_extension": p.suffix.lower().lstrip(".") if p.suffix else None,
        "original_format": p.suffix.lower().lstrip(".") if p.suffix else None,
        "rag_mode": rag_mode,
    }
    if page_count is not None:
        meta["page_count"] = page_count
    return meta
