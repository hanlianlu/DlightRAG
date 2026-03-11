"""Utility functions for citation processing."""

from __future__ import annotations

import re
from typing import Any

_IMAGE_METADATA_RE = re.compile(
    r"^\s*(Image Path|Caption|Figure|Alt Text|Image Description)\s*:", re.IGNORECASE
)
_REDUNDANT_PLACEHOLDER_RE = re.compile(
    r"^\s*(Caption|Description|Alt Text)\s*:\s*(None|N/A|n/a|\s*)$", re.IGNORECASE
)


def split_source_ids(source_id: Any) -> list[str]:
    """Split comma-separated source_id string into list of chunk IDs."""
    if source_id is None:
        return []
    return [part.strip() for part in str(source_id).split(",") if part.strip()]


def filter_content_for_display(content: str, max_chars: int | None = None) -> str:
    """Filter metadata lines while preserving structure for display."""
    lines = content.split("\n")
    filtered = []
    for line in lines:
        if _IMAGE_METADATA_RE.match(line):
            continue
        if _REDUNDANT_PLACEHOLDER_RE.match(line):
            continue
        filtered.append(line)

    result = "\n".join(filtered).strip()

    if max_chars and len(result) > max_chars:
        result = result[:max_chars] + "..."

    return result
