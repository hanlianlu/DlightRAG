# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Client-side conveniences for building answer and retrieve requests."""

from collections.abc import Sequence
from typing import Any


def query_image_blocks_from_urls(values: Sequence[str]) -> list[dict[str, Any]]:
    """Wrap CLI-supplied image URLs/data URIs as modern image_url content blocks."""
    return [{"type": "image_url", "image_url": {"url": value}} for value in values]


__all__ = [
    "query_image_blocks_from_urls",
]
