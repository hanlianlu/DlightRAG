# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer LLM image budgeting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dlightrag.utils.images import bounded_image_data_uri, image_url_block

logger = logging.getLogger(__name__)


@dataclass
class AnswerImageBudget:
    """Bound image payloads sent to an answer model."""

    max_images: int
    max_total_bytes: int
    max_bytes_per_image: int
    max_px: int
    quality: int
    count: int = 0
    used_bytes: int = 0

    def add_base64(self, value: str, *, label: str) -> dict[str, Any] | None:
        """Add a raw base64/data URI image if it fits the remaining budget."""
        if self.count >= self.max_images or self.used_bytes >= self.max_total_bytes:
            return None
        remaining = self.max_total_bytes - self.used_bytes
        max_bytes = min(self.max_bytes_per_image, remaining)
        bounded = bounded_image_data_uri(
            value,
            max_bytes=max_bytes,
            max_px=self.max_px,
            quality=self.quality,
        )
        if bounded is None:
            logger.info("Skipping answer image %s: cannot fit payload budget", label)
            return None
        uri, byte_count = bounded
        self.count += 1
        self.used_bytes += byte_count
        return {"type": "image_url", "image_url": {"url": uri}}

    def add_user_image(self, value: str | dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add a user image. URLs pass through while base64 is bounded."""
        if self.count >= self.max_images:
            return None
        if isinstance(value, dict):
            block = image_url_block(value)
            if block is None:
                return None
            self.count += 1
            return block
        text = value.strip()
        if text.startswith(("http://", "https://")):
            self.count += 1
            return {"type": "image_url", "image_url": {"url": text}}
        return self.add_base64(text, label=label)


__all__ = ["AnswerImageBudget"]
