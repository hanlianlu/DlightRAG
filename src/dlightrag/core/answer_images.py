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
    min_px: int
    quality: int
    min_quality: int
    count: int = 0
    used_bytes: int = 0

    def add_base64(self, value: str, *, label: str) -> dict[str, Any] | None:
        """Add a raw base64/data URI image if it fits the remaining budget."""
        bounded = self._bound_base64(value, label=label)
        if bounded is None:
            return None
        uri, _ = bounded
        return {"type": "image_url", "image_url": {"url": uri}}

    def _bound_base64(self, value: str, *, label: str) -> tuple[str, int] | None:
        """Bound a raw base64/data URI image and record consumed bytes."""
        if self.count >= self.max_images or self.used_bytes >= self.max_total_bytes:
            return None
        remaining = self.max_total_bytes - self.used_bytes
        max_bytes = min(self.max_bytes_per_image, remaining)
        bounded = bounded_image_data_uri(
            value,
            max_bytes=max_bytes,
            max_px=self.max_px,
            min_px=self.min_px,
            quality=self.quality,
            min_quality=self.min_quality,
        )
        if bounded is None:
            logger.info("Skipping answer image %s: cannot fit payload budget", label)
            return None
        uri, byte_count = bounded
        self.count += 1
        self.used_bytes += byte_count
        return uri, byte_count

    def add_user_image(self, value: str | dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add a user image. URLs pass through while base64 is bounded."""
        if self.count >= self.max_images:
            return None
        if isinstance(value, dict):
            return self._add_image_url_block(value, label=label)
        text = value.strip()
        if text.startswith(("http://", "https://")):
            self.count += 1
            return {"type": "image_url", "image_url": {"url": text}}
        return self.add_base64(text, label=label)

    def _add_image_url_block(self, value: dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add an OpenAI-style image block without letting data URIs bypass budget."""
        block = image_url_block(value)
        if block is None:
            return None
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = image_url.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        text = url.strip()
        if text.startswith(("http://", "https://")):
            self.count += 1
            return block
        bounded = self._bound_base64(text, label=label)
        if bounded is None:
            return None
        bounded_url, _ = bounded
        bounded_block = dict(block)
        bounded_image_url = dict(image_url)
        bounded_image_url["url"] = bounded_url
        bounded_block["image_url"] = bounded_image_url
        return bounded_block


__all__ = ["AnswerImageBudget"]
