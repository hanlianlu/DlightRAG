# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared image payload budgeting for model requests."""

import logging
from dataclasses import dataclass

from dlightrag.utils.images import bounded_image_data_uri

logger = logging.getLogger(__name__)


@dataclass
class ImagePayloadBudget:
    """Bound base64 image payloads before sending them to model APIs."""

    max_total_bytes: int
    max_bytes_per_image: int
    max_px: int
    min_px: int
    quality: int
    min_quality: int
    max_images: int | None = None
    count: int = 0
    used_bytes: int = 0

    def add_base64(self, value: str, *, label: str) -> tuple[str, int] | None:
        """Add an image if it can fit the remaining byte/image budget."""
        if self.max_images is not None and self.count >= self.max_images:
            return None
        if self.used_bytes >= self.max_total_bytes:
            return None

        remaining = max(0, self.max_total_bytes - self.used_bytes)
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
            logger.info("Skipping image %s: cannot fit model payload budget", label)
            return None

        uri, byte_count = bounded
        if self.used_bytes + byte_count > self.max_total_bytes:
            logger.info("Skipping image %s: would exceed model payload budget", label)
            return None

        self.count += 1
        self.used_bytes += byte_count
        return uri, byte_count


__all__ = ["ImagePayloadBudget"]
