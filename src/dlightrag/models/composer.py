# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned Composer analysis model resources."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ComposerImageTransportSettings:
    """Canonical image transport settings used by Composer analysis."""

    max_images: int
    image_max_bytes: int
    image_max_total_bytes: int
    image_max_px: int
    image_min_px: int
    image_quality: int
    image_min_quality: int

    @classmethod
    def from_config(cls, config: DlightragConfig) -> ComposerImageTransportSettings:
        answer = config.answer
        return cls(
            max_images=int(answer.max_images),
            image_max_bytes=int(answer.image_max_bytes),
            image_max_total_bytes=int(answer.image_max_total_bytes),
            image_max_px=int(answer.image_max_px),
            image_min_px=int(answer.image_min_px),
            image_quality=int(answer.image_quality),
            image_min_quality=int(answer.image_min_quality),
        )

    def fingerprint_payload(self) -> dict[str, int]:
        return {
            "max_images": self.max_images,
            "image_max_bytes": self.image_max_bytes,
            "image_max_total_bytes": self.image_max_total_bytes,
            "image_max_px": self.image_max_px,
            "image_min_px": self.image_min_px,
            "image_quality": self.image_quality,
            "image_min_quality": self.image_min_quality,
        }


@dataclass(slots=True)
class ComposerModelBundle:
    """Own the cache-neutral VLM and EXTRACT callables used by Composer."""

    vlm_func: Callable[..., Awaitable[Any]]
    extract_func: Callable[..., Awaitable[Any]]
    vlm_identity: dict[str, Any]
    extract_identity: dict[str, Any]
    _closers: tuple[Callable[[], Awaitable[None]], ...] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        config: DlightragConfig,
        *,
        bind: Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]],
    ) -> ComposerModelBundle:
        """Create both role adapters once and bind them to manager concurrency."""
        from dlightrag.models.llm import create_composer_analysis_adapter

        vlm_func, vlm_identity, close_vlm = create_composer_analysis_adapter(config, role="vlm")
        extract_func, extract_identity, close_extract = create_composer_analysis_adapter(
            config, role="extract"
        )
        return cls(
            vlm_func=bind(vlm_func),
            extract_func=bind(extract_func),
            vlm_identity=vlm_identity,
            extract_identity=extract_identity,
            _closers=(close_vlm, close_extract),
        )

    async def aclose(self) -> None:
        """Close each captured provider exactly once."""
        if self._closed:
            return
        self._closed = True
        for close in self._closers:
            try:
                await close()
            except Exception:
                logger.warning("Failed to close Composer model provider", exc_info=True)


__all__ = ["ComposerImageTransportSettings", "ComposerModelBundle"]
