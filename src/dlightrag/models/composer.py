# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned Composer analysis model resources."""

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True, slots=True)
class ComposerAnalysisSettings:
    """Effective LightRAG analysis limits plus Composer image transport."""

    enabled: bool
    vlm_max_image_bytes: int
    vlm_min_image_pixel: int
    surrounding_leading_max_tokens: int
    surrounding_trailing_max_tokens: int
    max_extract_input_tokens: int
    answer_image_transport: ComposerImageTransportSettings

    @classmethod
    def resolve(cls, config: DlightragConfig) -> ComposerAnalysisSettings:
        """Resolve limits with the same env defaults and clamps as LightRAG."""
        from lightrag import multimodal_context
        from lightrag.constants import (
            DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
            DEFAULT_MM_IMAGE_MIN_PIXEL,
        )
        from lightrag.utils import get_env_value

        leading_tokens, trailing_tokens = multimodal_context._resolve_surrounding_budget(  # pyright: ignore[reportPrivateUsage]
            None,
            None,
        )
        max_extract_input_tokens = get_env_value(
            "MAX_EXTRACT_INPUT_TOKENS",
            DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
            int,
        )
        return cls(
            enabled=bool(config.parser_sidecars.vlm.enabled),
            vlm_max_image_bytes=max(
                256 * 1024,
                int(os.getenv("VLM_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))),
            ),
            vlm_min_image_pixel=max(
                1,
                int(
                    os.getenv(
                        "VLM_MIN_IMAGE_PIXEL",
                        str(DEFAULT_MM_IMAGE_MIN_PIXEL),
                    )
                ),
            ),
            surrounding_leading_max_tokens=leading_tokens,
            surrounding_trailing_max_tokens=trailing_tokens,
            max_extract_input_tokens=max_extract_input_tokens,
            answer_image_transport=ComposerImageTransportSettings.from_config(config),
        )

    @property
    def vlm_image_transport(self) -> ComposerImageTransportSettings:
        """Apply upstream VLM gates to the existing answer-image transport."""
        return replace(
            self.answer_image_transport,
            image_max_bytes=min(
                self.answer_image_transport.image_max_bytes,
                self.vlm_max_image_bytes,
            ),
            image_min_px=max(
                self.answer_image_transport.image_min_px,
                self.vlm_min_image_pixel,
            ),
        )


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


__all__ = [
    "ComposerAnalysisSettings",
    "ComposerImageTransportSettings",
    "ComposerModelBundle",
]
