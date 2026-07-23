# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned Composer analysis model resources."""

import asyncio
import hashlib
import logging
import os
import posixpath
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)


class ComposerImagePayloadError(RuntimeError):
    """Composer analysis could not admit any supplied image payload."""


def normalized_endpoint_fingerprint(value: Any) -> str | None:
    """Hash a canonical endpoint without persisting recoverable routing data."""
    if not value:
        return None
    raw = str(value)
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return None
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            return None
        port = parsed.port
        if port == {"http": 80, "https": 443}.get(scheme):
            port = None
        authority = f"[{hostname}]" if ":" in hostname else hostname
        if port is not None:
            authority = f"{authority}:{port}"
        path = posixpath.normpath(parsed.path or "/")
        if not path.startswith("/"):
            path = f"/{path}"
        canonical = urlunsplit((scheme, authority, path, "", ""))
    except ValueError:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComposerImageTransportSettings:
    """Canonical image transport settings used by Composer analysis."""

    max_images: int
    image_max_bytes: int
    image_max_total_bytes: int
    image_max_px: int
    image_max_pixels: int
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
            image_max_pixels=int(answer.image_max_pixels),
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
            "image_max_pixels": self.image_max_pixels,
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
            enabled=get_env_value(
                "VLM_PROCESS_ENABLE",
                bool(config.parser_sidecars.vlm.enabled),
                bool,
            ),
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


class ComposerAnalysisAdapterFactory(Protocol):
    """Build one cache-neutral LightRAG analysis adapter for a Composer role.

    Injected into :meth:`ComposerModelBundle.acreate` so this module never
    imports the LLM factory layer, keeping it a dependency leaf.
    """

    def __call__(
        self,
        config: DlightragConfig,
        *,
        role: Literal["vlm", "extract"],
    ) -> tuple[
        Callable[..., Awaitable[Any]],
        dict[str, Any],
        Callable[[], Awaitable[None]],
    ]: ...


@dataclass(slots=True)
class ComposerModelBundle:
    """Own the cache-neutral VLM and EXTRACT callables used by Composer."""

    vlm_func: Callable[..., Awaitable[Any]]
    extract_func: Callable[..., Awaitable[Any]]
    vlm_identity: dict[str, Any]
    extract_identity: dict[str, Any]
    _closers: tuple[Callable[[], Awaitable[None]], ...] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _pending_closers: list[Callable[[], Awaitable[None]]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._pending_closers = list(self._closers)

    @property
    def is_closed(self) -> bool:
        """Return whether every owned provider has closed successfully."""
        return self._closed

    @classmethod
    async def acreate(
        cls,
        config: DlightragConfig,
        *,
        bind: Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]],
        adapter_factory: ComposerAnalysisAdapterFactory,
    ) -> ComposerModelBundle:
        """Create both role adapters once and bind them to manager concurrency.

        ``adapter_factory`` is injected by the composition root (normally
        :func:`dlightrag.models.llm.create_composer_analysis_adapter`) so this
        module stays a dependency leaf and never imports the LLM factory layer.
        """
        closers: list[Callable[[], Awaitable[None]]] = []
        try:
            vlm_func, vlm_identity, close_vlm = adapter_factory(config, role="vlm")
            closers.append(close_vlm)
            await asyncio.sleep(0)
            bound_vlm = bind(vlm_func)

            extract_func, extract_identity, close_extract = adapter_factory(config, role="extract")
            closers.append(close_extract)
            await asyncio.sleep(0)
            bound_extract = bind(extract_func)
        except BaseException:
            _clear_pending_cancellation()
            await _rollback_composer_closers(closers)
            raise

        return cls(
            vlm_func=bound_vlm,
            extract_func=bound_extract,
            vlm_identity=vlm_identity,
            extract_identity=extract_identity,
            _closers=tuple(closers),
        )

    async def aclose(self) -> None:
        """Close all providers, retaining unfinished closers for a later retry."""
        async with self._close_lock:
            if self._closed:
                return
            cancellation: asyncio.CancelledError | None = None
            pending: list[Callable[[], Awaitable[None]]] = []
            for close in self._pending_closers:
                try:
                    await close()
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                    pending.append(close)
                    _clear_pending_cancellation()
                except Exception:
                    pending.append(close)
                    logger.warning("Failed to close Composer model provider", exc_info=True)
            self._pending_closers = pending
            self._closed = not pending
            if cancellation is not None:
                raise cancellation


def _clear_pending_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


async def _rollback_composer_closers(
    closers: list[Callable[[], Awaitable[None]]],
) -> None:
    for close in closers:
        try:
            await close()
        except asyncio.CancelledError:
            _clear_pending_cancellation()
            logger.warning("Composer model rollback was cancelled", exc_info=True)
        except Exception:
            logger.warning("Failed to roll back Composer model provider", exc_info=True)


__all__ = [
    "ComposerAnalysisSettings",
    "ComposerImagePayloadError",
    "ComposerImageTransportSettings",
    "ComposerModelBundle",
    "normalized_endpoint_fingerprint",
]
