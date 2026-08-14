# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-level image capability probing for chat models."""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from dlightrag_ai.providers import get_provider
from dlightrag_ai.settings import ModelSettings
from dlightrag_ai.telemetry import NOOP_TELEMETRY, Telemetry

logger = logging.getLogger(__name__)

type ImageCapabilityStatus = Literal["supported", "unsupported", "unknown"]

# Some real vision providers reject sub-10px inputs, so the transport probe uses
# a 16px image rather than the tempting 1px minimum.
_PROBE_IMAGE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFElEQVR4nGOoJxEwjGoY1TB8NQAAjTl9"
    "EJLDg8QAAAAASUVORK5CYII="
)
_PROBE_IMAGE_DATA_URI = f"data:image/png;base64,{_PROBE_IMAGE_PNG_B64}"
_VISION_PROBE_MAX_TOKENS = 512
_REPROBE_COOLDOWN_SECONDS = 60.0

_UNSUPPORTED_MARKERS = (
    "does not support image",
    "image input is not supported",
    "no image support",
    "vision is not",
    "multimodal is not",
    "does not support vision",
    # Keep this OpenRouter marker image-specific: a generic "no endpoints"
    # response indicates a bad model/configuration, not text-only capability.
    "endpoints found that support image",
)


@dataclass(frozen=True, slots=True)
class ImageProbeOutcome:
    """Structured tri-state result of an image transport probe."""

    status: ImageCapabilityStatus
    failure_kind: str | None = None


def _classify_error(exc: Exception) -> ImageProbeOutcome:
    text = str(exc).lower()
    if any(marker in text for marker in _UNSUPPORTED_MARKERS):
        return ImageProbeOutcome(status="unsupported", failure_kind="explicit_unsupported")
    return ImageProbeOutcome(status="unknown", failure_kind=type(exc).__name__)


async def probe_image_capability(
    provider: Any,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
) -> ImageProbeOutcome:
    """Probe transport acceptance without interpreting generated text.

    A content-matching probe can falsely reject a capable model based on answer
    wording; a transport false positive only sends harmless extra image bytes.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "This is an image-capability probe."},
        {"type": "image_url", "image_url": {"url": _PROBE_IMAGE_DATA_URI}},
    ]
    try:
        await provider.complete(
            [{"role": "user", "content": content}],
            model=model,
            max_tokens=_VISION_PROBE_MAX_TOKENS,
            temperature=0,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - classification is the probe's job
        return _classify_error(exc)
    return ImageProbeOutcome(status="supported")


class ModelImageCapabilities:
    """Cache one tri-state transport fact per resolved model configuration."""

    def __init__(
        self,
        *,
        reprobe_cooldown_seconds: float = _REPROBE_COOLDOWN_SECONDS,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self._cooldown_seconds = reprobe_cooldown_seconds
        self._telemetry = telemetry
        self._identity_secret = secrets.token_bytes(32)
        self._outcomes: dict[str, ImageProbeOutcome] = {}
        self._last_probe: dict[str, float] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def resolve(self, settings: ModelSettings) -> ImageProbeOutcome:
        """Return the cached capability, probing unknown settings under single flight."""
        identity = self._identity(settings)
        cached = self._outcomes.get(identity)
        if cached is not None and cached.status != "unknown":
            return cached
        async with self._locks[identity]:
            cached = self._outcomes.get(identity)
            if cached is not None and cached.status != "unknown":
                return cached
            now = time.monotonic()
            last = self._last_probe.get(identity)
            if cached is not None and last is not None and now - last < self._cooldown_seconds:
                return cached
            outcome = await self._probe(settings)
            self._outcomes[identity] = outcome
            # Stamp after completion so a probe that consumes its timeout does
            # not also consume the cooldown and immediately permit a retry.
            self._last_probe[identity] = time.monotonic()
            return outcome

    async def _probe(self, settings: ModelSettings) -> ImageProbeOutcome:
        provider: Any = None
        async with self._telemetry.observe(
            "image_capability_probe",
            as_type="generation",
            metadata={"provider": settings.provider},
            model=settings.model,
        ) as observation:
            try:
                provider = get_provider(
                    settings.provider,
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    timeout=settings.timeout,
                    max_retries=settings.max_retries,
                )
                outcome = await probe_image_capability(
                    provider,
                    model=settings.model,
                    model_kwargs=settings.model_kwargs_copy() or None,
                )
            except Exception:
                logger.debug("Image capability probe failed", exc_info=True)
                outcome = ImageProbeOutcome(status="unknown", failure_kind="probe_error")
            finally:
                if provider is not None:
                    close_task = asyncio.create_task(provider.aclose())
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        try:
                            await close_task
                        except Exception:
                            logger.warning(
                                "Failed to close cancelled image-probe provider",
                                exc_info=True,
                            )
                        raise
            observation.update(
                output={"status": outcome.status, "failure_kind": outcome.failure_kind}
            )
        logger.info(
            "Image capability probe: status=%s model=%s provider=%s",
            outcome.status,
            settings.model,
            settings.provider,
        )
        return outcome

    def _identity(self, settings: ModelSettings) -> str:
        payload = json.dumps(
            [
                settings.provider,
                settings.base_url,
                settings.model,
                settings.model_kwargs_copy(),
            ],
            sort_keys=True,
            default=str,
        )
        return hmac.new(
            self._identity_secret,
            f"{payload}\0{settings.api_key or ''}".encode(),
            hashlib.sha256,
        ).hexdigest()


__all__ = [
    "ImageCapabilityStatus",
    "ImageProbeOutcome",
    "ModelImageCapabilities",
    "probe_image_capability",
]
