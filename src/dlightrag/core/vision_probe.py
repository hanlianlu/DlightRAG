# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup image capability probes for every image-bearing model role.

``probe_image_capability`` sends one small 16×16 PNG and treats transport
acceptance as the signal -- a completed request means the model accepts
``image_url`` blocks -- returning a tri-state outcome. The reply text is
deliberately not inspected. ``ModelImageCapabilities`` owns the shared cache:
answer, VLM, and rerank are separate role facts, but two roles that resolve to
the same model configuration share one probe. Results are recorded on the owning
``RAGServiceManager`` (never on the provider) and never persisted.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.config import ModelConfig

logger = logging.getLogger(__name__)

type ImageCapabilityStatus = Literal["supported", "unsupported", "unknown"]

# Minimal 16×16 gray PNG -- base64-encoded so no filesystem dependency. Kept at
# 16px (not 1px) because some real vision providers reject sub-10px images (e.g.
# Alibaba Qwen: "height/width must be larger than 10"), which the transport-only
# probe would otherwise misclassify as an ``unknown`` capability failure.
_PROBE_IMAGE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFElEQVR4nGOoJxEwjGoY1TB8NQAAjTl9"
    "EJLDg8QAAAAASUVORK5CYII="
)
_PROBE_IMAGE_DATA_URI = f"data:image/png;base64,{_PROBE_IMAGE_PNG_B64}"
_VISION_PROBE_MAX_TOKENS = 512
_REPROBE_COOLDOWN_SECONDS = 60.0


__all__ = ["ImageCapabilityStatus", "ModelImageCapabilities", "probe_image_capability"]


_UNSUPPORTED_MARKERS = (
    "does not support image",
    "image input is not supported",
    "no image support",
    "vision is not",
    "multimodal is not",
    "does not support vision",
    # OpenRouter rejects a text-only model's image request with a 404 whose body
    # reads "No endpoints found that support image input". Kept image-specific so
    # a wrong slug ("No endpoints found for <model>") stays classified unknown.
    "endpoints found that support image",
)


@dataclass(frozen=True, slots=True)
class ImageProbeOutcome:
    """Structured tri-state result of an answer-model image probe."""

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
    """Probe whether *model* accepts ``image_url`` blocks.

    Success means the transport accepted the image request; the reply text is
    deliberately NOT inspected. Content-grounded probing (asking the model to
    describe a probe image and matching the reply) is rejected on purpose: it
    would trade a benign false positive -- a lenient provider that silently
    ignores the image, costing only wasted bytes on a request that still
    succeeds -- for a harmful false negative that blocks a genuinely capable
    model whose phrasing failed to match. Explicit provider rejections classify
    as ``unsupported``; timeouts / 401 / 429 / 5xx / unclassified errors classify
    as ``unknown`` (never ``unsupported``). The result is a pure model fact: a
    deployment's own image ceiling is applied by the role that owns it, never
    here, so a disabled ceiling cannot poison a configuration another role shares.
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
    """Probe each distinct resolved model configuration once and share the outcome.

    Answer, VLM, and chat rerank are separate role facts because they may resolve
    to different models, but a fact is a property of the configuration, not of the
    role: two roles that resolve to the same provider, endpoint, model, credential,
    and model kwargs share one probe. ``supported`` and ``unsupported`` are
    terminal for the process; only ``unknown`` is re-probed, at most once per
    cooldown window, under a per-configuration single-flight lock.
    """

    def __init__(self, *, reprobe_cooldown_seconds: float = _REPROBE_COOLDOWN_SECONDS) -> None:
        self._cooldown_seconds = reprobe_cooldown_seconds
        # The credential is folded into the identity through a process-local HMAC
        # so two genuinely different configurations stay distinct without keeping
        # a reversible copy of the key in this cache.
        self._identity_secret = secrets.token_bytes(32)
        self._outcomes: dict[str, ImageProbeOutcome] = {}
        self._last_probe: dict[str, float] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def resolve(self, cfg: ModelConfig) -> ImageProbeOutcome:
        """Return the image capability of *cfg*, probing only when it is not settled."""
        identity = self._identity(cfg)
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
            self._last_probe[identity] = now
            outcome = await self._probe(cfg)
            self._outcomes[identity] = outcome
            return outcome

    async def _probe(self, cfg: ModelConfig) -> ImageProbeOutcome:
        from dlightrag.models.providers import get_provider

        provider: Any = None
        try:
            provider = get_provider(
                cfg.provider,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
            )
            outcome = await probe_image_capability(
                provider,
                model=cfg.model,
                model_kwargs=cfg.model_kwargs or None,
            )
        except Exception:
            logger.debug("Image capability probe failed", exc_info=True)
            outcome = ImageProbeOutcome(status="unknown", failure_kind="probe_error")
        finally:
            if provider is not None:
                await provider.aclose()
        logger.info(
            "Image capability probe: status=%s model=%s provider=%s",
            outcome.status,
            cfg.model,
            cfg.provider,
        )
        return outcome

    def _identity(self, cfg: ModelConfig) -> str:
        payload = json.dumps(
            [
                cfg.provider,
                cfg.base_url,
                cfg.model,
                cfg.model_kwargs,
            ],
            sort_keys=True,
            default=str,
        )
        return hmac.new(
            self._identity_secret,
            f"{payload}\0{cfg.api_key or ''}".encode(),
            hashlib.sha256,
        ).hexdigest()
