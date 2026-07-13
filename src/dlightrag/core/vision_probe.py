# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup vision capability probe for chat/answer and rerank models.

Sends one 1×1 pixel PNG image to the model and checks whether the response is a
valid text completion (``"ok"``) or an error. The result is recorded on the
owning ``RAGServiceManager`` (never on the provider).
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Minimal 1×1 white PNG — base64-encoded so no filesystem dependency.
_ONE_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+"
    "hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)
_ONE_PIXEL_DATA_URI = f"data:image/png;base64,{_ONE_PIXEL_PNG_B64}"
_VISION_PROBE_MAX_TOKENS = 512


async def probe_vision_support(
    provider: Any,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
) -> bool:
    """Probe whether *model* accepts ``image_url`` blocks.

    Sends a single-turn request with one 1×1 pixel image and expects
    the model to respond with ``"ok"``.  Returns ``True`` if the model
    handles images, ``False`` otherwise.  Reasoning-capable models may
    spend a few tokens before emitting final content, so the probe uses a
    generous but bounded content budget rather than the absolute minimum.

    The probe is idempotent — callers normally run it once at startup
    and record the result on the owning ``RAGServiceManager``.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reply with exactly 'ok' and nothing else."},
                {"type": "image_url", "image_url": {"url": _ONE_PIXEL_DATA_URI}},
            ],
        }
    ]
    try:
        response = await provider.complete(
            messages,
            model=model,
            max_tokens=_VISION_PROBE_MAX_TOKENS,
            temperature=0,
            model_kwargs=model_kwargs,
        )
        text = str(response).strip().lower()
        return "ok" in text
    except Exception:
        logger.debug("Vision probe failed for model %s", model, exc_info=True)
        return False


__all__ = ["probe_image_capability", "probe_vision_support"]


_UNSUPPORTED_MARKERS = (
    "does not support image",
    "image input is not supported",
    "no image support",
    "vision is not",
    "multimodal is not",
    "does not support vision",
)


@dataclass(frozen=True, slots=True)
class ImageProbeOutcome:
    """Structured tri-state result of an answer-model image probe."""

    status: Literal["supported", "unsupported", "unknown"]
    provider_max: int | None = None
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
    ceiling: int,
    model_kwargs: dict[str, Any] | None = None,
) -> ImageProbeOutcome:
    """Probe whether *model* accepts ``image_url`` blocks.

    Success means the transport accepted the image request; the reply text is
    NOT inspected for a magic word.  Explicit provider rejections classify as
    ``unsupported``; timeouts / 401 / 429 / 5xx / unclassified errors classify
    as ``unknown`` (never ``unsupported``).  A non-positive ``ceiling`` short
    circuits to ``unsupported`` with ``config_disabled`` and no model call.
    """
    if ceiling <= 0:
        return ImageProbeOutcome(status="unsupported", failure_kind="config_disabled")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "This is an image-capability probe."},
        {"type": "image_url", "image_url": {"url": _ONE_PIXEL_DATA_URI}},
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
