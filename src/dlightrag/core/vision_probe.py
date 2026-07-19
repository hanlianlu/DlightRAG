# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup image capability probes for answer and rerank models.

``probe_image_capability`` sends one small 16×16 PNG and treats transport
acceptance as the signal -- a completed request means the model accepts
``image_url`` blocks -- returning a tri-state outcome. The reply text is
deliberately not inspected. Both the answer path and the rerank path use it.
Results are recorded on the owning ``RAGServiceManager`` (never on the provider).
"""

from dataclasses import dataclass
from typing import Any, Literal

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


__all__ = ["probe_image_capability"]


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

    status: Literal["supported", "unsupported", "unknown"]
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
    deliberately NOT inspected. Content-grounded probing (asking the model to
    describe a probe image and matching the reply) is rejected on purpose: it
    would trade a benign false positive -- a lenient provider that silently
    ignores the image, costing only wasted bytes on a request that still
    succeeds -- for a harmful false negative that blocks a genuinely capable
    model whose phrasing failed to match. Explicit provider rejections classify
    as ``unsupported``; timeouts / 401 / 429 / 5xx / unclassified errors classify
    as ``unknown`` (never ``unsupported``). A non-positive ``ceiling`` short
    circuits to ``unsupported`` with ``config_disabled`` and no model call.
    """
    if ceiling <= 0:
        return ImageProbeOutcome(status="unsupported", failure_kind="config_disabled")
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
