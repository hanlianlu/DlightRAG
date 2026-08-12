# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-model image capability, discovered at startup (never persisted).

The startup probe records whether the *query-role* answer model accepts
``image_url`` blocks and how many, as a genuine tri-state.  This drives the
unified answer image transport budget and the Web upload gate; it is
re-validated every process start rather than cached in any store.
"""

from __future__ import annotations

from dataclasses import dataclass

from dlightrag.core.answer.errors import (
    ANSWER_IMAGE_CAPABILITY_UNKNOWN,
    CURRENT_IMAGES_UNSUPPORTED,
    AnswerImageError,
    CurrentImagePayloadError,
)
from dlightrag.core.vision_probe import ImageCapabilityStatus

_ERROR_IMAGES_NOT_SUPPORTED = (
    "Current model does not support image input. Use a vision-capable model or remove images."
)
_ERROR_CAPABILITY_UNKNOWN = (
    "Answer-model image capability is unknown: the startup probe did not confirm image support. "
    "Provide a vision-capable query model or retry once the model is reachable."
)


def derive_effective_max_images(
    status: ImageCapabilityStatus,
    configured_ceiling: int,
) -> int:
    """Effective final image-block count for the answer transport budget.

    ``supported`` uses the configured deployment ceiling; every other status
    (``unsupported``/``unknown``) or a non-positive ceiling yields ``0`` (no raw
    images, text descriptions only).
    """
    if status != "supported" or configured_ceiling <= 0:
        return 0
    return configured_ceiling


@dataclass(frozen=True, slots=True)
class AnswerImageCapability:
    """Request-independent answer-model image capability snapshot."""

    status: ImageCapabilityStatus
    configured_ceiling: int
    effective_max_images: int
    provider: str
    base_url: str | None
    model: str
    failure_kind: str | None


def check_answer_image_capability(
    *,
    image_count: int,
    capability: AnswerImageCapability | None,
) -> None:
    """Reject images unless the query-role answer model is confirmed to accept them."""
    if image_count <= 0:
        return
    if capability is None or capability.status == "unknown":
        raise AnswerImageError(
            f"[ANSWER_IMAGE_CAPABILITY_UNKNOWN] {_ERROR_CAPABILITY_UNKNOWN}",
            error_kind=ANSWER_IMAGE_CAPABILITY_UNKNOWN,
        )
    if capability.status == "unsupported":
        raise AnswerImageError(
            f"[IMAGES_NOT_SUPPORTED_BY_MODEL] {_ERROR_IMAGES_NOT_SUPPORTED}",
            error_kind=CURRENT_IMAGES_UNSUPPORTED,
        )
    if image_count > capability.effective_max_images:
        raise CurrentImagePayloadError(
            f"at most {capability.effective_max_images} current images are allowed"
        )


def answer_image_capability_summary(
    capability: AnswerImageCapability | None,
) -> dict[str, object]:
    """Client-facing capability summary shared by REST ``/health`` and MCP.

    Exposes only the fields a caller needs to decide whether and how many images
    to send; internal transport details (``base_url``) are omitted. A missing or
    unprobed capability is reported as ``unknown`` with zero slots, matching the
    fail-closed answer-image guard.
    """
    if capability is None:
        return {
            "status": "unknown",
            "effective_max_images": 0,
            "configured_ceiling": 0,
            "model": None,
        }
    return {
        "status": capability.status,
        "effective_max_images": capability.effective_max_images,
        "configured_ceiling": capability.configured_ceiling,
        "model": capability.model,
    }


__all__ = [
    "AnswerImageCapability",
    "answer_image_capability_summary",
    "check_answer_image_capability",
    "derive_effective_max_images",
]
