# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-model image capability, discovered at startup (never persisted).

The startup probe records whether the *query-role* answer model accepts
``image_url`` blocks and how many, as a genuine tri-state.  This drives the
unified answer image transport budget and the Web upload gate; it is
re-validated every process start rather than cached in any store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapabilityStatus = Literal["supported", "unsupported", "unknown"]


def derive_effective_max_images(
    status: CapabilityStatus,
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

    status: CapabilityStatus
    configured_ceiling: int
    effective_max_images: int
    provider: str
    base_url: str | None
    model: str
    failure_kind: str | None


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
    "CapabilityStatus",
    "answer_image_capability_summary",
    "derive_effective_max_images",
]
