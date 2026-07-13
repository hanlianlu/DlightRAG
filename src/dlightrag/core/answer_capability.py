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
    provider_max: int | None,
) -> int:
    """Effective final image-block count for the answer transport budget.

    ``supported`` clamps the deployment ceiling to any provider-declared max;
    every other status (``unsupported``/``unknown``) or a non-positive ceiling
    yields ``0`` (no raw images, text descriptions only).
    """
    if status != "supported" or configured_ceiling <= 0:
        return 0
    if provider_max is not None:
        return max(0, min(configured_ceiling, provider_max))
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


__all__ = ["AnswerImageCapability", "CapabilityStatus", "derive_effective_max_images"]
