# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared answer-image error taxonomy (design §14.2).

One error-kind vocabulary and classifier consumed by every surface (Web SSE,
REST, MCP) so callers can branch on a stable, machine-readable ``error_kind``
instead of parsing free-form messages.
"""

from __future__ import annotations

from dlightrag.core.answer_prompt import CurrentImagePayloadError

CURRENT_IMAGES_UNSUPPORTED = "CURRENT_IMAGES_UNSUPPORTED"
CURRENT_IMAGE_LIMIT_EXCEEDED = "CURRENT_IMAGE_LIMIT_EXCEEDED"
ANSWER_IMAGE_CAPABILITY_UNKNOWN = "ANSWER_IMAGE_CAPABILITY_UNKNOWN"
ANSWER_STREAM_FAILED = "ANSWER_STREAM_FAILED"

_IMAGES_NOT_SUPPORTED_MARKER = "[IMAGES_NOT_SUPPORTED_BY_MODEL]"


class AnswerImageError(ValueError):
    """Answer-image request rejected at the capability/transport boundary.

    Carries a stable ``error_kind`` from the answer-image taxonomy so every
    surface surfaces the same machine-readable classification.
    """

    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


def classify_answer_error(exc: BaseException) -> str:
    """Map an answer-stream failure to a stable answer-image error kind."""
    if isinstance(exc, AnswerImageError):
        return exc.error_kind
    if isinstance(exc, CurrentImagePayloadError):
        return CURRENT_IMAGE_LIMIT_EXCEEDED
    if _IMAGES_NOT_SUPPORTED_MARKER in str(exc):
        return CURRENT_IMAGES_UNSUPPORTED
    return ANSWER_STREAM_FAILED


__all__ = [
    "ANSWER_IMAGE_CAPABILITY_UNKNOWN",
    "ANSWER_STREAM_FAILED",
    "CURRENT_IMAGES_UNSUPPORTED",
    "CURRENT_IMAGE_LIMIT_EXCEEDED",
    "AnswerImageError",
    "classify_answer_error",
]
