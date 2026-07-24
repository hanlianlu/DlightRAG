# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared answer-input error taxonomy (design §14.2).

One error-kind vocabulary and classifier consumed by every surface (Web SSE,
REST, MCP) so callers can branch on a stable, machine-readable ``error_kind``
instead of parsing free-form messages.
"""

from __future__ import annotations

CURRENT_IMAGES_UNSUPPORTED = "CURRENT_IMAGES_UNSUPPORTED"
CURRENT_IMAGE_LIMIT_EXCEEDED = "CURRENT_IMAGE_LIMIT_EXCEEDED"
CURRENT_DOCUMENT_PARSE_FAILED = "CURRENT_DOCUMENT_PARSE_FAILED"
ANSWER_IMAGE_CAPABILITY_UNKNOWN = "ANSWER_IMAGE_CAPABILITY_UNKNOWN"
ANSWER_STREAM_FAILED = "ANSWER_STREAM_FAILED"

_IMAGES_NOT_SUPPORTED_MARKER = "[IMAGES_NOT_SUPPORTED_BY_MODEL]"


class AnswerInputError(ValueError):
    """Answer input rejected with a client-safe message and stable kind.

    Subclasses must construct ``public_message`` only from sanitized content.
    """

    def __init__(self, public_message: str, *, error_kind: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.error_kind = error_kind


class AnswerImageError(AnswerInputError):
    """Answer-image request rejected at the capability/transport boundary.

    Carries a stable ``error_kind`` from the answer-image taxonomy so every
    surface surfaces the same machine-readable classification.
    """

    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(public_message=message, error_kind=error_kind)


class CurrentImagePayloadError(AnswerImageError):
    """Explicit user images cannot fit the configured answer transport."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_kind=CURRENT_IMAGE_LIMIT_EXCEEDED)


class CurrentDocumentParseError(AnswerInputError):
    """A current Composer document could not be parsed safely."""

    def __init__(self, safe_filename: str) -> None:
        super().__init__(
            public_message=(
                f"Could not read {safe_filename}. Check that the document is valid and "
                "the document parser is available."
            ),
            error_kind=CURRENT_DOCUMENT_PARSE_FAILED,
        )


def classify_answer_error(exc: BaseException) -> str:
    """Map an answer-stream failure to a stable answer-input error kind."""
    if isinstance(exc, AnswerInputError):
        return exc.error_kind
    if _IMAGES_NOT_SUPPORTED_MARKER in str(exc):
        return CURRENT_IMAGES_UNSUPPORTED
    return ANSWER_STREAM_FAILED


__all__ = [
    "ANSWER_IMAGE_CAPABILITY_UNKNOWN",
    "ANSWER_STREAM_FAILED",
    "CURRENT_DOCUMENT_PARSE_FAILED",
    "CURRENT_IMAGES_UNSUPPORTED",
    "CURRENT_IMAGE_LIMIT_EXCEEDED",
    "AnswerInputError",
    "AnswerImageError",
    "CurrentDocumentParseError",
    "CurrentImagePayloadError",
    "classify_answer_error",
]
