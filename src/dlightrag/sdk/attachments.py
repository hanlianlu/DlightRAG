# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""HTTP multipart attachment records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnswerAttachmentUpload:
    """One multipart attachment sent with an Answer request."""

    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


__all__ = ["AnswerAttachmentUpload"]
