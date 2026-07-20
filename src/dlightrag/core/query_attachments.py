# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Content-addressed parse-cache value types for Web query attachments.

These primitive-only dataclasses are the storage contract for cached document
parses. The storage layer maps content-addressed cache rows to and from these
types without importing any Web transport module. The query-time attachment
service (added in a later slice) builds on the same types.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AttachmentContextChunk:
    """One parsed, retrievable unit of a query-time document attachment."""

    chunk_id: str
    attachment_id: str
    filename: str
    chunk_index: int
    content: str
    token_estimate: int = 0
    page_idx: int | None = None
    bbox: Any = None
    sidecar_type: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedAttachmentBundle:
    """An ordered set of parsed chunks tied to the parser that produced them."""

    chunks: list[AttachmentContextChunk]
    parser_signature: str


__all__ = [
    "AttachmentContextChunk",
    "ParsedAttachmentBundle",
]
