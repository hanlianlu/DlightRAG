# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed model-visible content carried by one Agent tool result."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolTextPart:
    """One text block in a tool result."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ToolResourceAttachmentPart:
    """A durable resource snapshot attached to a tool result.

    The part carries identity and integrity metadata for the journal. Raw bytes
    live in the owner-scoped Blob store durably and ride along only as the
    transport-private ``data`` field, which is never journaled, logged, or
    persisted by telemetry; providers encode it into their wire format at
    projection time.
    """

    resource_id: str
    safe_name: str
    media_type: str
    content_digest: str
    size_bytes: int
    type: Literal["resource_attachment"] = "resource_attachment"
    #: Transport-private original bytes; excluded from every durable encoding.
    data: bytes = b""

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("tool attachment resource id cannot be empty")
        if not self.safe_name.strip():
            raise ValueError("tool attachment safe name cannot be empty")
        if not self.media_type.strip():
            raise ValueError("tool attachment media type cannot be empty")
        if len(self.content_digest) != 64:
            raise ValueError("tool attachment digest must be a SHA-256 hex digest")
        if self.size_bytes < 0:
            raise ValueError("tool attachment size cannot be negative")


type ToolContentPart = ToolTextPart | ToolResourceAttachmentPart
type ToolContent = tuple[ToolContentPart, ...]


def tool_content_text(parts: ToolContent) -> str:
    """Join text parts without exposing attachment metadata as prose."""
    return "\n".join(part.text for part in parts if isinstance(part, ToolTextPart))


def tool_content_attachments(parts: ToolContent) -> tuple[ToolResourceAttachmentPart, ...]:
    """Return resource parts in their declared order."""
    return tuple(part for part in parts if isinstance(part, ToolResourceAttachmentPart))


def tool_content_message_fields(parts: ToolContent) -> dict[str, Any]:
    """Project typed content into provider-neutral tool-message fields."""
    fields: dict[str, Any] = {"content": tool_content_text(parts)}
    attachments = tool_content_attachments(parts)
    if attachments:
        projected: list[dict[str, Any]] = []
        for attachment in attachments:
            item: dict[str, Any] = {
                "resource_id": attachment.resource_id,
                "safe_name": attachment.safe_name,
                "media_type": attachment.media_type,
                "content_digest": attachment.content_digest,
                "size_bytes": attachment.size_bytes,
            }
            if attachment.data:
                item["data_url"] = (
                    f"data:{attachment.media_type};base64,"
                    f"{base64.b64encode(attachment.data).decode('ascii')}"
                )
            projected.append(item)
        fields["attachments"] = projected
    return fields


def encode_tool_content(parts: ToolContent) -> list[dict[str, Any]]:
    """Encode typed parts for an append-only session payload."""
    encoded: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, ToolTextPart):
            encoded.append({"type": "text", "text": part.text})
        else:
            encoded.append(
                {
                    "type": "resource_attachment",
                    "resource_id": part.resource_id,
                    "safe_name": part.safe_name,
                    "media_type": part.media_type,
                    "content_digest": part.content_digest,
                    "size_bytes": part.size_bytes,
                }
            )
    return encoded


def decode_tool_content(value: object) -> ToolContent:
    """Decode Tool Contract v2 parts; legacy string results are rejected."""
    if not isinstance(value, list):
        raise ValueError("tool result content must be a content-part list")
    parts: list[ToolContentPart] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("tool result content part must be an object")
        part_type = raw.get("type")
        if part_type == "text":
            parts.append(ToolTextPart(str(raw.get("text") or "")))
        elif part_type == "resource_attachment":
            parts.append(
                ToolResourceAttachmentPart(
                    resource_id=str(raw.get("resource_id") or ""),
                    safe_name=str(raw.get("safe_name") or ""),
                    media_type=str(raw.get("media_type") or ""),
                    content_digest=str(raw.get("content_digest") or ""),
                    size_bytes=int(raw.get("size_bytes") or 0),
                )
            )
        else:
            raise ValueError(f"unknown tool result content part: {part_type!r}")
    return tuple(parts)


__all__ = [
    "ToolContent",
    "ToolContentPart",
    "ToolResourceAttachmentPart",
    "ToolTextPart",
    "decode_tool_content",
    "encode_tool_content",
    "tool_content_attachments",
    "tool_content_message_fields",
    "tool_content_text",
]
