# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web current-turn attachment admission (images + documents) and request-local models."""

import base64
import binascii
import hashlib
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from dlightrag.core.ingestion.uploads import safe_upload_basename
from dlightrag.utils.images import (
    MODEL_IMAGE_MAX_PIXELS,
    split_data_uri,
    verify_web_image_bytes,
)

MAX_CURRENT_DOCUMENTS = 3
MAX_DOCUMENT_BYTES = 100 * 1024 * 1024

_DOCUMENT_SUFFIXES = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".md",
        ".markdown",
        ".textpack",
        ".txt",
        ".csv",
        ".json",
        ".html",
        ".htm",
        ".xml",
        ".yaml",
        ".yml",
        ".rtf",
        ".odt",
        ".epub",
        ".tex",
        ".log",
        ".py",
        ".js",
        ".ts",
        ".css",
        ".scss",
        ".sql",
        ".sh",
        ".conf",
        ".ini",
        ".properties",
    }
)

AttachmentKind = Literal["image", "document", "unsupported"]


@dataclass(frozen=True, slots=True)
class ValidatedWebDocument:
    """One current-turn Web Composer document after admission."""

    attachment_id: str
    ordinal: int
    filename: str
    mime_type: str
    suffix: str
    document_bytes: bytes
    byte_size: int
    content_sha256: str

    @property
    def safe_filename(self) -> str:
        return safe_upload_basename(self.filename)

    def as_catalog_row(self, *, turn_number: int | None = None) -> dict[str, object]:
        row: dict[str, object] = {
            "attachment_id": self.attachment_id,
            "ordinal": self.ordinal,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "suffix": self.suffix,
            "byte_size": self.byte_size,
        }
        if turn_number is not None:
            row["turn_number"] = turn_number
        return row


def _suffix(filename: str) -> str:
    safe = safe_upload_basename(filename)
    dot = safe.rfind(".")
    return safe[dot:].lower() if dot >= 0 else ""


def classify_web_attachment(filename: str, mime_type: str | None) -> AttachmentKind:
    mime = (mime_type or mimetypes.guess_type(filename)[0] or "").lower()
    if mime.startswith("image/"):
        return "image"
    return "document" if _suffix(filename) in _DOCUMENT_SUFFIXES else "unsupported"


def validate_web_documents(
    documents: Sequence[tuple[str, str | None, bytes]],
) -> tuple[ValidatedWebDocument, ...]:
    if len(documents) > MAX_CURRENT_DOCUMENTS:
        raise ValueError("Web answer accepts at most 3 documents per message")

    validated: list[ValidatedWebDocument] = []
    for ordinal, (filename, mime_type, payload) in enumerate(documents, start=1):
        safe_name = safe_upload_basename(filename)
        suffix = _suffix(safe_name)
        if classify_web_attachment(safe_name, mime_type) != "document":
            raise ValueError(f"Unsupported document attachment: {safe_name}")
        if not payload:
            raise ValueError(f"Document attachment is empty: {safe_name}")
        if len(payload) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document attachment exceeds 100 MB")
        detected_mime = (
            mime_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        )
        validated.append(
            ValidatedWebDocument(
                attachment_id=str(uuid4()),
                ordinal=ordinal,
                filename=safe_name,
                mime_type=detected_mime,
                suffix=suffix,
                document_bytes=payload,
                byte_size=len(payload),
                content_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(validated)


# --- Web current-turn image admission ---------------------------------------


@dataclass(frozen=True, slots=True)
class ValidatedWebImage:
    """One canonical browser image upload, safe for model use and persistence."""

    image_id: str
    ordinal: int
    mime_type: str
    image_bytes: bytes
    data_uri: str
    content_sha256: str

    @property
    def model_block(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.data_uri}}


def _strict_base64_decoded_size(payload: str) -> int | None:
    """Return decoded size from canonical padding without allocating decoded bytes."""
    if not payload or len(payload) % 4:
        return None
    padding = len(payload) - len(payload.rstrip("="))
    misplaced_padding = "=" in payload[:-padding] if padding else "=" in payload
    if padding > 2 or misplaced_padding:
        return None
    return (len(payload) // 4 * 3) - padding


def validate_web_images(
    values: list[str],
    *,
    max_images: int,
    max_bytes: int,
    max_pixels: int = MODEL_IMAGE_MAX_PIXELS,
) -> tuple[ValidatedWebImage, ...]:
    """Strictly validate browser image uploads and return their canonical form."""
    if len(values) > max_images:
        raise ValueError(f"at most {max_images} current images are allowed")
    validated: list[ValidatedWebImage] = []
    for ordinal, value in enumerate(values, start=1):
        _declared_mime, payload = split_data_uri(value)
        decoded_size = _strict_base64_decoded_size(payload)
        if decoded_size is not None and decoded_size > max_bytes:
            raise ValueError(f"current image {ordinal} exceeds the {max_bytes}-byte limit")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"current image {ordinal} is not valid base64") from exc
        if not raw:
            raise ValueError(f"current image {ordinal} is empty")
        if len(raw) > max_bytes:
            raise ValueError(f"current image {ordinal} exceeds the {max_bytes}-byte limit")
        try:
            mime_type = verify_web_image_bytes(raw, max_pixels=max_pixels)
        except ValueError as exc:
            raise ValueError(f"current image {ordinal} {exc}") from exc
        encoded = base64.b64encode(raw).decode("ascii")
        validated.append(
            ValidatedWebImage(
                image_id=str(uuid4()),
                ordinal=ordinal,
                mime_type=mime_type,
                image_bytes=raw,
                data_uri=f"data:{mime_type};base64,{encoded}",
                content_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return tuple(validated)


__all__ = [
    "MAX_CURRENT_DOCUMENTS",
    "MAX_DOCUMENT_BYTES",
    "ValidatedWebDocument",
    "ValidatedWebImage",
    "classify_web_attachment",
    "validate_web_documents",
    "validate_web_images",
]
