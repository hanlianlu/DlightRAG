# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web query document attachment admission and request-local models."""

import hashlib
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from dlightrag.core.ingestion.uploads import safe_upload_basename

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
    """One current-turn Web document attachment after admission."""

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


__all__ = [
    "MAX_CURRENT_DOCUMENTS",
    "MAX_DOCUMENT_BYTES",
    "ValidatedWebDocument",
    "classify_web_attachment",
    "validate_web_documents",
]
