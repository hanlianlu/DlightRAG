# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified Web current-turn attachment admission."""

import hashlib
import io
from collections.abc import Sequence

import pytest
from PIL import Image

from dlightrag.web.attachment_models import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    ValidatedWebAttachment,
    classify_web_attachment,
    validate_web_attachments,
)

_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
_MAX_TOTAL_ATTACHMENT_BYTES = 128 * 1024 * 1024


def _validate(
    items: Sequence[tuple[str, str | None, bytes]],
    *,
    max_attachments: int = 6,
    max_attachment_bytes: int = _MAX_ATTACHMENT_BYTES,
    max_total_attachment_bytes: int = _MAX_TOTAL_ATTACHMENT_BYTES,
    **kwargs: int,
) -> tuple[ValidatedWebAttachment, ...]:
    return validate_web_attachments(
        items,
        max_attachments=max_attachments,
        max_attachment_bytes=max_attachment_bytes,
        max_total_attachment_bytes=max_total_attachment_bytes,
        **kwargs,
    )


def _png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_classify_web_attachment_separates_images_and_documents() -> None:
    assert classify_web_attachment("chart.png", "image/png") == "image"
    assert classify_web_attachment("photo.jpg", None) == "image"
    assert classify_web_attachment("report.pdf", "application/pdf") == "document"
    assert classify_web_attachment("notes.md", "text/markdown") == "document"
    assert classify_web_attachment("archive.zip", "application/zip") == "unsupported"


def test_validate_rejects_total_attachment_bytes_before_processing() -> None:
    items = [
        ("first.txt", "text/plain", b"1234"),
        ("second.txt", "text/plain", b"5678"),
    ]

    with pytest.raises(ValueError, match="total attachment bytes exceed 7"):
        validate_web_attachments(
            items,
            max_attachments=6,
            max_attachment_bytes=10,
            max_total_attachment_bytes=7,
        )
    assert classify_web_attachment("payload.bin", None) == "unsupported"


def test_supported_document_extensions_cover_core_formats() -> None:
    assert SUPPORTED_DOCUMENT_EXTENSIONS
    assert {"pdf", "docx", "pptx", "xlsx", "md", "csv", "json", "html"} <= (
        SUPPORTED_DOCUMENT_EXTENSIONS
    )


def test_validate_admits_mixed_ordered_image_and_document() -> None:
    png = _png_bytes()
    doc = b"# report\nbody"
    validated = _validate(
        [
            ("chart.png", "image/png", png),
            ("notes.md", "text/markdown", doc),
        ],
    )

    assert [item.kind for item in validated] == ["image", "document"]
    assert [item.ordinal for item in validated] == [1, 2]
    image, document = validated
    assert isinstance(image, ValidatedWebAttachment)
    assert image.filename == "chart.png"
    assert image.mime_type == "image/png"
    assert image.attachment_bytes == png
    assert image.byte_size == len(png)
    assert image.content_sha256 == hashlib.sha256(png).hexdigest()
    assert document.suffix == ".md"
    assert document.attachment_bytes == doc


def test_validate_rejects_too_many_attachments() -> None:
    png = _png_bytes()
    items = [(f"c{i}.png", "image/png", png) for i in range(7)]

    with pytest.raises(ValueError, match="at most 6 attachments"):
        _validate(items)


def test_validate_enforces_a_lowered_attachment_count_limit() -> None:
    png = _png_bytes()
    items = [(f"c{i}.png", "image/png", png) for i in range(3)]

    with pytest.raises(ValueError, match="at most 2 attachments"):
        _validate(items, max_attachments=2)


def test_validate_image_uses_detected_mime_over_declared() -> None:
    png = _png_bytes()

    (image,) = _validate([("mislabeled.jpg", "image/jpeg", png)])

    assert image.kind == "image"
    assert image.mime_type == "image/png"


def test_validate_rejects_image_over_byte_limit() -> None:
    png = _png_bytes()

    with pytest.raises(ValueError, match="exceeds"):
        _validate([("chart.png", "image/png", png)], max_attachment_bytes=len(png) - 1)


def test_validate_rejects_image_over_pixel_limit() -> None:
    png = _png_bytes((32, 32))

    with pytest.raises(ValueError, match="pixel"):
        _validate([("chart.png", "image/png", png)], image_max_pixels=100)


def test_validate_rejects_corrupt_image_bytes() -> None:
    with pytest.raises(ValueError, match="image"):
        _validate([("chart.png", "image/png", b"not-a-real-image")])


def test_validate_rejects_document_over_byte_limit() -> None:
    with pytest.raises(ValueError, match="size limit"):
        _validate(
            [("huge.pdf", "application/pdf", b"x" * (_MAX_ATTACHMENT_BYTES + 1))],
            max_attachment_bytes=_MAX_ATTACHMENT_BYTES,
        )


def test_validate_rejects_unsupported_attachment() -> None:
    with pytest.raises(ValueError, match="Unsupported attachment"):
        _validate([("archive.zip", "application/zip", b"x")])


def test_validate_rejects_empty_attachment() -> None:
    with pytest.raises(ValueError, match="empty"):
        _validate([("empty.pdf", "application/pdf", b"")])


def test_validate_rejects_unsafe_filename() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _validate([("../secret.pdf", "application/pdf", b"body")])
