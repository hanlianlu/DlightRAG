# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web Composer document admission."""

from pathlib import Path

import pytest

from dlightrag.web.attachment_models import (
    MAX_CURRENT_DOCUMENTS,
    MAX_DOCUMENT_BYTES,
    ValidatedWebDocument,
    classify_web_attachment,
    validate_web_documents,
)


def test_classify_web_attachment_separates_images_and_documents() -> None:
    assert classify_web_attachment("chart.png", "image/png") == "image"
    assert classify_web_attachment("report.pdf", "application/pdf") == "document"
    assert classify_web_attachment("notes.md", "text/markdown") == "document"
    assert classify_web_attachment("archive.zip", "application/zip") == "unsupported"


def test_validate_web_documents_enforces_count_and_size() -> None:
    payload = b"x" * 12
    docs = [
        ("a.pdf", "application/pdf", payload),
        (
            "b.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            payload,
        ),
        ("c.md", "text/markdown", payload),
    ]

    validated = validate_web_documents(docs)

    assert len(validated) == MAX_CURRENT_DOCUMENTS
    assert all(isinstance(item, ValidatedWebDocument) for item in validated)
    assert validated[0].filename == "a.pdf"
    assert validated[0].suffix == ".pdf"
    assert validated[0].byte_size == len(payload)


def test_validate_web_documents_rejects_too_many_documents() -> None:
    docs = [(f"doc{i}.pdf", "application/pdf", b"x") for i in range(MAX_CURRENT_DOCUMENTS + 1)]

    with pytest.raises(ValueError, match="at most 3 documents"):
        validate_web_documents(docs)


def test_validate_web_documents_rejects_oversized_document() -> None:
    with pytest.raises(ValueError, match="100 MB"):
        validate_web_documents([("huge.pdf", "application/pdf", b"x" * (MAX_DOCUMENT_BYTES + 1))])


def test_validate_web_documents_rejects_unsafe_or_unsupported_names() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        validate_web_documents([("../secret.pdf", "application/pdf", b"x")])

    with pytest.raises(ValueError, match="Unsupported"):
        validate_web_documents([("archive.zip", "application/zip", b"x")])


def test_validated_document_model_block_is_not_an_image_block() -> None:
    doc = validate_web_documents([("report.pdf", "application/pdf", b"hello")])[0]

    assert doc.attachment_id
    assert doc.content_sha256
    assert doc.as_catalog_row(turn_number=1)["filename"] == "report.pdf"
    assert "image_url" not in doc.as_catalog_row(turn_number=1)
    assert Path(doc.safe_filename).name == "report.pdf"
