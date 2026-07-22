# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web Composer document admission."""

import hashlib

import pytest
from lightrag.parser.registry import suffix_capabilities
from lightrag.parser.routing import resolve_parser_directives

from dlightrag.config import ParserConfig
from dlightrag.web.attachment_models import (
    COMPOSER_DOCUMENT_EXTENSIONS,
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
    assert classify_web_attachment("notes.markdown", "text/markdown") == "unsupported"
    assert classify_web_attachment("archive.zip", "application/zip") == "unsupported"


def test_composer_document_policy_is_supported_by_default_parser_routing() -> None:
    assert COMPOSER_DOCUMENT_EXTENSIONS
    rules = ParserConfig().rules

    unsupported: list[str] = []
    for extension in sorted(COMPOSER_DOCUMENT_EXTENSIONS):
        directives = resolve_parser_directives(
            f"document.{extension}",
            parser_rules=rules,
            require_external_endpoint=False,
        )
        if extension not in suffix_capabilities(directives.engine):
            unsupported.append(f"{extension}:{directives.engine}")

    assert unsupported == []


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
    assert validated[0].document_bytes == payload
    assert validated[0].content_sha256 == hashlib.sha256(payload).hexdigest()


def test_validate_web_documents_rejects_too_many_documents() -> None:
    docs = [(f"doc{i}.pdf", "application/pdf", b"x") for i in range(MAX_CURRENT_DOCUMENTS + 1)]

    with pytest.raises(ValueError, match="at most 3 documents"):
        validate_web_documents(docs)


def test_validate_web_documents_rejects_oversized_document() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_web_documents([("huge.pdf", "application/pdf", b"x" * (MAX_DOCUMENT_BYTES + 1))])

    assert str(exc_info.value) == "Composer document exceeds 100 MB"


def test_validate_web_documents_rejects_unsafe_or_unsupported_names() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        validate_web_documents([("../secret.pdf", "application/pdf", b"x")])

    with pytest.raises(ValueError) as exc_info:
        validate_web_documents([("archive.zip", "application/zip", b"x")])

    assert str(exc_info.value) == "Unsupported Composer document: archive.zip"


def test_validate_web_documents_rejects_empty_document() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_web_documents([("empty.pdf", "application/pdf", b"")])

    assert str(exc_info.value) == "Composer document is empty: empty.pdf"
