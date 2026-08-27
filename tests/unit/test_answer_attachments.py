# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for SDK answer-attachment conveniences and their ResourceInput adapter."""

from pathlib import Path

import pytest

from dlightrag.engine.answer.resources.attachments import (
    AnswerAttachment,
    resource_inputs_from_attachments,
)
from dlightrag.engine.answer.resources.models import ResourceInput


def test_from_path_reads_bytes_and_hides_caller_path(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7 body")

    attachment = AnswerAttachment.from_path(source)
    [resource] = resource_inputs_from_attachments([attachment])

    assert isinstance(resource, ResourceInput)
    assert resource.content == b"%PDF-1.7 body"
    assert resource.filename == "report.pdf"
    assert resource.url is None
    # No ResourceInput field may leak the caller's absolute filesystem path.
    assert not any(
        str(source) in str(value) for value in vars(resource).values() if value is not None
    )


def test_from_path_missing_file_errors(tmp_path: Path) -> None:
    attachment = AnswerAttachment.from_path(tmp_path / "missing.pdf")

    with pytest.raises(OSError):
        resource_inputs_from_attachments([attachment])


def test_from_path_bounds_bytes(tmp_path: Path) -> None:
    source = tmp_path / "big.bin"
    source.write_bytes(b"x" * 2048)
    attachment = AnswerAttachment.from_path(source)

    with pytest.raises(ValueError):
        resource_inputs_from_attachments([attachment], max_attachment_bytes=1024)


def test_from_bytes_builds_and_bounds() -> None:
    attachment = AnswerAttachment.from_bytes(
        b"hello", filename="note.txt", declared_mime="text/plain"
    )
    [resource] = resource_inputs_from_attachments([attachment])

    assert resource.content == b"hello"
    assert resource.filename == "note.txt"
    assert resource.declared_mime == "text/plain"
    assert resource.url is None

    with pytest.raises(ValueError):
        resource_inputs_from_attachments(
            [AnswerAttachment.from_bytes(b"x" * 10)], max_attachment_bytes=4
        )


def test_from_url_validates_http_and_builds() -> None:
    attachment = AnswerAttachment.from_url("https://example.com/report.pdf", filename="report.pdf")
    [resource] = resource_inputs_from_attachments([attachment])

    assert resource.url == "https://example.com/report.pdf"
    assert resource.filename == "report.pdf"
    assert resource.content is None

    http_attachment = AnswerAttachment.from_url("http://example.com/report.pdf")
    [http_resource] = resource_inputs_from_attachments([http_attachment])
    assert http_resource.url == "http://example.com/report.pdf"

    with pytest.raises(ValueError):
        AnswerAttachment.from_url("ftp://example.com/report.pdf")
