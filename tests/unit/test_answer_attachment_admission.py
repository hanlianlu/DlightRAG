# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The uploads an Answer Run reads, decided once by the Engine."""

import io
import zipfile

import pytest
from PIL import Image

from dlightrag.engine.answer.errors import (
    UNSUPPORTED_ATTACHMENT_TYPE,
    UnsupportedAttachmentTypeError,
)
from dlightrag.engine.answer.resources.admission import (
    READABLE_DOCUMENT_EXTENSIONS,
    TEXT_EXTENSIONS,
    require_readable_attachment,
)
from dlightrag.engine.answer.resources.converters import CONVERTED_EXTENSIONS, is_convertible
from dlightrag.engine.answer.resources.models import ResourceDecodeError, ResourceInput
from dlightrag.engine.answer.resources.registry import ResourceRegistry

_OCTET = "application/octet-stream"
#: Bytes no Run reads: not an image, and not text in any encoding.
_BINARY = bytes(range(256)) * 8


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def _package() -> bytes:
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("content.xml", "<office:document/>")
    return package.getvalue()


async def _run_reads(filename: str, content: bytes) -> str:
    """What a Run that admitted this upload reads from it, or raises."""
    async with ResourceRegistry() as registry:
        handle = registry.register(
            ResourceInput(filename=filename, content=content, declared_mime=_OCTET)
        )
        page = await registry.read(handle, max_window_tokens=400)
    return page.extraction_status


def test_the_listed_documents_are_the_converted_and_the_decoded_ones() -> None:
    assert set(READABLE_DOCUMENT_EXTENSIONS) == CONVERTED_EXTENSIONS | TEXT_EXTENSIONS
    assert list(READABLE_DOCUMENT_EXTENSIONS) == sorted(READABLE_DOCUMENT_EXTENSIONS)
    assert {"pdf", "docx", "pptx", "xlsx", "csv", "html", "htm"} == CONVERTED_EXTENSIONS
    assert all(is_convertible(f"file.{extension}", None) for extension in CONVERTED_EXTENSIONS)
    # No listed text extension reaches a converter: each one is read by decoding.
    assert not any(is_convertible(f"file.{extension}", None) for extension in TEXT_EXTENSIONS)


@pytest.mark.parametrize("extension", sorted(TEXT_EXTENSIONS))
async def test_every_listed_text_type_is_read_as_decoded_text(extension: str) -> None:
    registry = ResourceRegistry()
    handle = registry.register(
        ResourceInput(filename=f"notes.{extension}", content=b"revenue grew 11%\n")
    )

    page = await registry.read(handle, max_window_tokens=200)

    assert "revenue grew 11%" in page.content
    assert page.extraction_status == "text"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("main.go", b"package main\n\nfunc main() {}\n"),
        ("data.tsv", b"quarter\trevenue\nQ1\t11\n"),
        ("Cargo.toml", b'[package]\nname = "report"\n'),
        ("events.jsonl", b'{"event": "open"}\n{"event": "close"}\n'),
        ("Makefile", b"report:\n\tpython report.py\n"),
        ("notes.weird", "Umsatz stieg um 11 %.\n".encode("latin-1")),
    ],
)
async def test_an_unlisted_upload_a_run_decodes_is_admitted_by_its_bytes(
    filename: str, content: bytes
) -> None:
    # A client that names no type sends application/octet-stream, as curl does; a
    # Run decodes these bytes as text, so they are admitted.
    require_readable_attachment(filename, _OCTET, content)

    assert await _run_reads(filename, content) == "text"


async def test_an_image_sent_under_another_name_is_admitted_by_its_bytes() -> None:
    require_readable_attachment("scan.bin", _OCTET, _png())

    assert await _run_reads("scan.bin", _png()) == "image"


@pytest.mark.parametrize(
    ("filename", "content", "label"),
    [
        ("draft.odt", _package(), ".odt"),
        ("novel.epub", _package(), ".epub"),
        ("notes.textpack", _package(), ".textpack"),
        ("archive.zip", _package(), ".zip"),
        ("model.bin", _BINARY, ".bin"),
        ("blob", _BINARY, _OCTET),
    ],
)
async def test_an_upload_no_run_reads_is_refused_naming_its_type(
    filename: str, content: bytes, label: str
) -> None:
    # No converter routes these and their bytes are neither an image nor text, so a
    # Run that admitted one could only fail when it read it.
    with pytest.raises(ResourceDecodeError):
        await _run_reads(filename, content)

    with pytest.raises(UnsupportedAttachmentTypeError) as refused:
        require_readable_attachment(filename, _OCTET, content)

    assert refused.value.error_kind == UNSUPPORTED_ATTACHMENT_TYPE
    assert refused.value.attachment_type == label
    assert refused.value.public_message.startswith(f"Attachment type {label} cannot be read")
    assert "pdf, pptx, properties" in refused.value.public_message


@pytest.mark.parametrize(
    ("filename", "declared_mime"),
    [
        ("chart.png", "image/png"),
        ("scan", "image/tiff"),
        ("report.PDF", None),
        ("report", "application/pdf"),
        ("notes.md", "application/pdf"),
        ("notes", "text/plain; charset=utf-8"),
        ("main.go", "text/x-go"),
        ("table.csv", _OCTET),
    ],
)
def test_a_type_a_run_reads_is_admitted_without_reading_its_bytes(
    filename: str, declared_mime: str | None
) -> None:
    # The type decides these, so even bytes that decode as nothing are admitted: a
    # listed type's bytes are read, and refused, when the Run reads them.
    require_readable_attachment(filename, declared_mime, _BINARY)


@pytest.mark.parametrize(
    ("filename", "declared_mime", "label"),
    [
        ("blob", None, None),
        ("blob", "not a media type", None),
        ("weird.<script>", None, None),
    ],
)
def test_an_unreadable_upload_is_named_without_echoing_what_the_caller_sent(
    filename: str, declared_mime: str | None, label: str | None
) -> None:
    with pytest.raises(UnsupportedAttachmentTypeError) as refused:
        require_readable_attachment(filename, declared_mime, _BINARY)

    assert refused.value.attachment_type == label
    assert refused.value.public_message.startswith("An attachment without a file type")
    assert "script" not in refused.value.public_message
