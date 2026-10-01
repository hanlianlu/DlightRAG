# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for direct-text detection, decoding, and the windows a read pages through."""

import pytest

from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.resources.formatting import format_resource_read
from dlightrag.engine.answer.resources.models import (
    ResourceDecodeError,
    ResourceInput,
    ResourceReadResult,
    TextWindowLocator,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.text import decode_text

_WINDOW_TOKENS = 100


def test_decodes_utf8_bom() -> None:
    assert decode_text("café\nline".encode("utf-8-sig"), declared_charset=None) == "café\nline"


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_decodes_utf16_bom(encoding: str) -> None:
    raw = "hello wörld".encode(encoding)
    bom = b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff"
    assert decode_text(bom + raw, declared_charset=None) == "hello wörld"


def test_decodes_utf32_bom() -> None:
    assert decode_text("z".encode("utf-32"), declared_charset=None) == "z"


def test_uses_declared_charset_without_bom() -> None:
    raw = "café crème".encode("iso-8859-1")
    assert decode_text(raw, declared_charset="iso-8859-1") == "café crème"


def test_declared_utf16_le_without_bom() -> None:
    raw = "hi wörld".encode("utf-16-le")
    assert decode_text(raw, declared_charset="utf-16-le") == "hi wörld"


def test_uses_charset_normalizer_fallback() -> None:
    text = "The quick brown fox jumped over the lazy dog. héllo wörld café. " * 8
    assert decode_text(text.encode("utf-8"), declared_charset=None) == text


@pytest.mark.parametrize(
    "text",
    [
        "# Title\n\nSome *markdown* body with a [link](https://example.com).",
        "plain text without any structure at all",
        '{"key": "value", "n": 1}',
        '{"a": 1}\n{"a": 2}\n{"a": 3}',
        "root:\n  child: value\n  list:\n    - one\n    - two",
        "<root><child>value</child></root>",
        'title = "demo"\n[owner]\nname = "sam"',
        "[section]\nkey = value\nother = 2",
        "app.name=demo\napp.port=8080",
        "2026-08-09 12:00:00 INFO started\n2026-08-09 12:00:01 WARN slow",
        "def main() -> int:\n    return 0\n",
    ],
)
def test_decodes_textual_formats(text: str) -> None:
    assert decode_text(text.encode("utf-8"), declared_charset=None) == text


def test_rejects_binary_disguised_as_txt() -> None:
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    with pytest.raises(ResourceDecodeError):
        decode_text(png, declared_charset=None)


def test_rejects_invalid_declared_utf8() -> None:
    with pytest.raises(ResourceDecodeError):
        decode_text(b"\xc3\x28 broken", declared_charset="utf-8")


def test_rejects_mismatched_declared_charset() -> None:
    # Latin-1 bytes claiming to be UTF-8 do not decode strictly.
    with pytest.raises(ResourceDecodeError):
        decode_text("café".encode("iso-8859-1"), declared_charset="utf-8")


def test_empty_content_decodes_to_empty_string() -> None:
    assert decode_text(b"", declared_charset=None) == ""


async def _pages(text: str) -> list[ResourceReadResult]:
    """Every page a model reads of ``text``, following each cursor to the end."""
    async with ResourceRegistry() as registry:
        resource_id = registry.register(ResourceInput(filename="notes.txt", content=text.encode()))
        page = await registry.read(resource_id, max_window_tokens=_WINDOW_TOKENS)
        pages = [page]
        while page.next_cursor is not None:
            page = await registry.read(
                resource_id, cursor=page.next_cursor, max_window_tokens=_WINDOW_TOKENS
            )
            pages.append(page)
    return pages


def _locators(pages: list[ResourceReadResult]) -> list[TextWindowLocator]:
    locators = [page.locator for page in pages]
    assert all(locator is not None for locator in locators)
    return [locator for locator in locators if locator is not None]


async def test_pages_above_one_window_name_contiguous_whole_lines() -> None:
    lines = [f"line {index} " + "x" * 30 for index in range(200)]
    text = "\n".join(lines)

    pages = await _pages(text)

    assert len(pages) >= 2
    # What the model is shown of each page fits the window, and the pages
    # together are the text with nothing dropped or repeated.
    assert all(estimate_tokens(format_resource_read(page)) <= _WINDOW_TOKENS for page in pages)
    assert "".join(page.content for page in pages) == text
    locators = _locators(pages)
    assert locators[0].start == 1
    assert locators[-1].end == len(lines)
    for previous, current in zip(locators, locators[1:], strict=False):
        assert current.start == previous.end + 1
    assert all(locator.char_start is None for locator in locators)


async def test_a_line_longer_than_a_page_reads_as_character_spans_of_that_line() -> None:
    # One physical line (no newline) far larger than a single observation budget.
    line = "x" * (_WINDOW_TOKENS * 8)

    pages = await _pages(line)

    assert len(pages) >= 2
    assert all(estimate_tokens(format_resource_read(page)) <= _WINDOW_TOKENS for page in pages)
    assert "".join(page.content for page in pages) == line
    # Every page names the one line and the characters of it that it holds.
    locators = _locators(pages)
    assert {(locator.start, locator.end) for locator in locators} == {(1, 1)}
    assert locators[0].char_start == 1
    assert locators[-1].char_end == len(line)
    for previous, current in zip(locators, locators[1:], strict=False):
        assert previous.char_end is not None
        assert current.char_start == previous.char_end + 1
