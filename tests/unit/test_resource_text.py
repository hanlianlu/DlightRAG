# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for direct-text detection, decoding, and structural windows."""

import pytest

from dlightrag.engine.answer.resources.models import ResourceDecodeError, TextWindowLocator
from dlightrag.engine.answer.resources.text import build_text_windows, decode_text

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


def test_single_window_has_structural_line_locator() -> None:
    windows = build_text_windows("alpha\nbeta\ngamma", max_window_tokens=_WINDOW_TOKENS)

    assert len(windows) == 1
    locator, content = windows[0]
    assert locator == TextWindowLocator(unit="line", start=1, end=3)
    assert content == "alpha\nbeta\ngamma"


def test_windows_split_above_observation_budget() -> None:
    lines = [f"line {index} " + "x" * 30 for index in range(2000)]
    text = "\n".join(lines)

    windows = build_text_windows(text, max_window_tokens=_WINDOW_TOKENS)

    assert len(windows) >= 2
    # Every window stays within the per-observation token budget.
    from dlightrag.engine.ai.tokens import estimate_tokens

    for _, content in windows:
        assert estimate_tokens(content) <= _WINDOW_TOKENS
    # Windows are contiguous and cover every line exactly once.
    assert windows[0][0].start == 1
    assert windows[-1][0].end == len(lines)
    rebuilt = "".join(content for _, content in windows)
    assert rebuilt == text


def test_single_line_over_budget_splits_into_subline_windows() -> None:
    from dlightrag.engine.ai.tokens import estimate_tokens

    # One physical line (no newline) far larger than a single observation budget.
    line = "x" * (_WINDOW_TOKENS * 8)

    windows = build_text_windows(line, max_window_tokens=_WINDOW_TOKENS)

    assert len(windows) >= 2
    for _, content in windows:
        assert estimate_tokens(content) <= _WINDOW_TOKENS
    # Character sub-windows reconstruct the original line with no drop/duplication.
    assert "".join(content for _, content in windows) == line
    # Locators stay truthful: every sub-window lives on the same single line and
    # carries an explicit intra-line character span covering the whole line.
    first_locator = windows[0][0]
    assert first_locator.unit == "line"
    assert first_locator.start == 1
    assert first_locator.end == 1
    assert first_locator.char_start == 1
    assert windows[-1][0].char_end == len(line)
    spans = [(loc.char_start, loc.char_end) for loc, _ in windows]
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert prev_end is not None and next_start is not None
        assert next_start == prev_end + 1
