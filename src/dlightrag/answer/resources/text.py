# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Direct-text detection, strict decoding, and structural line windows.

Decoding order is deterministic: a UTF byte-order mark, then a validated declared
charset, then a conservative ``charset-normalizer`` guess. Every path decodes
strictly and rejects replacement-character recovery so binary, mismatched, or
undecodable payloads surface as errors instead of lossy text.
"""

from __future__ import annotations

from charset_normalizer import from_bytes

from dlightrag.ai.tokens import estimate_tokens
from dlightrag.answer.resources.models import ResourceDecodeError, TextWindowLocator

# Bytes that legitimately appear in decoded single-/multi-byte text. High bytes
# stay in the set so UTF-8 and Latin text are not misread as binary; the decoder
# below still rejects them if no charset decodes the payload strictly.
_TEXT_BYTES = bytes(range(0x20, 0x7F)) + b"\t\n\r\f\b\x1b" + bytes(range(0x80, 0x100))
_BINARY_SAMPLE = 8192
_BINARY_RATIO = 0.30

# The token estimator's loosest bucket is four characters per token, so a token
# budget can never span more than this many characters.
_MAX_CHARS_PER_TOKEN = 4


def decode_text(content: bytes, *, declared_charset: str | None) -> str:
    """Decode *content* to text or raise :class:`ResourceDecodeError`."""
    if not content:
        return ""

    detected = _detect_bom(content)
    if detected is not None:
        encoding, offset = detected
        return _strict_decode(content[offset:], encoding)

    if declared_charset:
        return _strict_decode(content, declared_charset)

    if _looks_binary(content):
        raise ResourceDecodeError("resource content is not decodable text")

    best = from_bytes(content).best()
    if best is None:
        raise ResourceDecodeError("resource text encoding could not be determined")
    return _strict_decode(content, str(best.encoding))


def _strict_decode(data: bytes, encoding: str) -> str:
    try:
        decoded = data.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise ResourceDecodeError(f"resource bytes are not valid {encoding}") from exc
    if "\ufffd" in decoded:
        raise ResourceDecodeError("resource text contains replacement characters")
    return decoded


def _detect_bom(content: bytes) -> tuple[str, int] | None:
    if content.startswith(b"\xef\xbb\xbf"):
        return ("utf-8-sig", 0)
    if content.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return ("utf-32", 0)
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ("utf-16", 0)
    return None


def _looks_binary(content: bytes) -> bool:
    sample = content[:_BINARY_SAMPLE]
    if b"\x00" in sample:
        return True
    nontext = sample.translate(None, _TEXT_BYTES)
    return len(nontext) / len(sample) > _BINARY_RATIO


def build_text_windows(
    text: str,
    *,
    max_window_tokens: int,
) -> list[tuple[TextWindowLocator, str]]:
    """Split *text* into windows within the observation budget.

    Each window's content is an exact contiguous slice of *text*; concatenating
    every window's content in order reconstructs *text* with no drop or
    duplication. Windows normally span whole lines. A single line larger than one
    observation budget is split into character sub-windows whose locators carry
    an explicit intra-line character span so the structural locator stays
    truthful.
    """
    if max_window_tokens < 1:
        raise ValueError("max_window_tokens must be positive")
    segments = text.splitlines(keepends=True)
    if not segments:
        return []

    windows: list[tuple[TextWindowLocator, str]] = []
    pending: list[str] = []
    pending_tokens = 0
    pending_start = 1

    def flush(end_line: int) -> None:
        nonlocal pending, pending_tokens
        if pending:
            windows.append(
                (
                    TextWindowLocator(unit="line", start=pending_start, end=end_line),
                    "".join(pending),
                )
            )
            pending = []
            pending_tokens = 0

    for offset, segment in enumerate(segments):
        line_no = offset + 1
        segment_tokens = max(1, estimate_tokens(segment))
        if segment_tokens > max_window_tokens:
            flush(line_no - 1)
            windows.extend(
                _split_oversized_line(
                    segment,
                    line_no,
                    max_window_tokens=max_window_tokens,
                )
            )
            continue
        if pending and pending_tokens + segment_tokens > max_window_tokens:
            flush(line_no - 1)
        if not pending:
            pending_start = line_no
        pending.append(segment)
        pending_tokens += segment_tokens
    flush(len(segments))
    return windows


def _split_oversized_line(
    line: str,
    line_no: int,
    *,
    max_window_tokens: int,
) -> list[tuple[TextWindowLocator, str]]:
    """Split one over-budget physical line into intra-line character windows."""
    windows: list[tuple[TextWindowLocator, str]] = []
    start = 0
    length = len(line)
    while start < length:
        span = _fit_char_span(line, start, max_window_tokens=max_window_tokens)
        end = start + span
        locator = TextWindowLocator(
            unit="line",
            start=line_no,
            end=line_no,
            char_start=start + 1,
            char_end=end,
        )
        windows.append((locator, line[start:end]))
        start = end
    return windows


def _fit_char_span(line: str, start: int, *, max_window_tokens: int) -> int:
    """Return the largest character count from *start* within the token budget.

    The estimator never decreases as characters are appended, so a bisection
    finds the longest prefix that fits; at least one character always advances.
    """
    remaining = len(line) - start
    high = min(remaining, max_window_tokens * _MAX_CHARS_PER_TOKEN)
    if estimate_tokens(line[start : start + high]) <= max_window_tokens:
        return high
    low = 1
    while low < high:
        midpoint = (low + high + 1) // 2
        if estimate_tokens(line[start : start + midpoint]) <= max_window_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    return low


__all__ = ["build_text_windows", "decode_text"]
