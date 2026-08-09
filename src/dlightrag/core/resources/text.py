# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Direct-text detection, strict decoding, and structural line windows.

Decoding order is deterministic: a UTF byte-order mark, then a validated declared
charset, then a conservative ``charset-normalizer`` guess. Every path decodes
strictly and rejects replacement-character recovery so binary, mismatched, or
undecodable payloads surface as errors instead of lossy text.
"""

from __future__ import annotations

from charset_normalizer import from_bytes

from dlightrag.core.answer.capacity import MAX_TOOL_OBSERVATION_TOKENS
from dlightrag.core.resources.models import ResourceDecodeError, TextWindowLocator
from dlightrag.utils.tokens import estimate_tokens

# Bytes that legitimately appear in decoded single-/multi-byte text. High bytes
# stay in the set so UTF-8 and Latin text are not misread as binary; the decoder
# below still rejects them if no charset decodes the payload strictly.
_TEXT_BYTES = bytes(range(0x20, 0x7F)) + b"\t\n\r\f\b\x1b" + bytes(range(0x80, 0x100))
_BINARY_SAMPLE = 8192
_BINARY_RATIO = 0.30


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


def build_text_windows(text: str) -> list[tuple[TextWindowLocator, str]]:
    """Split *text* into contiguous line windows within the observation budget."""
    lines = text.split("\n")
    windows: list[tuple[TextWindowLocator, str]] = []
    index = 0
    total = len(lines)
    while index < total:
        start = index
        window_tokens = 0
        while index < total:
            line_tokens = max(1, estimate_tokens(lines[index] + "\n"))
            if index > start and window_tokens + line_tokens > MAX_TOOL_OBSERVATION_TOKENS:
                break
            window_tokens += line_tokens
            index += 1
        locator = TextWindowLocator(unit="line", start=start + 1, end=index)
        windows.append((locator, "\n".join(lines[start:index])))
    return windows


__all__ = ["build_text_windows", "decode_text"]
