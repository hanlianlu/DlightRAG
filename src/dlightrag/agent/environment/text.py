# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Strict text decoding and line-ending preservation for path tools."""

from __future__ import annotations

from dataclasses import dataclass

_UTF16_LE = b"\xff\xfe"
_UTF16_BE = b"\xfe\xff"
_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True, slots=True)
class DecodedText:
    """File bytes decoded with the encoding and newline style to restore."""

    text: str
    encoding: str
    newline: str
    mixed_newlines: bool


def decode_workspace_text(data: bytes) -> DecodedText:
    """Decode UTF-8 or BOM-tagged UTF-16. Refuse charset guessing."""
    if data.startswith(_UTF16_LE):
        return _from_decoded(data.decode("utf-16-le").lstrip("\ufeff"), "utf-16-le")
    if data.startswith(_UTF16_BE):
        return _from_decoded(data.decode("utf-16-be").lstrip("\ufeff"), "utf-16-be")
    if data.startswith(_UTF8_BOM):
        return _from_decoded(data.decode("utf-8-sig"), "utf-8-sig")
    try:
        return _from_decoded(data.decode("utf-8"), "utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "file is not UTF-8 or BOM-tagged UTF-16; convert it before reading"
        ) from exc


def encode_workspace_text(decoded: DecodedText, text: str) -> bytes:
    """Restore the original encoding and newline style."""
    body = text
    if decoded.newline == "\r\n":
        body = text.replace("\n", "\r\n")
    elif decoded.newline == "\r":
        body = text.replace("\n", "\r")
    return body.encode(decoded.encoding)


def _from_decoded(text: str, encoding: str) -> DecodedText:
    crlf = text.count("\r\n")
    stripped = text.replace("\r\n", "")
    cr = stripped.count("\r")
    lf = stripped.count("\n")
    mixed = sum(1 for count in (crlf, cr, lf) if count > 0) > 1
    if crlf and not cr and not lf:
        newline = "\r\n"
        presented = text.replace("\r\n", "\n")
    elif cr and not crlf and not lf:
        newline = "\r"
        presented = text.replace("\r", "\n")
    else:
        newline = "\n"
        presented = text.replace("\r\n", "\n").replace("\r", "\n")
    return DecodedText(text=presented, encoding=encoding, newline=newline, mixed_newlines=mixed)


__all__ = ["DecodedText", "decode_workspace_text", "encode_workspace_text"]
