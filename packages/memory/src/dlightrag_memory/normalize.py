# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical body normalization for the exact-match recall leg.

One pure function shared by every storage adapter so equality semantics never
drift between backends: NFKC (fullwidth/halfwidth and compatibility forms),
casefold, and internal-whitespace collapse. Language-agnostic — CJK fullwidth
and Latin case both normalize here.
"""

from __future__ import annotations

import unicodedata

_WHITESPACE = " \t\r\n"


def normalized_body(text: str) -> str:
    """Return one canonical comparison key for a memory body."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    stripped = "".join(" " if char in _WHITESPACE else char for char in folded)
    return " ".join(stripped.split())


def single_spaced(text: str) -> str:
    """``text`` with every run of whitespace, line breaks included, made one space.

    Nothing else changes: not the case, not the width of a character. It is how a quote is held
    to what the owner wrote, which may differ from it in whitespace alone.
    """
    return " ".join(text.split())


__all__ = ["normalized_body", "single_spaced"]
