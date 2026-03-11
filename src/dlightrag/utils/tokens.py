# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight token estimation and conversation history truncation.

Provides a consistent token estimator used by both backend (Python) and
frontend (JS — see citation.js ``estimateTokens``). The two implementations
must stay in sync.

Three density buckets:
  - **Dense** (~1 token / 1.5 chars): CJK ideographs, Japanese kana, Korean hangul
  - **Latin-extended** (~1 token / 3 chars): accented Latin, Cyrillic, Greek, Thai,
    Arabic, Hebrew, Devanagari, and other non-ASCII scripts
  - **ASCII** (~1 token / 4 chars): basic Latin, digits, punctuation
"""

from __future__ import annotations

import math
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate the number of LLM tokens in *text*.

    Uses character-class heuristics — no tokenizer dependency.
    """
    ascii_chars = 0
    dense_chars = 0  # CJK + kana + hangul
    latin_ext_chars = 0  # everything else non-ASCII

    for ch in text:
        code = ord(ch)
        if code <= 0x7F:
            ascii_chars += 1
        elif (
            (0x2E80 <= code <= 0x9FFF)  # CJK radicals, ideographs
            or (0xF900 <= code <= 0xFAFF)  # CJK compatibility ideographs
            or (0xFE30 <= code <= 0xFE4F)  # CJK compatibility forms
            or (0x3040 <= code <= 0x30FF)  # Japanese hiragana + katakana
            or (0x31F0 <= code <= 0x31FF)  # Katakana phonetic extensions
            or (0xAC00 <= code <= 0xD7AF)  # Korean hangul syllables
        ):
            dense_chars += 1
        else:
            latin_ext_chars += 1

    return (
        math.ceil(ascii_chars / 4) + math.ceil(dense_chars / 1.5) + math.ceil(latin_ext_chars / 3)
    )


def truncate_conversation_history(
    history: list[dict[str, Any]],
    *,
    max_messages: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Return the most recent slice of *history* within budget.

    Walks backwards from the end and stops when either *max_messages* or
    *max_tokens* is exceeded.
    """
    if not history:
        return []

    if len(history) > max_messages:
        history = history[-max_messages:]

    total = 0
    cutoff = 0
    for i in range(len(history) - 1, -1, -1):
        total += estimate_tokens(history[i].get("content", ""))
        if total > max_tokens:
            cutoff = i + 1
            break

    return history[cutoff:] if cutoff else history
