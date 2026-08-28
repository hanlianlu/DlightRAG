# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight token estimation and conversation history truncation.

Provides a consistent backend token estimator for planner history truncation.

Three density buckets:
  - **Dense** (~1 token / 1.5 chars): CJK ideographs, Japanese kana, Korean hangul
  - **Latin-extended** (~1 token / 3 chars): accented Latin, Cyrillic, Greek, Thai,
    Arabic, Hebrew, Devanagari, and other non-ASCII scripts
  - **ASCII** (~1 token / 4 chars): basic Latin, digits, punctuation
"""

import json
import math
import re
from typing import Any

# Dense-bucket class and its complement. The CJK range already contains kana and
# the katakana extensions, so those need no separate clause.
_DENSE = r"\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\ufe30-\ufe4f"
_DENSE_RE = re.compile(f"[{_DENSE}]")
_NON_DENSE_RE = re.compile(f"[^{_DENSE}]")

# Loosest bucket is 4 chars per token, so a budget can never span more chars.
_MAX_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the number of LLM tokens in *text*.

    Uses character-class heuristics — no tokenizer dependency.
    """
    total = len(text)
    if not total:
        return 0
    if text.isascii():
        return math.ceil(total / 4)

    ascii_chars = len(text.encode("ascii", "ignore"))
    non_ascii = total - ascii_chars
    # A substitution costs what it deletes, so strip whichever class is smaller.
    if non_ascii * 2 < total:
        dense_chars = total - len(_DENSE_RE.sub("", text))
    else:
        dense_chars = len(_NON_DENSE_RE.sub("", text))

    return (
        math.ceil(ascii_chars / 4)
        + math.ceil(dense_chars / 1.5)
        + math.ceil((non_ascii - dense_chars) / 3)
    )


def truncate_to_estimated_tokens(text: str, token_budget: int) -> str:
    """Return the longest text prefix within the estimator-based token budget."""
    if token_budget <= 0:
        return ""
    if estimate_tokens(text) <= token_budget:
        return text.strip()
    low = 0
    high = min(len(text), token_budget * _MAX_CHARS_PER_TOKEN)
    while low < high:
        midpoint = (low + high + 1) // 2
        if estimate_tokens(text[:midpoint]) <= token_budget:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low].rstrip()


def estimate_content_tokens(content: Any) -> int:
    """Estimate tokens for one message content string or multimodal block list."""
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += estimate_tokens(block)
            elif isinstance(block, dict):
                text = str(block.get("text", ""))
                total += estimate_tokens(text)
                # Image token accounting is provider/model-specific. Without a
                # resolved model fact, count/byte budgets govern image blocks and
                # this text estimator deliberately adds no guessed token charge.
            # Skip unknown types
        return total
    return estimate_tokens(str(content))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate a rendered chat request, including per-message framing."""
    total = 2  # assistant priming / provider framing reserve
    for message in messages:
        total += 4
        total += estimate_tokens(str(message.get("role") or ""))
        total += estimate_content_tokens(message.get("content", ""))
        # Tool calls and replayed provider reasoning are sent too, and the reasoning
        # of a thinking model dwarfs the text beside it.
        for key in ("tool_calls", "provider_state"):
            payload = message.get(key)
            if payload:
                total += estimate_tokens(json.dumps(payload, ensure_ascii=False, default=str))
    return total
