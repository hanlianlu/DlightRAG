# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Text processing utilities for LLM output."""

from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Extract JSON from LLM response text.

    Handles common wrapping patterns:
    - Markdown code fences (```json ... ```, ``` ... ```)
    - Preamble text before the JSON object
    - Trailing text after the JSON object

    Returns the extracted JSON string, or the original text if no JSON found.
    """
    # 1. Try markdown code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # 2. Try to find raw JSON object — match outermost braces
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]
    return text
