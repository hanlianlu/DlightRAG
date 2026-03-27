# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Text processing utilities for LLM output."""

from __future__ import annotations

import re

from json_repair import repair_json


def extract_json(text: str) -> str:
    """Extract JSON from LLM response text.

    Handles common wrapping patterns:
    - Markdown code fences (```json ... ```, ``` ... ```)
    - Preamble text before the JSON object
    - Trailing text after the JSON object
    - Unclosed braces / malformed JSON (via json-repair)

    Returns the extracted JSON string, or the original text if no JSON found.
    """
    # 1. Try markdown code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return str(repair_json(match.group(1).strip()))

    # 2. Try to find raw JSON object
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start != -1:
        return str(repair_json(text[start:]))

    return text
