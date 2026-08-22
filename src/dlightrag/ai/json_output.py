# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""JSON extraction for model output."""

import re

from json_repair import repair_json


def extract_json(text: str) -> str:
    """Extract and repair JSON from common model response wrappers."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return str(repair_json(match.group(1).strip()))

    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start != -1:
        return str(repair_json(text[start:]))
    return text


__all__ = ["extract_json"]
