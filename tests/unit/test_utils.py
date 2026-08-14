# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared utility helpers."""

from dlightrag_ai.telemetry import safe_log_text


def test_safe_log_text_removes_line_breaks_and_bounds_length() -> None:
    assert safe_log_text("alpha\nbeta\rgamma", max_length=80) == "alpha\\nbeta\\rgamma"
    assert safe_log_text("abcdef", max_length=5) == "ab..."
