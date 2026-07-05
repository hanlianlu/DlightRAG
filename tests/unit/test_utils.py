# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared utility helpers."""

from dlightrag.utils import log_safe


def test_log_safe_removes_line_breaks_and_bounds_length() -> None:
    assert log_safe("alpha\nbeta\rgamma", max_length=80) == "alpha\\nbeta\\rgamma"
    assert log_safe("abcdef", max_length=5) == "ab..."
