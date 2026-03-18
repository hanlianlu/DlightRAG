# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified mode structural context propagation."""

from __future__ import annotations


class TestStructuralContextPrompt:
    """Verify the prompt constant exists and has expected content."""

    def test_prompt_exists_and_mentions_json(self) -> None:
        from dlightrag.unifiedrepresent.prompts import STRUCTURAL_CONTEXT_PROMPT

        assert "update" in STRUCTURAL_CONTEXT_PROMPT.lower()
        assert "keep" in STRUCTURAL_CONTEXT_PROMPT.lower()
        assert "json" in STRUCTURAL_CONTEXT_PROMPT.lower()
