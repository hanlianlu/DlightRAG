# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer prompt composition."""

from __future__ import annotations

from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt


class TestGetAnswerSystemPrompt:
    def test_unified_prompt_contains_references_heading(self) -> None:
        """Unified prompt includes ### References format instructions."""
        prompt = get_answer_system_prompt()
        assert "### References" in prompt

    def test_unified_prompt_no_json_format_instruction(self) -> None:
        """Unified prompt must NOT contain JSON output instructions."""
        prompt = get_answer_system_prompt()
        # The prompt should not instruct JSON object output
        assert '"answer"' not in prompt
        assert '"references"' not in prompt

    def test_unified_prompt_contains_inline_citations(self) -> None:
        """Unified prompt includes inline citation instructions."""
        prompt = get_answer_system_prompt()
        assert "[n-m]" in prompt
        assert "inline" in prompt.lower()

    def test_unified_prompt_contains_example(self) -> None:
        """Unified prompt includes citation example."""
        prompt = get_answer_system_prompt()
        assert "[1-1]" in prompt
        assert "36.89%" in prompt

    def test_no_structured_parameter(self) -> None:
        """get_answer_system_prompt() takes no parameters."""
        # Just verify it works with zero args
        prompt = get_answer_system_prompt()
        assert len(prompt) > 0
