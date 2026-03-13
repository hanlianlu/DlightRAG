# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer prompt composition."""

from __future__ import annotations

from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt


class TestGetAnswerSystemPrompt:
    def test_freetext_contains_references_heading(self) -> None:
        prompt = get_answer_system_prompt(structured=False)
        assert "### References" in prompt
        assert "json" not in prompt.lower() or "JSON" not in prompt

    def test_structured_contains_json_instruction(self) -> None:
        prompt = get_answer_system_prompt(structured=True)
        assert "JSON" in prompt or "json" in prompt
        assert '"references"' in prompt
        assert "### References" not in prompt

    def test_both_contain_inline_citations(self) -> None:
        for structured in (True, False):
            prompt = get_answer_system_prompt(structured=structured)
            assert "[n-m]" in prompt
            assert "inline" in prompt.lower()

    def test_both_contain_example(self) -> None:
        for structured in (True, False):
            prompt = get_answer_system_prompt(structured=structured)
            assert "[1-1]" in prompt
            assert "36.89%" in prompt

    def test_backward_compat_default_is_freetext(self) -> None:
        prompt = get_answer_system_prompt()
        assert "### References" in prompt
