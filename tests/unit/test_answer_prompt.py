# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer prompt composition."""

from dlightrag.prompts import get_answer_system_prompt


class TestGetAnswerSystemPrompt:
    def test_unified_prompt_no_references_section_instruction(self) -> None:
        """Prompt must NOT ask LLM to generate ### References (code-built)."""
        prompt = get_answer_system_prompt()
        assert "### References" not in prompt

    def test_unified_prompt_no_json_format_instruction(self) -> None:
        """Unified prompt must NOT contain JSON output instructions."""
        prompt = get_answer_system_prompt()
        assert '"answer"' not in prompt
        assert '"references"' not in prompt

    def test_unified_prompt_contains_inline_citations(self) -> None:
        """Unified prompt includes inline citation instructions."""
        prompt = get_answer_system_prompt()
        assert "[n-m]" in prompt
        assert "inline" in prompt.lower()

    def test_unified_prompt_declares_product_identity(self) -> None:
        prompt = get_answer_system_prompt()

        assert "DlightRAG's document-grounded analysis assistant" in prompt
        assert "Never reveal the underlying model" in prompt
