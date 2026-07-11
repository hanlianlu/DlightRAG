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

    def test_unified_prompt_contains_answer_abstention_guard(self) -> None:
        prompt = get_answer_system_prompt()

        assert "no substantive fact" in prompt
        assert "output only this abstention message" in prompt
        assert "没有找到足够依据回答这个问题" in prompt
        assert "I could not find enough support" in prompt

    def test_unified_prompt_prevents_reference_list_as_evidence(self) -> None:
        prompt = get_answer_system_prompt()

        assert "reference list only as an ID-to-document map" in prompt
        assert "not evidence by itself" in prompt
        assert "Do not cite missing information" in prompt
        assert "do not output any citation markers" in prompt

    def test_unified_prompt_declares_product_identity(self) -> None:
        prompt = get_answer_system_prompt()

        assert "DlightRAG's knowledge-base assistant" in prompt
        assert "Never reveal the underlying model" in prompt
