# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer prompt composition."""

from dlightrag.engine.answer.prompts import answer_core


def test_answer_system_prompt_omits_forbidden_clauses() -> None:
    """Prompt must NOT ask LLM to generate ### References (code-built) or JSON output."""
    prompt = answer_core()

    assert "### References" not in prompt
    assert '"answer"' not in prompt
    assert '"references"' not in prompt
