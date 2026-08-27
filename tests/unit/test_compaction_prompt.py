# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Compaction prompts keep one parsed shape and a distinct merge contract."""

from dlightrag.engine.answer.prompts.compaction import (
    COMPACTION_SYSTEM_PROMPT,
    compaction_user_prompt,
)

_HEADINGS = (
    "## Goal",
    "## Constraints & Preferences",
    "## Progress",
    "### Done",
    "### In Progress",
    "### Blocked",
    "## Key Decisions",
    "## Next Steps",
    "## Critical Context",
)


def test_system_prompt_owns_the_heading_schema_once() -> None:
    for heading in _HEADINGS:
        assert COMPACTION_SYSTEM_PROMPT.count(heading) == 1
    assert "Do NOT continue the conversation" in COMPACTION_SYSTEM_PROMPT
    assert "Output ONLY the structured summary" in COMPACTION_SYSTEM_PROMPT


def test_first_pass_user_prompt_does_not_carry_merge_rules() -> None:
    prompt = compaction_user_prompt(previous_summary=None, transcript="ask then retrieve")

    assert "<transcript>" in prompt
    assert "ask then retrieve" in prompt
    assert "<previous-summary>" not in prompt
    assert "PRESERVE every still-relevant fact" not in prompt
    assert "UPDATE Progress" not in prompt
    assert prompt.count("## Goal") == 0


def test_merge_user_prompt_folds_without_forking_the_schema() -> None:
    prompt = compaction_user_prompt(
        previous_summary="## Goal\nShip the runtime.",
        transcript="new exchange",
    )

    assert "<previous-summary>" in prompt
    assert "## Goal\nShip the runtime." in prompt
    assert "PRESERVE every still-relevant fact" in prompt
    assert "ADD progress, decisions, and context" in prompt
    assert "move finished In Progress items into Done" in prompt
    assert "UPDATE Next Steps" in prompt
    assert "Drop blockers that the new transcript resolved" in prompt
    assert "no longer needed to continue" in prompt
    assert prompt.count("## Goal") == 1
