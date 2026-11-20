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


def test_a_merge_quotes_the_previous_summary_and_a_first_pass_has_none() -> None:
    first = compaction_user_prompt(previous_summary=None, transcript="ask then retrieve")
    merge = compaction_user_prompt(
        previous_summary="## Goal\nShip the runtime.",
        transcript="new exchange",
    )

    assert "ask then retrieve" in first
    assert "<previous-summary>" not in first
    assert "## Goal" not in first
    assert "new exchange" in merge
    assert "## Goal\nShip the runtime." in merge
    # The schema stays the system prompt's: the merge prompt quotes the previous summary only.
    assert merge.count("## Goal") == 1
