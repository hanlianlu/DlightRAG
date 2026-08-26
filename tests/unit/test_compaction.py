# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Automatic checkpoint compaction parsing and projection contracts."""

import pytest

from dlightrag.answer.agent.compaction import parse_compaction_summary


def test_compaction_summary_parser_preserves_typed_sections() -> None:
    summary = parse_compaction_summary(
        """## Goal
Ship the runtime.

## Constraints & Preferences
No generic workflow.

## Progress
Done: Runtime.

## Key Decisions
Total state.

## Next Steps
Review.

## Critical Context
Recovery uses the same interpreter.
"""
    )
    assert summary.goal == "Ship the runtime."
    assert summary.constraints_preferences == "No generic workflow."
    assert summary.progress == "Done: Runtime."
    assert summary.decisions == "Total state."
    assert summary.next_steps == "Review."
    assert summary.critical_context == "Recovery uses the same interpreter."


def test_compaction_summary_requires_a_goal() -> None:
    with pytest.raises(ValueError, match="goal"):
        parse_compaction_summary("## Progress\nNothing")


def test_unknown_compaction_sections_are_not_silently_dropped() -> None:
    summary = parse_compaction_summary("## Goal\nShip.\n\n## Extra\nKeep this.")
    assert "## Extra" in summary.critical_context
    assert "Keep this." in summary.critical_context
