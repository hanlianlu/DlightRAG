# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for one shared conversation-history projection."""

import pytest
from dlightrag_ai.capacity import ModelProfile

from dlightrag.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)


def _measure(fixed: int):
    def measure(messages: list[dict[str, object]]) -> int:
        return fixed + sum(len(str(message.get("content") or "")) for message in messages)

    return measure


def _history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "incomplete turn is dropped"},
    ]


def test_projector_keeps_newest_complete_pairs_within_every_call_allowance() -> None:
    profile = ModelProfile(context_window_tokens=100)
    projected = project_history(
        _history(),
        targets=(
            HistoryProjectionTarget("planner", profile, _measure(0)),
            HistoryProjectionTarget("fast", profile, _measure(76)),
        ),
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]


def test_zero_allowance_produces_empty_history() -> None:
    profile = ModelProfile(context_window_tokens=100)

    projected = project_history(
        _history(),
        targets=(HistoryProjectionTarget("planner", profile, _measure(85)),),
    )

    assert projected.messages == []


def test_research_seed_uses_compaction_trigger_as_acceptance_target() -> None:
    profile = ModelProfile(context_window_tokens=100)

    projected = project_history(
        _history(),
        targets=(
            HistoryProjectionTarget(
                "research_seed",
                profile,
                _measure(63),
                proactive_compaction=True,
            ),
        ),
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]


def test_fixed_envelope_overflow_names_the_failing_call() -> None:
    profile = ModelProfile(context_window_tokens=100)

    with pytest.raises(HistoryProjectionOverflowError) as caught:
        project_history(
            _history(),
            targets=(HistoryProjectionTarget("planner", profile, _measure(86)),),
        )

    assert caught.value.target == "planner"
    assert caught.value.fixed_input_tokens == 86
    assert caught.value.acceptance_limit_tokens == 85
