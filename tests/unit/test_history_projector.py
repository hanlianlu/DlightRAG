# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for one shared conversation-history projection."""

import pytest

from dlightrag.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)


def _measure(fixed: int):
    def measure(messages: list[dict[str, object]]) -> int:
        return fixed + sum(len(str(message.get("content") or "")) for message in messages)

    return measure


_POLICY = ContextPolicy(
    requested_output_reserve_tokens=0,
    observation_reserve_tokens=13,
    safety_reserve_tokens=15,
    minimum_input_tokens=0,
)


def _history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "incomplete turn is dropped"},
    ]


def test_projector_keeps_newest_pairs_and_summarizes_omitted_history() -> None:
    profile = ModelProfile(context_window_tokens=100)
    projected = project_history(
        _history(),
        targets=(
            HistoryProjectionTarget("planner", profile, _measure(0)),
            HistoryProjectionTarget("fast", profile, _measure(76)),
        ),
        context_policy=_POLICY,
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
    assert "user: old" in projected.episodic_summary
    assert "assistant: reply" in projected.episodic_summary


def test_zero_allowance_produces_only_episodic_continuation() -> None:
    profile = ModelProfile(context_window_tokens=100)

    projected = project_history(
        _history(),
        targets=(HistoryProjectionTarget("planner", profile, _measure(85)),),
        context_policy=_POLICY,
    )

    assert projected.messages == []
    assert "new" in projected.episodic_summary


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
        context_policy=_POLICY,
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
            context_policy=_POLICY,
        )

    assert caught.value.target == "planner"
    assert caught.value.fixed_input_tokens == 86
    assert caught.value.acceptance_limit_tokens == 85
