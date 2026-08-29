# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for one shared conversation-history projection."""

from typing import Any

import pytest

from dlightrag.engine.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.engine.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    IncrementalHistoryProjector,
    project_history,
)


def _measure(fixed: int, *, pinned_summary: str = ""):
    def measure(messages: list[dict[str, Any]], projected_summary: str = "") -> int:
        return (
            fixed
            + len(pinned_summary)
            + len(projected_summary)
            + sum(len(str(message.get("content") or "")) for message in messages)
        )

    return measure


_POLICY = ContextPolicy(
    requested_output_reserve_tokens=0,
    dynamic_context_reserve_tokens=13,
    safety_reserve_tokens=15,
    minimum_input_tokens=0,
)


def _history() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "incomplete turn is dropped"},
    ]


def test_projector_keeps_newest_pairs_before_fitting_omitted_summary() -> None:
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
    assert projected.episodic_summary == ""


def test_zero_allowance_drops_even_the_generated_continuation() -> None:
    profile = ModelProfile(context_window_tokens=100)

    projected = project_history(
        _history(),
        targets=(HistoryProjectionTarget("planner", profile, _measure(85)),),
        context_policy=_POLICY,
    )

    assert projected.messages == []
    assert projected.episodic_summary == ""


def test_generated_summary_is_exactly_remeasured_in_remaining_residual() -> None:
    profile = ModelProfile(context_window_tokens=100)
    measure = _measure(69, pinned_summary="pin")

    projected = project_history(
        _history(),
        targets=(HistoryProjectionTarget("fast", profile, measure),),
        context_policy=_POLICY,
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
    assert projected.episodic_summary
    assert measure(projected.messages, projected.episodic_summary) - measure([], "") <= 13


def test_new_fast_session_projects_external_history_to_compaction_trigger() -> None:
    profile = ModelProfile(context_window_tokens=100)
    history = [
        {"role": "user", "content": "u" * 30},
        {"role": "assistant", "content": "a" * 30},
    ]

    hard_limit_projection = project_history(
        history,
        targets=(HistoryProjectionTarget("generation", profile, _measure(20)),),
        context_policy=_POLICY,
    )
    fast_projection = project_history(
        history,
        targets=(
            HistoryProjectionTarget(
                "fast_generation",
                profile,
                _measure(20),
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
        ),
        context_policy=_POLICY,
    )

    assert hard_limit_projection.messages == history
    assert fast_projection.messages == []
    assert fast_projection.episodic_summary
    assert _measure(20)([], fast_projection.episodic_summary) <= 72


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


def test_incremental_durable_projection_matches_sequence_beyond_100_turns() -> None:
    profile = ModelProfile(context_window_tokens=220)
    target = HistoryProjectionTarget("durable", profile, _measure(40))
    pairs = [
        (
            {"role": "user", "content": f"q{index}"},
            {"role": "assistant", "content": f"a{index}"},
        )
        for index in range(205)
    ]
    expected = project_history(
        [message for pair in pairs for message in pair],
        targets=(target,),
        context_policy=_POLICY,
    )
    projector = IncrementalHistoryProjector(targets=(target,), context_policy=_POLICY)
    retained = 0
    for pair in reversed(pairs):
        if not projector.offer_newest_pair(*pair):
            break
        retained += 1
    for pair in pairs[: len(pairs) - retained]:
        if not projector.offer_oldest_omitted_pair(*pair):
            break

    actual = projector.finish()
    assert actual.messages == expected.messages
    assert actual.episodic_summary == expected.episodic_summary
    assert retained < 205


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
