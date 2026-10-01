# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for one shared conversation-history projection."""

from typing import Any

import pytest

from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
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


# Every call is measured against the product's own policy on one profile; a test
# names the history room it leaves by placing a call's fixed input just under
# the limit that call is accepted against.
_PROFILE = ModelProfile(context_window_tokens=200_000)
_HARD_LIMIT = CONTEXT_POLICY.hard_input_limit(_PROFILE)
_TRIGGER = CONTEXT_POLICY.compaction_trigger(_PROFILE)
_FULL_RESERVE_TRIGGER = CONTEXT_POLICY.compaction_trigger(
    _PROFILE, require_full_dynamic_reserve=True
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
    projected = project_history(
        _history(),
        targets=(
            HistoryProjectionTarget("planner", _PROFILE, _measure(0)),
            HistoryProjectionTarget("fast", _PROFILE, _measure(_HARD_LIMIT - 9)),
        ),
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
    assert projected.episodic_summary == ""


def test_zero_allowance_drops_even_the_generated_continuation() -> None:
    projected = project_history(
        _history(),
        targets=(HistoryProjectionTarget("planner", _PROFILE, _measure(_HARD_LIMIT)),),
    )

    assert projected.messages == []
    assert projected.episodic_summary == ""


def test_generated_summary_is_exactly_remeasured_in_remaining_residual() -> None:
    measure = _measure(_HARD_LIMIT - 16, pinned_summary="pin")

    projected = project_history(
        _history(),
        targets=(HistoryProjectionTarget("fast", _PROFILE, measure),),
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
    assert projected.episodic_summary
    assert measure(projected.messages, projected.episodic_summary) - measure([], "") <= 13


def test_new_fast_session_projects_external_history_to_compaction_trigger() -> None:
    history = [
        {"role": "user", "content": "u" * 30},
        {"role": "assistant", "content": "a" * 30},
    ]
    # Under the hard limit the pair fits; Fast keeps its full dynamic reserve and
    # leaves 52 tokens of history, too few for it.
    measure = _measure(_FULL_RESERVE_TRIGGER - 52)

    hard_limit_projection = project_history(
        history,
        targets=(HistoryProjectionTarget("generation", _PROFILE, measure),),
    )
    fast_projection = project_history(
        history,
        targets=(
            HistoryProjectionTarget(
                "fast_generation",
                _PROFILE,
                measure,
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
        ),
    )

    assert hard_limit_projection.messages == history
    assert fast_projection.messages == []
    assert fast_projection.episodic_summary
    assert measure([], fast_projection.episodic_summary) <= _FULL_RESERVE_TRIGGER


def test_research_seed_uses_compaction_trigger_as_acceptance_target() -> None:
    projected = project_history(
        _history(),
        targets=(
            HistoryProjectionTarget(
                "research_seed",
                _PROFILE,
                _measure(_TRIGGER - 9),
                proactive_compaction=True,
            ),
        ),
    )

    assert projected.messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]


def test_incremental_durable_projection_matches_sequence_beyond_100_turns() -> None:
    target = HistoryProjectionTarget("durable", _PROFILE, _measure(_HARD_LIMIT - 165))
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
    )
    projector = IncrementalHistoryProjector(targets=(target,))
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
    with pytest.raises(HistoryProjectionOverflowError) as caught:
        project_history(
            _history(),
            targets=(HistoryProjectionTarget("planner", _PROFILE, _measure(_HARD_LIMIT + 1)),),
        )

    assert caught.value.target == "planner"
    assert caught.value.fixed_input_tokens == _HARD_LIMIT + 1
    assert caught.value.acceptance_limit_tokens == _HARD_LIMIT
