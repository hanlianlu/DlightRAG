# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The session goal is a compare-and-set state machine and nothing else (ADR 0060, 0077)."""

from __future__ import annotations

import pytest

from dlightrag.engine.answer.goal import (
    DEFAULT_MAX_ROUNDS,
    GOAL_INVALID_BLOCK_REASON,
    GOAL_INVALID_OBJECTIVE,
    GOAL_INVALID_ROUNDS,
    GOAL_INVALID_TRANSITION,
    GOAL_ROUNDS_EXHAUSTED,
    GOAL_STALE_REVISION,
    MAX_BLOCK_REASON_CHARS,
    MAX_OBJECTIVE_CHARS,
    Goal,
    GoalError,
    GoalPhase,
)


def _goal(**kwargs: object) -> Goal:
    objective = kwargs.pop("objective", "value Polestar before the sealed model is finished")
    return Goal.create(str(objective), **kwargs)  # type: ignore[arg-type]


def _refused(code: str, call: object) -> None:
    with pytest.raises(GoalError) as raised:
        call()  # type: ignore[operator]
    assert raised.value.code == code


def test_a_new_goal_is_active_at_revision_one_with_the_house_cap() -> None:
    goal = _goal()

    assert DEFAULT_MAX_ROUNDS == 16, "the house cap is the round budget a goal gets by default"
    assert goal.id.startswith("goal-")
    assert goal.revision == 1
    assert goal.phase is GoalPhase.ACTIVE
    assert goal.max_rounds == 16
    assert goal.rounds_started == 0
    assert goal.rounds_left == 16
    assert goal.blocked_reason is None


def test_the_goal_holds_no_process_activation_field() -> None:
    """Continuation is the durable phase; there is no armed flag to persist or recover (ADR 0077)."""
    assert set(Goal.__dataclass_fields__) == {
        "id",
        "revision",
        "objective",
        "phase",
        "max_rounds",
        "rounds_started",
        "blocked_reason",
    }


def test_an_objective_is_a_task_statement_not_a_document() -> None:
    _refused(GOAL_INVALID_OBJECTIVE, lambda: Goal.create("   "))
    _refused(GOAL_INVALID_OBJECTIVE, lambda: Goal.create("x" * (MAX_OBJECTIVE_CHARS + 1)))

    assert Goal.create("  value Polestar  ").objective == "value Polestar"


def test_the_round_cap_is_positive_and_has_no_product_ceiling() -> None:
    _refused(GOAL_INVALID_ROUNDS, lambda: Goal.create("objective", max_rounds=0))
    _refused(GOAL_INVALID_ROUNDS, lambda: Goal.create("objective", max_rounds=-1))

    assert Goal.create("objective", max_rounds=1_000).max_rounds == 1_000


@pytest.mark.parametrize(
    "call",
    [
        lambda goal: goal.edit(revision=99, objective="new"),
        lambda goal: goal.pause(revision=99),
        lambda goal: goal.resume(revision=99),
        lambda goal: goal.complete(revision=99),
        lambda goal: goal.block(revision=99, reason="condition"),
        lambda goal: goal.admit_round(revision=99),
    ],
    ids=["edit", "pause", "resume", "complete", "block", "admit_round"],
)
def test_every_mutation_is_a_compare_and_set(call: object) -> None:
    _refused(GOAL_STALE_REVISION, lambda: call(_goal()))  # type: ignore[operator]


def test_edit_changes_only_the_objective_and_the_cap() -> None:
    goal = _goal(max_rounds=4)

    edited = goal.edit(revision=goal.revision, objective="value Volvo", max_rounds=2)

    assert edited.revision == goal.revision + 1
    assert edited.objective == "value Volvo"
    assert edited.max_rounds == 2
    assert edited.phase is GoalPhase.ACTIVE
    assert edited.rounds_started == 0


def test_edit_needs_something_to_change_and_refuses_a_completed_goal() -> None:
    goal = _goal()
    _refused(GOAL_INVALID_OBJECTIVE, lambda: goal.edit(revision=goal.revision))

    completed = goal.complete(revision=goal.revision)
    with pytest.raises(GoalError) as raised:
        completed.edit(revision=completed.revision, objective="too late")
    assert raised.value.code == GOAL_INVALID_TRANSITION
    # A completed goal is replaceable; the refusal names the way forward, not a new conversation.
    assert "start a new goal with `/goal <objective>`" in str(raised.value)
    assert "new conversation" not in str(raised.value)


def test_pause_and_resume_move_between_two_phases() -> None:
    goal = _goal()

    paused = goal.pause(revision=goal.revision)
    assert paused.phase is GoalPhase.PAUSED

    resumed = paused.resume(revision=paused.revision)
    assert resumed.phase is GoalPhase.ACTIVE

    _refused(GOAL_INVALID_TRANSITION, lambda: paused.pause(revision=paused.revision))


def test_resume_clears_the_blocking_condition() -> None:
    goal = _goal()
    blocked = goal.block(revision=goal.revision, reason="the filings portal is unreachable")
    assert blocked.phase is GoalPhase.BLOCKED
    assert blocked.blocked_reason == "the filings portal is unreachable"

    resumed = blocked.resume(revision=blocked.revision)
    assert resumed.phase is GoalPhase.ACTIVE
    assert resumed.blocked_reason is None


def test_resume_refuses_a_goal_whose_rounds_are_all_started() -> None:
    goal = _goal(max_rounds=1)
    started, _ = goal.admit_round(revision=goal.revision)
    paused = started.pause(revision=started.revision)

    _refused(GOAL_ROUNDS_EXHAUSTED, lambda: paused.resume(revision=paused.revision))


def test_block_needs_a_concrete_condition_and_an_active_goal() -> None:
    goal = _goal()
    _refused(GOAL_INVALID_BLOCK_REASON, lambda: goal.block(revision=goal.revision, reason="  "))

    paused = goal.pause(revision=goal.revision)
    _refused(
        GOAL_INVALID_TRANSITION,
        lambda: paused.block(revision=paused.revision, reason="condition"),
    )


def test_complete_is_accepted_before_the_end_and_only_once() -> None:
    for goal in (
        _goal(),
        _goal().pause(revision=1),
        _goal().block(revision=1, reason="condition"),
    ):
        completed = goal.complete(revision=goal.revision)
        assert completed.phase is GoalPhase.COMPLETE
        _refused(
            GOAL_INVALID_TRANSITION,
            lambda done=completed: done.complete(revision=done.revision),
        )


def test_admit_round_numbers_the_rounds_and_stops_at_the_cap() -> None:
    goal = _goal(max_rounds=2)

    first, first_number = goal.admit_round(revision=goal.revision)
    assert first_number == 1
    assert first.rounds_started == 1
    assert first.rounds_left == 1

    second, second_number = first.admit_round(revision=first.revision)
    assert second_number == 2
    assert second.rounds_started == 2
    assert second.rounds_left == 0

    _refused(GOAL_ROUNDS_EXHAUSTED, lambda: second.admit_round(revision=second.revision))


def test_a_paused_goal_cannot_start_a_round() -> None:
    paused = _goal().pause(revision=1)

    _refused(GOAL_INVALID_TRANSITION, lambda: paused.admit_round(revision=paused.revision))


def test_a_goal_is_immutable() -> None:
    goal = _goal()

    with pytest.raises(AttributeError):
        goal.phase = GoalPhase.COMPLETE  # type: ignore[misc]


def test_edit_cannot_push_the_cap_below_the_rounds_already_started() -> None:
    goal = _goal(max_rounds=5)
    started = goal
    for _ in range(3):
        started, _ = started.admit_round(revision=started.revision)

    _refused(GOAL_INVALID_ROUNDS, lambda: started.edit(revision=started.revision, max_rounds=2))
    assert started.edit(revision=started.revision, max_rounds=3).max_rounds == 3


def test_release_round_gives_back_one_admitted_round() -> None:
    goal = _goal(max_rounds=4)
    first, _ = goal.admit_round(revision=goal.revision)
    second, _ = first.admit_round(revision=first.revision)

    released = second.release_round(revision=second.revision)

    assert released.rounds_started == 1
    assert released.revision == second.revision + 1
    _refused(GOAL_INVALID_TRANSITION, lambda: goal.release_round(revision=goal.revision))


def test_a_blocking_condition_is_bounded_like_the_objective() -> None:
    goal = _goal()

    _refused(
        GOAL_INVALID_BLOCK_REASON,
        lambda: goal.block(revision=goal.revision, reason="x" * (MAX_BLOCK_REASON_CHARS + 1)),
    )
