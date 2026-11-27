# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The pure rules of a due Schedule: collapsing missed slots, grace, streaks, the harness line."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dlightrag.application.access.principal import UserContext
from dlightrag.application.schedules.cadence import Cadence, CronExpression
from dlightrag.application.schedules.errors import ScheduleInputError, ScheduleStateError
from dlightrag.application.schedules.model import (
    MANUAL_RETRY_FLOOR,
    MAX_GRACE,
    MISSED_SCAN_CAP,
    FireState,
    Schedule,
    ScheduleState,
    Target,
    failure_streak_reached,
    plan_due,
    retry_window,
    scheduled_turn_text,
)

NOW = datetime(2026, 12, 7, 12, 0, tzinfo=UTC)
USER = UserContext(user_id="u1", auth_mode="jwt", claims={"iss": "https://issuer"})


def _cadence(cron: str, zone: str = "UTC") -> Cadence:
    return Cadence(zone, CronExpression.parse(cron), None)


def _at(moment: datetime) -> Cadence:
    return Cadence("UTC", None, moment)


def _schedule(cadence: Cadence | None = None) -> Schedule:
    return Schedule.create(
        owner_id="owner",
        principal=USER,
        name="Weekly review",
        prompt="Summarize the week.",
        mode="research",
        workspaces=["alpha"],
        target=Target.NEW_CONVERSATION,
        cadence=cadence or _cadence("0 * * * *"),
        max_tokens_per_day=1_000,
        now=NOW,
    )


# -- plan_due ---------------------------------------------------------------------------------


def test_the_one_due_slot_inside_its_grace_is_started() -> None:
    cadence = _cadence("0 * * * *")
    due = datetime(2026, 12, 7, 11, 0, tzinfo=UTC)

    plan = plan_due(cadence, due, now=due + timedelta(seconds=20))

    assert [(fire.slot, fire.state) for fire in plan.fires] == [(due, FireState.PENDING)]
    assert plan.next_fire_at == datetime(2026, 12, 7, 12, 0, tzinfo=UTC)


def test_a_slot_past_half_its_interval_is_missed_and_starts_nothing() -> None:
    """An hourly slot is worth starting for 30 minutes; at 31 it is one `missed` Fire."""
    cadence = _cadence("0 * * * *")
    due = datetime(2026, 12, 7, 11, 0, tzinfo=UTC)

    on_time = plan_due(cadence, due, now=due + timedelta(minutes=30))
    late = plan_due(cadence, due, now=due + timedelta(minutes=31))

    assert [fire.state for fire in on_time.fires] == [FireState.PENDING]
    assert [fire.state for fire in late.fires] == [FireState.MISSED]
    assert late.fires[0].missed_count == 1
    assert late.fires[0].slot == late.fires[0].last_slot == due
    assert late.next_fire_at == datetime(2026, 12, 7, 12, 0, tzinfo=UTC)


def test_the_grace_of_a_sparse_recurrence_is_capped_at_six_hours() -> None:
    cadence = _cadence("0 8 * * *")
    due = datetime(2026, 12, 7, 8, 0, tzinfo=UTC)

    assert plan_due(cadence, due, now=due + MAX_GRACE).fires[0].state is FireState.PENDING
    assert (
        plan_due(cadence, due, now=due + MAX_GRACE + timedelta(minutes=1)).fires[0].state
        is FireState.MISSED
    )


def test_many_missed_slots_collapse_into_one_gap_and_one_late_fire() -> None:
    """The deployment was down for four hourly slots; only the latest runs, if it is in grace."""
    cadence = _cadence("0 * * * *")
    first = datetime(2026, 12, 7, 8, 0, tzinfo=UTC)

    plan = plan_due(cadence, first, now=datetime(2026, 12, 7, 12, 10, tzinfo=UTC))

    gap, late = plan.fires
    assert (gap.state, gap.slot, gap.missed_count) == (FireState.MISSED, first, 4)
    assert gap.last_slot == datetime(2026, 12, 7, 11, 0, tzinfo=UTC)
    assert (late.state, late.slot) == (FireState.PENDING, datetime(2026, 12, 7, 12, 0, tzinfo=UTC))
    assert plan.next_fire_at == datetime(2026, 12, 7, 13, 0, tzinfo=UTC)


def test_when_the_latest_missed_slot_is_too_late_every_slot_is_one_gap() -> None:
    cadence = _cadence("0 * * * *")
    first = datetime(2026, 12, 7, 8, 0, tzinfo=UTC)

    plan = plan_due(cadence, first, now=datetime(2026, 12, 7, 12, 40, tzinfo=UTC))

    (gap,) = plan.fires
    assert (gap.state, gap.slot, gap.missed_count) == (FireState.MISSED, first, 5)
    assert gap.last_slot == datetime(2026, 12, 7, 12, 0, tzinfo=UTC)
    assert plan.next_fire_at == datetime(2026, 12, 7, 13, 0, tzinfo=UTC)


def test_an_outage_longer_than_the_scan_cap_is_one_gap_and_never_a_late_fire() -> None:
    cadence = _cadence("0 * * * *")
    first = datetime(2024, 1, 1, tzinfo=UTC)

    plan = plan_due(cadence, first, now=datetime(2026, 12, 7, 12, 0, 20, tzinfo=UTC))

    (gap,) = plan.fires
    assert gap.state is FireState.MISSED
    assert gap.missed_count == MISSED_SCAN_CAP
    assert "at least" in (gap.reason or "")
    assert plan.next_fire_at == datetime(2026, 12, 7, 13, 0, tzinfo=UTC)


def test_a_single_time_starts_within_an_hour_and_is_missed_after() -> None:
    due = datetime(2026, 12, 7, 11, 0, tzinfo=UTC)
    cadence = _at(due)

    on_time = plan_due(cadence, due, now=due + timedelta(hours=1))
    late = plan_due(cadence, due, now=due + timedelta(hours=1, minutes=1))

    assert [fire.state for fire in on_time.fires] == [FireState.PENDING]
    assert on_time.next_fire_at is None
    assert [fire.state for fire in late.fires] == [FireState.MISSED]
    assert late.next_fire_at is None


def test_an_unstarted_fire_is_worth_retrying_for_the_grace_of_its_slot() -> None:
    slot = datetime(2026, 12, 7, 11, 0, tzinfo=UTC)
    assert retry_window(_cadence("0 * * * *"), slot) == timedelta(minutes=30)
    assert retry_window(_cadence("0 8 * * *"), slot) == MAX_GRACE
    assert retry_window(_at(slot), slot) == timedelta(hours=1)


# -- streaks ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcomes", "tripped"),
    [
        ([FireState.FAILED, FireState.FAILED, FireState.FAILED], True),
        ([FireState.REFUSED, FireState.FAILED, FireState.REFUSED], True),
        ([FireState.FAILED, FireState.FAILED, FireState.SUCCEEDED], False),
        ([FireState.FAILED, FireState.CANCELLED, FireState.FAILED], False),
        ([FireState.FAILED, FireState.FAILED], False),
        ([FireState.FAILED, FireState.FAILED, FireState.FAILED, FireState.SUCCEEDED], True),
    ],
)
def test_a_streak_is_the_newest_fires_that_ran_or_were_refused_all_ending_badly(
    outcomes: list[FireState], tripped: bool
) -> None:
    assert failure_streak_reached(outcomes, 3) is tripped


# -- the turn the Agent reads -----------------------------------------------------------------


def test_the_turn_is_one_harness_line_then_the_owners_prompt() -> None:
    cadence = _cadence("0 8 * * 1", "Europe/Stockholm")
    text = scheduled_turn_text(
        name='Weekly "review"\n of  things',
        slot=datetime(2026, 12, 7, 7, 0, tzinfo=UTC),
        cadence=cadence,
        prompt="Summarize the week.",
    )

    first, blank, rest = text.split("\n", 2)
    assert first == "[Scheduled: \"Weekly 'review' of things\", 2026-12-07 08:00 Europe/Stockholm]"
    assert blank == ""
    assert rest == "Summarize the week."


# -- the Schedule itself ----------------------------------------------------------------------


def test_a_new_schedule_waits_for_the_first_slot_after_now() -> None:
    schedule = _schedule(_cadence("0 * * * *"))
    assert schedule.state is ScheduleState.ACTIVE
    assert schedule.next_fire_at == datetime(2026, 12, 7, 13, 0, tzinfo=UTC)
    assert schedule.owner_id == "owner"
    assert schedule.principal == USER


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "  "},
        {"name": "x" * 121},
        {"prompt": " "},
        {"prompt": "x" * 20_001},
        {"mode": "auto"},
        {"workspaces": []},
        {"max_tokens_per_day": 0},
        {"goal_max_rounds": 0},
        {"goal_max_rounds": 2, "mode": "fast"},
        {"goal_max_rounds": 2, "prompt": "x" * 2_001},
    ],
)
def test_a_schedule_with_a_bad_part_is_refused(changes: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "owner_id": "owner",
        "principal": USER,
        "name": "Weekly review",
        "prompt": "Summarize.",
        "mode": "research",
        "workspaces": ["alpha"],
        "target": Target.NEW_CONVERSATION,
        "cadence": _cadence("0 * * * *"),
        "max_tokens_per_day": 10,
        "now": NOW,
    }
    with pytest.raises(ScheduleInputError):
        Schedule.create(**{**arguments, **changes})


def test_editing_a_cadence_recomputes_the_next_slot_and_editing_the_rest_does_not() -> None:
    schedule = _schedule(_cadence("0 * * * *"))
    later = NOW + timedelta(minutes=5)
    other = UserContext(user_id="u2", auth_mode="jwt", claims={})

    renamed = schedule.edit(principal=other, now=later, name="Daily", workspaces=["a", "a", "b"])
    assert renamed.name == "Daily"
    assert renamed.workspaces == ("a", "b")
    assert renamed.principal == other
    assert renamed.next_fire_at == schedule.next_fire_at

    moved = schedule.edit(principal=USER, now=later, cadence=_cadence("30 9 * * *"))
    assert moved.next_fire_at == datetime(2026, 12, 8, 9, 30, tzinfo=UTC)


def test_a_completed_schedule_takes_no_edit() -> None:
    done = replace(_schedule(), state=ScheduleState.COMPLETED, next_fire_at=None)
    with pytest.raises(ScheduleStateError):
        done.edit(principal=USER, now=NOW, name="again")


def _goal_schedule(rounds: int = 4) -> Schedule:
    return Schedule.create(
        owner_id="owner",
        principal=USER,
        name="Goal review",
        prompt="Finish the review.",
        mode="research",
        workspaces=["alpha"],
        target=Target.SAME_CONVERSATION,
        cadence=_cadence("0 * * * *"),
        max_tokens_per_day=1_000,
        now=NOW,
        goal_max_rounds=rounds,
    )


def test_a_goal_schedule_keeps_its_cap_and_edits_it_under_the_goals_own_rules() -> None:
    goal = _goal_schedule()
    assert goal.goal_max_rounds == 4
    assert _schedule().goal_max_rounds is None

    assert goal.edit(principal=USER, now=NOW, goal_max_rounds=9).goal_max_rounds == 9
    assert goal.edit(principal=USER, now=NOW, name="Other").goal_max_rounds == 4
    with pytest.raises(ScheduleInputError, match="positive integer"):
        goal.edit(principal=USER, now=NOW, goal_max_rounds=0)
    with pytest.raises(ScheduleInputError, match="research mode"):
        goal.edit(principal=USER, now=NOW, mode="fast")
    with pytest.raises(ScheduleInputError, match="at most 2000 characters"):
        goal.edit(principal=USER, now=NOW, prompt="x" * 2_001)


def test_a_schedule_that_starts_no_goal_has_no_cap_to_edit() -> None:
    with pytest.raises(ScheduleInputError, match="does not start a goal"):
        _schedule().edit(principal=USER, now=NOW, goal_max_rounds=3)


def test_a_manual_fire_is_worth_at_least_five_minutes_whatever_the_cadence_says() -> None:
    slot = datetime(2026, 12, 7, 11, 58, tzinfo=UTC)  # two minutes before the next hourly slot
    hourly = _cadence("0 * * * *")
    assert retry_window(hourly, slot) == timedelta(minutes=1)
    assert retry_window(hourly, slot, manual=True) == MANUAL_RETRY_FLOOR
    assert retry_window(_cadence("0 8 * * *"), slot, manual=True) == MAX_GRACE


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "a\x00b"},
        {"prompt": "a\x00b"},
        {"workspaces": ["alpha\x00"]},
        {"max_tokens_per_day": 2**63},
        {"goal_max_rounds": 2**63},
    ],
)
def test_a_schedule_a_database_cannot_hold_is_refused_before_it_is_stored(
    changes: dict[str, Any],
) -> None:
    arguments: dict[str, Any] = {
        "owner_id": "owner",
        "principal": USER,
        "name": "Weekly review",
        "prompt": "Summarize.",
        "mode": "research",
        "workspaces": ["alpha"],
        "target": Target.NEW_CONVERSATION,
        "cadence": _cadence("0 * * * *"),
        "max_tokens_per_day": 10,
        "now": NOW,
    }
    with pytest.raises(ScheduleInputError):
        Schedule.create(**{**arguments, **changes})


def test_the_largest_ceiling_a_column_holds_is_allowed() -> None:
    edited = _schedule().edit(principal=USER, now=NOW, max_tokens_per_day=2**63 - 1)
    assert edited.max_tokens_per_day == 2**63 - 1


def test_an_edit_to_the_same_cadence_keeps_the_next_slot_and_restamps_the_identity() -> None:
    schedule = _schedule(_cadence("0 * * * *"))
    later = NOW + timedelta(minutes=45)

    unchanged = schedule.edit(principal=USER, now=later, cadence=_cadence("0 * * * *"))

    assert unchanged.next_fire_at == schedule.next_fire_at
    assert unchanged.principal_stamped_at == later
    assert schedule.principal_stamped_at == NOW
