# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What an owner does with a Schedule, and that nobody else can (ADR 0103 Decision 12)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.config.sections import SchedulesSettings
from dlightrag.application.schedules.cadence import CadenceError
from dlightrag.application.schedules.errors import (
    ScheduleInputError,
    ScheduleLimitError,
    ScheduleStateError,
)
from dlightrag.application.schedules.model import FireState, ScheduleState, Target
from dlightrag.application.schedules.service import ScheduleService
from tests.support.schedules import Clock, FakeScheduleStore

NOW = datetime(2026, 12, 7, 10, 0, 20, tzinfo=UTC)
DAVE = UserContext(user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer"})
ERIN = UserContext(user_id="erin", auth_mode="jwt", claims={"iss": "https://issuer"})
DAVE_AGAIN = UserContext(user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer", "x": 1})


def _service(**settings: Any) -> tuple[ScheduleService, FakeScheduleStore, Clock]:
    clock = Clock(NOW)
    store = FakeScheduleStore(clock)
    service = ScheduleService(
        store=store,
        firer=cast(Any, None),
        settings=SchedulesSettings(**settings),
        clock=clock,
    )
    return service, store, clock


async def _create(service: ScheduleService, user: UserContext = DAVE, **changes: Any) -> Any:
    arguments: dict[str, Any] = {
        "name": "Weekly review",
        "prompt": "Summarize the week.",
        "workspaces": ["alpha"],
        "zone": "Europe/Stockholm",
        "cron": "0 8 * * 1",
    }
    return await service.create(user, **{**arguments, **changes})


async def test_a_new_schedule_acts_as_its_creator_with_the_default_ceiling() -> None:
    service, _store, _clock = _service(default_max_tokens_per_day=123_456)

    schedule = await _create(service)

    assert schedule.owner_id == owner_id_from_user(DAVE)
    assert schedule.principal == DAVE
    assert schedule.max_tokens_per_day == 123_456
    assert schedule.state is ScheduleState.ACTIVE
    assert schedule.target is Target.NEW_CONVERSATION
    assert schedule.next_fire_at == datetime(2026, 12, 14, 7, 0, tzinfo=UTC)

    capped = await _create(service, max_tokens_per_day=50)
    assert capped.max_tokens_per_day == 50


async def test_a_recurrence_closer_than_the_minimum_interval_is_refused_and_the_minimum_is_configurable() -> (
    None
):
    service, store, _clock = _service()
    with pytest.raises(CadenceError, match="at least 60 minutes"):
        await _create(service, cron="*/30 * * * *")
    assert store.schedules == {}

    relaxed, _store, _clock = _service(min_interval_minutes=15)
    assert (await _create(relaxed, cron="*/30 * * * *")).cadence.cron == "*/30 * * * *"


async def test_a_single_time_is_a_schedule_that_completes() -> None:
    service, _store, _clock = _service()
    at = NOW + timedelta(days=2)

    schedule = await _create(service, cron=None, at=at)

    assert schedule.cadence.at == at
    assert schedule.next_fire_at == at
    with pytest.raises(CadenceError, match="in the future"):
        await _create(service, cron=None, at=NOW - timedelta(minutes=1))


async def test_an_owner_holds_at_most_the_configured_active_schedules() -> None:
    service, _store, _clock = _service(max_active_per_owner=2)
    first = await _create(service)
    await _create(service)
    with pytest.raises(ScheduleLimitError, match="at most 2"):
        await _create(service)
    await _create(service, user=ERIN)

    await service.pause(DAVE, first.schedule_id)
    await _create(service)
    with pytest.raises(ScheduleLimitError):
        await service.resume(DAVE, first.schedule_id)


async def test_nobody_but_the_owner_can_see_or_change_a_schedule() -> None:
    service, _store, _clock = _service()
    schedule = await _create(service)

    assert await service.get(ERIN, schedule.schedule_id) is None
    assert await service.list(ERIN) == ()
    assert await service.update(ERIN, schedule.schedule_id, name="mine now") is None
    assert await service.pause(ERIN, schedule.schedule_id) is None
    assert await service.resume(ERIN, schedule.schedule_id) is None
    assert await service.list_fires(ERIN, schedule.schedule_id) is None
    assert await service.run_now(ERIN, schedule.schedule_id) is None
    assert await service.delete(ERIN, schedule.schedule_id) is False

    assert await service.get(DAVE, schedule.schedule_id) == schedule
    assert await service.list(DAVE) == (schedule,)
    assert await service.delete(DAVE, schedule.schedule_id) is True
    assert await service.get(DAVE, schedule.schedule_id) is None


async def test_an_edit_that_leaves_the_cadence_alone_keeps_its_place_in_line() -> None:
    service, _store, clock = _service()
    schedule = await _create(service)
    clock.advance(minutes=30)

    edited = await service.update(
        DAVE_AGAIN, schedule.schedule_id, name="Monday review", workspaces=["alpha", "beta"]
    )

    assert edited is not None
    assert edited.name == "Monday review"
    assert edited.workspaces == ("alpha", "beta")
    assert edited.principal == DAVE_AGAIN
    assert edited.next_fire_at == schedule.next_fire_at


async def test_a_new_cadence_or_zone_recomputes_the_next_slot_from_now() -> None:
    service, _store, clock = _service()
    schedule = await _create(service)
    clock.advance(hours=1)

    moved = await service.update(DAVE, schedule.schedule_id, cron="30 9 * * *")
    assert moved is not None
    assert moved.cadence.cron == "30 9 * * *"
    assert moved.next_fire_at == datetime(2026, 12, 8, 8, 30, tzinfo=UTC)  # 09:30 Stockholm

    rezoned = await service.update(DAVE, schedule.schedule_id, zone="America/New_York")
    assert rezoned is not None
    assert rezoned.cadence.cron == "30 9 * * *"
    assert rezoned.next_fire_at == datetime(2026, 12, 7, 14, 30, tzinfo=UTC)  # 09:30 New York

    once = await service.update(DAVE, schedule.schedule_id, at=NOW + timedelta(days=3))
    assert once is not None
    assert (once.cadence.cron, once.next_fire_at) == (None, NOW + timedelta(days=3))

    with pytest.raises(ScheduleInputError):
        await service.update(DAVE, schedule.schedule_id, cron="*/5 * * * *")
    with pytest.raises(ScheduleInputError):
        await service.update(DAVE, schedule.schedule_id, prompt="   ")


async def test_pausing_stops_it_and_resuming_acts_as_whoever_resumes_it() -> None:
    service, _store, _clock = _service()
    schedule = await _create(service)

    paused = await service.pause(DAVE, schedule.schedule_id)
    assert paused is not None
    assert (paused.state, paused.pause_reason) == (ScheduleState.PAUSED, None)
    assert await service.pause(DAVE, schedule.schedule_id) == paused

    resumed = await service.resume(DAVE_AGAIN, schedule.schedule_id)
    assert resumed is not None
    assert (resumed.state, resumed.pause_reason, resumed.principal) == (
        ScheduleState.ACTIVE,
        None,
        DAVE_AGAIN,
    )
    assert resumed.next_fire_at == schedule.next_fire_at
    assert await service.resume(DAVE, schedule.schedule_id) == resumed


async def test_a_completed_schedule_cannot_be_paused_resumed_or_edited() -> None:
    service, store, _clock = _service()
    schedule = await _create(service, cron=None, at=NOW + timedelta(hours=1))
    store.schedules[schedule.schedule_id] = replace(
        schedule, state=ScheduleState.COMPLETED, next_fire_at=None
    )

    for action in (
        service.pause(DAVE, schedule.schedule_id),
        service.resume(DAVE, schedule.schedule_id),
        service.update(DAVE, schedule.schedule_id, name="again"),
    ):
        with pytest.raises(ScheduleStateError):
            await action


async def test_fires_are_listed_newest_first_within_a_bounded_page() -> None:
    service, store, clock = _service()
    schedule = await _create(service)
    for hour in range(3):
        await store.insert_manual_fire(schedule, clock.now + timedelta(hours=hour), fires_kept=200)

    fires = await service.list_fires(DAVE, schedule.schedule_id, limit=2)

    assert fires is not None
    assert [fire.slot for fire in fires] == [
        clock.now + timedelta(hours=2),
        clock.now + timedelta(hours=1),
    ]
    assert all(fire.state is FireState.PENDING for fire in fires)
    everything = await service.list_fires(DAVE, schedule.schedule_id, limit=10_000)
    assert everything is not None and len(everything) == 3


async def test_a_goal_schedule_is_made_with_a_round_cap_and_edited_under_the_goals_rules() -> None:
    service, _store, _clock = _service()

    goal = await _create(service, goal_max_rounds=4, target=Target.SAME_CONVERSATION)
    assert goal.goal_max_rounds == 4
    assert (await _create(service)).goal_max_rounds is None

    raised = await service.update(DAVE, goal.schedule_id, goal_max_rounds=10)
    assert raised is not None and raised.goal_max_rounds == 10
    with pytest.raises(ScheduleInputError, match="research mode"):
        await _create(service, goal_max_rounds=4, mode="fast")
    with pytest.raises(ScheduleInputError, match="positive integer"):
        await _create(service, goal_max_rounds=0)
    with pytest.raises(ScheduleInputError, match="at most 2000 characters"):
        await service.update(DAVE, goal.schedule_id, prompt="x" * 2_001)


async def test_a_completed_schedule_is_a_state_error_before_its_cadence_is_judged() -> None:
    service, store, _clock = _service()
    schedule = await _create(service, cron=None, at=NOW + timedelta(hours=1))
    store.schedules[schedule.schedule_id] = replace(
        schedule, state=ScheduleState.COMPLETED, next_fire_at=None
    )

    with pytest.raises(ScheduleStateError):
        await service.update(DAVE, schedule.schedule_id, at=NOW - timedelta(days=1))
    with pytest.raises(ScheduleStateError):
        await service.update(DAVE, schedule.schedule_id, zone="Asia/Tokyo")


async def test_a_zone_only_edit_of_a_single_time_that_has_passed_does_not_ask_for_the_future() -> (
    None
):
    service, store, clock = _service()
    schedule = await _create(service, cron=None, at=NOW + timedelta(hours=1))
    clock.advance(hours=3)  # the time passed, and the Schedule waits paused for a new one
    store.schedules[schedule.schedule_id] = replace(
        schedule, state=ScheduleState.PAUSED, pause_reason="missed", next_fire_at=None
    )

    moved = await service.update(DAVE, schedule.schedule_id, zone="Asia/Tokyo")

    assert moved is not None
    assert (moved.cadence.zone_name, moved.cadence.at) == ("Asia/Tokyo", schedule.cadence.at)
    assert moved.next_fire_at is None


async def test_an_edit_that_leaves_the_cadence_as_it_is_does_not_move_the_next_slot() -> None:
    service, _store, clock = _service()
    schedule = await _create(service)
    clock.advance(days=3)

    same = await service.update(DAVE, schedule.schedule_id, cron="0 8 * * 1")
    rezoned_to_the_same = await service.update(DAVE, schedule.schedule_id, zone="Europe/Stockholm")

    assert same is not None and rezoned_to_the_same is not None
    assert same.next_fire_at == rezoned_to_the_same.next_fire_at == schedule.next_fire_at


def test_the_default_ceiling_and_the_age_limit_are_held_to_what_the_store_can_keep() -> None:
    from pydantic import ValidationError

    assert SchedulesSettings(default_max_tokens_per_day=2**63 - 1).principal_max_age_days == 30
    with pytest.raises(ValidationError):
        SchedulesSettings(default_max_tokens_per_day=2**63)
    with pytest.raises(ValidationError):
        SchedulesSettings(principal_max_age_days=0)
