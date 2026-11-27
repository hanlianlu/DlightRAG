# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent meets the owner's rules, as the Run's caller, and holds nothing the owner does not."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from dlightrag.adapters.http.schedules import schedule_payload
from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.config.sections import SchedulesSettings
from dlightrag.application.schedules.agent import AgentSchedules, OwnerCallers
from dlightrag.application.schedules.model import ScheduleState
from dlightrag.application.schedules.service import ScheduleService
from dlightrag.engine.answer.tools.schedules import (
    SCHEDULE_INVALID,
    SCHEDULE_LIMIT,
    SCHEDULE_NOT_FOUND,
    SCHEDULE_STATE,
    SCHEDULE_UNAVAILABLE,
    ScheduleRefusal,
)
from tests.support.schedules import Clock, FakeScheduleStore

NOW = datetime(2026, 12, 7, 10, 0, 20, tzinfo=UTC)
DAVE = UserContext(user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer"})
ERIN = UserContext(user_id="erin", auth_mode="jwt", claims={"iss": "https://issuer"})
DAVE_OWNER = owner_id_from_user(DAVE)
ERIN_OWNER = owner_id_from_user(ERIN)


class Callers:
    """Which caller each Run was submitted by."""

    def __init__(self) -> None:
        self.by_run: dict[str, UserContext] = {"run-dave": DAVE, "run-erin": ERIN}

    async def caller_of(self, *, owner_id: str, run_id: str) -> UserContext | None:
        return self.by_run.get(run_id)


def _agent(**settings: Any) -> tuple[AgentSchedules, ScheduleService, FakeScheduleStore]:
    clock = Clock(NOW)
    store = FakeScheduleStore(clock)
    service = ScheduleService(
        store=store, firer=cast(Any, None), settings=SchedulesSettings(**settings), clock=clock
    )
    agent = AgentSchedules(callers=Callers())
    agent.bind(service)
    return agent, service, store


_CREATED: dict[str, Any] = {
    "name": "Weekday digest",
    "prompt": "Summarise the new filings.",
    "workspaces": ("alpha",),
    "zone": "Europe/Stockholm",
    "cron": "0 9 * * 1-5",
    "at": None,
    "mode": "research",
    "target": "new_conversation",
}


async def _create(agent: AgentSchedules, run: str = "run-dave", **changes: Any) -> Any:
    owner = DAVE_OWNER if run == "run-dave" else ERIN_OWNER
    return await agent.create(owner_id=owner, run_id=run, **{**_CREATED, **changes})


async def _refusal(call: Any) -> ScheduleRefusal:
    with pytest.raises(ScheduleRefusal) as caught:
        await call
    return caught.value


async def test_a_schedule_the_agent_makes_is_the_callers_with_the_default_ceiling_and_no_goal() -> (
    None
):
    agent, service, store = _agent(default_max_tokens_per_day=123_456)

    record = await _create(agent)

    stored = store.schedules[record.schedule_id]
    assert stored.principal == DAVE
    assert stored.owner_id == DAVE_OWNER
    assert stored.max_tokens_per_day == 123_456
    assert stored.goal_max_rounds is None
    assert (record.zone, record.cron, record.state) == ("Europe/Stockholm", "0 9 * * 1-5", "active")
    assert record.next_fire_at == datetime(2026, 12, 8, 8, 0, tzinfo=UTC)
    assert [item.schedule_id for item in await service.list(DAVE)] == [record.schedule_id]


@pytest.mark.parametrize(
    ("changes", "code", "needle"),
    [
        ({"cron": "*/30 * * * *"}, SCHEDULE_INVALID, "at least 60 minutes"),
        ({"cron": "every day at nine"}, SCHEDULE_INVALID, "five fields"),
        ({"zone": "Mars/Olympus"}, SCHEDULE_INVALID, "is not an IANA time zone"),
        (
            {"cron": None, "at": datetime(2026, 12, 1, 9, 0, tzinfo=UTC)},
            SCHEDULE_INVALID,
            "must be in the future",
        ),
        ({"workspaces": ()}, SCHEDULE_INVALID, "at least one workspace"),
    ],
)
async def test_the_owners_cadence_rules_refuse_the_agent_too(
    changes: dict[str, Any], code: str, needle: str
) -> None:
    agent, _service, store = _agent()

    refusal = await _refusal(_create(agent, **changes))

    assert (refusal.code, needle in str(refusal)) == (code, True)
    assert store.schedules == {}


async def test_the_cap_on_active_schedules_binds_the_agent_and_resuming_counts_against_it() -> None:
    agent, _service, _store = _agent(max_active_per_owner=1)
    first = await _create(agent)

    refusal = await _refusal(_create(agent))
    assert (refusal.code, "at most 1 active Schedules" in str(refusal)) == (SCHEDULE_LIMIT, True)

    await _update(agent, first.schedule_id, paused=True)
    second = await _create(agent)
    refused_resume = await _refusal(_update(agent, first.schedule_id, paused=False))
    assert refused_resume.code == SCHEDULE_LIMIT
    assert second.state == "active"


def _update(agent: AgentSchedules, schedule_id: str, run: str = "run-dave", **changes: Any) -> Any:
    """An edit that names only what ``changes`` names."""
    owner = DAVE_OWNER if run == "run-dave" else ERIN_OWNER
    untouched: dict[str, Any] = dict.fromkeys(
        ("name", "prompt", "workspaces", "mode", "zone", "cron", "at", "paused")
    )
    return agent.update(
        owner_id=owner, run_id=run, schedule_id=schedule_id, **{**untouched, **changes}
    )


async def test_update_changes_what_is_named_and_leaves_the_owners_ceiling_and_goal_alone() -> None:
    agent, service, store = _agent()
    kept: dict[str, Any] = {
        "workspaces": ["alpha"],
        "zone": "Europe/Stockholm",
        "cron": "0 8 * * 1",
        "max_tokens_per_day": 777,
    }
    plain = await service.create(DAVE, name="Watch", prompt="Review.", **kept)
    goal = await service.create(
        DAVE, name="Goal watch", prompt="Finish.", goal_max_rounds=3, **kept
    )

    record = await _update(agent, plain.schedule_id, name="Monday watch", cron="0 9 * * 1")
    renamed = await _update(agent, goal.schedule_id, name="Monday goal watch")

    assert (record.name, record.cron, record.goal_max_rounds) == ("Monday watch", "0 9 * * 1", None)
    assert record.next_fire_at == datetime(2026, 12, 14, 8, 0, tzinfo=UTC)
    assert (renamed.name, renamed.goal_max_rounds) == ("Monday goal watch", 3)
    assert [store.schedules[item.schedule_id].max_tokens_per_day for item in (plain, goal)] == [
        777,
        777,
    ]


async def test_pause_and_resume_move_the_state_and_a_completed_schedule_takes_no_edit() -> None:
    agent, _service, store = _agent()
    record = await _create(agent)

    paused = await _update(agent, record.schedule_id, paused=True)
    resumed = await _update(agent, record.schedule_id, paused=False)
    assert (paused.state, paused.next_fire_at is not None) == ("paused", True)
    assert resumed.state == "active"

    done = store.schedules[record.schedule_id]
    store.schedules[record.schedule_id] = replace(done, state=ScheduleState.COMPLETED)
    refusal = await _refusal(_update(agent, record.schedule_id, name="Renamed"))
    assert (refusal.code, "completed" in str(refusal)) == (SCHEDULE_STATE, True)


async def test_the_agent_sees_edits_and_deletes_only_its_callers_schedules() -> None:
    agent, _service, store = _agent()
    mine = await _create(agent)
    theirs = await _create(agent, run="run-erin", name="Erin's")

    listed = await agent.list(owner_id=DAVE_OWNER, run_id="run-dave")
    assert [item.name for item in listed] == ["Weekday digest"]

    for call in (
        _update(agent, theirs.schedule_id, name="Taken"),
        agent.delete(owner_id=DAVE_OWNER, run_id="run-dave", schedule_id=theirs.schedule_id),
    ):
        assert (await _refusal(call)).code == SCHEDULE_NOT_FOUND
    assert store.schedules[theirs.schedule_id].name == "Erin's"

    removed = await agent.delete(
        owner_id=DAVE_OWNER, run_id="run-dave", schedule_id=mine.schedule_id
    )
    assert removed.name == "Weekday digest"
    assert mine.schedule_id not in store.schedules


async def test_a_run_whose_caller_is_unknown_or_not_its_owner_cannot_touch_schedules() -> None:
    agent, _service, store = _agent()

    unknown = await _refusal(agent.list(owner_id=DAVE_OWNER, run_id="run-nobody"))
    mismatched = await _refusal(agent.list(owner_id=ERIN_OWNER, run_id="run-dave"))
    unbound = AgentSchedules(callers=Callers())
    detached = await _refusal(unbound.list(owner_id=DAVE_OWNER, run_id="run-dave"))

    assert {unknown.code, mismatched.code, detached.code} == {SCHEDULE_UNAVAILABLE}
    assert store.schedules == {}


async def test_the_agent_acts_as_the_owners_latest_human_turn_kept_one_per_owner() -> None:
    clock = Clock(NOW)
    store = FakeScheduleStore(clock)
    service = ScheduleService(
        store=store, firer=cast(Any, None), settings=SchedulesSettings(), clock=clock
    )
    callers = OwnerCallers(store)
    agent = AgentSchedules(callers=callers)
    agent.bind(service)
    refreshed = UserContext(
        user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer", "n": 2}
    )

    nothing_kept = await _refusal(agent.list(owner_id=DAVE_OWNER, run_id="run-1"))
    await callers.remember(owner_id=DAVE_OWNER, caller=DAVE)
    await callers.remember(owner_id=ERIN_OWNER, caller=ERIN)
    await callers.remember(owner_id=DAVE_OWNER, caller=refreshed)
    record = await agent.create(owner_id=DAVE_OWNER, run_id="run-1", **_CREATED)

    assert nothing_kept.code == SCHEDULE_UNAVAILABLE
    assert store.principals == {DAVE_OWNER: refreshed, ERIN_OWNER: ERIN}
    assert store.schedules[record.schedule_id].principal == refreshed
    # The record is the owner's, whatever Run asks.
    assert await callers.caller_of(owner_id=DAVE_OWNER, run_id="another-run") == refreshed


async def test_an_identity_that_is_not_the_owners_is_not_kept() -> None:
    store = FakeScheduleStore(Clock(NOW))

    await OwnerCallers(store).remember(owner_id=DAVE_OWNER, caller=ERIN)

    assert store.principals == {}


async def test_what_the_agent_wrote_is_marked_and_an_owners_own_change_clears_the_mark() -> None:
    agent, service, store = _agent()

    made = await _create(agent)
    assert store.schedules[made.schedule_id].changed_by == "agent"
    assert schedule_payload(store.schedules[made.schedule_id])["changed_by"] == "agent"

    await _update(agent, made.schedule_id, paused=True)
    # Pausing leaves the mark as it was; the owner editing it takes the mark back.
    assert store.schedules[made.schedule_id].changed_by == "agent"
    resumed_by_owner = await service.resume(DAVE, made.schedule_id)
    assert resumed_by_owner is not None and resumed_by_owner.changed_by == "owner"

    await _update(agent, made.schedule_id, name="Agent rename")
    assert store.schedules[made.schedule_id].changed_by == "agent"
    await service.update(DAVE, made.schedule_id, name="Owner rename")
    assert store.schedules[made.schedule_id].changed_by == "owner"

    await _update(agent, made.schedule_id, paused=True)
    await _update(agent, made.schedule_id, paused=False)
    assert store.schedules[made.schedule_id].changed_by == "agent"

    own = await service.create(
        DAVE,
        name="Own",
        prompt="p",
        workspaces=["alpha"],
        zone="Europe/Stockholm",
        cron="0 8 * * 1",
    )
    assert own.changed_by == "owner"


async def test_a_schedule_that_starts_a_goal_is_the_owners_to_change_but_not_to_pause_or_delete() -> (
    None
):
    agent, service, store = _agent()
    goal = await service.create(
        DAVE,
        name="Goal watch",
        prompt="Finish the review.",
        workspaces=["alpha"],
        zone="Europe/Stockholm",
        cron="0 8 * * 1",
        goal_max_rounds=3,
    )

    edits: list[dict[str, Any]] = [
        {"prompt": "Do something else."},
        {"workspaces": ("beta",)},
        {"mode": "fast"},
        {"zone": "UTC"},
        {"cron": "0 9 * * 1"},
        {"at": datetime(2030, 1, 1, 9, 0)},
    ]
    for edit in edits:
        refusal = await _refusal(_update(agent, goal.schedule_id, **edit))
        assert refusal.code == SCHEDULE_STATE, edit
        assert "starts a goal" in str(refusal)
    renamed = await _update(agent, goal.schedule_id, name="Renamed")
    paused = await _update(agent, goal.schedule_id, paused=True)
    refused_resume = await _refusal(_update(agent, goal.schedule_id, paused=False))
    assert (renamed.name, paused.state) == ("Renamed", "paused")
    assert refused_resume.code == SCHEDULE_STATE
    assert store.schedules[goal.schedule_id].state is ScheduleState.PAUSED

    await agent.delete(owner_id=DAVE_OWNER, run_id="run-dave", schedule_id=goal.schedule_id)
    assert goal.schedule_id not in store.schedules


async def test_an_edit_saved_before_a_refused_resume_says_what_was_saved_and_what_was_not() -> None:
    agent, _service, store = _agent(max_active_per_owner=1)
    first = await _create(agent)
    await _update(agent, first.schedule_id, paused=True)
    await _create(agent)

    refusal = await _refusal(_update(agent, first.schedule_id, name="Renamed", paused=False))

    assert refusal.code == SCHEDULE_LIMIT
    assert str(refusal).startswith("Saved: name. Not done: resume \u2014 ")
    assert "at most 1 active Schedules" in str(refusal)
    stored = store.schedules[first.schedule_id]
    assert (stored.name, stored.state) == ("Renamed", ScheduleState.PAUSED)
    # With nothing saved the refusal is the service's own.
    bare = await _refusal(_update(agent, first.schedule_id, paused=False))
    assert str(bare).startswith("an owner may hold")


async def test_a_time_with_no_offset_is_read_on_the_named_zone_or_the_schedules_own() -> None:
    agent, _service, _store = _agent()
    once = await _create(agent, cron=None, at=datetime(2026, 12, 10, 9, 0, tzinfo=UTC))

    on_own_zone = await _update(agent, once.schedule_id, at=datetime(2026, 12, 20, 9, 0))
    on_named_zone = await _update(
        agent, once.schedule_id, at=datetime(2026, 12, 21, 9, 0), zone="America/New_York"
    )

    assert on_own_zone.at == datetime(2026, 12, 20, 8, 0, tzinfo=UTC)
    assert on_named_zone.at == datetime(2026, 12, 21, 14, 0, tzinfo=UTC)
