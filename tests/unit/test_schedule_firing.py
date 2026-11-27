# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The look: due Schedules become Fires, Fires become Runs, and the rules between them (ADR 0103)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from dlightrag.application.access.control import AccessSubject, AllowAllAccessControl
from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.answer.goal_rounds import GoalRoundDriver, goal_round_prompt
from dlightrag.application.answer_runs.service import (
    ActiveRunLimitError,
    AnswerConnectionsChangedError,
    AnswerRequestError,
)
from dlightrag.application.config.sections import SchedulesSettings
from dlightrag.application.errors import ApplicationNotFoundError
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunView,
    SessionWorkBusyError,
)
from dlightrag.application.schedules.cadence import Cadence, CronExpression
from dlightrag.application.schedules.errors import ScheduleStateError
from dlightrag.application.schedules.firing import ScheduleFirer, fire_key
from dlightrag.application.schedules.model import (
    Fire,
    FireState,
    Schedule,
    ScheduledConversation,
    ScheduledGoalRound,
    ScheduledTurn,
    ScheduleState,
    Target,
)
from dlightrag.application.schedules.service import ScheduleService
from dlightrag.application.web_conversations.models import ConversationSubmissionConflict
from dlightrag.engine.answer.goal import Goal, GoalPhase
from dlightrag.engine.answer.goal_service import AnswerGoalService
from dlightrag.engine.runtime.coordinator import SettledRun
from tests.support.goals import FakeGoalStore
from tests.support.schedules import Clock, FakeScheduleStore

USER = UserContext(user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer"})
OWNER = owner_id_from_user(USER)
SETTINGS = SchedulesSettings(look_interval_seconds=30, failure_streak=3)

#: Created at 10:00:20; its first slot is 11:00, and the tests look at 11:00:20 unless they move.
CREATED = datetime(2026, 12, 7, 10, 0, 20, tzinfo=UTC)
FIRST_SLOT = datetime(2026, 12, 7, 11, 0, tzinfo=UTC)


class Alpha(AllowAllAccessControl):
    """An access policy under which the caller reads one workspace, and no other."""

    async def filter_workspaces(
        self, subject: AccessSubject, action: str, workspaces: Sequence[str]
    ) -> list[str]:
        return [name for name in workspaces if name == "alpha"]


@dataclass
class FakeRuns:
    views: dict[str, SimpleNamespace] = field(default_factory=dict)

    async def get(self, *, owner_id: str, run_id: str) -> RunView | None:
        del owner_id
        return cast(Any, self.views.get(run_id))

    def finish(
        self,
        run_id: str,
        status: str = "succeeded",
        *,
        session: str = "s1",
        request: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.views[run_id] = SimpleNamespace(
            run_id=run_id,
            run_kind="answer",
            submitted_by=OWNER,
            access_scope_kind="owner",
            terminal=True,
            status=status,
            error_kind=extra.get("error_kind"),
            error_message=extra.get("error_message"),
            result=None,
            request=request or {"agent_session_id": session},
        )

    def live(self, run_id: str) -> None:
        self.views[run_id] = SimpleNamespace(terminal=False, status="running", request={})


@dataclass
class FakeTurns:
    """The conversations; each accepted turn is a new, working Run in a conversation of its own."""

    runs: FakeRuns
    calls: list[dict[str, Any]] = field(default_factory=list)
    ensured: list[dict[str, Any]] = field(default_factory=list)
    raises: list[BaseException] = field(default_factory=list)
    #: The turns made, by the key they were accepted under: a key accepted twice is one turn.
    accepted: dict[str, ScheduledTurn] = field(default_factory=dict)

    async def ensure_scheduled_conversation(self, **call: Any) -> ScheduledConversation:
        """A conversation is its own Session, and its id is the key's when it had none."""
        self.ensured.append(call)
        conversation_id = call["conversation_id"] or f"conversation-of-{call['idempotency_key']}"
        return ScheduledConversation(conversation_id, conversation_id)

    async def replay_scheduled_turn(self, **call: Any) -> ScheduledTurn | None:
        return self.accepted.get(call["idempotency_key"])

    async def accept_scheduled_turn(self, **call: Any) -> ScheduledTurn:
        self.calls.append(call)
        if call["idempotency_key"] in self.accepted:
            return self.accepted[call["idempotency_key"]]
        if self.raises:
            raise self.raises.pop(0)
        run_id = f"00000000-0000-4000-8000-{len(self.calls):012d}"
        self.runs.live(run_id)
        turn = ScheduledTurn(
            run_id=run_id,
            conversation_id=call["conversation_id"] or f"conversation-{len(self.calls)}",
        )
        self.accepted[call["idempotency_key"]] = turn
        return turn


@dataclass
class World:
    clock: Clock
    store: FakeScheduleStore
    turns: FakeTurns
    runs: FakeRuns
    goals: AnswerGoalService
    goal_store: FakeGoalStore
    firer: ScheduleFirer

    async def schedule(
        self,
        *,
        cron: str = "0 * * * *",
        at: datetime | None = None,
        target: Target = Target.NEW_CONVERSATION,
        workspaces: Sequence[str] = ("alpha", "beta"),
        principal: UserContext = USER,
        owner: str = OWNER,
        goal_max_rounds: int | None = None,
    ) -> Schedule:
        cadence = Cadence("UTC", None if at else CronExpression.parse(cron), at)
        return await self.store.create(
            Schedule.create(
                owner_id=owner,
                principal=principal,
                name="Weekly review",
                prompt="Summarize the week.",
                mode="research",
                workspaces=workspaces,
                target=target,
                cadence=cadence,
                max_tokens_per_day=1_000,
                now=CREATED,
                goal_max_rounds=goal_max_rounds,
            ),
            max_active=20,
        )

    def fires(self, schedule: Schedule) -> list[Fire]:
        return self.store.fires_of(schedule.schedule_id)

    def states(self, schedule: Schedule) -> list[str]:
        return [str(fire.state) for fire in self.fires(schedule)]

    def current(self, schedule: Schedule) -> Schedule:
        return self.store.schedules[schedule.schedule_id]


def _world(access: AllowAllAccessControl | None = None) -> World:
    clock = Clock(FIRST_SLOT + timedelta(seconds=20))
    store = FakeScheduleStore(clock)
    runs = FakeRuns()
    turns, goal_store = FakeTurns(runs), FakeGoalStore()
    goals = AnswerGoalService(store=goal_store)
    firer = ScheduleFirer(
        store=store,
        turns=turns,
        runs=runs,
        goals=goals,
        access=access or AllowAllAccessControl(),
        settings=SETTINGS,
        clock=clock,
    )
    return World(clock, store, turns, runs, goals, goal_store, firer)


@pytest.fixture
def world() -> World:
    return _world()


def _settled(run_id: str, status: str = "succeeded") -> SettledRun:
    return SettledRun(owner_id=OWNER, run_id=run_id, run_kind="answer", status=cast(Any, status))


# -- a due slot starts one Run --------------------------------------------------------------


async def test_a_due_slot_starts_one_run_as_its_owner_in_a_new_conversation(world: World) -> None:
    schedule = await world.schedule()

    await world.firer.look()

    (call,) = world.turns.calls
    assert call["caller"] == USER
    assert call["conversation_id"] is None
    assert call["workspaces"] == ("alpha", "beta")
    assert call["mode"] == "research"
    assert call["title"] == "Weekly review · 2026-12-07 11:00"
    assert call["idempotency_key"] == fire_key(schedule.schedule_id, FIRST_SLOT)
    assert call["query"] == (
        '[Scheduled: "Weekly review", 2026-12-07 11:00 UTC]\n\nSummarize the week.'
    )
    (fire,) = world.fires(schedule)
    assert fire.state is FireState.STARTED
    assert fire.run_id == "00000000-0000-4000-8000-000000000001"
    assert fire.conversation_id == "conversation-1"
    assert world.current(schedule).next_fire_at == datetime(2026, 12, 7, 12, 0, tzinfo=UTC)


async def test_looking_again_in_the_same_moment_starts_nothing_more(world: World) -> None:
    await world.schedule()

    await world.firer.look()
    await world.firer.look()

    assert len(world.turns.calls) == 1


async def test_the_key_of_a_slot_is_the_same_wherever_it_is_computed() -> None:
    slot = datetime(2026, 12, 7, 11, 0, tzinfo=UTC)
    local = slot.astimezone(UTC).replace(tzinfo=UTC)
    assert fire_key("s", slot) == fire_key("s", local)
    assert fire_key("s", slot) != fire_key("t", slot)
    assert fire_key("s", slot) != fire_key("s", slot + timedelta(hours=1))
    assert len(fire_key("s", slot)) == 36


async def test_a_same_conversation_schedule_opens_its_conversation_once_and_keeps_it(
    world: World,
) -> None:
    schedule = await world.schedule(target=Target.SAME_CONVERSATION)

    await world.firer.look()
    world.runs.finish(world.fires(schedule)[0].run_id or "")
    await world.firer.settle(_settled(world.fires(schedule)[0].run_id or ""))
    world.clock.advance(hours=1)
    await world.firer.look()

    first, second = world.turns.calls
    assert first["conversation_id"] is None
    assert second["conversation_id"] == "conversation-1"
    assert first["title"] == second["title"] == "Weekly review"
    assert world.current(schedule).conversation_id == "conversation-1"


async def test_slots_missed_while_nothing_looked_are_one_gap_and_one_late_run(
    world: World,
) -> None:
    schedule = await world.schedule()
    world.clock.advance(hours=3)  # 14:00:20, so 11:00 to 14:00 were all due

    await world.firer.look()

    assert world.states(schedule) == ["missed", "started"]
    gap, late = world.fires(schedule)
    assert (gap.slot, gap.missed_count) == (FIRST_SLOT, 3)
    assert late.slot == datetime(2026, 12, 7, 14, 0, tzinfo=UTC)
    assert len(world.turns.calls) == 1
    assert world.current(schedule).next_fire_at == datetime(2026, 12, 7, 15, 0, tzinfo=UTC)


async def test_a_slot_too_late_is_missed_and_starts_nothing(world: World) -> None:
    schedule = await world.schedule()
    world.clock.advance(minutes=40)  # 11:40:20 is past half an hour

    await world.firer.look()

    assert world.states(schedule) == ["missed"]
    assert world.turns.calls == []
    assert world.current(schedule).state is ScheduleState.ACTIVE


async def test_a_single_time_fires_once_and_is_then_completed(world: World) -> None:
    schedule = await world.schedule(at=FIRST_SLOT)

    await world.firer.look()
    world.clock.advance(hours=5)
    await world.firer.look()

    assert world.states(schedule) == ["started"]
    done = world.current(schedule)
    assert (done.state, done.next_fire_at) == (ScheduleState.COMPLETED, None)
    assert len(world.turns.calls) == 1


# -- the rules before a Run -----------------------------------------------------------------


async def test_a_fire_is_skipped_while_the_previous_run_is_still_working(world: World) -> None:
    schedule = await world.schedule()
    await world.firer.look()
    (first,) = world.fires(schedule)
    assert first.run_id is not None
    world.runs.live(first.run_id)
    world.clock.advance(hours=1)

    await world.firer.look()

    assert world.states(schedule) == ["started", "skipped_busy"]
    assert "still working" in (world.fires(schedule)[1].reason or "")
    assert len(world.turns.calls) == 1
    assert world.current(schedule).state is ScheduleState.ACTIVE

    # Skipped, not queued: once the Run ends the next slot starts normally.
    world.runs.finish(first.run_id)
    world.clock.advance(hours=1)
    await world.firer.look()
    assert world.states(schedule) == ["succeeded", "skipped_busy", "started"]


async def test_a_fire_is_skipped_while_the_goal_the_previous_run_started_is_active(
    world: World,
) -> None:
    schedule = await world.schedule()
    await world.firer.look()
    run_id = world.fires(schedule)[0].run_id or ""
    world.runs.finish(run_id, session="s1")
    world.goal_store.rows[(OWNER, "s1")] = Goal.create("finish the model")
    world.clock.advance(hours=1)

    await world.firer.look()

    assert world.states(schedule) == ["succeeded", "skipped_busy"]
    assert "goal" in (world.fires(schedule)[1].reason or "")

    world.goal_store.rows[(OWNER, "s1")] = world.goal_store.rows[(OWNER, "s1")].pause(revision=1)
    world.clock.advance(hours=1)
    await world.firer.look()
    assert world.states(schedule)[-1] == "started"


async def test_a_busy_session_or_a_reached_limit_is_a_skip_and_the_schedule_stays_active(
    world: World,
) -> None:
    schedule = await world.schedule()
    world.turns.raises = [SessionWorkBusyError("run-live"), ActiveRunLimitError(2)]

    await world.firer.look()
    world.clock.advance(hours=1)
    await world.firer.look()
    world.turns.raises = [RunAdmissionLimitExceededError()]
    world.clock.advance(hours=1)
    await world.firer.look()

    assert world.states(schedule) == ["skipped_busy", "skipped_limit", "skipped_limit"]
    assert world.current(schedule).state is ScheduleState.ACTIVE
    assert world.current(schedule).pause_reason is None


async def test_an_identity_that_no_longer_belongs_to_the_owner_is_refused_and_pauses(
    world: World,
) -> None:
    stranger = UserContext(user_id="mallory", auth_mode="jwt", claims={"iss": "https://issuer"})
    schedule = await world.schedule(principal=stranger)

    await world.firer.look()

    assert world.states(schedule) == ["refused"]
    paused = world.current(schedule)
    assert paused.state is ScheduleState.PAUSED
    assert "no longer belongs to its owner" in (paused.pause_reason or "")
    assert world.turns.calls == []


async def test_a_workspace_the_owner_can_no_longer_read_is_dropped_from_the_turn() -> None:
    world = _world(Alpha())
    schedule = await world.schedule()

    await world.firer.look()

    assert world.turns.calls[0]["workspaces"] == ("alpha",)
    assert world.states(schedule) == ["started"]


async def test_a_schedule_with_no_readable_workspace_is_refused_and_pauses() -> None:
    world = _world(Alpha())
    schedule = await world.schedule(workspaces=("beta",))

    await world.firer.look()

    assert world.states(schedule) == ["refused"]
    assert world.current(schedule).state is ScheduleState.PAUSED
    assert "can no longer read any of its workspaces" in (
        world.current(schedule).pause_reason or ""
    )
    assert world.turns.calls == []


async def test_a_schedule_at_its_daily_ceiling_is_refused_and_pauses(world: World) -> None:
    schedule = await world.schedule()
    world.store.spent = 1_000

    await world.firer.look()

    assert world.states(schedule) == ["refused"]
    paused = world.current(schedule)
    assert paused.state is ScheduleState.PAUSED
    assert "ceiling of 1000" in (paused.pause_reason or "")
    assert world.turns.calls == []


async def test_a_schedule_under_its_ceiling_runs(world: World) -> None:
    schedule = await world.schedule()
    world.store.spent = 999

    await world.firer.look()

    assert world.states(schedule) == ["started"]


@pytest.mark.parametrize(
    "error",
    [
        AnswerRequestError("at least one canonical workspace is required"),
        ApplicationNotFoundError("the conversation this Schedule keeps is gone"),
    ],
)
async def test_a_turn_admission_refuses_for_a_reason_only_the_owner_can_change(
    world: World, error: Exception
) -> None:
    schedule = await world.schedule()
    world.turns.raises = [error]

    await world.firer.look()

    assert world.states(schedule) == ["refused"]
    paused = world.current(schedule)
    assert paused.state is ScheduleState.PAUSED
    assert paused.pause_reason == str(error)


# -- how a Fire ends ------------------------------------------------------------------------


async def test_a_settled_run_ends_its_fire_as_succeeded_failed_or_cancelled(world: World) -> None:
    schedule = await world.schedule()
    for status, error in (("succeeded", None), ("failed", "model_error"), ("cancelled", None)):
        await world.firer.look()
        run_id = world.fires(schedule)[-1].run_id or ""
        world.runs.finish(run_id, status, error_kind=error, error_message="the provider refused")
        await world.firer.settle(_settled(run_id, status))
        world.clock.advance(hours=1)

    first, second, third = world.fires(schedule)
    assert (first.state, first.reason) == (FireState.SUCCEEDED, None)
    assert second.state is FireState.FAILED
    assert second.reason == "model_error the provider refused"
    assert (third.state, third.reason) == (FireState.CANCELLED, "the run was cancelled")


async def test_three_failed_fires_in_a_row_pause_the_schedule_with_the_last_reason(
    world: World,
) -> None:
    schedule = await world.schedule()
    for number in range(3):
        await world.firer.look()
        run_id = world.fires(schedule)[-1].run_id or ""
        world.runs.finish(
            run_id, "failed", error_kind="model_error", error_message=f"boom {number}"
        )
        await world.firer.settle(_settled(run_id, "failed"))
        world.clock.advance(hours=1)
    assert world.current(schedule).state is ScheduleState.PAUSED
    assert "boom 2" in (world.current(schedule).pause_reason or "")

    # Nothing retries on its own: a paused Schedule fires nothing, however long it waits.
    world.clock.advance(hours=1)
    await world.firer.look()
    assert len(world.turns.calls) == 3


async def test_a_success_between_failures_keeps_the_schedule_running(world: World) -> None:
    schedule = await world.schedule()
    for status in ("failed", "failed", "succeeded", "failed", "failed"):
        await world.firer.look()
        run_id = world.fires(schedule)[-1].run_id or ""
        world.runs.finish(run_id, status, error_kind="model_error" if status == "failed" else None)
        await world.firer.settle(_settled(run_id, status))
        world.clock.advance(hours=1)

    assert world.current(schedule).state is ScheduleState.ACTIVE


async def test_a_run_the_observer_never_reported_is_settled_by_a_later_look(
    world: World,
) -> None:
    schedule = await world.schedule()
    await world.firer.look()
    run_id = world.fires(schedule)[0].run_id or ""
    world.runs.finish(run_id, "failed", error_kind="worker_lost")

    world.clock.advance(minutes=2)
    await world.firer.look()
    assert world.states(schedule) == ["started"], "too soon to doubt the observer"

    world.clock.advance(minutes=4)
    await world.firer.look()
    assert world.states(schedule) == ["failed"]
    assert world.fires(schedule)[0].reason == "worker_lost"


async def test_a_started_fire_whose_run_is_gone_ends_cancelled(world: World) -> None:
    schedule = await world.schedule()
    await world.firer.look()
    del world.runs.views[world.fires(schedule)[0].run_id or ""]
    world.clock.advance(minutes=6)

    await world.firer.look()

    assert world.states(schedule) == ["cancelled"]
    assert world.fires(schedule)[0].reason == "its run no longer exists"


async def test_settling_a_run_no_fire_started_changes_nothing(world: World) -> None:
    schedule = await world.schedule()

    await world.firer.settle(_settled("someone-elses-run"))

    assert world.fires(schedule) == []


# -- a crash between the Fire and the Run ---------------------------------------------------


async def test_a_pending_fire_is_retried_under_the_same_key_after_the_lease(world: World) -> None:
    schedule = await world.schedule()
    world.turns.raises = [RuntimeError("the database blinked")]

    await world.firer.look()
    assert world.states(schedule) == ["pending"]

    world.clock.advance(seconds=30)
    await world.firer.look()
    assert world.states(schedule) == ["pending"], "not before the lease has run out"
    assert len(world.turns.calls) == 1

    world.clock.advance(seconds=40)
    await world.firer.look()
    assert world.states(schedule) == ["started"]
    first, retry = world.turns.calls
    assert first["idempotency_key"] == retry["idempotency_key"]
    assert first["query"] == retry["query"]


async def test_a_pending_fire_too_old_to_be_worth_starting_is_missed(world: World) -> None:
    schedule = await world.schedule()
    world.turns.raises = [RuntimeError("the providers are down")]
    await world.firer.look()

    world.clock.advance(minutes=45)  # past the half hour an hourly slot is worth
    await world.firer.look()

    assert world.states(schedule)[0] == "missed"
    assert len(world.turns.calls) == 1


# -- Run now --------------------------------------------------------------------------------


async def test_without_a_clock_of_its_own_the_look_and_the_service_read_the_stores(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    far = datetime(2031, 3, 1, 9, 0, 20, tzinfo=UTC)  # nowhere near the clock of this machine
    store = FakeScheduleStore(Clock(far))
    claim_due = AsyncMock(wraps=store.claim_due)
    monkeypatch.setattr(store, "claim_due", claim_due)
    firer = ScheduleFirer(
        store=store,
        turns=world.turns,
        runs=world.runs,
        goals=world.goals,
        access=AllowAllAccessControl(),
        settings=SETTINGS,
    )
    service = ScheduleService(store=store, firer=firer, settings=SETTINGS)

    await firer.look()
    created = await service.create(
        USER, name="Hourly", prompt="Look.", workspaces=["alpha"], zone="UTC", cron="0 * * * *"
    )

    (claim,) = claim_due.await_args_list
    assert claim.args[0] == far
    assert created.created_at == far
    assert created.next_fire_at == datetime(2031, 3, 1, 10, 0, tzinfo=UTC)


async def test_a_manual_fire_goes_through_the_same_rules_as_a_clock_fire(world: World) -> None:
    schedule = await world.schedule()
    service = ScheduleService(
        store=world.store, firer=world.firer, settings=SETTINGS, clock=world.clock
    )
    other = UserContext(user_id="erin", auth_mode="jwt", claims={"iss": "https://issuer"})

    assert await service.run_now(other, schedule.schedule_id) is None
    started = await service.run_now(USER, schedule.schedule_id)

    assert started is not None and started.state is FireState.STARTED
    assert started.slot == world.clock.now
    assert world.turns.calls[0]["caller"] == USER
    assert "11:00" in world.turns.calls[0]["query"].splitlines()[0]

    assert started.manual is True

    # Its Run is still working, so a second request is refused rather than started beside it.
    world.clock.advance(seconds=5)
    with pytest.raises(ScheduleStateError, match="already has a fire"):
        await service.run_now(USER, schedule.schedule_id)


# -- goal Schedules -------------------------------------------------------------------------

KEPT = "kept-conversation"


async def _keeping(world: World, **changes: Any) -> Schedule:
    """A goal Schedule that already keeps a conversation, so the goal there decides its fire."""
    schedule = await world.schedule(target=Target.SAME_CONVERSATION, goal_max_rounds=3, **changes)
    world.store.schedules[schedule.schedule_id] = replace(schedule, conversation_id=KEPT)
    return world.store.schedules[schedule.schedule_id]


async def test_a_goal_fire_makes_its_goal_and_starts_round_one_as_the_run(world: World) -> None:
    schedule = await world.schedule(goal_max_rounds=5)

    await world.firer.look()

    (ensured,) = world.turns.ensured
    assert ensured["conversation_id"] is None
    (call,) = world.turns.calls
    (goal,) = world.goal_store.rows.values()
    assert world.goal_store.rows == {(OWNER, call["conversation_id"]): goal}
    assert (goal.objective, goal.max_rounds, goal.rounds_started) == (
        "Summarize the week.",
        5,
        1,
    )
    assert goal.phase is GoalPhase.ACTIVE
    # The Run is the goal's round 1: it carries the goal's identity and the harness line, then the
    # same prose the goal command writes for that objective.
    assert call["goal"] == ScheduledGoalRound(goal.id, goal.revision, 1)
    assert call["mode"] == "research"
    assert call["caller"] == USER
    heading, rest = call["query"].split("\n\n", 1)
    assert heading == '[Scheduled: "Weekly review", 2026-12-07 11:00 UTC]'
    assert rest == goal_round_prompt(goal=goal, round_number=1)
    (fire,) = world.fires(schedule)
    assert fire.state is FireState.STARTED
    assert fire.conversation_id == call["conversation_id"]


async def test_the_same_conversation_goal_fire_goes_into_the_conversation_it_keeps(
    world: World,
) -> None:
    await _keeping(world)

    await world.firer.look()

    assert world.turns.ensured[0]["conversation_id"] == KEPT
    assert world.turns.calls[0]["conversation_id"] == KEPT
    assert world.goal_store.rows[(OWNER, KEPT)].rounds_started == 1


async def test_a_conversation_whose_goal_is_still_working_skips_the_fire(world: World) -> None:
    schedule = await _keeping(world)
    working = Goal.create("something the owner asked for")
    world.goal_store.rows[(OWNER, KEPT)] = working

    await world.firer.look()

    assert world.states(schedule) == ["skipped_busy"]
    assert world.turns.calls == []
    assert world.goal_store.rows[(OWNER, KEPT)] == working
    assert world.current(schedule).state is ScheduleState.ACTIVE


@pytest.mark.parametrize(
    ("waiting", "phase"),
    [
        (Goal.create("old").pause(revision=1), "paused"),
        (Goal.create("old").block(revision=1, reason="needs the owner"), "blocked"),
    ],
)
async def test_a_goal_waiting_for_its_owner_refuses_the_fire_and_only_the_streak_pauses(
    world: World, waiting: Goal, phase: str
) -> None:
    schedule = await _keeping(world)
    world.goal_store.rows[(OWNER, KEPT)] = waiting

    await world.firer.look()

    assert world.states(schedule) == ["refused"]
    assert phase in (world.fires(schedule)[0].reason or "")
    assert world.turns.calls == []
    assert world.goal_store.rows[(OWNER, KEPT)] == waiting
    still = world.current(schedule)
    assert (still.state, still.pause_reason) == (ScheduleState.ACTIVE, None)

    for _ in range(2):
        world.clock.advance(hours=1)
        await world.firer.look()
    assert world.states(schedule) == ["refused", "refused", "refused"]
    paused = world.current(schedule)
    assert paused.state is ScheduleState.PAUSED
    assert phase in (paused.pause_reason or "")


async def test_a_finished_goal_is_replaced_by_the_fires_own(world: World) -> None:
    schedule = await _keeping(world)
    finished = Goal.create("last week's objective").complete(revision=1)
    world.goal_store.rows[(OWNER, KEPT)] = finished

    await world.firer.look()

    replaced = world.goal_store.rows[(OWNER, KEPT)]
    assert replaced.id != finished.id
    assert (replaced.objective, replaced.phase, replaced.rounds_started) == (
        "Summarize the week.",
        GoalPhase.ACTIVE,
        1,
    )
    assert world.states(schedule) == ["started"]
    assert world.turns.calls[0]["goal"].goal_id == replaced.id


async def test_a_failed_first_round_fails_the_fire_and_blocks_the_goal_with_no_second_round(
    world: World,
) -> None:
    """The goal driver settles the Run as a round: failed means blocked, not another round."""
    schedule = await world.schedule(goal_max_rounds=3)
    await world.firer.look()
    (call,) = world.turns.calls
    run_id = world.fires(schedule)[0].run_id or ""
    session = call["conversation_id"]
    goal_round = call["goal"]
    world.runs.finish(
        run_id,
        "failed",
        error_kind="model_error",
        error_message="the provider refused",
        request={
            "agent_session_id": session,
            "continuation_kind": "goal_round",
            "goal_id": goal_round.goal_id,
            "goal_revision": goal_round.goal_revision,
            "goal_round": goal_round.goal_round,
        },
    )
    answers = SimpleNamespace(create=AsyncMock())
    driver = GoalRoundDriver(
        goals=world.goals, answers=cast(Any, answers), runs=cast(Any, world.runs)
    )

    settled = _settled(run_id, "failed")
    await driver.settle(settled)
    await world.firer.settle(settled)

    assert world.states(schedule) == ["failed"]
    assert world.fires(schedule)[0].reason == "model_error the provider refused"
    blocked = world.goal_store.rows[(OWNER, session)]
    assert blocked.phase is GoalPhase.BLOCKED
    assert "failed" in (blocked.blocked_reason or "")
    answers.create.assert_not_awaited()
    assert len(world.turns.calls) == 1


async def test_a_retried_goal_fire_finds_its_goal_and_admits_no_second_round(
    world: World,
) -> None:
    schedule = await world.schedule(goal_max_rounds=3)
    world.turns.raises = [RuntimeError("the database blinked")]
    await world.firer.look()
    assert world.states(schedule) == ["pending"]
    (stored,) = world.goal_store.rows.values()
    assert stored.rounds_started == 1

    world.clock.advance(seconds=70)
    await world.firer.look()

    assert world.states(schedule) == ["started"]
    first, retry = world.turns.calls
    assert first["idempotency_key"] == retry["idempotency_key"]
    assert first["goal"] == retry["goal"]
    assert first["query"] == retry["query"]
    (after,) = world.goal_store.rows.values()
    assert (after.id, after.rounds_started, after.revision) == (
        stored.id,
        1,
        stored.revision,
    )


async def test_a_retry_after_the_goal_was_made_but_its_round_was_not_admits_round_one_once(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = await world.schedule(goal_max_rounds=3)
    admit = world.goals.admit_round
    failures = [RuntimeError("the database blinked")]

    async def flaky(**call: Any) -> tuple[Goal, int]:
        if failures:
            raise failures.pop()
        return await admit(**call)

    monkeypatch.setattr(world.goals, "admit_round", flaky)

    await world.firer.look()
    assert world.states(schedule) == ["pending"]
    (made,) = world.goal_store.rows.values()
    assert made.rounds_started == 0
    assert world.turns.calls == []

    world.clock.advance(seconds=70)
    await world.firer.look()

    assert world.states(schedule) == ["started"]
    (after,) = world.goal_store.rows.values()
    assert (after.id, after.rounds_started) == (made.id, 1)
    assert len(world.turns.calls) == 1


async def test_a_goal_whose_round_cannot_start_is_cleared_so_recovery_does_not_start_it(
    world: World,
) -> None:
    schedule = await world.schedule(goal_max_rounds=3)
    world.turns.raises = [SessionWorkBusyError("run-live")]

    await world.firer.look()

    assert world.states(schedule) == ["skipped_busy"]
    assert world.goal_store.rows == {}


async def test_a_goal_fire_that_misses_its_window_clears_the_goal_it_made(world: World) -> None:
    schedule = await world.schedule(goal_max_rounds=3)
    world.turns.raises = [RuntimeError("the providers are down")]
    await world.firer.look()
    assert len(world.goal_store.rows) == 1

    world.clock.advance(minutes=45)
    await world.firer.look()

    assert world.states(schedule) == ["missed"]
    assert world.goal_store.rows == {}


# -- review: retries, paused Schedules, overlap, errors and the identity's age ------------------


def _service(world: World) -> ScheduleService:
    return ScheduleService(
        store=world.store, firer=world.firer, settings=SETTINGS, clock=world.clock
    )


class _SlowTurns(FakeTurns):
    """Accepting a turn takes a while: admission, connection binding, policy reads."""

    gate: asyncio.Event

    async def accept_scheduled_turn(self, **call: Any) -> ScheduledTurn:
        await self.gate.wait()
        return await super().accept_scheduled_turn(**call)


async def test_two_run_now_requests_never_start_two_runs(world: World) -> None:
    turns = _SlowTurns(world.runs)
    turns.gate = asyncio.Event()
    world.firer._turns = turns
    service = _service(world)
    schedule = await world.schedule()

    first = asyncio.create_task(service.run_now(USER, schedule.schedule_id))
    await asyncio.sleep(0)
    world.clock.advance(seconds=1)
    with pytest.raises(ScheduleStateError, match="already has a fire"):
        await service.run_now(USER, schedule.schedule_id)
    turns.gate.set()

    started = await first
    assert started is not None and started.state is FireState.STARTED
    assert len(turns.calls) == 1


async def test_a_fire_waits_behind_an_earlier_fire_that_is_still_starting(world: World) -> None:
    schedule = await world.schedule()
    first = await world.store.insert_manual_fire(
        schedule, world.clock.now - timedelta(minutes=2), fires_kept=200
    )
    later = await world.store.insert_manual_fire(schedule, world.clock.now, fires_kept=200)
    assert first is not None and later is not None

    ended = await world.firer.start(later)

    assert ended is not None and ended.state is FireState.SKIPPED_BUSY
    assert "earlier fire" in (ended.reason or "")
    assert world.turns.calls == []


async def test_a_retried_fire_whose_run_exists_is_recorded_before_any_rule_is_asked(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process died after the Run was made and before the Fire heard of it."""
    schedule = await world.schedule()
    finish = world.store.finish_fire
    attempts = {"made": 0}

    async def crash_once(*args: Any, **kwargs: Any) -> Fire | None:
        attempts["made"] += 1
        if attempts["made"] == 1:
            raise ConnectionError("process died")
        return await finish(*args, **kwargs)

    monkeypatch.setattr(world.store, "finish_fire", crash_once)
    await world.firer.look()
    assert world.states(schedule) == ["pending"]
    (turn,) = world.turns.accepted.values()

    world.store.spent = 5_000  # the ceiling was reached meanwhile; the Run exists all the same
    world.clock.advance(seconds=61)
    await world.firer.look()

    (fire,) = world.fires(schedule)
    assert (fire.state, fire.run_id, fire.conversation_id) == (
        FireState.STARTED,
        turn.run_id,
        turn.conversation_id,
    )
    assert world.current(schedule).state is ScheduleState.ACTIVE
    assert len(world.turns.calls) == 1


async def test_a_retried_fire_past_its_window_still_records_a_run_that_exists(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = await world.schedule(goal_max_rounds=3)
    finish = world.store.finish_fire
    attempts = {"made": 0}

    async def crash_once(*args: Any, **kwargs: Any) -> Fire | None:
        attempts["made"] += 1
        if attempts["made"] == 1:
            raise ConnectionError("process died")
        return await finish(*args, **kwargs)

    monkeypatch.setattr(world.store, "finish_fire", crash_once)
    await world.firer.look()
    world.clock.advance(minutes=45)  # past the half hour an hourly slot is worth

    await world.firer.look()

    assert world.states(schedule) == ["started"]
    (goal,) = world.goal_store.rows.values()
    assert (goal.phase, goal.rounds_started) == (GoalPhase.ACTIVE, 1)


async def test_a_key_that_already_names_a_run_is_that_fires_run(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = await world.schedule()
    world.turns.raises = [IdempotencyKeyConflict()]
    existing = ScheduledTurn(run_id="00000000-0000-4000-8000-0000000000aa", conversation_id="c-1")

    async def found(**_call: Any) -> ScheduledTurn | None:
        return existing

    monkeypatch.setattr(world.turns, "replay_scheduled_turn", found)
    await world.firer.look()

    (fire,) = world.fires(schedule)
    assert (fire.state, fire.run_id, fire.conversation_id) == (
        FireState.STARTED,
        existing.run_id,
        "c-1",
    )


async def test_a_key_conflict_with_no_run_to_find_is_refused_with_a_reason(world: World) -> None:
    schedule = await world.schedule()
    world.turns.raises = [ConversationSubmissionConflict("reused")]

    await world.firer.look()

    assert world.states(schedule) == ["refused"]
    assert "different request" in (world.fires(schedule)[0].reason or "")


async def test_a_pending_fire_of_a_schedule_paused_since_is_cancelled_not_started(
    world: World,
) -> None:
    schedule = await world.schedule()
    world.turns.raises = [RuntimeError("the runtime is briefly unavailable")]
    await world.firer.look()
    assert world.states(schedule) == ["pending"]

    await world.store.pause(OWNER, schedule.schedule_id, reason="paused by its owner")
    world.clock.advance(seconds=61)
    await world.firer.look()

    (fire,) = world.fires(schedule)
    assert fire.state is FireState.CANCELLED
    assert fire.reason == "the Schedule was paused before it started"
    assert len(world.turns.calls) == 1
    assert world.current(schedule).state is ScheduleState.PAUSED


async def test_the_owner_may_run_a_paused_schedule_now_and_it_stays_paused(world: World) -> None:
    schedule = await world.schedule()
    await world.store.pause(OWNER, schedule.schedule_id, reason="paused by its owner")

    started = await _service(world).run_now(USER, schedule.schedule_id)

    assert started is not None and started.state is FireState.STARTED and started.manual
    paused = world.current(schedule)
    assert (paused.state, paused.pause_reason) == (ScheduleState.PAUSED, "paused by its owner")


async def test_settling_the_previous_fire_into_a_pause_cancels_the_fire_after_it(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = await world.schedule()
    for _ in range(2):  # two failures the observer reported, and a third it never did
        await world.firer.look()
        run_id = world.fires(schedule)[-1].run_id or ""
        world.runs.finish(run_id, "failed", error_kind="model_error")
        await world.firer.settle(_settled(run_id, "failed"))
        world.clock.advance(hours=1)
    await world.firer.look()
    lost = world.fires(schedule)[-1].run_id or ""
    world.runs.finish(lost, "failed", error_kind="model_error")

    async def nothing(_now: datetime) -> None:
        return None

    monkeypatch.setattr(world.firer, "_settle_lost", nothing)
    world.clock.advance(hours=1)
    await world.firer.look()

    assert world.states(schedule) == ["failed", "failed", "failed", "cancelled"]
    assert world.current(schedule).state is ScheduleState.PAUSED
    assert len(world.turns.calls) == 3


async def test_a_connection_change_while_admitting_is_a_wait_not_a_refusal(world: World) -> None:
    schedule = await world.schedule()
    world.turns.raises = [AnswerConnectionsChangedError()]

    await world.firer.look()

    (fire,) = world.fires(schedule)
    assert fire.state is FireState.PENDING
    assert "AnswerConnectionsChangedError" in (fire.last_error or "")
    assert world.current(schedule).state is ScheduleState.ACTIVE

    world.clock.advance(seconds=61)
    await world.firer.look()
    assert world.states(schedule) == ["started"]


async def test_fires_that_keep_failing_to_start_are_missed_with_their_error_and_pause_the_schedule(
    world: World,
) -> None:
    schedule = await world.schedule()
    for _ in range(3):
        world.turns.raises = [RuntimeError("the store is unreadable")]
        await world.firer.look()
        world.clock.advance(minutes=45)  # past what an hourly slot is worth
        await world.firer.look()
        world.clock.advance(minutes=15)

    assert world.states(schedule) == ["missed", "missed", "missed"]
    assert all(
        "RuntimeError: the store is unreadable" in (f.reason or "") for f in world.fires(schedule)
    )
    paused = world.current(schedule)
    assert paused.state is ScheduleState.PAUSED
    assert "the store is unreadable" in (paused.pause_reason or "")


async def test_a_missed_gap_is_not_a_failure_and_does_not_count_toward_the_streak(
    world: World,
) -> None:
    schedule = await world.schedule()
    for _ in range(3):  # the slot was too late to start: missed, with no error behind it
        world.clock.advance(minutes=40)
        await world.firer.look()
        world.clock.advance(minutes=20)

    assert set(world.states(schedule)) == {"missed"}
    assert world.current(schedule).state is ScheduleState.ACTIVE


async def test_one_fire_the_sweep_cannot_read_does_not_stop_the_others(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await world.schedule()
    second = await world.schedule()
    await world.firer.look()
    runs = [fire.run_id or "" for fire in (*world.fires(first), *world.fires(second))]
    for run_id in runs:
        world.runs.finish(run_id)
    get = world.runs.get

    async def flaky(*, owner_id: str, run_id: str) -> RunView | None:
        if run_id == runs[0]:
            raise ConnectionError("the run store blinked")
        return await get(owner_id=owner_id, run_id=run_id)

    monkeypatch.setattr(world.runs, "get", flaky)
    world.clock.advance(minutes=6)
    await world.firer.look()

    assert world.states(first) == ["started"]
    assert world.states(second) == ["succeeded"]


async def test_a_run_still_working_is_looked_at_once_per_patience_not_every_look(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = await world.schedule()
    await world.firer.look()
    run_id = world.fires(schedule)[0].run_id or ""
    reads = {"n": 0}
    get = world.runs.get

    async def counting(*, owner_id: str, run_id: str) -> RunView | None:
        reads["n"] += 1
        return await get(owner_id=owner_id, run_id=run_id)

    monkeypatch.setattr(world.runs, "get", counting)
    world.clock.advance(minutes=6)
    await world.firer.look()
    assert reads["n"] == 1
    assert world.fires(schedule)[0].updated_at == world.clock.now

    world.clock.advance(minutes=1)
    await world.firer.look()
    assert reads["n"] == 1, "a Run read a minute ago is not read again"
    world.runs.finish(run_id)
    world.clock.advance(minutes=5)
    await world.firer.look()
    assert world.states(schedule) == ["succeeded"]


async def test_a_signed_in_identity_older_than_the_limit_is_refused_until_the_owner_resumes(
    world: World,
) -> None:
    schedule = await world.schedule()
    world.clock.now = datetime(2027, 1, 9, 11, 0, 20, tzinfo=UTC)  # a month on, at a slot

    await world.firer.look()

    assert world.states(schedule)[-1] == "refused"
    paused = world.current(schedule)
    assert paused.state is ScheduleState.PAUSED
    assert "more than 30 days ago" in (paused.pause_reason or "")
    assert "resume" in (paused.pause_reason or "")
    assert world.turns.calls == []

    resumed = await _service(world).resume(USER, schedule.schedule_id)
    assert resumed is not None and resumed.principal_stamped_at == world.clock.now
    world.clock.advance(hours=1)
    await world.firer.look()
    assert world.states(schedule)[-1] == "started"


async def test_an_identity_that_is_not_signed_in_does_not_age(world: World) -> None:
    anonymous = UserContext(user_id="anonymous", auth_mode="none")
    schedule = await world.schedule(principal=anonymous, owner=owner_id_from_user(anonymous))
    world.clock.now = datetime(2027, 1, 9, 11, 0, 20, tzinfo=UTC)

    await world.firer.look()

    assert world.states(schedule)[-1] == "started"


async def test_a_single_time_that_starts_nothing_waits_paused_for_a_new_time(
    world: World,
) -> None:
    schedule = await world.schedule(at=FIRST_SLOT)
    world.turns.raises = [SessionWorkBusyError("run-live")]

    await world.firer.look()

    assert world.states(schedule) == ["skipped_busy"]
    waiting = world.current(schedule)
    assert (waiting.state, waiting.next_fire_at) == (ScheduleState.PAUSED, None)
    assert "already working" in (waiting.pause_reason or "")

    service = _service(world)
    with pytest.raises(ScheduleStateError, match="no time"):
        await service.resume(USER, schedule.schedule_id)
    later = world.clock.now + timedelta(hours=2)
    edited = await service.update(USER, schedule.schedule_id, at=later)
    assert edited is not None and edited.next_fire_at == later
    resumed = await service.resume(USER, schedule.schedule_id)
    assert resumed is not None and resumed.state is ScheduleState.ACTIVE

    world.clock.now = later + timedelta(seconds=20)
    await world.firer.look()
    assert world.states(schedule) == ["skipped_busy", "started"]
    assert world.current(schedule).state is ScheduleState.COMPLETED


async def test_a_single_time_that_was_missed_is_paused_not_completed(world: World) -> None:
    schedule = await world.schedule(at=FIRST_SLOT)
    world.clock.advance(hours=3)

    await world.firer.look()

    assert world.states(schedule) == ["missed"]
    waiting = world.current(schedule)
    assert waiting.state is ScheduleState.PAUSED
    assert "too late" in (waiting.pause_reason or "")


async def test_a_manual_fire_on_a_single_time_leaves_its_state_alone(world: World) -> None:
    schedule = await world.schedule(at=FIRST_SLOT + timedelta(days=1))

    started = await _service(world).run_now(USER, schedule.schedule_id)

    assert started is not None and started.state is FireState.STARTED
    assert world.current(schedule).state is ScheduleState.ACTIVE
