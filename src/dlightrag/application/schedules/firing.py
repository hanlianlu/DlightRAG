# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The look: turn due Schedules into Fires, Fires into Runs, and settled Runs back into Fires (ADR 0103).

Each look does three things, in this order. It retries Fires that were written but never started
(the process died between the Fire and the Run); it settles Fires whose Run ended while nothing
was listening; and it claims the Schedules whose time has come. Claiming is one database
transaction (``ScheduleStore.claim_due``): the slot's Fire is written under a unique key and the
Schedule moves on, so any number of processes can look and a slot is still taken once. The Run
starts afterwards, with an idempotency key built from the slot, so a retry replays the Run that
was created instead of creating another.

A Fire that is retried first asks whether its Run already exists, and records it as started if
it does: the rules judge a Run that has not been made, never one that has. A Fire the Schedule's
owner paused or deleted before it started is cancelled, unless the owner asked for it (Run now).

Starting a Run applies the rules in one order, and the first that holds ends the Fire:

1. the Schedule's identity must still be its owner, and still be recent enough to trust;
2. the owner must still read a workspace;
3. the previous Fire's Run, the goal it left running, or an earlier Fire still starting must be
   finished (a skip, not a queue);
4. the Schedule must be under its daily token ceiling;
5. a goal Schedule's conversation must hold no goal still working or waiting for its owner;
6. admission may refuse: the Session is busy, or a run limit is reached (a skip).

A goal Schedule's fire makes its Goal (the prompt is the objective), admits round 1, and starts
that round as the Run through the same admission and prompt the ``/goal`` command uses, so the goal
driver settles it as any round: a failed first round blocks the goal, and a good one goes on to the
next. The goal's id comes from the Fire's key, so a retried Fire finds the goal it made and neither
makes a second nor admits a second round.

Nothing here retries a failed Run, and a Fire that does not start is never queued: the next slot
is the next chance.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

from dlightrag.application.access.control import AccessAction, AccessControl
from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.access.workspaces import AccessGate
from dlightrag.application.answer.goal_rounds import goal_round_prompt
from dlightrag.application.answer_runs.service import ActiveRunLimitError
from dlightrag.application.config.sections import SchedulesSettings
from dlightrag.application.errors import ApplicationInputError, ApplicationNotFoundError
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunView,
    SessionWorkBusyError,
)
from dlightrag.application.schedules.model import (
    Fire,
    FireState,
    Schedule,
    ScheduledConversation,
    ScheduledGoalRound,
    ScheduledTurn,
    ScheduleState,
    Target,
    plan_due,
    retry_window,
    scheduled_turn_text,
)
from dlightrag.application.schedules.store import DueFire, ScheduleStore
from dlightrag.application.web_conversations.models import ConversationSubmissionConflict
from dlightrag.engine.answer.goal import Goal, GoalError, GoalPhase
from dlightrag.engine.runtime.coordinator import SettledRun

logger = logging.getLogger(__name__)

#: Due Schedules one claim transaction takes, and the Fires one sweep reads.
_CLAIM_BATCH = 20
_SWEEP_BATCH = 50

#: The window the daily token ceiling is counted over.
_SPEND_WINDOW = timedelta(hours=24)

#: How long a started Fire waits for its Run's settlement to arrive before the look asks the Run.
_SETTLEMENT_PATIENCE = timedelta(minutes=5)

_NAMESPACE = "dlightrag-schedule-fire"


class ScheduledTurns(Protocol):
    """The conversations that take a scheduled turn: a new one, or the Schedule's own."""

    async def ensure_scheduled_conversation(
        self, *, caller: UserContext, idempotency_key: str, conversation_id: str | None
    ) -> ScheduledConversation: ...

    async def replay_scheduled_turn(
        self, *, caller: UserContext, idempotency_key: str
    ) -> ScheduledTurn | None: ...

    async def accept_scheduled_turn(
        self,
        *,
        caller: UserContext,
        idempotency_key: str,
        conversation_id: str | None,
        query: str,
        workspaces: tuple[str, ...],
        mode: str,
        title: str,
        goal: ScheduledGoalRound | None = None,
    ) -> ScheduledTurn: ...


class FireRuns(Protocol):
    """The Run service, as far as the look reads a Run."""

    async def get(self, *, owner_id: str, run_id: str) -> RunView | None: ...


class SessionGoals(Protocol):
    """The Session goal service, as far as a fire reads a goal and starts or drops its own."""

    async def get(self, *, owner_id: str, session_id: str) -> Goal | None: ...

    async def create(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        objective: str,
        max_rounds: int,
        goal_id: str | None = None,
    ) -> Goal: ...

    async def admit_round(
        self, *, owner_id: str, session_id: str, run_id: str, goal_id: str, revision: int
    ) -> tuple[Goal, int]: ...

    async def clear(self, *, owner_id: str, session_id: str, run_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class _RoundOne:
    """A goal fire that has made its goal and admitted round 1, ready to start that round."""

    conversation: ScheduledConversation
    goal: Goal


@dataclass(frozen=True, slots=True)
class _Stopped:
    """A goal fire that ended before starting any Run: the Fire as it now stands."""

    fire: Fire | None


def fire_key(schedule_id: str, slot: datetime) -> str:
    """The idempotency key of one slot's Run: the same slot always names the same key."""
    stamp = slot.astimezone(UTC).isoformat(timespec="microseconds")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_NAMESPACE}:{schedule_id}:{stamp}"))


def _goal_id(schedule: Schedule, fire: Fire) -> str:
    """The id of the goal one Fire makes: the same Fire always names the same goal."""
    key = fire_key(schedule.schedule_id, fire.slot)
    return f"goal-{uuid.uuid5(uuid.NAMESPACE_URL, f'{_NAMESPACE}-goal:{key}')}"


def _now() -> datetime:
    return datetime.now(UTC)


class ScheduleFirer:
    """Start the Runs of due Schedules and record how each Fire went."""

    def __init__(
        self,
        *,
        store: ScheduleStore,
        turns: ScheduledTurns,
        runs: FireRuns,
        goals: SessionGoals,
        access: AccessControl,
        settings: SchedulesSettings,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._store = store
        self._turns = turns
        self._runs = runs
        self._goals = goals
        self._access = access
        self._settings = settings
        self._clock = clock
        self._tasks: set[asyncio.Task[None]] = set()

    # -- the look ----------------------------------------------------------------------------

    async def look(self) -> None:
        """One pass: retry unstarted Fires, settle lost ones, then claim and start what is due."""
        now = self._clock()
        for sweep in (self._retry_pending, self._settle_lost):
            try:
                await sweep(now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("A schedule sweep failed", exc_info=True)
        while True:
            claim = await self._store.claim_due(
                now,
                limit=_CLAIM_BATCH,
                fires_kept=self._settings.fires_kept,
                plan=lambda schedule: plan_due(schedule.cadence, schedule.next_fire_at or now, now),
            )
            for due in claim.due:
                await self._start_guarded(due)
            if claim.schedules < _CLAIM_BATCH:
                return

    async def _retry_pending(self, now: datetime) -> None:
        """Start the Fires a process wrote and died before starting, while their slot is still worth it."""
        lease = timedelta(seconds=2 * self._settings.look_interval_seconds)
        for due in await self._store.claim_pending(now, lease=lease, limit=_SWEEP_BATCH):
            await self._start_guarded(due, retried=True)

    async def _drop_unstarted_goal(self, due: DueFire) -> None:
        """Clear the goal a missed goal Fire made before it could start round 1."""
        schedule, fire = due.schedule, due.fire
        if schedule.goal_max_rounds is None:
            return
        try:
            conversation = await self._turns.ensure_scheduled_conversation(
                caller=schedule.principal,
                idempotency_key=fire_key(schedule.schedule_id, fire.slot),
                conversation_id=(
                    schedule.conversation_id
                    if schedule.target is Target.SAME_CONVERSATION
                    else None
                ),
            )
        except ApplicationNotFoundError:
            return
        await self._clear_goal(schedule.owner_id, conversation.session_id, _goal_id(schedule, fire))

    async def _settle_lost(self, now: datetime) -> None:
        """Settle the Fires whose Run ended without the settlement observer reaching this process.

        A Run still working is touched, so the sweep reads the Fires nobody has looked at before it
        reads that one again: a few long Runs cannot keep the others out of a batch.
        """
        older_than = now - _SETTLEMENT_PATIENCE
        for fire in await self._store.unsettled_fires(older_than=older_than, limit=_SWEEP_BATCH):
            if fire.run_id is None:
                continue
            try:
                view = await self._runs.get(owner_id=fire.owner_id, run_id=fire.run_id)
                if view is None:
                    await self._end(
                        fire,
                        FireState.CANCELLED,
                        reason="its run no longer exists",
                        expected=(FireState.STARTED,),
                    )
                elif view.terminal:
                    await self._settle(fire, view)
                else:
                    await self._store.touch_fire(fire.owner_id, fire.fire_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Could not settle fire %s", fire.fire_id, exc_info=True)

    async def _start_guarded(self, due: DueFire, *, retried: bool = False) -> None:
        try:
            await self.start(due, retried=retried)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "Fire %s of Schedule %s could not be started; a later look retries it",
                due.fire.fire_id,
                due.schedule.schedule_id,
                exc_info=True,
            )
            await self._note_error(due.fire, error)

    async def _note_error(self, fire: Fire, error: Exception) -> None:
        """Keep what stopped this attempt on the Fire, to say why it was missed if it never starts."""
        try:
            await self._store.record_error(
                fire.owner_id, fire.fire_id, f"{type(error).__name__}: {error}"[:500]
            )
        except Exception:
            logger.warning("Could not record the error of fire %s", fire.fire_id, exc_info=True)

    # -- starting one Fire -------------------------------------------------------------------

    async def start(self, due: DueFire, *, retried: bool = False) -> Fire | None:
        """Apply the rules to one pending Fire; start its Run, or end it with the reason it did not.

        A ``retried`` Fire (one a process wrote and did not finish) first looks for the Run it may
        already have started and records that, before any rule is asked; and it is missed once its
        slot is no longer worth starting. Returns the Fire as it now stands, or ``None`` when
        another process had already moved it. What the rules cannot decide (a database or
        dependency that is briefly out) raises, and the Fire stays pending for the next look.
        """
        fire = due.fire
        key = fire_key(due.schedule.schedule_id, fire.slot)
        if retried:
            existing = await self._turns.replay_scheduled_turn(
                caller=due.schedule.principal, idempotency_key=key
            )
            if existing is not None:
                return await self._started(fire, existing)
            if self._clock() - fire.slot > retry_window(
                due.schedule.cadence, fire.slot, manual=fire.manual
            ):
                return await self._expire(due)
        schedule, stopped = await self._reread(due)
        if schedule is None:
            return None
        if stopped is not None:
            return await self._cancel(fire, stopped)
        principal = schedule.principal
        if owner_id_from_user(principal) != schedule.owner_id:
            return await self._refuse(
                fire, "the identity this Schedule runs as no longer belongs to its owner"
            )
        max_age = timedelta(days=self._settings.principal_max_age_days)
        if principal.auth_mode == "jwt" and self._clock() - schedule.principal_stamped_at > max_age:
            return await self._refuse(
                fire,
                f"the sign-in this Schedule acts under was last confirmed more than "
                f"{self._settings.principal_max_age_days} days ago; open the Schedule and resume "
                "it to renew it",
            )
        gate = AccessGate(self._access, principal)
        readable = await gate.authorized_workspace_ids(
            AccessAction.WORKSPACE_QUERY, schedule.workspaces
        )
        workspaces = tuple(name for name in schedule.workspaces if name in readable)
        if not workspaces:
            return await self._refuse(fire, "its owner can no longer read any of its workspaces")
        if busy := await self._previous_work(schedule, fire):
            return await self._end(
                fire, FireState.SKIPPED_BUSY, reason=busy, expected=(FireState.PENDING,)
            )
        # Settling the previous Fire may have paused the Schedule, so it is read again.
        schedule, stopped = await self._reread(due)
        if schedule is None:
            return None
        if stopped is not None:
            return await self._cancel(fire, stopped)
        spent = await self._store.spent_tokens(
            schedule.owner_id, schedule.schedule_id, since=self._clock() - _SPEND_WINDOW
        )
        if spent >= schedule.max_tokens_per_day:
            return await self._refuse(
                fire,
                f"it used {spent} tokens in the last 24 hours, at or over its ceiling of "
                f"{schedule.max_tokens_per_day}",
            )
        conversation_id = (
            schedule.conversation_id if schedule.target is Target.SAME_CONVERSATION else None
        )
        prompt = schedule.prompt
        round_one: _RoundOne | None = None
        goal: ScheduledGoalRound | None = None
        if schedule.goal_max_rounds is not None:
            begun = await self._begin_goal(schedule, fire, schedule.goal_max_rounds)
            if isinstance(begun, _Stopped):
                return begun.fire
            round_one = begun
            conversation_id = begun.conversation.conversation_id
            prompt = goal_round_prompt(goal=begun.goal, round_number=1)
            goal = ScheduledGoalRound(begun.goal.id, begun.goal.revision, 1)
        try:
            turn = await self._turns.accept_scheduled_turn(
                caller=principal,
                idempotency_key=key,
                conversation_id=conversation_id,
                query=scheduled_turn_text(
                    name=schedule.name, slot=fire.slot, cadence=schedule.cadence, prompt=prompt
                ),
                workspaces=workspaces,
                mode=schedule.mode,
                title=_title(schedule, fire),
                goal=goal,
            )
        except SessionWorkBusyError:
            await self._drop_goal(schedule, round_one)
            return await self._end(
                fire,
                FireState.SKIPPED_BUSY,
                reason="its conversation is already working on another run",
                expected=(FireState.PENDING,),
            )
        except (ActiveRunLimitError, RunAdmissionLimitExceededError) as limit:
            await self._drop_goal(schedule, round_one)
            return await self._end(
                fire, FireState.SKIPPED_LIMIT, reason=str(limit), expected=(FireState.PENDING,)
            )
        except ConversationSubmissionConflict, IdempotencyKeyConflict:
            # The key already names a Run, made from a request that read differently: that Run is
            # this Fire's, so the Fire records it rather than judging it again.
            existing = await self._turns.replay_scheduled_turn(
                caller=principal, idempotency_key=key
            )
            if existing is not None:
                return await self._started(fire, existing)
            return await self._refuse(
                fire, "the key of its run was already used by a different request"
            )
        except (ApplicationInputError, ApplicationNotFoundError) as refused:
            await self._drop_goal(schedule, round_one)
            return await self._refuse(fire, str(refused))
        return await self._started(fire, turn)

    async def _reread(self, due: DueFire) -> tuple[Schedule | None, str | None]:
        """The Schedule as it stands now, and why this Fire may no longer start (None if it may).

        A Fire the owner asked for runs on a Schedule that is paused or completed, and does not
        resume it; any other Fire of a Schedule that was paused or completed after the Fire was
        written must not start. A Schedule that was deleted has no Fire left to start.
        """
        stored = await self._store.get(due.schedule.owner_id, due.schedule.schedule_id)
        if stored is None:
            return None, None
        if due.fire.manual:
            as_asked = replace(
                stored,
                principal=due.schedule.principal,
                principal_stamped_at=due.schedule.principal_stamped_at,
            )
            return as_asked, None
        if stored.state is ScheduleState.PAUSED:
            return stored, "the Schedule was paused before it started"
        if stored.state is ScheduleState.COMPLETED:
            return stored, "the Schedule was completed before it started"
        return stored, None

    async def _cancel(self, fire: Fire, reason: str) -> Fire | None:
        return await self._end(
            fire, FireState.CANCELLED, reason=reason, expected=(FireState.PENDING,)
        )

    async def _started(self, fire: Fire, turn: ScheduledTurn) -> Fire | None:
        return await self._end(
            fire,
            FireState.STARTED,
            run_id=turn.run_id,
            conversation_id=turn.conversation_id,
            expected=(FireState.PENDING,),
        )

    async def _expire(self, due: DueFire) -> Fire | None:
        """End a Fire no Run was made for as missed, saying what stopped its last attempt."""
        reason = "it could not be started while it was still worth starting"
        if due.fire.last_error:
            reason = f"{reason}: {due.fire.last_error}"
        ended = await self._end(
            due.fire, FireState.MISSED, reason=reason, expected=(FireState.PENDING,)
        )
        await self._drop_unstarted_goal(due)
        return ended

    async def _begin_goal(
        self, schedule: Schedule, fire: Fire, max_rounds: int
    ) -> _RoundOne | _Stopped:
        """Make the Fire's goal and admit its round 1, or end the Fire when the conversation holds one.

        What the conversation's goal says decides first: one still working skips the Fire, one
        waiting for its owner (paused or blocked) refuses it, and that refusal counts toward the
        failure streak without pausing at once. A finished goal, or none, is replaced by the Fire's
        own, whose id the Fire's key decides. A retry finds that goal and does not make or admit
        another: it admits round 1 only when no round has been admitted.
        """
        owner_id, goal_id = schedule.owner_id, _goal_id(schedule, fire)
        conversation = await self._turns.ensure_scheduled_conversation(
            caller=schedule.principal,
            idempotency_key=fire_key(schedule.schedule_id, fire.slot),
            conversation_id=(
                schedule.conversation_id if schedule.target is Target.SAME_CONVERSATION else None
            ),
        )
        session_id = conversation.session_id
        goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
        if goal is not None and goal.id != goal_id and goal.phase is not GoalPhase.COMPLETE:
            if goal.phase is GoalPhase.ACTIVE:
                return _Stopped(
                    await self._end(
                        fire,
                        FireState.SKIPPED_BUSY,
                        reason="the goal in its conversation is still working",
                        expected=(FireState.PENDING,),
                    )
                )
            return _Stopped(
                await self._end(
                    fire,
                    FireState.REFUSED,
                    reason=f"the goal in its conversation is {goal.phase}; resume or clear it",
                    expected=(FireState.PENDING,),
                )
            )
        try:
            if goal is None or goal.id != goal_id:
                goal = await self._goals.create(
                    owner_id=owner_id,
                    session_id=session_id,
                    run_id="",
                    objective=schedule.prompt,
                    max_rounds=max_rounds,
                    goal_id=goal_id,
                )
            if goal.rounds_started == 0:
                goal, _ = await self._goals.admit_round(
                    owner_id=owner_id,
                    session_id=session_id,
                    run_id="",
                    goal_id=goal.id,
                    revision=goal.revision,
                )
        except GoalError as error:
            # Another attempt at this very Fire may have made or admitted it a moment before.
            goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
            if goal is None or goal.id != goal_id or goal.rounds_started != 1:
                return _Stopped(
                    await self._end(
                        fire,
                        FireState.SKIPPED_BUSY,
                        reason=str(error),
                        expected=(FireState.PENDING,),
                    )
                )
        return _RoundOne(conversation, goal)

    async def _drop_goal(self, schedule: Schedule, round_one: _RoundOne | None) -> None:
        """Clear the goal a fire made whose round 1 never started.

        Left active it would be picked up by goal recovery, which would start a round outside the
        Schedule's rules.
        """
        if round_one is None:
            return
        await self._clear_goal(
            schedule.owner_id, round_one.conversation.session_id, round_one.goal.id
        )

    async def _clear_goal(self, owner_id: str, session_id: str, goal_id: str) -> None:
        try:
            goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
            if (
                goal is not None
                and goal.id == goal_id
                and goal.phase is GoalPhase.ACTIVE
                and goal.rounds_started <= 1
            ):
                await self._goals.clear(owner_id=owner_id, session_id=session_id, run_id="")
        except Exception:
            logger.warning("Could not clear the goal %s of a fire that did not start", goal_id)

    async def _previous_work(self, schedule: Schedule, fire: Fire) -> str | None:
        """Why the previous Fire's work keeps this one from starting, or ``None`` when it is done.

        Work is done when the previous Run has settled and the goal its Session carries is no
        longer active. A Run that settled without its Fire hearing of it is settled here. An
        earlier Fire of the Schedule that is still being started counts as work too: its Run is
        about to exist, and starting this one beside it would be two at once.
        """
        if await self._store.earlier_pending_fire(
            schedule.owner_id, schedule.schedule_id, fire.slot, exclude=fire.fire_id
        ):
            return "an earlier fire of this Schedule is still starting"
        previous = await self._store.previous_fire(
            schedule.owner_id, schedule.schedule_id, fire.slot
        )
        if previous is None or previous.run_id is None:
            return None
        view = await self._runs.get(owner_id=schedule.owner_id, run_id=previous.run_id)
        if view is None:
            return None
        if not view.terminal:
            return "the previous run is still working"
        if previous.state is FireState.STARTED:
            await self._settle(previous, view)
        session_id = str(view.request.get("agent_session_id") or "")
        if session_id:
            goal = await self._goals.get(owner_id=schedule.owner_id, session_id=session_id)
            if goal is not None and goal.phase is GoalPhase.ACTIVE:
                return "the goal the previous run started is still working"
        return None

    # -- settlement --------------------------------------------------------------------------

    def run_settled(self, settled: SettledRun) -> None:
        """The coordinator's synchronous, schedule-only callback for every terminal Run."""
        if settled.run_kind != "answer":
            return
        try:
            task = asyncio.get_running_loop().create_task(self.settle(settled))
        except RuntimeError:  # no loop: the composition is shutting down
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def settle(self, settled: SettledRun) -> None:
        """Record how the Fire this Run started went; a failure here never reaches the Run."""
        try:
            fire = await self._store.fire_for_run(settled.owner_id, settled.run_id)
            if fire is None or fire.state is not FireState.STARTED:
                return
            view = await self._runs.get(owner_id=settled.owner_id, run_id=settled.run_id)
            if view is not None and view.terminal:
                await self._settle(fire, view)
        except Exception:
            logger.warning(
                "Schedule settlement failed after run %s ended", settled.run_id, exc_info=True
            )

    async def _settle(self, fire: Fire, view: RunView) -> Fire | None:
        if view.status == "succeeded":
            return await self._end(fire, FireState.SUCCEEDED, expected=(FireState.STARTED,))
        if view.status == "cancelled":
            return await self._end(
                fire,
                FireState.CANCELLED,
                reason="the run was cancelled",
                expected=(FireState.STARTED,),
            )
        reason = " ".join(part for part in (view.error_kind, view.error_message) if part)
        return await self._end(
            fire,
            FireState.FAILED,
            reason=(reason or "the run failed")[:500],
            expected=(FireState.STARTED,),
        )

    # -- endings -----------------------------------------------------------------------------

    async def _refuse(self, fire: Fire, reason: str) -> Fire | None:
        """End a Fire refused for a reason only the owner can change: the Schedule pauses."""
        return await self._end(
            fire,
            FireState.REFUSED,
            reason=reason,
            pause_reason=reason,
            expected=(FireState.PENDING,),
        )

    async def _end(
        self,
        fire: Fire,
        state: FireState,
        *,
        expected: tuple[FireState, ...],
        reason: str | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        pause_reason: str | None = None,
    ) -> Fire | None:
        return await self._store.finish_fire(
            fire.owner_id,
            fire.fire_id,
            expected=expected,
            state=state,
            reason=reason,
            run_id=run_id,
            conversation_id=conversation_id,
            pause_reason=pause_reason,
            failure_streak=self._settings.failure_streak,
        )


def _title(schedule: Schedule, fire: Fire) -> str:
    if schedule.target is Target.SAME_CONVERSATION:
        return schedule.name
    local = fire.slot.astimezone(schedule.cadence.zone)
    return f"{schedule.name} · {local:%Y-%m-%d %H:%M}"[:120]


__all__ = [
    "FireRuns",
    "ScheduleFirer",
    "ScheduledTurns",
    "SessionGoals",
    "fire_key",
]
