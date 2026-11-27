# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Schedule and its Fires (ADR 0103): what the owner wrote, and what each slot did.

This module has no I/O and no clock. A Schedule is the owner's record of one turn to start at
times the cadence names; a Fire is one slot's outcome. The pure rules that decide what a due
Schedule does live here too, so they are tested without a database: how a slot that was missed is
collapsed, how long a late slot is still worth running, and when failures stop a Schedule.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Self

from dlightrag.application.access.principal import UserContext
from dlightrag.application.schedules.cadence import Cadence
from dlightrag.application.schedules.errors import ScheduleInputError, ScheduleStateError
from dlightrag.engine.answer.goal import Goal, GoalError

#: The longest a slot may be late and still run: half its interval, and never more than this.
MAX_GRACE = timedelta(hours=6)

#: A one-time Schedule has no interval to halve; an hour late is still the time the owner meant.
ONE_TIME_GRACE = timedelta(hours=1)

#: Slots a single look counts when it finds a Schedule that missed many. Past this a deployment
#: has been down for over a year of hourly slots; the gap is recorded as missed and nothing runs late.
MISSED_SCAN_CAP = 10_000

#: A manual Fire has no slot on the cadence to take a grace from, so it is worth this long at least.
MANUAL_RETRY_FLOOR = timedelta(minutes=5)

NAME_MAX_CHARS = 120

#: The largest number a column of BIGINT holds: what a ceiling or a round cap may be.
MAX_COUNT = 2**63 - 1

#: The prompt is a turn the owner wrote, bounded like a message sent to a live Run.
PROMPT_MAX_CHARS = 20_000

type ScheduleMode = Literal["research", "fast"]
SCHEDULE_MODES: tuple[ScheduleMode, ...] = ("research", "fast")

#: Who last made or edited a Schedule: its owner, or the Agent on the owner's behalf from a chat.
type ChangedBy = Literal["owner", "agent"]

#: What reads "now" for a Schedule. It is asynchronous because the one clock every process shares
#: is the database's (``ScheduleStore.now``), so replicas whose own clocks differ agree on what is due.
type Clock = Callable[[], Awaitable[datetime]]


class ScheduleState(StrEnum):
    """Whether a Schedule fires. A completed Schedule was a single time, and it has fired."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class Target(StrEnum):
    """Where a fire's turn goes: a conversation of its own, or the one the Schedule keeps."""

    NEW_CONVERSATION = "new_conversation"
    SAME_CONVERSATION = "same_conversation"


class FireState(StrEnum):
    """What one slot did. Every state but ``pending`` and ``started`` is final."""

    PENDING = "pending"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUSED = "refused"
    SKIPPED_BUSY = "skipped_busy"
    SKIPPED_LIMIT = "skipped_limit"
    MISSED = "missed"


#: The states a Run's own outcome ends a Fire in, and the ones that read as a bad outcome.
RUN_OUTCOMES = frozenset({FireState.SUCCEEDED, FireState.FAILED, FireState.CANCELLED})
_BAD_OUTCOMES = frozenset({FireState.FAILED, FireState.REFUSED})
#: The states that say how a Fire went, as opposed to one that started nothing by design.
STREAK_STATES = (
    FireState.SUCCEEDED,
    FireState.FAILED,
    FireState.CANCELLED,
    FireState.REFUSED,
)


@dataclass(frozen=True, slots=True)
class Schedule:
    """One owner's recurring or one-time turn.

    ``principal`` is who last created, edited or resumed it: the identity a fire acts under, whose
    owner must still be this Schedule's owner (ADR 0103 Decision 6). ``conversation_id`` is the one
    conversation a ``same_conversation`` Schedule keeps; its first fire opens it.
    ``goal_max_rounds`` makes it a goal Schedule: each fire starts a Goal whose objective is the
    prompt, with that round cap, and the fire's Run is the Goal's round 1 (Decision 10).
    ``principal_stamped_at`` is when the identity was last confirmed by its owner (creating,
    editing or resuming), and ``resumed_at`` when the Schedule last resumed: a failure streak
    counts only the Fires made after it. ``changed_by`` is who last made, edited or resumed it, so
    an owner can tell what the Agent wrote; pausing, deleting and running now leave it as it was.
    """

    schedule_id: str
    owner_id: str
    name: str
    prompt: str
    mode: ScheduleMode
    workspaces: tuple[str, ...]
    target: Target
    conversation_id: str | None
    cadence: Cadence
    max_tokens_per_day: int
    goal_max_rounds: int | None
    state: ScheduleState
    pause_reason: str | None
    next_fire_at: datetime | None
    principal: UserContext
    principal_stamped_at: datetime
    resumed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    changed_by: ChangedBy = "owner"

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        principal: UserContext,
        name: str,
        prompt: str,
        mode: str,
        workspaces: Sequence[str],
        target: Target,
        cadence: Cadence,
        max_tokens_per_day: int,
        now: datetime,
        goal_max_rounds: int | None = None,
        changed_by: ChangedBy = "owner",
    ) -> Self:
        """A new active Schedule, waiting for the cadence's first slot."""
        schedule = cls(
            schedule_id=str(uuid.uuid4()),
            owner_id=owner_id,
            name=_name(name),
            prompt=_prompt(prompt),
            mode=_mode(mode),
            workspaces=_workspaces(workspaces),
            target=target,
            conversation_id=None,
            cadence=cadence,
            max_tokens_per_day=_ceiling(max_tokens_per_day),
            goal_max_rounds=goal_max_rounds,
            state=ScheduleState.ACTIVE,
            pause_reason=None,
            next_fire_at=cadence.first_slot(now),
            principal=principal,
            principal_stamped_at=now,
            resumed_at=None,
            created_at=now,
            updated_at=now,
            changed_by=changed_by,
        )
        _check_goal(schedule)
        return schedule

    def edit(
        self,
        *,
        principal: UserContext,
        now: datetime,
        name: str | None = None,
        prompt: str | None = None,
        mode: str | None = None,
        workspaces: Sequence[str] | None = None,
        cadence: Cadence | None = None,
        max_tokens_per_day: int | None = None,
        goal_max_rounds: int | None = None,
        changed_by: ChangedBy = "owner",
    ) -> Self:
        """The Schedule with the named parts changed and ``principal`` as the identity it acts as.

        A new cadence recomputes ``next_fire_at`` from ``now`` (ADR 0103 Decision 12). A completed
        Schedule has fired its one time and takes no edit: the owner makes a new one. Whether a
        Schedule starts a Goal is fixed when it is made; only a goal Schedule's round cap changes.
        """
        if self.state is ScheduleState.COMPLETED:
            raise ScheduleStateError("a completed Schedule cannot be edited; create a new one")
        changed = replace(
            self,
            name=self.name if name is None else _name(name),
            prompt=self.prompt if prompt is None else _prompt(prompt),
            mode=self.mode if mode is None else _mode(mode),
            workspaces=self.workspaces if workspaces is None else _workspaces(workspaces),
            max_tokens_per_day=(
                self.max_tokens_per_day
                if max_tokens_per_day is None
                else _ceiling(max_tokens_per_day)
            ),
            goal_max_rounds=(self.goal_max_rounds if goal_max_rounds is None else goal_max_rounds),
            principal=principal,
            principal_stamped_at=now,
            updated_at=now,
            changed_by=changed_by,
        )
        if self.goal_max_rounds is None and goal_max_rounds is not None:
            raise ScheduleInputError("this Schedule does not start a goal, so it has no round cap")
        _check_goal(changed)
        if cadence is None:
            return changed
        if cadence.same_slots(self.cadence):
            return replace(changed, cadence=cadence)
        return replace(changed, cadence=cadence, next_fire_at=cadence.first_slot(now))


@dataclass(frozen=True, slots=True)
class Fire:
    """One slot of one Schedule: its state, and the Run it started if it started one.

    A gap of missed slots is one Fire (``missed_count`` slots from ``slot`` to ``last_slot``).
    ``reason`` states why a Fire that started nothing did not, or what a failed Run said.
    ``manual`` is a Fire the owner asked for (Run now), whose slot is the request instant.
    ``last_error`` is what stopped the latest attempt to start a Fire that is still pending; a
    Fire that is missed with one ended badly, and counts toward the failure streak.
    """

    fire_id: str
    schedule_id: str
    owner_id: str
    slot: datetime
    state: FireState
    reason: str | None
    run_id: str | None
    conversation_id: str | None
    missed_count: int | None
    last_slot: datetime | None
    created_at: datetime
    updated_at: datetime
    manual: bool = False
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledTurn:
    """The Run a fire started, and the conversation it went into."""

    run_id: str
    conversation_id: str


@dataclass(frozen=True, slots=True)
class ScheduledConversation:
    """A conversation a goal fire goes into, and the Session its goal hangs from."""

    conversation_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ScheduledGoalRound:
    """The goal and round a fire's Run is: what its settlement tells the goal driver it served."""

    goal_id: str
    goal_revision: int
    goal_round: int


@dataclass(frozen=True, slots=True)
class FireDraft:
    """A Fire the look is about to write for a due Schedule."""

    slot: datetime
    state: FireState
    reason: str | None = None
    missed_count: int | None = None
    last_slot: datetime | None = None


@dataclass(frozen=True, slots=True)
class DuePlan:
    """What one due Schedule gets: its Fires, and where it waits next.

    ``next_fire_at`` is ``None`` for a single time, which then completes.
    """

    fires: tuple[FireDraft, ...]
    next_fire_at: datetime | None


def plan_due(cadence: Cadence, due: datetime, now: datetime) -> DuePlan:
    """Decide a Schedule whose ``next_fire_at`` (``due``) has passed (ADR 0103 Decision 4).

    Every slot from ``due`` up to ``now`` is due. Only the latest may still run, and only within
    its grace: half the interval to the slot after it (at most ``MAX_GRACE``), or an hour for a
    single time. The slots before it, and the latest too when it is past its grace, are one
    ``missed`` Fire that starts nothing. ``next_fire_at`` moves to the first slot after ``now``.
    """
    expression = cadence.expression
    if expression is None:
        if now - due > ONE_TIME_GRACE:
            return DuePlan((_gap(due, due, 1, superseded=False),), None)
        return DuePlan((FireDraft(due, FireState.PENDING),), None)
    zone = cadence.zone
    count, latest, before = 1, due, due
    following = expression.next_instant(due, zone)
    while following <= now:
        count, before, latest = count + 1, latest, following
        if count >= MISSED_SCAN_CAP:
            return DuePlan(
                (_gap(due, latest, count, superseded=False),), expression.next_instant(now, zone)
            )
        following = expression.next_instant(latest, zone)
    if now - latest > grace(latest, following):
        return DuePlan((_gap(due, latest, count, superseded=False),), following)
    late = FireDraft(latest, FireState.PENDING)
    if count == 1:
        return DuePlan((late,), following)
    return DuePlan((_gap(due, before, count - 1, superseded=True), late), following)


def grace(slot: datetime, following: datetime) -> timedelta:
    """How late ``slot`` may still start: half the way to ``following``, and no more than ``MAX_GRACE``."""
    return min(MAX_GRACE, (following - slot) / 2)


def retry_window(cadence: Cadence, slot: datetime, *, manual: bool = False) -> timedelta:
    """How long a Fire for ``slot`` that has not started is still worth starting.

    A manual Fire's slot is the moment it was asked for, not a slot of the cadence, so half the
    way to the next slot says nothing about it; it keeps at least ``MANUAL_RETRY_FLOOR``.
    """
    if cadence.expression is None:
        window = ONE_TIME_GRACE
    else:
        window = grace(slot, cadence.expression.next_instant(slot, cadence.zone))
    return max(window, MANUAL_RETRY_FLOOR) if manual else window


def _gap(first: datetime, last: datetime, count: int, *, superseded: bool) -> FireDraft:
    """The one ``missed`` Fire for ``count`` slots from ``first`` to ``last``."""
    times = f"{'at least ' if count >= MISSED_SCAN_CAP else ''}{count} scheduled time(s)"
    if superseded:
        reason = f"{times} came while nothing was looking; only the latest one is started"
    else:
        reason = f"{times} came while nothing was looking, and it is now too late to start"
    return FireDraft(first, FireState.MISSED, reason, count, last)


def failure_streak_reached(outcomes: Sequence[FireState], streak: int) -> bool:
    """Whether the newest ``streak`` Fires that ran or were refused all ended badly.

    ``outcomes`` are those Fires' states, newest first. A Fire that started nothing by design (a
    skip, or a missed gap) is not among them, and one success or cancellation ends a streak.
    """
    return len(outcomes) >= streak and all(state in _BAD_OUTCOMES for state in outcomes[:streak])


def scheduled_turn_text(*, name: str, slot: datetime, cadence: Cadence, prompt: str) -> str:
    """The user message a fired Run carries: one line naming the Schedule and the slot, then the prompt.

    The line gives the Agent the slot's local time ("this week" in a prompt has a date) and shows a
    reader that nobody typed this turn.
    """
    label = " ".join(name.replace('"', "'").split())
    local = slot.astimezone(cadence.zone)
    return f'[Scheduled: "{label}", {local:%Y-%m-%d %H:%M} {cadence.zone_name}]\n\n{prompt}'


def _check_goal(schedule: Schedule) -> None:
    """A goal Schedule's prompt is a goal objective and its rounds are research Runs.

    The objective and the cap are held to the Goal's own rules, so the Schedule cannot hold what
    the goal service would refuse at the first fire.
    """
    if schedule.goal_max_rounds is None:
        return
    if schedule.goal_max_rounds > MAX_COUNT:
        raise ScheduleInputError("a goal's round cap is too large")
    if schedule.mode != "research":
        raise ScheduleInputError("a Schedule that starts a goal runs in research mode")
    try:
        Goal.create(schedule.prompt, max_rounds=schedule.goal_max_rounds)
    except GoalError as error:
        raise ScheduleInputError(str(error)) from error


def _text(value: str, what: str) -> str:
    """Text a column of TEXT can hold: PostgreSQL stores no NUL character."""
    if "\x00" in value:
        raise ScheduleInputError(f"{what} cannot hold a NUL character")
    return value


def _name(value: str) -> str:
    text = " ".join(_text(value, "a name").split())
    if not text:
        raise ScheduleInputError("a Schedule needs a name")
    if len(text) > NAME_MAX_CHARS:
        raise ScheduleInputError(f"a name is at most {NAME_MAX_CHARS} characters")
    return text


def _prompt(value: str) -> str:
    text = _text(value, "a prompt").strip()
    if not text:
        raise ScheduleInputError("a Schedule needs a prompt")
    if len(text) > PROMPT_MAX_CHARS:
        raise ScheduleInputError(f"a prompt is at most {PROMPT_MAX_CHARS} characters")
    return text


def _mode(value: str) -> ScheduleMode:
    for mode in SCHEDULE_MODES:
        if value == mode:
            return mode
    raise ScheduleInputError("a Schedule runs in research or fast mode")


def _workspaces(values: Sequence[str]) -> tuple[str, ...]:
    chosen = tuple(dict.fromkeys(_text(value, "a workspace id") for value in values if value))
    if not chosen:
        raise ScheduleInputError("a Schedule needs at least one workspace")
    return chosen


def _ceiling(value: int) -> int:
    if isinstance(value, bool) or not 1 <= value <= MAX_COUNT:
        raise ScheduleInputError(f"the daily token ceiling must be a number from 1 to {MAX_COUNT}")
    return value


__all__ = [
    "MANUAL_RETRY_FLOOR",
    "MAX_COUNT",
    "MAX_GRACE",
    "MISSED_SCAN_CAP",
    "NAME_MAX_CHARS",
    "ONE_TIME_GRACE",
    "PROMPT_MAX_CHARS",
    "RUN_OUTCOMES",
    "SCHEDULE_MODES",
    "STREAK_STATES",
    "ChangedBy",
    "Clock",
    "DuePlan",
    "Fire",
    "FireDraft",
    "FireState",
    "Schedule",
    "ScheduleMode",
    "ScheduledTurn",
    "ScheduleState",
    "ScheduledConversation",
    "ScheduledGoalRound",
    "Target",
    "failure_streak_reached",
    "grace",
    "plan_due",
    "retry_window",
    "scheduled_turn_text",
]
