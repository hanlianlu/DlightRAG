# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Schedule store stand-in that keeps the PostgreSQL adapter's claim, fence and streak rules."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from dlightrag.application.access.principal import UserContext
from dlightrag.application.schedules.errors import ScheduleLimitError
from dlightrag.application.schedules.model import (
    STREAK_STATES,
    DuePlan,
    Fire,
    FireState,
    Schedule,
    ScheduleState,
    Target,
    failure_streak_reached,
)
from dlightrag.application.schedules.store import Claim, DueFire


class Clock:
    """A settable instant, so a test moves time instead of waiting for it."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **span: float) -> None:
        self.now += timedelta(**span)


@dataclass
class _Held:
    fire: Fire
    claimed_at: datetime


class FakeScheduleStore:
    """In-memory Schedules and Fires, with the transactions of the real store as plain methods."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.schedules: dict[str, Schedule] = {}
        self.held: dict[str, _Held] = {}
        #: What the Runs of any Schedule's Fires are said to have spent; the real store reads it.
        self.spent = 0

    @property
    def fires(self) -> list[Fire]:
        return sorted((held.fire for held in self.held.values()), key=lambda fire: fire.slot)

    def fires_of(self, schedule_id: str) -> list[Fire]:
        return [fire for fire in self.fires if fire.schedule_id == schedule_id]

    async def create(self, schedule: Schedule, *, max_active: int) -> Schedule:
        active = [
            item
            for item in self.schedules.values()
            if item.owner_id == schedule.owner_id and item.state is ScheduleState.ACTIVE
        ]
        if len(active) >= max_active:
            raise ScheduleLimitError(max_active)
        self.schedules[schedule.schedule_id] = schedule
        return schedule

    async def get(self, owner_id: str, schedule_id: str) -> Schedule | None:
        found = self.schedules.get(schedule_id)
        return found if found is not None and found.owner_id == owner_id else None

    async def list(self, owner_id: str) -> tuple[Schedule, ...]:
        return tuple(item for item in self.schedules.values() if item.owner_id == owner_id)

    async def edit(self, schedule: Schedule, *, reschedule: bool) -> Schedule | None:
        current = await self.get(schedule.owner_id, schedule.schedule_id)
        if current is None or current.state is ScheduleState.COMPLETED:
            return None
        kept = schedule if reschedule else replace(schedule, next_fire_at=current.next_fire_at)
        self.schedules[schedule.schedule_id] = kept
        return kept

    async def pause(
        self, owner_id: str, schedule_id: str, *, reason: str | None
    ) -> Schedule | None:
        current = await self.get(owner_id, schedule_id)
        if current is None or current.state is not ScheduleState.ACTIVE:
            return None
        return self._put(replace(current, state=ScheduleState.PAUSED, pause_reason=reason))

    async def resume(
        self, owner_id: str, schedule_id: str, *, principal: UserContext, max_active: int
    ) -> Schedule | None:
        current = await self.get(owner_id, schedule_id)
        if current is None or current.state is not ScheduleState.PAUSED:
            return None
        active = [
            item
            for item in self.schedules.values()
            if item.owner_id == owner_id and item.state is ScheduleState.ACTIVE
        ]
        if len(active) >= max_active:
            raise ScheduleLimitError(max_active)
        now = self.clock.now
        return self._put(
            replace(
                current,
                state=ScheduleState.ACTIVE,
                pause_reason=None,
                principal=principal,
                principal_stamped_at=now,
                resumed_at=now,
            )
        )

    async def delete(self, owner_id: str, schedule_id: str) -> bool:
        if await self.get(owner_id, schedule_id) is None:
            return False
        del self.schedules[schedule_id]
        for fire_id in [
            key for key, held in self.held.items() if held.fire.schedule_id == schedule_id
        ]:
            del self.held[fire_id]
        return True

    async def claim_due(
        self,
        now: datetime,
        *,
        limit: int,
        fires_kept: int,
        plan: Callable[[Schedule], DuePlan],
    ) -> Claim:
        due = sorted(
            (
                item
                for item in self.schedules.values()
                if item.state is ScheduleState.ACTIVE
                and item.next_fire_at is not None
                and item.next_fire_at <= now
            ),
            key=lambda item: item.next_fire_at or now,
        )[:limit]
        pending: list[DueFire] = []
        for schedule in due:
            decision = plan(schedule)
            written: list[Fire] = []
            for draft in decision.fires:
                if any(f.slot == draft.slot for f in self.fires_of(schedule.schedule_id)):
                    continue
                fire = self._new_fire(
                    schedule,
                    draft.slot,
                    draft.state,
                    reason=draft.reason,
                    missed_count=draft.missed_count,
                    last_slot=draft.last_slot,
                )
                if draft.state is FireState.PENDING:
                    written.append(fire)
            moved = self._put(replace(schedule, next_fire_at=decision.next_fire_at))
            self._trim(schedule.schedule_id, fires_kept)
            if decision.next_fire_at is None and not written:
                moved = self._put(
                    replace(
                        moved,
                        state=ScheduleState.PAUSED,
                        pause_reason=decision.fires[-1].reason or "its time passed without a start",
                    )
                )
            pending.extend(DueFire(moved, fire) for fire in written)
        return Claim(len(due), tuple(pending))

    async def claim_pending(
        self, now: datetime, *, lease: timedelta, limit: int
    ) -> tuple[DueFire, ...]:
        taken = [
            held
            for held in sorted(self.held.values(), key=lambda held: held.fire.slot)
            if held.fire.state is FireState.PENDING and held.claimed_at < now - lease
        ][:limit]
        for held in taken:
            held.claimed_at = now
        return tuple(DueFire(self.schedules[held.fire.schedule_id], held.fire) for held in taken)

    async def insert_manual_fire(
        self, schedule: Schedule, slot: datetime, *, fires_kept: int
    ) -> DueFire | None:
        if any(fire.slot == slot for fire in self.fires_of(schedule.schedule_id)):
            return None
        fire = self._new_fire(schedule, slot, FireState.PENDING, manual=True)
        self._trim(schedule.schedule_id, fires_kept)
        return DueFire(schedule, fire)

    async def has_open_fire(self, owner_id: str, schedule_id: str) -> bool:
        return any(
            fire.owner_id == owner_id and fire.state in (FireState.PENDING, FireState.STARTED)
            for fire in self.fires_of(schedule_id)
        )

    async def earlier_pending_fire(
        self, owner_id: str, schedule_id: str, before: datetime, *, exclude: str
    ) -> Fire | None:
        return next(
            (
                fire
                for fire in self.fires_of(schedule_id)
                if fire.owner_id == owner_id
                and fire.state is FireState.PENDING
                and fire.slot < before
                and fire.fire_id != exclude
            ),
            None,
        )

    async def record_error(self, owner_id: str, fire_id: str, error: str) -> None:
        held = self.held.get(fire_id)
        if held is not None and held.fire.owner_id == owner_id:
            held.fire = replace(held.fire, last_error=error)

    async def touch_fire(self, owner_id: str, fire_id: str) -> None:
        held = self.held.get(fire_id)
        if held is not None and held.fire.owner_id == owner_id:
            held.fire = replace(held.fire, updated_at=self.clock.now)

    async def finish_fire(
        self,
        owner_id: str,
        fire_id: str,
        *,
        expected: Sequence[FireState],
        state: FireState,
        reason: str | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        pause_reason: str | None = None,
        failure_streak: int,
    ) -> Fire | None:
        held = self.held.get(fire_id)
        if held is None or held.fire.owner_id != owner_id or held.fire.state not in expected:
            return None
        fire = replace(
            held.fire,
            state=state,
            reason=reason,
            run_id=run_id or held.fire.run_id,
            conversation_id=conversation_id or held.fire.conversation_id,
            updated_at=self.clock.now,
        )
        held.fire = fire
        schedule = self.schedules[fire.schedule_id]
        if (
            conversation_id is not None
            and schedule.target is Target.SAME_CONVERSATION
            and schedule.conversation_id is None
        ):
            schedule = self._put(replace(schedule, conversation_id=conversation_id))
        active = schedule.state is ScheduleState.ACTIVE
        if (
            active
            and schedule.cadence.expression is None
            and schedule.next_fire_at is None
            and not fire.manual
            and FireState.PENDING in expected
        ):
            if state is FireState.STARTED:
                schedule = self._put(replace(schedule, state=ScheduleState.COMPLETED))
            else:
                schedule = self._put(
                    replace(
                        schedule,
                        state=ScheduleState.PAUSED,
                        pause_reason=fire.reason or str(state),
                    )
                )
            active = False
        bad = state in (FireState.FAILED, FireState.REFUSED) or (
            state is FireState.MISSED and fire.last_error is not None
        )
        if active and bad:
            why = pause_reason
            if why is None:
                since = schedule.resumed_at
                outcomes = [
                    FireState.FAILED if other.state is FireState.MISSED else other.state
                    for other in reversed(self.fires_of(fire.schedule_id))
                    if (since is None or other.created_at >= since)
                    and (
                        other.state in STREAK_STATES
                        or (other.state is FireState.MISSED and other.last_error is not None)
                    )
                ][:failure_streak]
                if failure_streak_reached(outcomes, failure_streak):
                    why = f"the last {failure_streak} fires failed or were refused: {reason}"
            if why is not None:
                self._put(replace(schedule, state=ScheduleState.PAUSED, pause_reason=why))
        return fire

    async def fire_for_run(self, owner_id: str, run_id: str) -> Fire | None:
        return next((f for f in self.fires if f.owner_id == owner_id and f.run_id == run_id), None)

    async def previous_fire(self, owner_id: str, schedule_id: str, before: datetime) -> Fire | None:
        earlier = [
            f
            for f in self.fires_of(schedule_id)
            if f.owner_id == owner_id and f.run_id is not None and f.slot < before
        ]
        return earlier[-1] if earlier else None

    async def unsettled_fires(self, *, older_than: datetime, limit: int) -> tuple[Fire, ...]:
        return tuple(
            f for f in self.fires if f.state is FireState.STARTED and f.updated_at < older_than
        )[:limit]

    async def list_fires(
        self, owner_id: str, schedule_id: str, *, limit: int
    ) -> tuple[Fire, ...] | None:
        if await self.get(owner_id, schedule_id) is None:
            return None
        return tuple(reversed(self.fires_of(schedule_id)))[:limit]

    async def spent_tokens(self, owner_id: str, schedule_id: str, *, since: datetime) -> int:
        del owner_id, schedule_id, since
        return self.spent

    def _put(self, schedule: Schedule) -> Schedule:
        self.schedules[schedule.schedule_id] = schedule
        return schedule

    def _new_fire(
        self,
        schedule: Schedule,
        slot: datetime,
        state: FireState,
        *,
        reason: str | None = None,
        missed_count: int | None = None,
        last_slot: datetime | None = None,
        manual: bool = False,
    ) -> Fire:
        now = self.clock.now
        fire = Fire(
            fire_id=str(uuid.uuid4()),
            schedule_id=schedule.schedule_id,
            owner_id=schedule.owner_id,
            slot=slot,
            state=state,
            reason=reason,
            run_id=None,
            conversation_id=None,
            missed_count=missed_count,
            last_slot=last_slot,
            created_at=now,
            updated_at=now,
            manual=manual,
        )
        self.held[fire.fire_id] = _Held(fire, now)
        return fire

    def _trim(self, schedule_id: str, fires_kept: int) -> None:
        for old in self.fires_of(schedule_id)[:-fires_kept]:
            del self.held[old.fire_id]
