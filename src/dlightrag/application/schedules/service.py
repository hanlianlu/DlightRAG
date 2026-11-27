# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What an owner does with a Schedule (ADR 0103 Decision 12).

Every method is scoped to the caller's owner: a Schedule that is not theirs is the same miss as
one that does not exist, as a conversation is. Creating, editing and resuming record the caller as
the identity the Schedule acts as from then on; the Agent has no way into this service. The
workspaces a caller names are the ones the transport already authorized for them. The firing rules
read access again at every fire, so what was readable when the Schedule was written is not trusted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.config.sections import SchedulesSettings
from dlightrag.application.schedules.cadence import Cadence
from dlightrag.application.schedules.errors import ScheduleStateError
from dlightrag.application.schedules.firing import ScheduleFirer
from dlightrag.application.schedules.model import (
    Fire,
    Schedule,
    ScheduleState,
    Target,
)
from dlightrag.application.schedules.store import DueFire, ScheduleStore

FIRES_PAGE_DEFAULT = 50
FIRES_PAGE_MAX = 200


def _now() -> datetime:
    return datetime.now(UTC)


class ScheduleService:
    """Create, edit, pause, resume, run and delete one owner's Schedules, and read their Fires."""

    def __init__(
        self,
        *,
        store: ScheduleStore,
        firer: ScheduleFirer,
        settings: SchedulesSettings,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._store = store
        self._firer = firer
        self._settings = settings
        self._clock = clock

    async def create(
        self,
        user: UserContext,
        *,
        name: str,
        prompt: str,
        workspaces: Sequence[str],
        zone: str,
        mode: str = "research",
        target: Target = Target.NEW_CONVERSATION,
        cron: str | None = None,
        at: datetime | None = None,
        max_tokens_per_day: int | None = None,
        goal_max_rounds: int | None = None,
    ) -> Schedule:
        """Save a new active Schedule: a recurrence (``cron``) or a single time (``at``).

        With ``goal_max_rounds`` each fire starts a Goal whose objective is ``prompt`` and which
        runs at most that many rounds; the fire's Run is round 1.

        Refused when the cadence is invalid or recurs closer than the minimum interval, and when
        the owner already holds the most active Schedules the deployment allows.
        """
        now = self._clock()
        cadence = Cadence.create(
            zone_name=zone, cron=cron, at=at, now=now, min_interval=self._min_interval
        )
        schedule = Schedule.create(
            owner_id=owner_id_from_user(user),
            principal=user,
            name=name,
            prompt=prompt,
            mode=mode,
            workspaces=workspaces,
            target=target,
            cadence=cadence,
            max_tokens_per_day=(
                self._settings.default_max_tokens_per_day
                if max_tokens_per_day is None
                else max_tokens_per_day
            ),
            now=now,
            goal_max_rounds=goal_max_rounds,
        )
        return await self._store.create(schedule, max_active=self._settings.max_active_per_owner)

    async def get(self, user: UserContext, schedule_id: str) -> Schedule | None:
        return await self._store.get(owner_id_from_user(user), schedule_id)

    async def list(self, user: UserContext) -> tuple[Schedule, ...]:
        return await self._store.list(owner_id_from_user(user))

    async def update(
        self,
        user: UserContext,
        schedule_id: str,
        *,
        name: str | None = None,
        prompt: str | None = None,
        workspaces: Sequence[str] | None = None,
        mode: str | None = None,
        zone: str | None = None,
        cron: str | None = None,
        at: datetime | None = None,
        max_tokens_per_day: int | None = None,
        goal_max_rounds: int | None = None,
    ) -> Schedule | None:
        """Change the parts that are named; ``None`` when the Schedule is not this owner's.

        Naming ``cron`` or ``at`` replaces the cadence, and naming only ``zone`` keeps its
        recurrence, or its single time, in the new zone. A cadence that differs from the current
        one recomputes the next slot from now; one that is the same changes nothing. A completed
        Schedule takes no edit.
        """
        owner_id = owner_id_from_user(user)
        current = await self._store.get(owner_id, schedule_id)
        if current is None:
            return None
        if current.state is ScheduleState.COMPLETED:
            raise ScheduleStateError("a completed Schedule cannot be edited; create a new one")
        now = self._clock()
        cadence = None
        if cron is not None or at is not None:
            cadence = Cadence.create(
                zone_name=zone or current.cadence.zone_name,
                cron=cron,
                at=at,
                now=now,
                min_interval=self._min_interval,
            )
        elif zone is not None and current.cadence.expression is None:
            cadence = current.cadence.in_zone(zone)
        elif zone is not None:
            cadence = Cadence.create(
                zone_name=zone,
                cron=current.cadence.cron,
                at=None,
                now=now,
                min_interval=self._min_interval,
            )
        edited = current.edit(
            principal=user,
            now=now,
            name=name,
            prompt=prompt,
            mode=mode,
            workspaces=workspaces,
            cadence=cadence,
            max_tokens_per_day=max_tokens_per_day,
            goal_max_rounds=goal_max_rounds,
        )
        return await self._store.edit(
            edited, reschedule=not edited.cadence.same_slots(current.cadence)
        )

    async def pause(self, user: UserContext, schedule_id: str) -> Schedule | None:
        """Stop a Schedule firing until the owner resumes it; pausing a paused one is a no-op."""
        owner_id = owner_id_from_user(user)
        current = await self._store.get(owner_id, schedule_id)
        if current is None or current.state is ScheduleState.PAUSED:
            return current
        if current.state is ScheduleState.COMPLETED:
            raise ScheduleStateError("a completed Schedule has nothing left to pause")
        paused = await self._store.pause(owner_id, schedule_id, reason=None)
        return paused or await self._store.get(owner_id, schedule_id)

    async def resume(self, user: UserContext, schedule_id: str) -> Schedule | None:
        """Let a paused Schedule fire again, acting as the caller from now on.

        Slots it slept through are not replayed beyond the one late fire the grace window allows.
        """
        owner_id = owner_id_from_user(user)
        current = await self._store.get(owner_id, schedule_id)
        if current is None or current.state is ScheduleState.ACTIVE:
            return current
        if current.state is ScheduleState.COMPLETED:
            raise ScheduleStateError("a completed Schedule has fired its time; create a new one")
        if current.next_fire_at is None:
            raise ScheduleStateError("this Schedule has no time to wait for; edit it to name one")
        resumed = await self._store.resume(
            owner_id,
            schedule_id,
            principal=user,
            max_active=self._settings.max_active_per_owner,
        )
        return resumed or await self._store.get(owner_id, schedule_id)

    async def delete(self, user: UserContext, schedule_id: str) -> bool:
        """Remove a Schedule and its Fires. The conversations its Runs wrote stay."""
        return await self._store.delete(owner_id_from_user(user), schedule_id)

    async def run_now(self, user: UserContext, schedule_id: str) -> Fire | None:
        """Fire once now, under every rule a clock fire meets, as the caller.

        The Fire's slot is this moment, and it is manual: it runs on a Schedule that is paused or
        completed and leaves it as it was. Refused while the Schedule has a Fire still starting or
        working, since a second Run beside it is what the rules exist to prevent. ``None`` when the
        Schedule is not this owner's, or when another process moved the Fire first.
        """
        owner_id = owner_id_from_user(user)
        schedule = await self._store.get(owner_id, schedule_id)
        if schedule is None:
            return None
        if await self._store.has_open_fire(owner_id, schedule_id):
            raise ScheduleStateError("this Schedule already has a fire starting or working")
        now = self._clock()
        due = await self._store.insert_manual_fire(
            schedule, now, fires_kept=self._settings.fires_kept
        )
        if due is None:
            raise ScheduleStateError("a run was requested at this very moment; try again")
        asked = replace(schedule, principal=user, principal_stamped_at=now)
        return await self._firer.start(DueFire(asked, due.fire))

    async def list_fires(
        self, user: UserContext, schedule_id: str, *, limit: int = FIRES_PAGE_DEFAULT
    ) -> tuple[Fire, ...] | None:
        """The Schedule's newest Fires first; ``None`` when it is not this owner's."""
        return await self._store.list_fires(
            owner_id_from_user(user), schedule_id, limit=max(1, min(limit, FIRES_PAGE_MAX))
        )

    @property
    def _min_interval(self) -> timedelta:
        return timedelta(minutes=self._settings.min_interval_minutes)


__all__ = ["FIRES_PAGE_DEFAULT", "FIRES_PAGE_MAX", "ScheduleService"]
