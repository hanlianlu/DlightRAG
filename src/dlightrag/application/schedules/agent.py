# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent's way to an owner's Schedules: the Schedule service, as the owner's latest human turn.

The engine's Schedule tools name a port, ``AnswerScheduleService``; this is its implementation. It
adds nothing to what an owner may do and takes nothing away: every call goes through
``ScheduleService`` as the caller ``OwnerCallers`` kept from the owner's latest human turn, so the
minimum interval, the cap on active Schedules, cadence validation and the refusal to edit a
completed Schedule are the owner's own. What the Agent does not get is the two things only an owner
decides: a Schedule made here has no goal and no spend ceiling of its own (the deployment's default
applies), and an edit here never names either (ADR 0103 Decision 12).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from dlightrag.application.access.principal import Principal, UserContext, owner_id_from_user
from dlightrag.application.errors import ApplicationConflictError, ApplicationInputError
from dlightrag.application.schedules.cadence import zone_of
from dlightrag.application.schedules.errors import ScheduleLimitError
from dlightrag.application.schedules.model import Schedule, Target
from dlightrag.application.schedules.service import ScheduleService
from dlightrag.application.schedules.store import ScheduleStore
from dlightrag.engine.answer.tools.schedules import (
    SCHEDULE_INVALID,
    SCHEDULE_LIMIT,
    SCHEDULE_NOT_FOUND,
    SCHEDULE_STATE,
    SCHEDULE_UNAVAILABLE,
    SCHEDULES_UNAVAILABLE,
    ScheduleRecord,
    ScheduleRefusal,
)

_NOT_YOURS = "the user has no Schedule with this id; read list_schedules for the ids"
_GOAL_SCHEDULE = (
    "this Schedule starts a goal, so only its owner changes or resumes it, in the Scheduled list "
    "that the Scheduled row in the sidebar opens; you can rename or pause it"
)


class RunCallers(Protocol):
    """Who an owner's Agent acts as, for the Run that asks."""

    async def caller_of(self, *, owner_id: str, run_id: str) -> UserContext | None: ...


class OwnerCallers:
    """Who an owner's Agent acts as: the caller of the owner's latest human turn.

    One record per owner, replaced by each such turn, so a Schedule the Agent makes acts under the
    identity the owner last presented, exactly as one made in the Scheduled list does.
    """

    def __init__(self, store: ScheduleStore) -> None:
        self._store = store

    async def remember(self, *, owner_id: str, caller: Principal) -> None:
        """Keep ``caller`` as the owner's identity. One that does not map to ``owner_id`` is not kept."""
        user = UserContext(
            user_id=caller.user_id, auth_mode=caller.auth_mode, claims=dict(caller.claims)
        )
        if owner_id_from_user(user) == owner_id:
            await self._store.remember_principal(owner_id, user)

    async def caller_of(self, *, owner_id: str, run_id: str) -> UserContext | None:
        del run_id  # the record is the owner's latest human turn, not this Run's
        return await self._store.principal_of(owner_id)


class AgentSchedules:
    """The owner's Schedules for the Agent of one Run, over the same service the owner uses.

    The service is bound after construction: it needs the conversations the executor that holds
    this object helps to build. A call made before it is bound is refused as unavailable.
    """

    def __init__(self, *, callers: RunCallers) -> None:
        self._callers = callers
        self._schedules: ScheduleService | None = None

    def bind(self, schedules: ScheduleService) -> None:
        self._schedules = schedules

    async def list(self, *, owner_id: str, run_id: str) -> tuple[ScheduleRecord, ...]:
        service, user = await self._acting(owner_id, run_id)
        return tuple(_record(schedule) for schedule in await service.list(user))

    async def create(
        self,
        *,
        owner_id: str,
        run_id: str,
        name: str,
        prompt: str,
        workspaces: Sequence[str],
        zone: str,
        cron: str | None,
        at: datetime | None,
        mode: str,
        target: str,
    ) -> ScheduleRecord:
        service, user = await self._acting(owner_id, run_id)
        try:
            return _record(
                await service.create(
                    user,
                    name=name,
                    prompt=prompt,
                    workspaces=workspaces,
                    zone=zone,
                    mode=mode,
                    target=Target(target),
                    cron=cron,
                    at=at,
                    changed_by="agent",
                )
            )
        except (ApplicationInputError, ApplicationConflictError) as error:
            raise _refusal(error) from error

    async def update(
        self,
        *,
        owner_id: str,
        run_id: str,
        schedule_id: str,
        name: str | None,
        prompt: str | None,
        workspaces: Sequence[str] | None,
        mode: str | None,
        zone: str | None,
        cron: str | None,
        at: datetime | None,
        paused: bool | None,
    ) -> ScheduleRecord:
        """Edit the named parts, then pause or resume; an edit already made stays if that is refused.

        A Schedule that starts a goal is the owner's: the Agent may rename or pause it, nothing more.
        A time with no UTC offset is read on the zone named, or on the Schedule's own.
        """
        service, user = await self._acting(owner_id, run_id)
        current = await service.get(user, schedule_id)
        if current is None:
            raise ScheduleRefusal(SCHEDULE_NOT_FOUND, _NOT_YOURS)
        if current.goal_max_rounds is not None and (
            paused is False
            or any(value is not None for value in (prompt, workspaces, mode, zone, cron, at))
        ):
            raise ScheduleRefusal(SCHEDULE_STATE, _GOAL_SCHEDULE)
        saved = [
            label
            for label, named in (
                ("name", name is not None),
                ("prompt", prompt is not None),
                ("workspaces", workspaces is not None),
                ("mode", mode is not None),
                ("when", any(value is not None for value in (zone, cron, at))),
            )
            if named
        ]
        schedule = current
        try:
            if saved:
                if at is not None and at.tzinfo is None:
                    at = at.replace(tzinfo=zone_of(zone or current.cadence.zone_name))
                schedule = await service.update(
                    user,
                    schedule_id,
                    name=name,
                    prompt=prompt,
                    workspaces=workspaces,
                    mode=mode,
                    zone=zone,
                    cron=cron,
                    at=at,
                    changed_by="agent",
                )
        except (ApplicationInputError, ApplicationConflictError) as error:
            raise _refusal(error) from error
        if paused is not None and schedule is not None:
            try:
                schedule = await (
                    service.pause(user, schedule_id)
                    if paused
                    else service.resume(user, schedule_id, changed_by="agent")
                )
            except (ApplicationInputError, ApplicationConflictError) as error:
                refusal = _refusal(error)
                if not saved:
                    raise refusal from error
                raise ScheduleRefusal(
                    refusal.code,
                    f"Saved: {', '.join(saved)}. Not done: {'pause' if paused else 'resume'}"
                    f" \u2014 {refusal}",
                ) from error
        if schedule is None:
            raise ScheduleRefusal(SCHEDULE_NOT_FOUND, _NOT_YOURS)
        return _record(schedule)

    async def delete(self, *, owner_id: str, run_id: str, schedule_id: str) -> ScheduleRecord:
        service, user = await self._acting(owner_id, run_id)
        schedule = await service.get(user, schedule_id)
        if schedule is None or not await service.delete(user, schedule_id):
            raise ScheduleRefusal(SCHEDULE_NOT_FOUND, _NOT_YOURS)
        return _record(schedule)

    async def _acting(self, owner_id: str, run_id: str) -> tuple[ScheduleService, UserContext]:
        """The service and the Run's caller, who must be the owner the Run runs for."""
        user = await self._callers.caller_of(owner_id=owner_id, run_id=run_id)
        if self._schedules is None or user is None or owner_id_from_user(user) != owner_id:
            raise ScheduleRefusal(SCHEDULE_UNAVAILABLE, SCHEDULES_UNAVAILABLE)
        return self._schedules, user


def _refusal(error: ApplicationInputError | ApplicationConflictError) -> ScheduleRefusal:
    if isinstance(error, ScheduleLimitError):
        return ScheduleRefusal(SCHEDULE_LIMIT, str(error))
    if isinstance(error, ApplicationConflictError):
        return ScheduleRefusal(SCHEDULE_STATE, str(error))
    return ScheduleRefusal(SCHEDULE_INVALID, str(error))


def _record(schedule: Schedule) -> ScheduleRecord:
    cadence = schedule.cadence
    return ScheduleRecord(
        schedule_id=schedule.schedule_id,
        name=schedule.name,
        prompt=schedule.prompt,
        mode=schedule.mode,
        target=schedule.target.value,
        workspaces=schedule.workspaces,
        zone=cadence.zone_name,
        cron=cadence.cron,
        at=cadence.at,
        state=schedule.state.value,
        pause_reason=schedule.pause_reason,
        next_fire_at=schedule.next_fire_at,
        goal_max_rounds=schedule.goal_max_rounds,
    )


__all__ = ["AgentSchedules", "OwnerCallers", "RunCallers"]
