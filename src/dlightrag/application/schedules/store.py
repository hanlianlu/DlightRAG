# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the Schedule service and the look need from durable storage (ADR 0103).

A narrow port, like the goal store: the PostgreSQL adapter and the unit fake are interchangeable.
The port is where the queue lives. ``claim_due`` is the one transaction that turns a due Schedule
into its Fires, and ``finish_fire`` is the one that ends a Fire and decides, from what is stored,
whether its Schedule has now failed often enough to pause.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from dlightrag.application.access.principal import UserContext
from dlightrag.application.schedules.model import (
    ChangedBy,
    DuePlan,
    Fire,
    FireState,
    Schedule,
)


@dataclass(frozen=True, slots=True)
class DueFire:
    """A pending Fire with the Schedule it belongs to, as the look starts it."""

    schedule: Schedule
    fire: Fire


@dataclass(frozen=True, slots=True)
class Claim:
    """What one claim transaction found: how many due Schedules, and the Fires to start."""

    schedules: int
    due: tuple[DueFire, ...]


class ScheduleStore(Protocol):
    """Schedules and their Fires. Every owner-facing method is scoped to one owner."""

    async def now(self) -> datetime:
        """The instant the store's clock reads, as an aware UTC time.

        Every decision the Schedules make by time reads this one clock, so processes whose own
        clocks differ still agree on what is due.
        """
        ...

    async def create(self, schedule: Schedule, *, max_active: int) -> Schedule:
        """Store a new active Schedule; raises ``ScheduleLimitError`` at ``max_active``."""
        ...

    async def remember_principal(self, owner_id: str, principal: UserContext) -> None:
        """Keep ``principal`` as the identity of the owner's latest human turn: one record per owner."""
        ...

    async def principal_of(self, owner_id: str) -> UserContext | None:
        """The identity the owner's latest human turn kept, or ``None`` when none was or it is over a day old."""
        ...

    async def get(self, owner_id: str, schedule_id: str) -> Schedule | None: ...

    async def list(self, owner_id: str) -> tuple[Schedule, ...]: ...

    async def edit(self, schedule: Schedule, *, reschedule: bool) -> Schedule | None:
        """Write the owner-editable parts; ``next_fire_at`` moves only when ``reschedule``."""
        ...

    async def pause(
        self, owner_id: str, schedule_id: str, *, reason: str | None
    ) -> Schedule | None:
        """Pause an active Schedule; ``None`` when it is missing or not active.

        ``reason`` says why the look paused it; an owner's own pause has none to give.
        """
        ...

    async def resume(
        self,
        owner_id: str,
        schedule_id: str,
        *,
        principal: UserContext,
        max_active: int,
        changed_by: ChangedBy = "owner",
    ) -> Schedule | None:
        """Resume a paused Schedule as ``principal``; ``None`` when missing or not paused."""
        ...

    async def delete(self, owner_id: str, schedule_id: str) -> bool: ...

    async def claim_due(
        self,
        now: datetime,
        *,
        limit: int,
        fires_kept: int,
        plan: Callable[[Schedule], DuePlan],
    ) -> Claim:
        """Lock up to ``limit`` due Schedules, write the Fires ``plan`` names, and advance them."""
        ...

    async def claim_pending(
        self, now: datetime, *, lease: timedelta, limit: int
    ) -> tuple[DueFire, ...]:
        """Take pending Fires nobody has touched for ``lease``, so one process retries each."""
        ...

    async def insert_manual_fire(
        self, schedule: Schedule, slot: datetime, *, fires_kept: int
    ) -> DueFire | None:
        """A pending manual Fire for ``slot``, trimming to ``fires_kept``; ``None`` if the slot has one."""
        ...

    async def has_open_fire(self, owner_id: str, schedule_id: str) -> bool:
        """Whether any Fire of the Schedule is still ``pending`` or ``started``."""
        ...

    async def earlier_pending_fire(
        self, owner_id: str, schedule_id: str, before: datetime, *, exclude: str
    ) -> Fire | None:
        """A Fire of the Schedule that is still ``pending`` with a slot before ``before``."""
        ...

    async def record_error(self, owner_id: str, fire_id: str, error: str) -> None:
        """Remember what stopped the latest attempt to start a pending Fire."""
        ...

    async def touch_fire(self, owner_id: str, fire_id: str) -> None:
        """Say a started Fire was just looked at, so the sweep reads others before it again."""
        ...

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
        """Move a Fire from one of ``expected`` states to ``state``; ``None`` when it was elsewhere.

        A Fire that ends ``failed`` or ``refused`` pauses its active Schedule at once when
        ``pause_reason`` is given, and otherwise when the Schedule's newest ``failure_streak``
        Fires that ran or were refused all ended that way.
        """
        ...

    async def fire_for_run(self, owner_id: str, run_id: str) -> Fire | None: ...

    async def previous_fire(self, owner_id: str, schedule_id: str, before: datetime) -> Fire | None:
        """The newest Fire before ``before`` that started a Run."""
        ...

    async def unsettled_fires(self, *, older_than: datetime, limit: int) -> tuple[Fire, ...]:
        """Fires still ``started`` that nothing has touched since ``older_than``."""
        ...

    async def list_fires(
        self, owner_id: str, schedule_id: str, *, limit: int
    ) -> tuple[Fire, ...] | None:
        """The Schedule's newest Fires first; ``None`` when it is not this owner's."""
        ...

    async def spent_tokens(self, owner_id: str, schedule_id: str, *, since: datetime) -> int:
        """Tokens spent since ``since`` by the Runs this Schedule's Fires started."""
        ...


__all__ = ["Claim", "DueFire", "ScheduleStore"]
