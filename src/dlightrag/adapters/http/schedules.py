# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the REST and Web surfaces say about Schedules (ADR 0103): one request vocabulary, one
projection, and the two reads the service does not make for them.

The routes of each surface authorize the workspaces a caller names, pass the authenticated caller
to the owner-scoped service, and answer through the functions here, so a Schedule is spelled one
way on both. Times are UTC ISO 8601 with a ``Z``; ``zone`` is the IANA zone the owner thinks of
the cadence in.
"""

from datetime import datetime, timedelta
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from dlightrag.adapters.http.errors import invalid_request
from dlightrag.application.access.principal import UserContext
from dlightrag.application.config.sections import SchedulesSettings
from dlightrag.application.opaque_cursor import format_utc_timestamp, parse_utc_timestamp
from dlightrag.application.schedules.cadence import Cadence
from dlightrag.application.schedules.model import (
    NAME_MAX_CHARS,
    PROMPT_MAX_CHARS,
    Fire,
    Schedule,
    ScheduleMode,
    Target,
)
from dlightrag.application.schedules.service import (
    FIRES_PAGE_DEFAULT,
    FIRES_PAGE_MAX,
    ScheduleService,
)

#: How many upcoming slots a preview names.
PREVIEW_SLOTS = 5


class CadenceInput(BaseModel):
    """A recurrence (``cron``, five fields) or one instant (``at``), read in ``zone``."""

    model_config = ConfigDict(extra="forbid")

    zone: str = Field(min_length=1, max_length=64)
    cron: str | None = Field(default=None, max_length=256)
    at: AwareDatetime | None = None


class ScheduleInput(CadenceInput):
    """A new Schedule. Without ``max_tokens_per_day`` it starts with the deployment's default."""

    name: str = Field(min_length=1, max_length=NAME_MAX_CHARS)
    prompt: str = Field(min_length=1, max_length=PROMPT_MAX_CHARS)
    mode: ScheduleMode = "research"
    workspaces: list[str] = Field(min_length=1)
    target: Target = Target.NEW_CONVERSATION
    max_tokens_per_day: int | None = Field(default=None, ge=1)
    goal_max_rounds: int | None = Field(default=None, ge=1)


class ScheduleEdit(BaseModel):
    """The parts of a Schedule to change; what is not named stays.

    Naming ``cron`` or ``at`` replaces the cadence, and naming only ``zone`` moves its recurrence
    to the new zone. Whether a Schedule starts a goal and where its fires go are fixed when it is
    made.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=NAME_MAX_CHARS)
    prompt: str | None = Field(default=None, min_length=1, max_length=PROMPT_MAX_CHARS)
    mode: ScheduleMode | None = None
    workspaces: list[str] | None = Field(default=None, min_length=1)
    zone: str | None = Field(default=None, min_length=1, max_length=64)
    cron: str | None = Field(default=None, max_length=256)
    at: AwareDatetime | None = None
    max_tokens_per_day: int | None = Field(default=None, ge=1)
    goal_max_rounds: int | None = Field(default=None, ge=1)


def _utc(value: datetime | None) -> str | None:
    return None if value is None else format_utc_timestamp(value)


def schedule_payload(schedule: Schedule) -> dict[str, Any]:
    """One Schedule as a client reads it; the principal it acts as stays on the server."""
    cadence = schedule.cadence
    return {
        "schedule_id": schedule.schedule_id,
        "name": schedule.name,
        "prompt": schedule.prompt,
        "mode": schedule.mode,
        "workspaces": list(schedule.workspaces),
        "target": schedule.target.value,
        "conversation_id": schedule.conversation_id,
        "zone": cadence.zone_name,
        "cron": cadence.cron,
        "at": _utc(cadence.at),
        "max_tokens_per_day": schedule.max_tokens_per_day,
        "goal_max_rounds": schedule.goal_max_rounds,
        "state": schedule.state.value,
        "pause_reason": schedule.pause_reason,
        "next_fire_at": _utc(schedule.next_fire_at),
        "created_at": _utc(schedule.created_at),
        "updated_at": _utc(schedule.updated_at),
    }


def schedules_payload(
    schedules: tuple[Schedule, ...], settings: SchedulesSettings
) -> dict[str, Any]:
    """The owner's Schedules, and the ceiling a new one starts with."""
    return {
        "schedules": [schedule_payload(schedule) for schedule in schedules],
        "default_max_tokens_per_day": settings.default_max_tokens_per_day,
    }


def fire_payload(fire: Fire) -> dict[str, Any]:
    """One Fire: the slot's state, why it started nothing when it did not, and the Run it began."""
    return {
        "fire_id": fire.fire_id,
        "schedule_id": fire.schedule_id,
        "slot": _utc(fire.slot),
        "state": fire.state.value,
        "reason": fire.reason,
        "run_id": fire.run_id,
        "conversation_id": fire.conversation_id,
        "missed_count": fire.missed_count,
        "last_slot": _utc(fire.last_slot),
        "manual": fire.manual,
        "last_error": fire.last_error,
        "created_at": _utc(fire.created_at),
        "updated_at": _utc(fire.updated_at),
    }


def preview_payload(
    settings: SchedulesSettings, body: CadenceInput, now: datetime
) -> dict[str, Any]:
    """The next ``PREVIEW_SLOTS`` slots of a cadence after ``now``, in UTC and on the zone's own clock.

    ``now`` is the Schedules' clock (``ScheduleService.now``), so a preview names the slots that
    saving it now would. The cadence is held to every rule saving it would be, so a refusal here
    is the one the owner would meet there; nothing is stored.
    """
    cadence = Cadence.create(
        zone_name=body.zone,
        cron=body.cron,
        at=body.at,
        now=now,
        min_interval=timedelta(minutes=settings.min_interval_minutes),
    )
    slots: list[dict[str, str]] = []
    slot = cadence.first_slot(now)
    while slot is not None and len(slots) < PREVIEW_SLOTS:
        slots.append(
            {
                "at": format_utc_timestamp(slot),
                "local": slot.astimezone(cadence.zone).isoformat(),
            }
        )
        slot = cadence.next_slot(slot)
    return {"zone": cadence.zone_name, "slots": slots}


async def fires_page_payload(
    service: ScheduleService,
    user: UserContext,
    schedule_id: str,
    *,
    limit: int,
    cursor: str | None,
) -> dict[str, Any] | None:
    """One page of a Schedule's Fires, newest slot first; ``None`` when it is not this owner's.

    The history is bounded (``schedules.fires_kept``) and a slot is unique within a Schedule, so
    a page is read from the whole of it and the cursor is the last slot the page ended on: Fires
    that arrive while the reader pages never shift a page.
    """
    after = None
    if cursor is not None:
        try:
            after = parse_utc_timestamp(cursor)
        except ValueError:
            raise invalid_request("is not a cursor this API issued", "query", "cursor") from None
    fires = await service.list_fires(user, schedule_id, limit=FIRES_PAGE_MAX)
    if fires is None:
        return None
    older = [fire for fire in fires if after is None or fire.slot < after]
    page = older[:limit]
    return {
        "fires": [fire_payload(fire) for fire in page],
        "next_cursor": format_utc_timestamp(page[-1].slot) if len(older) > limit else None,
    }


__all__ = [
    "FIRES_PAGE_DEFAULT",
    "FIRES_PAGE_MAX",
    "PREVIEW_SLOTS",
    "CadenceInput",
    "ScheduleEdit",
    "ScheduleInput",
    "fire_payload",
    "fires_page_payload",
    "preview_payload",
    "schedule_payload",
    "schedules_payload",
]
