# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST Adapter for an owner's Schedules (ADR 0103)."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.schedules import (
    FIRES_PAGE_DEFAULT,
    FIRES_PAGE_MAX,
    CadenceInput,
    ScheduleEdit,
    ScheduleInput,
    fire_payload,
    fires_page_payload,
    preview_payload,
    schedule_payload,
    schedules_payload,
)
from dlightrag.application.access.principal import UserContext

from .deps import get_application, resolve_authorized_query_workspaces

router = APIRouter()


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")


@router.get("/schedules")
async def list_schedules(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
    return schedules_payload(await application.schedules.list(user), application.config.schedules)


@router.post("/schedules", status_code=status.HTTP_201_CREATED)
async def create_schedule(
    body: ScheduleInput, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
    workspaces = await resolve_authorized_query_workspaces(
        request, user, workspaces=body.workspaces, all_workspaces=False
    )
    schedule = await application.schedules.create(
        user,
        name=body.name,
        prompt=body.prompt,
        workspaces=workspaces,
        zone=body.zone,
        mode=body.mode,
        target=body.target,
        cron=body.cron,
        at=body.at,
        max_tokens_per_day=body.max_tokens_per_day,
        goal_max_rounds=body.goal_max_rounds,
    )
    return schedule_payload(schedule)


@router.post("/schedules/preview", dependencies=[Depends(get_current_user)])
async def preview_cadence(body: CadenceInput, request: Request) -> dict[str, Any]:
    """The next slots of a cadence, or the 422 saving it would meet; nothing is stored."""
    application = get_application(request)
    return preview_payload(application.config.schedules, body, await application.schedules.now())


@router.get("/schedules/{schedule_id}")
async def get_schedule(
    schedule_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    schedule = await get_application(request).schedules.get(user, schedule_id)
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.patch("/schedules/{schedule_id}")
async def edit_schedule(
    schedule_id: str,
    body: ScheduleEdit,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    workspaces = (
        None
        if body.workspaces is None
        else await resolve_authorized_query_workspaces(
            request, user, workspaces=body.workspaces, all_workspaces=False
        )
    )
    schedule = await get_application(request).schedules.update(
        user,
        schedule_id,
        name=body.name,
        prompt=body.prompt,
        workspaces=workspaces,
        mode=body.mode,
        zone=body.zone,
        cron=body.cron,
        at=body.at,
        max_tokens_per_day=body.max_tokens_per_day,
        goal_max_rounds=body.goal_max_rounds,
    )
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.delete("/schedules/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> None:
    if not await get_application(request).schedules.delete(user, schedule_id):
        raise _not_found()


@router.post("/schedules/{schedule_id}/pause")
async def pause_schedule(
    schedule_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    schedule = await get_application(request).schedules.pause(user, schedule_id)
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.post("/schedules/{schedule_id}/resume")
async def resume_schedule(
    schedule_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    schedule = await get_application(request).schedules.resume(user, schedule_id)
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.post("/schedules/{schedule_id}/run")
async def run_schedule_now(
    schedule_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Fire once now, under every rule a clock fire meets; the Fire says what became of it."""
    fire = await get_application(request).schedules.run_now(user, schedule_id)
    if fire is None:
        raise _not_found()
    return fire_payload(fire)


@router.get("/schedules/{schedule_id}/fires")
async def list_schedule_fires(
    schedule_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
    limit: Annotated[int, Query(ge=1, le=FIRES_PAGE_MAX)] = FIRES_PAGE_DEFAULT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
) -> dict[str, Any]:
    page = await fires_page_payload(
        get_application(request).schedules, user, schedule_id, limit=limit, cursor=cursor
    )
    if page is None:
        raise _not_found()
    return page


__all__ = ["router"]
