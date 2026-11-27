# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin owner-only Settings routes for Schedules (ADR 0103).

The same operations as the REST adapter, answering the same shapes, for the Settings page. The
caller is the browser's signed-in owner; a Schedule that is not theirs is as unknown as one that
does not exist.
"""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
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
from dlightrag.application.access.control import AccessAction
from dlightrag.application.access.principal import UserContext
from dlightrag.application.corpus_admin.service import workspace_ids_for_names

router = APIRouter(prefix="/schedules")


def _user(request: Request) -> UserContext:
    return request.state.user_context


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")


async def _authorized(request: Request, names: list[str]) -> list[str]:
    """The canonical ids of workspaces the viewer may query, the way a chat turn names them."""
    workspaces = workspace_ids_for_names(names)
    for workspace in workspaces:
        await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace)
    return workspaces


@router.get("")
async def list_schedules(request: Request) -> dict[str, Any]:
    application = get_application(request)
    return schedules_payload(
        await application.schedules.list(_user(request)), application.config.schedules
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_schedule(request: Request, body: ScheduleInput) -> dict[str, Any]:
    workspaces = await _authorized(request, body.workspaces)
    schedule = await get_application(request).schedules.create(
        _user(request),
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


@router.post("/preview")
async def preview_cadence(request: Request, body: CadenceInput) -> dict[str, Any]:
    """The next slots of a cadence, or the 422 saving it would meet; nothing is stored."""
    return preview_payload(get_application(request).config.schedules, body)


@router.get("/{schedule_id}")
async def read_schedule(schedule_id: str, request: Request) -> dict[str, Any]:
    schedule = await get_application(request).schedules.get(_user(request), schedule_id)
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.patch("/{schedule_id}")
async def edit_schedule(schedule_id: str, request: Request, body: ScheduleEdit) -> dict[str, Any]:
    workspaces = None if body.workspaces is None else await _authorized(request, body.workspaces)
    schedule = await get_application(request).schedules.update(
        _user(request),
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


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(schedule_id: str, request: Request) -> None:
    if not await get_application(request).schedules.delete(_user(request), schedule_id):
        raise _not_found()


@router.post("/{schedule_id}/pause")
async def pause_schedule(schedule_id: str, request: Request) -> dict[str, Any]:
    schedule = await get_application(request).schedules.pause(_user(request), schedule_id)
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.post("/{schedule_id}/resume")
async def resume_schedule(schedule_id: str, request: Request) -> dict[str, Any]:
    schedule = await get_application(request).schedules.resume(_user(request), schedule_id)
    if schedule is None:
        raise _not_found()
    return schedule_payload(schedule)


@router.post("/{schedule_id}/run")
async def run_schedule_now(schedule_id: str, request: Request) -> dict[str, Any]:
    """Fire once now, under every rule a clock fire meets; the Fire says what became of it."""
    fire = await get_application(request).schedules.run_now(_user(request), schedule_id)
    if fire is None:
        raise _not_found()
    return fire_payload(fire)


@router.get("/{schedule_id}/fires")
async def list_schedule_fires(
    schedule_id: str,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=FIRES_PAGE_MAX)] = FIRES_PAGE_DEFAULT,
    cursor: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
) -> dict[str, Any]:
    page = await fires_page_payload(
        get_application(request).schedules, _user(request), schedule_id, limit=limit, cursor=cursor
    )
    if page is None:
        raise _not_found()
    return page


__all__ = ["router"]
