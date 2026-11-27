# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The REST and Web routes for an owner's Schedules (ADR 0103 Decision 12).

The service is the real one over the Schedule store stand-in, so what these tests observe through
HTTP is what an owner meets: how a refusal is spelled, whose Schedules a caller sees, and what the
workspaces a caller names must clear before the service stores them.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import jwt
import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.browser.auth import WEB_CSRF_COOKIE
from dlightrag.adapters.http.errors import install_error_handlers
from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.routes import schedules as rest_routes
from dlightrag.adapters.http.server import create_app
from dlightrag.application.access.control import AccessAction
from dlightrag.application.access.principal import UserContext
from dlightrag.application.config.sections import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
    SchedulesSettings,
)
from dlightrag.application.schedules.errors import ScheduleStateError
from dlightrag.application.schedules.model import FireState, ScheduleState
from dlightrag.application.schedules.service import ScheduleService
from dlightrag.application.schedules.store import DueFire
from tests.support.application_double import application_double
from tests.support.config_helpers import mutate_config
from tests.support.schedules import Clock, FakeScheduleStore

#: A Monday, ten minutes past ten UTC.
NOW = datetime(2026, 12, 7, 10, 0, 20, tzinfo=UTC)
ISSUER = "https://team.example.com"
KEY = "test-only-web-jwt-key-32-bytes!!"
FAR = "2099-06-01T08:00:00+02:00"

BODY: dict[str, Any] = {
    "name": "Weekly review",
    "prompt": "Summarize the week.",
    "workspaces": ["alpha"],
    "zone": "Europe/Stockholm",
    "cron": "0 8 * * 1",
}


class _Firer:
    """What starts a Fire's Run, answering as a Run that began, or refusing as a busy moment."""

    refusal: ScheduleStateError | None = None

    async def start(self, due: DueFire) -> Any:
        if self.refusal is not None:
            raise self.refusal
        return replace(
            due.fire, state=FireState.STARTED, run_id="run-1", conversation_id="conversation-1"
        )


class _Parts:
    def __init__(self, config: DlightragConfig, **settings: Any) -> None:
        self.clock = Clock(NOW)
        self.store = FakeScheduleStore(self.clock)
        self.firer = _Firer()
        self.service = ScheduleService(
            store=self.store,
            firer=cast(Any, self.firer),
            settings=SchedulesSettings(**settings) if settings else config.schedules,
            clock=self.clock,
        )
        self.application = application_double(config, schedules=self.service)


@pytest.fixture
def parts(test_config: DlightragConfig) -> _Parts:
    return _Parts(test_config)


def _only_alpha_for_alice(config: DlightragConfig) -> None:
    """Everyone from the team signs in, and may query one workspace: ``alpha``."""
    mutate_config(
        config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="iss",
                    value=ISSUER,
                    workspaces=["alpha"],
                    actions=[AccessAction.WORKSPACE_QUERY],
                )
            ]
        ),
    )


def _rest_app(application: Any) -> FastAPI:
    """The REST routes alone, with whoever the ``X-Test-User`` header names signed in."""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(rest_routes.router)
    app.state.application = application

    async def signed_in(request: Request) -> UserContext:
        name = request.headers.get("X-Test-User", "alice")
        return UserContext(user_id=name, auth_mode="jwt", claims={"iss": ISSUER})

    app.dependency_overrides[get_current_user] = signed_in
    return app


@pytest.fixture
async def rest(parts: _Parts):
    async with AsyncClient(
        transport=ASGITransport(app=_rest_app(parts.application)), base_url="http://test"
    ) as client:
        yield client


async def _create(client: AsyncClient, user: str = "alice", **changes: Any) -> dict[str, Any]:
    response = await client.post(
        "/schedules", json={**BODY, **changes}, headers={"X-Test-User": user}
    )
    assert response.status_code == 201, response.text
    return response.json()


def _as(user: str) -> dict[str, str]:
    return {"X-Test-User": user}


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


async def test_an_owner_creates_lists_and_reads_a_schedule_with_the_deployments_default_ceiling(
    rest: AsyncClient, test_config: DlightragConfig
) -> None:
    created = await _create(rest)

    assert created == {
        "schedule_id": created["schedule_id"],
        "name": "Weekly review",
        "prompt": "Summarize the week.",
        "mode": "research",
        "workspaces": ["alpha"],
        "target": "new_conversation",
        "conversation_id": None,
        "zone": "Europe/Stockholm",
        "cron": "0 8 * * 1",
        "at": None,
        "max_tokens_per_day": test_config.schedules.default_max_tokens_per_day,
        "goal_max_rounds": None,
        "state": "active",
        "pause_reason": None,
        "next_fire_at": "2026-12-14T07:00:00.000000Z",
        "created_at": "2026-12-07T10:00:20.000000Z",
        "updated_at": "2026-12-07T10:00:20.000000Z",
    }
    listed = (await rest.get("/schedules", headers=_as("alice"))).json()
    assert listed == {
        "schedules": [created],
        "default_max_tokens_per_day": test_config.schedules.default_max_tokens_per_day,
    }
    assert (
        await rest.get(f"/schedules/{created['schedule_id']}", headers=_as("alice"))
    ).json() == created


async def test_a_single_time_and_a_goal_are_stored_as_named(rest: AsyncClient) -> None:
    once = await _create(rest, cron=None, at=FAR, target="same_conversation", mode="fast")
    goal = await _create(rest, goal_max_rounds=4, max_tokens_per_day=500)

    assert (once["cron"], once["at"], once["next_fire_at"]) == (
        None,
        "2099-06-01T06:00:00.000000Z",
        "2099-06-01T06:00:00.000000Z",
    )
    assert (once["target"], once["mode"]) == ("same_conversation", "fast")
    assert (goal["goal_max_rounds"], goal["max_tokens_per_day"]) == (4, 500)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"cron": "*/5 * * * *"}, "at least 60 minutes"),
        ({"cron": "0 8 * *"}, "five fields"),
        ({"cron": None}, "needs a recurrence or a single time"),
        ({"cron": None, "at": "2020-01-01T00:00:00Z"}, "in the future"),
        ({"cron": None, "at": "2099-06-01T08:00:00"}, "body.at"),
        ({"zone": "Mars/Olympus"}, "not an IANA time zone"),
        ({"workspaces": []}, "body.workspaces"),
        ({"mode": "slow"}, "body.mode"),
        ({"name": "   "}, "needs a name"),
        ({"goal_max_rounds": 3, "mode": "fast"}, "research mode"),
        ({"max_tokens_per_day": 0}, "body.max_tokens_per_day"),
        ({"nickname": "x"}, "body.nickname"),
    ],
)
async def test_a_schedule_the_owner_must_change_is_refused_with_422_and_the_reason(
    rest: AsyncClient, parts: _Parts, changes: dict[str, Any], reason: str
) -> None:
    response = await rest.post("/schedules", json={**BODY, **changes}, headers=_as("alice"))

    assert response.status_code == 422
    assert response.json()["error_type"] == "validation"
    assert reason in response.json()["detail"]
    assert parts.store.schedules == {}


async def test_the_per_owner_active_cap_is_a_409_that_says_how_many(
    test_config: DlightragConfig,
) -> None:
    capped = _Parts(test_config, max_active_per_owner=1)
    async with AsyncClient(
        transport=ASGITransport(app=_rest_app(capped.application)), base_url="http://test"
    ) as client:
        first = await _create(client)
        refused = await client.post("/schedules", json=BODY, headers=_as("alice"))
        other = await client.post("/schedules", json=BODY, headers=_as("bob"))
        await client.post(f"/schedules/{first['schedule_id']}/pause", headers=_as("alice"))
        again = await client.post("/schedules", json=BODY, headers=_as("alice"))

    assert refused.status_code == 409
    assert refused.json() == {
        "detail": "an owner may hold at most 1 active Schedules",
        "error_type": "conflict",
    }
    assert (other.status_code, again.status_code) == (201, 201)


async def test_nobody_but_the_owner_sees_or_changes_a_schedule(rest: AsyncClient) -> None:
    created = await _create(rest)
    path = f"/schedules/{created['schedule_id']}"

    assert (await rest.get("/schedules", headers=_as("bob"))).json()["schedules"] == []
    for method, url, body in (
        ("GET", path, None),
        ("PATCH", path, {"name": "Mine now"}),
        ("DELETE", path, None),
        ("POST", f"{path}/pause", None),
        ("POST", f"{path}/resume", None),
        ("POST", f"{path}/run", None),
        ("GET", f"{path}/fires", None),
    ):
        response = await rest.request(method, url, json=body, headers=_as("bob"))
        assert response.status_code == 404, (method, url)
        assert response.json()["error_type"] == "not_found"
    assert (await rest.get(path, headers=_as("alice"))).json() == created


async def test_an_edit_changes_the_named_parts_and_a_new_cadence_recomputes_the_next_slot(
    rest: AsyncClient,
) -> None:
    created = await _create(rest)
    path = f"/schedules/{created['schedule_id']}"

    renamed = await rest.patch(
        path, json={"name": "Monday review", "mode": "fast"}, headers=_as("alice")
    )
    assert renamed.status_code == 200
    assert (renamed.json()["name"], renamed.json()["mode"]) == ("Monday review", "fast")
    assert renamed.json()["next_fire_at"] == created["next_fire_at"]

    moved = await rest.patch(
        path, json={"cron": "30 9 * * 2", "zone": "Asia/Tokyo"}, headers=_as("alice")
    )
    assert (moved.json()["cron"], moved.json()["zone"]) == ("30 9 * * 2", "Asia/Tokyo")
    assert moved.json()["next_fire_at"] == "2026-12-08T00:30:00.000000Z"

    refused = await rest.patch(path, json={"cron": "* * * * *"}, headers=_as("alice"))
    assert refused.status_code == 422
    assert (await rest.get(path, headers=_as("alice"))).json()["cron"] == "30 9 * * 2"

    plain = await rest.patch(path, json={"goal_max_rounds": 3}, headers=_as("alice"))
    assert plain.status_code == 422
    assert "does not start a goal" in plain.json()["detail"]


async def test_pausing_resuming_and_deleting_a_schedule(rest: AsyncClient) -> None:
    created = await _create(rest)
    path = f"/schedules/{created['schedule_id']}"

    paused = (await rest.post(f"{path}/pause", headers=_as("alice"))).json()
    assert (paused["state"], paused["pause_reason"]) == ("paused", None)
    resumed = (await rest.post(f"{path}/resume", headers=_as("alice"))).json()
    assert (resumed["state"], resumed["pause_reason"]) == ("active", None)

    gone = await rest.delete(path, headers=_as("alice"))
    assert (gone.status_code, gone.content) == (204, b"")
    assert (await rest.get(path, headers=_as("alice"))).status_code == 404
    assert (await rest.delete(path, headers=_as("alice"))).status_code == 404


async def test_a_completed_schedule_refuses_to_pause_resume_or_edit_with_409(
    rest: AsyncClient, parts: _Parts
) -> None:
    created = await _create(rest, cron=None, at=FAR)
    path = f"/schedules/{created['schedule_id']}"
    stored = parts.store.schedules[created["schedule_id"]]
    parts.store.schedules[created["schedule_id"]] = replace(stored, state=ScheduleState.COMPLETED)

    for method, url, body in (
        ("POST", f"{path}/pause", None),
        ("POST", f"{path}/resume", None),
        ("PATCH", path, {"name": "again"}),
    ):
        response = await rest.request(method, url, json=body, headers=_as("alice"))
        assert response.status_code == 409, (method, url)
        assert response.json()["error_type"] == "conflict"


async def test_running_now_answers_the_fire_and_a_refused_moment_is_a_409(
    rest: AsyncClient, parts: _Parts
) -> None:
    created = await _create(rest)
    path = f"/schedules/{created['schedule_id']}/run"

    started = await rest.post(path, headers=_as("alice"))
    assert started.status_code == 200
    assert started.json() | {"fire_id": "-"} == {
        "fire_id": "-",
        "schedule_id": created["schedule_id"],
        "slot": "2026-12-07T10:00:20.000000Z",
        "state": "started",
        "reason": None,
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "missed_count": None,
        "last_slot": None,
        "manual": True,
        "last_error": None,
        "created_at": "2026-12-07T10:00:20.000000Z",
        "updated_at": "2026-12-07T10:00:20.000000Z",
    }

    # Another request while that Fire is still working is refused: one open Fire at a time.
    taken = await rest.post(path, headers=_as("alice"))
    assert taken.status_code == 409
    assert "already has a fire" in taken.json()["detail"]


async def test_fires_page_newest_first_by_a_cursor_that_names_the_last_slot(
    rest: AsyncClient, parts: _Parts
) -> None:
    created = await _create(rest)
    schedule = parts.store.schedules[created["schedule_id"]]
    for day in range(5):
        await parts.store.insert_manual_fire(schedule, NOW + timedelta(days=day), fires_kept=200)
    path = f"/schedules/{created['schedule_id']}/fires"

    first = (await rest.get(path, params={"limit": 2}, headers=_as("alice"))).json()
    assert [fire["slot"][:10] for fire in first["fires"]] == ["2026-12-11", "2026-12-10"]
    assert first["next_cursor"] == "2026-12-10T10:00:20.000000Z"

    second = (
        await rest.get(
            path, params={"limit": 2, "cursor": first["next_cursor"]}, headers=_as("alice")
        )
    ).json()
    assert [fire["slot"][:10] for fire in second["fires"]] == ["2026-12-09", "2026-12-08"]

    last = (
        await rest.get(
            path, params={"limit": 2, "cursor": second["next_cursor"]}, headers=_as("alice")
        )
    ).json()
    assert [fire["slot"][:10] for fire in last["fires"]] == ["2026-12-07"]
    assert last["next_cursor"] is None

    everything = (await rest.get(path, headers=_as("alice"))).json()
    assert len(everything["fires"]) == 5 and everything["next_cursor"] is None

    refused = await rest.get(path, params={"cursor": "2026-12-10"}, headers=_as("alice"))
    assert refused.status_code == 422
    assert "query.cursor" in refused.json()["detail"]
    assert (await rest.get(path, params={"limit": 201}, headers=_as("alice"))).status_code == 422


async def test_a_preview_names_the_next_five_slots_in_utc_and_on_the_zones_clock(
    rest: AsyncClient,
) -> None:
    response = await rest.post(
        "/schedules/preview",
        json={"zone": "Europe/Stockholm", "cron": "0 8 * * 1"},
        headers=_as("alice"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["zone"] == "Europe/Stockholm"
    slots = body["slots"]
    assert len(slots) == 5
    instants = [datetime.fromisoformat(slot["at"]) for slot in slots]
    assert instants == sorted(instants) and len(set(instants)) == 5
    assert instants[0] > datetime.now(UTC)
    for slot, instant in zip(slots, instants, strict=True):
        local = datetime.fromisoformat(slot["local"])
        assert local == instant
        assert (local.hour, local.minute, local.weekday()) == (8, 0, 0)
        assert local.utcoffset() in (timedelta(hours=1), timedelta(hours=2))


async def test_a_preview_of_a_single_time_is_that_time_and_a_bad_cadence_is_the_422_saving_would_meet(
    rest: AsyncClient,
) -> None:
    once = await rest.post(
        "/schedules/preview", json={"zone": "Asia/Tokyo", "at": FAR}, headers=_as("alice")
    )
    assert once.json()["slots"] == [
        {"at": "2099-06-01T06:00:00.000000Z", "local": "2099-06-01T15:00:00+09:00"}
    ]

    for changes, reason in (
        ({"cron": "*/5 * * * *"}, "at least 60 minutes"),
        ({"cron": "nonsense"}, "five fields"),
        ({"cron": "0 8 * * 1", "at": FAR}, "not both"),
        ({"cron": "0 8 * * 1", "zone": "Nowhere/Land"}, "not an IANA time zone"),
    ):
        refused = await rest.post(
            "/schedules/preview",
            json={"zone": "Europe/Stockholm", **changes},
            headers=_as("alice"),
        )
        assert refused.status_code == 422, changes
        assert reason in refused.json()["detail"], changes


async def test_the_workspaces_a_caller_names_must_clear_the_query_check_before_the_service_stores_them(
    test_config: DlightragConfig,
) -> None:
    _only_alpha_for_alice(test_config)
    parts = _Parts(test_config)
    async with AsyncClient(
        transport=ASGITransport(app=_rest_app(parts.application)), base_url="http://test"
    ) as client:
        created = await _create(client)
        denied = await client.post(
            "/schedules", json={**BODY, "workspaces": ["alpha", "beta"]}, headers=_as("alice")
        )
        edit_denied = await client.patch(
            f"/schedules/{created['schedule_id']}",
            json={"workspaces": ["beta"]},
            headers=_as("alice"),
        )

    assert denied.status_code == edit_denied.status_code == 403
    assert denied.json()["error_type"] == "auth"
    assert [schedule.workspaces for schedule in parts.store.schedules.values()] == [("alpha",)]


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------


@pytest.fixture
async def web(parts: _Parts):
    app = create_app()
    app.state.application = parts.application
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", follow_redirects=False
    ) as client:
        yield client


async def test_settings_creates_edits_runs_and_deletes_a_schedule_through_the_web_routes(
    web: AsyncClient, parts: _Parts, test_config: DlightragConfig
) -> None:
    empty = (await web.get("/web/api/schedules")).json()
    assert empty == {
        "schedules": [],
        "default_max_tokens_per_day": test_config.schedules.default_max_tokens_per_day,
    }

    created = await web.post("/web/api/schedules", json=BODY)
    assert created.status_code == 201
    schedule = created.json()
    path = f"/web/api/schedules/{schedule['schedule_id']}"
    assert schedule["next_fire_at"] == "2026-12-14T07:00:00.000000Z"

    assert (await web.get(path)).json() == schedule
    edited = (await web.patch(path, json={"name": "Monday review"})).json()
    assert edited["name"] == "Monday review"
    assert (await web.post(f"{path}/pause")).json()["state"] == "paused"
    assert (await web.post(f"{path}/resume")).json()["state"] == "active"

    fire = (await web.post(f"{path}/run")).json()
    assert (fire["state"], fire["run_id"], fire["conversation_id"]) == (
        "started",
        "run-1",
        "conversation-1",
    )
    fires = (await web.get(f"{path}/fires")).json()
    assert [item["fire_id"] for item in fires["fires"]] == [fire["fire_id"]]
    assert fires["next_cursor"] is None

    preview = (
        await web.post("/web/api/schedules/preview", json={"zone": "UTC", "cron": "0 * * * *"})
    ).json()
    assert len(preview["slots"]) == 5

    assert (await web.delete(path)).status_code == 204
    assert (await web.get(path)).status_code == 404
    assert parts.store.schedules == {}


async def test_the_web_routes_answer_a_refusal_in_the_general_envelope(web: AsyncClient) -> None:
    refused = await web.post("/web/api/schedules", json={**BODY, "cron": "0 8 * *"})
    assert refused.status_code == 422
    assert refused.json()["error_type"] == "validation"
    assert "five fields" in refused.json()["detail"]

    missing = await web.post("/web/api/schedules/11111111-1111-4111-8111-111111111111/pause")
    assert (missing.status_code, missing.json()["error_type"]) == (404, "not_found")


async def test_a_signed_in_owner_sees_only_their_schedules_and_writes_carry_the_csrf_token(
    web: AsyncClient, parts: _Parts, test_config: DlightragConfig
) -> None:
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", KEY)
    _only_alpha_for_alice(test_config)

    def owner(subject: str) -> dict[str, str]:
        """What the browser of one signed-in owner sends, the double-submit token included."""
        claims = {"iss": ISSUER, "sub": subject, "exp": datetime.now(UTC) + timedelta(hours=1)}
        headers = {"Authorization": f"Bearer {jwt.encode(claims, KEY, algorithm='HS256')}"}
        if csrf := web.cookies.get(WEB_CSRF_COOKIE):
            headers["X-CSRF-Token"] = csrf
        return headers

    assert (await web.get("/web/api/schedules", headers=owner("alice"))).status_code == 200

    unguarded = await web.post(
        "/web/api/schedules", json=BODY, headers=_without_csrf(owner("alice"))
    )
    assert unguarded.status_code == 403
    assert parts.store.schedules == {}

    created = await web.post("/web/api/schedules", json=BODY, headers=owner("alice"))
    assert created.status_code == 201
    path = f"/web/api/schedules/{created.json()['schedule_id']}"

    assert (await web.get("/web/api/schedules", headers=owner("bob"))).json()["schedules"] == []
    assert (await web.get(path, headers=owner("bob"))).status_code == 404
    assert (await web.delete(path, headers=owner("bob"))).status_code == 404
    assert (await web.get(path, headers=owner("alice"))).status_code == 200

    denied = await web.post(
        "/web/api/schedules", json={**BODY, "workspaces": ["beta"]}, headers=owner("alice")
    )
    assert denied.status_code == 403
    assert len(parts.store.schedules) == 1


def _without_csrf(headers: dict[str, str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name != "X-CSRF-Token"}
