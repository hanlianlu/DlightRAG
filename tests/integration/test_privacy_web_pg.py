# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An owner's Conversation Contribution in Settings, over the real Web app, its authentication,
CSRF and PostgreSQL.

The routes are driven as a browser drives them, in each authentication mode, against a scratch
database whose rows the test reads back. What is checked is what an owner is told, that nobody
else's choice moves it, and that only a write the page made is kept.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.access.principal import (
    DEPLOYMENT_OWNER_ID,
    owner_id_from_principal,
)
from dlightrag.application.config.loading import set_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.privacy import Privacy, PrivacySettings
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_agent_accounts_web_pg import (
    CSRF_COOKIE,
    JWT_KEY,
    MODES,
    SIMPLE_TOKEN,
    writing,
)
from tests.support.application_double import application_double
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

URL = "/web/api/privacy/settings"
# UTC ISO 8601 with a Z, as every Web view of a time has it.
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z")
NOT_CONTRIBUTING = {"contribute": False, "since": None}


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


@dataclass
class Deployment:
    """A Web app serving the Settings route, and the store behind it."""

    mode: str
    app: FastAPI
    store: PGRunStore

    def owner(self, subject: str = "a") -> str:
        """The owner a caller who signs in as ``subject`` is: only a jwt tells people apart."""
        if self.mode != "jwt":
            return DEPLOYMENT_OWNER_ID
        return owner_id_from_principal(auth_mode="jwt", user_id=subject, issuer="fixture-issuer")

    def browser(self, subject: str = "a") -> AsyncClient:
        """A browser signed in as ``subject``, which has not yet been issued a CSRF token."""
        headers: dict[str, str] = {}
        if self.mode != "none":
            token = (
                SIMPLE_TOKEN
                if self.mode == "simple"
                else jwt.encode(
                    {"iss": "fixture-issuer", "sub": subject}, JWT_KEY, algorithm="HS256"
                )
            )
            headers = {"Authorization": f"Bearer {token}"}
        return AsyncClient(
            transport=ASGITransport(self.app), base_url="http://test", headers=headers
        )


@asynccontextmanager
async def deployed(mode: str) -> AsyncIterator[Deployment]:
    config = DlightragConfig(
        _env_file=None,
        models={
            "chat": {
                "roles": {
                    role: {"model": "fixture-model"}
                    for role in ("extract", "query", "keyword", "vlm")
                }
            }
        },
        access={
            "auth_mode": mode,
            "jwt_verification_key": JWT_KEY,
            "api_token": SIMPLE_TOKEN if mode == "simple" else None,
        },
    )
    # The app is the production one, so the route is reached the way a browser reaches it.
    set_config(config)
    async with isolated_run_runtime("privacy_web") as (store, _):
        app = create_app()
        app.state.application = application_double(config, privacy=Privacy(store=store))
        yield Deployment(mode, app, store)


def _moment(stamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(stamp)


@pytest.mark.parametrize("mode", MODES)
async def test_the_permission_is_off_until_given_and_can_be_withdrawn_and_given_again(
    mode: str,
) -> None:
    async with deployed(mode) as web, web.browser("a") as client:
        first = await client.get(URL)
        assert first.status_code == 200
        assert first.json() == NOT_CONTRIBUTING

        on = await client.put(URL, json={"contribute": True}, headers=writing(client))
        assert on.status_code == 200
        given = on.json()
        assert set(given) == {"contribute", "since"} and given["contribute"] is True
        assert STAMP.fullmatch(given["since"])
        assert (await client.get(URL)).json() == given
        # Turning it on while it is on keeps the moment it began.
        again = await client.put(URL, json={"contribute": True}, headers=writing(client))
        assert again.json() == given

        off = await client.put(URL, json={"contribute": False}, headers=writing(client))
        assert off.status_code == 200 and off.json() == NOT_CONTRIBUTING
        assert (await client.get(URL)).json() == NOT_CONTRIBUTING
        assert await web.store.read_privacy_settings(owner_id=web.owner("a")) == PrivacySettings(
            contribute=False, since=None
        )

        back = (await client.put(URL, json={"contribute": True}, headers=writing(client))).json()
        assert _moment(back["since"]) > _moment(given["since"])


async def test_a_body_that_is_not_exactly_one_boolean_is_refused_and_changes_nothing() -> None:
    async with deployed("jwt") as web, web.browser("a") as client:
        await client.get(URL)
        given = (await client.put(URL, json={"contribute": True}, headers=writing(client))).json()

        bodies = (
            {},
            {"contribute": "yes"},
            {"contribute": 1},
            {"contribute": None},
            {"contribute": False, "reason": "x"},
            [False],
        )
        refused = [
            (await client.put(URL, json=body, headers=writing(client))).status_code
            for body in bodies
        ]

        assert refused == [422] * len(bodies)
        assert (await client.get(URL)).json() == given


async def test_the_permission_is_each_owners_own_in_a_deployment_that_tells_owners_apart() -> None:
    async with deployed("jwt") as web:
        async with web.browser("a") as first, web.browser("b") as second:
            await first.get(URL)
            await second.get(URL)

            await first.put(URL, json={"contribute": True}, headers=writing(first))
            assert [(await who.get(URL)).json()["contribute"] for who in (first, second)] == [
                True,
                False,
            ]

            await second.put(URL, json={"contribute": False}, headers=writing(second))
            assert (await first.get(URL)).json()["contribute"] is True


@pytest.mark.parametrize("mode", MODES)
async def test_a_write_the_page_did_not_make_is_refused_and_changes_nothing(mode: str) -> None:
    async with deployed(mode) as web, web.browser("a") as client:
        await client.get(URL)
        token = client.cookies.get(CSRF_COOKIE, "")
        assert token

        forged = [
            {"Origin": "https://evil.example", "X-CSRF-Token": token},
            {"Origin": "http://test", "X-CSRF-Token": "not-the-token"},
            {"Origin": "http://test"},
        ]
        refused = [
            (await client.put(URL, json={"contribute": True}, headers=headers)).status_code
            for headers in forged
        ]

        assert refused == [403, 403, 403]
        assert await web.store.read_privacy_settings(owner_id=web.owner("a")) == PrivacySettings(
            contribute=False, since=None
        )
        assert (
            await client.put(URL, json={"contribute": True}, headers=writing(client))
        ).status_code == 200
