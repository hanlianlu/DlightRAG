# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An owner's Agent Accounts in Settings, over real Web authentication, CSRF and PostgreSQL.

The routes are driven as a browser drives them, in each authentication mode, against a scratch
database whose rows the test reads back. What an owner is shown is checked for what it holds and
for what it must never hold: no password, envelope, key or id.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.browser.auth import WebAuthMiddleware
from dlightrag.adapters.http.browser.routes.agent_accounts import router
from dlightrag.adapters.http.errors import install_error_handlers
from dlightrag.adapters.postgres.answer.agent_accounts import (
    PGAgentAccountSettingsStore,
    PGAgentAccountStore,
)
from dlightrag.application.access import DEPLOYMENT_OWNER_ID, owner_id_from_principal
from dlightrag.application.agent_accounts import AgentAccounts
from dlightrag.application.config import DlightragConfig
from dlightrag.engine.answer.agent_browser import StoredAgentAccount
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.support.application_double import application_double
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MODES = ["jwt", "none", "simple"]
JWT_KEY = "test-only-web-jwt-key-not-for-production"
SIMPLE_TOKEN = "test-only-simple-token"
ENVELOPE, KEY_ID = "sealed-envelope-of-the-password", "ring-key-7"
CSRF_COOKIE = "dlightrag_web_csrf"
VIEW_KEYS = {"available", "registration", "accounts"}
ACCOUNT_KEYS = {"site", "email", "username", "created_at", "last_used_at"}


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


@dataclass
class Deployment:
    """A Web app serving the Settings routes, and the stores behind them."""

    mode: str
    app: FastAPI
    accounts: PGAgentAccountStore
    settings: PGAgentAccountSettingsStore

    def owner(self, subject: str = "a") -> str:
        """The owner a caller who signs in as ``subject`` is: only a jwt tells people apart."""
        if self.mode != "jwt":
            return DEPLOYMENT_OWNER_ID
        return owner_id_from_principal(auth_mode="jwt", user_id=subject, issuer="fixture-issuer")

    async def seed(
        self,
        owner: str,
        site: str,
        *,
        email: str | None = "agent@alias.example",
        username: str | None = None,
        used: bool = False,
    ) -> None:
        account = StoredAgentAccount(
            owner, site, f"id-of-{site}", email, username, KEY_ID, ENVELOPE
        )
        await self.accounts.save(account)
        if used:
            await self.accounts.mark_used(account)

    def headers(self, subject: str) -> dict[str, str]:
        if self.mode == "none":
            return {}
        token = (
            SIMPLE_TOKEN
            if self.mode == "simple"
            else jwt.encode({"iss": "fixture-issuer", "sub": subject}, JWT_KEY, algorithm="HS256")
        )
        return {"Authorization": f"Bearer {token}"}

    def browser(self, subject: str = "a") -> AsyncClient:
        """A browser signed in as ``subject``, which has not yet been issued a CSRF token."""
        return AsyncClient(
            transport=ASGITransport(self.app),
            base_url="http://test",
            headers=self.headers(subject),
        )


@asynccontextmanager
async def deployed(
    mode: str, *, available: bool = True, registration_allowed: bool = True
) -> AsyncIterator[Deployment]:
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
    async with isolated_run_runtime("agent_accounts_web") as (_, pool):
        accounts, settings = PGAgentAccountStore(pool=pool), PGAgentAccountSettingsStore(pool=pool)
        app = FastAPI()
        install_error_handlers(app)
        app.include_router(router, prefix="/web/api")
        app.state.application = application_double(
            config,
            agent_accounts=AgentAccounts(
                store=accounts,
                settings_store=settings,
                available=available,
                registration_allowed=registration_allowed,
            ),
        )
        app.add_middleware(WebAuthMiddleware, config_getter=lambda: config)
        yield Deployment(mode, app, accounts, settings)


def writing(client: AsyncClient) -> dict[str, str]:
    """The headers a page sends with a write: its own origin and the token it was issued."""
    return {"Origin": "http://test", "X-CSRF-Token": client.cookies.get(CSRF_COOKIE) or ""}


@pytest.mark.parametrize("mode", MODES)
async def test_an_owner_is_shown_their_accounts_and_never_a_secret(mode: str) -> None:
    async with deployed(mode) as web:
        await web.seed(web.owner("a"), "shop.example", used=True)
        await web.seed(web.owner("a"), "alpha.example", email=None, username="agent-77")
        await web.seed(web.owner("b"), "other.example")
        async with web.browser("a") as client:
            response = await client.get("/web/api/agent-accounts")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == VIEW_KEYS
        assert body["available"] is True
        assert body["registration"] == {"allowed": True, "enabled": True}
        accounts = body["accounts"]
        # An owner's own accounts, in the order of their sites; in a deployment of one owner
        # that owner has every account there is.
        assert [account["site"] for account in accounts] == (
            ["alpha.example", "shop.example"]
            if mode == "jwt"
            else ["alpha.example", "other.example", "shop.example"]
        )
        assert all(set(account) == ACCOUNT_KEYS for account in accounts)
        by_site = {account["site"]: account for account in accounts}
        assert (by_site["alpha.example"]["email"], by_site["alpha.example"]["username"]) == (
            None,
            "agent-77",
        )
        assert by_site["shop.example"]["email"] == "agent@alias.example"
        stamp = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
        assert all(stamp.fullmatch(account["created_at"]) for account in accounts)
        assert by_site["alpha.example"]["last_used_at"] is None
        assert stamp.fullmatch(by_site["shop.example"]["last_used_at"])
        for secret in (ENVELOPE, KEY_ID, "id-of-shop.example", "envelope", "password", "key_id"):
            assert secret not in response.text


@pytest.mark.parametrize("mode", MODES)
async def test_the_switch_for_new_sign_ups_is_set_and_read_back(mode: str) -> None:
    async with deployed(mode) as web, web.browser("a") as client:
        await client.get("/web/api/agent-accounts")

        off = await client.put(
            "/web/api/agent-accounts/settings",
            json={"registration_enabled": False},
            headers=writing(client),
        )

        assert off.status_code == 200
        assert off.json()["registration"] == {"allowed": True, "enabled": False}
        assert await web.settings.sign_ups_enabled(owner_id=web.owner("a")) is False
        again = (await client.get("/web/api/agent-accounts")).json()
        assert again["registration"] == {"allowed": True, "enabled": False}
        on = await client.put(
            "/web/api/agent-accounts/settings",
            json={"registration_enabled": True},
            headers=writing(client),
        )
        assert on.json()["registration"] == {"allowed": True, "enabled": True}
        # A body that is not exactly the one boolean is refused, and the switch stays.
        for body in ({}, {"registration_enabled": "no"}, {"registration_enabled": True, "x": 1}):
            refused = await client.put(
                "/web/api/agent-accounts/settings", json=body, headers=writing(client)
            )
            assert refused.status_code == 422
        assert await web.settings.sign_ups_enabled(owner_id=web.owner("a")) is True


async def test_a_switch_is_each_owners_own_in_a_deployment_that_tells_owners_apart() -> None:
    async with deployed("jwt") as web:
        async with web.browser("a") as first, web.browser("b") as second:
            await first.get("/web/api/agent-accounts")
            await first.put(
                "/web/api/agent-accounts/settings",
                json={"registration_enabled": False},
                headers=writing(first),
            )

            enabled = [
                (await who.get("/web/api/agent-accounts")).json()["registration"]["enabled"]
                for who in (first, second)
            ]

        assert enabled == [False, True]


@pytest.mark.parametrize("mode", MODES)
async def test_a_deployment_without_sign_ups_or_accounts_says_so_and_still_removes_accounts(
    mode: str,
) -> None:
    async with deployed(mode, available=False, registration_allowed=False) as web:
        await web.seed(web.owner("a"), "shop.example")
        async with web.browser("a") as client:
            view = (await client.get("/web/api/agent-accounts")).json()

            assert view["available"] is False
            assert view["registration"] == {"allowed": False, "enabled": True}
            # The owner's switch is theirs to set even where the deployment bounds it.
            switched = await client.put(
                "/web/api/agent-accounts/settings",
                json={"registration_enabled": False},
                headers=writing(client),
            )
            assert switched.json()["registration"] == {"allowed": False, "enabled": False}
            removed = await client.delete(
                "/web/api/agent-accounts/shop.example", headers=writing(client)
            )

        assert removed.status_code == 200 and removed.json()["accounts"] == []


@pytest.mark.parametrize("mode", MODES)
async def test_removing_an_account_answers_the_fresh_view_and_a_missing_one_is_not_found(
    mode: str,
) -> None:
    async with deployed(mode) as web, web.browser("a") as client:
        await web.seed(web.owner("a"), "shop.example")
        await web.seed(web.owner("a"), "keep.example")
        await client.get("/web/api/agent-accounts")

        removed = await client.delete(
            "/web/api/agent-accounts/shop.example", headers=writing(client)
        )
        gone = await client.delete("/web/api/agent-accounts/shop.example", headers=writing(client))

        assert removed.status_code == 200
        assert [account["site"] for account in removed.json()["accounts"]] == ["keep.example"]
        assert gone.status_code == 404
        assert gone.json()["error_type"] == "not_found"
        assert await web.accounts.account(owner_id=web.owner("a"), site="keep.example") is not None


async def test_another_owners_account_is_as_unknown_as_one_nobody_has() -> None:
    async with deployed("jwt") as web:
        await web.seed(web.owner("a"), "shop.example")
        async with web.browser("b") as client:
            await client.get("/web/api/agent-accounts")

            denied = await client.delete(
                "/web/api/agent-accounts/shop.example", headers=writing(client)
            )
            listed = (await client.get("/web/api/agent-accounts")).json()

        assert denied.status_code == 404
        assert listed["accounts"] == []
        assert await web.accounts.account(owner_id=web.owner("a"), site="shop.example") is not None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "site",
    ["Shop.Example", "shop..example", ".example", "shop.example.", "a%20b", "sh%C3%B6p.example"],
)
async def test_a_site_that_is_not_a_lowercase_hostname_is_refused(mode: str, site: str) -> None:
    async with deployed(mode) as web, web.browser("a") as client:
        await web.seed(web.owner("a"), "shop.example")
        await client.get("/web/api/agent-accounts")

        refused = await client.delete(f"/web/api/agent-accounts/{site}", headers=writing(client))

        assert refused.status_code == 422
        assert await web.accounts.account(owner_id=web.owner("a"), site="shop.example") is not None


def delete_shop(client: AsyncClient, headers: dict[str, str]) -> Any:
    return client.delete("/web/api/agent-accounts/shop.example", headers=headers)


def switch_off(client: AsyncClient, headers: dict[str, str]) -> Any:
    return client.put(
        "/web/api/agent-accounts/settings", json={"registration_enabled": False}, headers=headers
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("write", [delete_shop, switch_off], ids=["remove", "switch"])
async def test_a_write_the_page_did_not_make_is_refused_and_changes_nothing(
    mode: str, write: Callable[[AsyncClient, dict[str, str]], Any]
) -> None:
    async with deployed(mode) as web, web.browser("a") as client:
        await web.seed(web.owner("a"), "shop.example")
        await client.get("/web/api/agent-accounts")
        token = client.cookies.get(CSRF_COOKIE, "")
        assert token

        forged = [
            {"Origin": "https://evil.example", "X-CSRF-Token": token},
            {"Origin": "http://test", "X-CSRF-Token": "not-the-token"},
            {"Origin": "http://test"},
        ]
        refused = [(await write(client, headers)).status_code for headers in forged]

        assert refused == [403, 403, 403]
        assert await web.accounts.account(owner_id=web.owner("a"), site="shop.example") is not None
        assert await web.settings.sign_ups_enabled(owner_id=web.owner("a")) is True
        assert (await write(client, writing(client))).status_code == 200
