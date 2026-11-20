# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An owner's Profile Memory switch in Settings, and what a Run's writers left under an answer,
over the real Web app, its authentication, CSRF and PostgreSQL.

The routes are driven as a browser drives them, in each authentication mode, against a scratch
database whose settings row the test reads back. What is checked is that only a write the page
made is kept, and that a reader is shown only their own Run's changes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import jwt
import pytest
from dlightrag_memory import MemoryProvenance
from dlightrag_memory.ports import ChineseAndEnglish, NullEmbedder
from dlightrag_memory.postgres import PostgresMemoryStore
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.postgres.answer.memory_settings import PGMemorySettingsStore
from dlightrag.application.access.principal import (
    DEPLOYMENT_OWNER_ID,
    owner_id_from_principal,
)
from dlightrag.application.config.loading import set_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.memory.service import MemoryService
from dlightrag.engine.answer.memory import MemoryCapability
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

SETTINGS_URL = "/web/api/memory/settings"
CLEAR_URL = "/web/api/memory/clear"
# An owner who never chose has memory on, and no run invalidated.
UNTOUCHED = MemoryCapability(enabled=True, epoch=0)


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


@dataclass
class Deployment:
    """A Web app serving the Settings routes, and the settings row behind them."""

    mode: str
    app: FastAPI
    settings: PGMemorySettingsStore
    memory: MemoryService

    def owner(self, subject: str = "a") -> str:
        """The owner a caller who signs in as ``subject`` is: only a jwt tells people apart."""
        if self.mode != "jwt":
            return DEPLOYMENT_OWNER_ID
        return owner_id_from_principal(auth_mode="jwt", user_id=subject, issuer="fixture-issuer")

    async def state(self) -> MemoryCapability:
        return await self.settings.state(owner_id=self.owner())

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
    # The app is the production one, so the routes are reached the way a browser reaches them.
    set_config(config)
    async with isolated_run_runtime("memory_web") as (run_store, pool):
        records = PostgresMemoryStore(
            pool=pool, embedder=NullEmbedder(), languages=ChineseAndEnglish()
        )
        await records.initialize()
        settings = PGMemorySettingsStore(pool=pool)
        app = create_app()
        memory = MemoryService(
            records,
            settings_store=settings,
            sessions=run_store,
            memory_list_cursor_secret=b"memory-web-test",
        )
        app.state.application = application_double(config, memory=memory)
        try:
            yield Deployment(mode, app, settings, memory)
        finally:
            await records.aclose()


@pytest.mark.parametrize("mode", MODES)
async def test_the_switch_is_set_and_read_back(mode: str) -> None:
    async with deployed(mode) as web, web.browser() as client:
        assert (await client.get(SETTINGS_URL)).json()["enabled"] is True

        off = await client.put(SETTINGS_URL, json={"enabled": False}, headers=writing(client))

        assert off.status_code == 200 and off.json() == {"enabled": False, "active_count": None}
        assert (await client.get(SETTINGS_URL)).json()["enabled"] is False
        assert (await web.state()).enabled is False


@pytest.mark.parametrize("mode", MODES)
async def test_a_write_the_page_did_not_make_is_refused_and_changes_nothing(mode: str) -> None:
    async with deployed(mode) as web, web.browser() as client:
        await client.get(SETTINGS_URL)
        token = client.cookies.get(CSRF_COOKIE, "")
        assert token

        forged = [
            {"Origin": "https://evil.example", "X-CSRF-Token": token},
            {"Origin": "http://test", "X-CSRF-Token": "not-the-token"},
            {"Origin": "http://test"},
        ]
        switched = [
            (await client.put(SETTINGS_URL, json={"enabled": False}, headers=headers)).status_code
            for headers in forged
        ]
        cleared = [
            (await client.post(CLEAR_URL, headers=headers)).status_code for headers in forged
        ]

        assert switched == cleared == [403, 403, 403]
        assert await web.state() == UNTOUCHED
        assert (
            await client.put(SETTINGS_URL, json={"enabled": False}, headers=writing(client))
        ).status_code == 200


RUN = "0190a1b2-0000-7000-8000-00000000000a"
# The Web contract and the REST one answer the same question at their own roots.
CHANGES_URLS = (f"/web/api/memory/runs/{RUN}/changes", f"/memory/runs/{RUN}/changes")


async def _left_by(
    web: Deployment, owner: str, run: str, origin: str, kind: str, body: str, key: str
) -> str:
    """One record a writer left in ``owner``'s profile under ``run``; the change that made it."""
    receipt = await web.memory.remember(
        owner_id=owner,
        kind=kind,  # type: ignore[arg-type]
        body=body,
        provenance=MemoryProvenance(
            origin_kind=origin,  # type: ignore[arg-type]
            origin_id=run,
            run_id=run,
            session_id="session-1",
        ),
        idempotency_key=key,
    )
    assert receipt.outcome == "changed"
    return receipt.change_id


@pytest.mark.parametrize("url", CHANGES_URLS)
@pytest.mark.parametrize("mode", MODES)
async def test_a_reader_is_shown_what_their_runs_writers_left_and_nothing_of_anyone_elses(
    mode: str, url: str
) -> None:
    async with deployed(mode) as web, web.browser() as client:
        mine = web.owner()
        agent = await _left_by(web, mine, RUN, "answer_run", "preference", "Tables first.", "k1")
        learned = await _left_by(web, mine, RUN, "session_pass", "fact", "Lives in Munich.", "k2")
        await _left_by(web, mine, "another-run", "answer_run", "fact", "Another run's.", "k3")
        if mode == "jwt":  # the only mode with more than one owner
            await _left_by(web, web.owner("b"), RUN, "answer_run", "fact", "Not mine.", "k4")

        response = await client.get(url)

        assert response.status_code == 200
        shown = response.json()["changes"]
        assert [(c["body"], c["writer"], c["action"]) for c in shown] == [
            ("Tables first.", "agent", "add"),
            ("Lives in Munich.", "conversation", "add"),
        ]
        assert [c["change_id"] for c in shown] == [agent, learned]
        assert {c["kind"] for c in shown} == {"preference", "fact"}


@pytest.mark.parametrize("url", CHANGES_URLS)
@pytest.mark.parametrize("mode", MODES)
async def test_a_reader_whose_memory_is_off_is_shown_nothing(mode: str, url: str) -> None:
    async with deployed(mode) as web, web.browser() as client:
        await _left_by(web, web.owner(), RUN, "answer_run", "fact", "Lives in Munich.", "k1")
        assert len((await client.get(url)).json()["changes"]) == 1

        await web.memory.set_enabled(owner_id=web.owner(), enabled=False)

        response = await client.get(url)
        assert response.status_code == 200 and response.json() == {"changes": []}
        await web.memory.set_enabled(owner_id=web.owner(), enabled=True)
        assert len((await client.get(url)).json()["changes"]) == 1


async def test_a_reader_who_is_not_signed_in_is_shown_no_changes() -> None:
    async with deployed("simple") as web:
        anonymous = AsyncClient(transport=ASGITransport(web.app), base_url="http://test")
        async with anonymous:
            for url in CHANGES_URLS:
                assert (await anonymous.get(url)).status_code in (401, 403)


# The Web contract and the REST one take the same mutations at their own roots.
ROOTS = ("/web/api", "")
NOT_AN_ID = "not-a-uuid"
UNKNOWN_ID = "33333333-3333-3333-3333-333333333333"


async def _write(client: AsyncClient, method: str, url: str, **kwargs: Any) -> Response:
    """One mutation as a page or a REST caller makes it, under its own origin and CSRF token."""
    await client.get(SETTINGS_URL)
    headers = {**writing(client), "Idempotency-Key": "key-1"}
    return await client.request(method, url, headers=headers, **kwargs)


@pytest.mark.parametrize("root", ROOTS)
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        pytest.param("DELETE", f"/memory/{NOT_AN_ID}", None, id="forget"),
        pytest.param("POST", f"/memory/changes/{NOT_AN_ID}/undo", None, id="undo"),
        pytest.param(
            "POST",
            "/memory",
            {"kind": "fact", "body": "Likes tea.", "supersedes_id": NOT_AN_ID},
            id="remember-supersedes",
        ),
    ],
)
async def test_a_memory_id_that_is_not_an_id_is_refused_as_invalid_not_a_server_error(
    root: str, method: str, path: str, payload: dict[str, Any] | None
) -> None:
    async with deployed("none") as web, web.browser() as client:
        response = await _write(client, method, f"{root}{path}", json=payload)

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation"


async def test_a_fuse_source_that_is_not_an_id_is_refused_as_invalid() -> None:
    async with deployed("none") as web, web.browser() as client:
        response = await _write(
            client,
            "POST",
            "/memory/fuse",
            json={"memory_ids": [UNKNOWN_ID, NOT_AN_ID], "body": "Likes tea and coffee."},
        )

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation"


@pytest.mark.parametrize("root", ROOTS)
async def test_a_well_formed_id_no_memory_has_is_still_an_unchanged_receipt(root: str) -> None:
    async with deployed("none") as web, web.browser() as client:
        forgotten = await _write(client, "DELETE", f"{root}/memory/{UNKNOWN_ID}")

        assert forgotten.status_code == 200
        assert forgotten.json()["outcome"] == "unchanged"
