# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Browser's leases and Compose provider, over disposable PostgreSQL and a real run-server.

Runs are claimed through the Run store, so a lease is live exactly as long as its holder's Run
lease is. The pool is real Playwright servers driving Chromium; the web they render is a loopback
proxy serving ``http://*.example`` pages, as the egress proxy would carry the public Web.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import pytest

from dlightrag.adapters.agent_browser import ComposeBrowserProvider
from dlightrag.adapters.postgres.runtime.browser_leases import PGAgentBrowserLeaseStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    AgentBrowserSettings,
    BrowserHolder,
    RunAgentBrowser,
)
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope
from tests.support.agent_browser import (
    Served,
    WebProxy,
    run_server,
    sandbox_refusal,
    web_proxy,
)
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

OWNER = "owner-browser"
SHELL = (
    "<html><body><div id='app'>loading</div>"
    "<script>document.getElementById('app').textContent = 'quote from script'</script>"
    "</body></html>"
)
PAGES = {
    "http://js.example/": Served(SHELL),
    "http://forbidden.example/": Served("<html><body>no</body></html>", status=403),
    "http://files.example/report.csv": Served(
        "a,b\n1,2\n",
        headers={"content-type": "text/csv", "content-disposition": "attachment; filename=r.csv"},
    ),
}
COMPOSE_LOGGER = "dlightrag.adapters.agent_browser.compose"


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


@pytest.fixture
async def pg() -> AsyncIterator[tuple[PGRunStore, Any]]:
    async with isolated_run_runtime("agent_browser") as pair:
        yield pair


async def live_run(store: PGRunStore, key: str, *, worker: str | None = None) -> BrowserHolder:
    """A Run this test has claimed, as the holder its worker would lease a browser for."""
    await store.accept_run(
        envelope=run_envelope("answer", key=key, owner=OWNER), run_id=str(uuid.uuid7())
    )
    claim = await store.claim_next(worker_id=worker or f"worker-{key}")
    assert claim is not None
    execution = claim.execution
    return BrowserHolder(
        execution.owner_id, execution.run_id, execution.worker_id, execution.fencing_epoch
    )


async def holder_of(pool: Any, endpoint: str) -> str | None:
    """The Run the lease table says holds ``endpoint``, or None when it is free."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT run_id::text FROM dlightrag_agent_browser_leases WHERE endpoint = $1", endpoint
        )


async def eventually(condition: Callable[[], Awaitable[bool]], *, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not await condition():
        assert time.monotonic() < deadline, "the condition never held"
        await asyncio.sleep(0.05)


def settings(
    *, wait: float = 5.0, idle: float = 600.0, settle: float = 0.5
) -> AgentBrowserSettings:
    return AgentBrowserSettings(
        lease_wait_seconds=wait,
        navigation_timeout_seconds=15.0,
        settle_timeout_seconds=settle,
        idle_release_seconds=idle,
        max_page_bytes=1_000_000,
    )


@asynccontextmanager
async def pool_of(
    pool: Any, *servers: str, proxy: WebProxy, connect_timeout: float = 20.0
) -> AsyncIterator[tuple[ComposeBrowserProvider, PGAgentBrowserLeaseStore]]:
    leases = PGAgentBrowserLeaseStore(pool=pool)
    provider = ComposeBrowserProvider(
        endpoints=servers,
        egress_proxy=proxy.url,
        connect_timeout_seconds=connect_timeout,
        leases=leases,
    )
    try:
        yield provider, leases
    finally:
        await provider.aclose()


# -- leases ----------------------------------------------------------------------------------


ONLY = ("ws://pool-1/",)


async def test_a_live_run_holds_the_only_endpoint_until_it_releases_it(pg) -> None:
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    first = await live_run(store, "first")
    second = await live_run(store, "second")

    assert await leases.claim(first, ONLY) == "ws://pool-1/"
    assert await leases.claim(second, ONLY) is None
    await leases.release(second, "ws://pool-1/")  # not its lease to give back
    assert await leases.claim(second, ONLY) is None

    await leases.release(first, "ws://pool-1/")
    assert await leases.claim(second, ONLY) == "ws://pool-1/"
    assert await holder_of(pool, "ws://pool-1/") == second.run_id


async def test_a_holders_repeat_claim_returns_its_own_endpoint_and_honours_exclusion(pg) -> None:
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    both = ("ws://pool-1/", "ws://pool-2/")
    await leases.register_endpoints(both)
    holder = await live_run(store, "holder")

    taken = await leases.claim(holder, both)
    assert taken is not None
    assert await leases.claim(holder, both) == taken
    other = next(endpoint for endpoint in both if endpoint != taken)
    # An excluded endpoint is neither returned nor taken, even when the holder holds it.
    assert await leases.claim(holder, both, exclude=[taken]) == other


async def test_a_run_whose_lease_lapsed_frees_its_endpoint(pg) -> None:
    """A worker that died stops renewing its Run's lease, and its browser frees with it."""
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    dead = await live_run(store, "dead")
    live = await live_run(store, "live")
    assert await leases.claim(dead, ONLY) == "ws://pool-1/"
    assert await leases.claim(live, ONLY) is None

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second'"
            " WHERE run_id = $1",
            uuid.UUID(dead.run_id),
        )

    assert await leases.claim(live, ONLY) == "ws://pool-1/"


async def test_a_reclaimed_run_takes_over_the_endpoint_its_dead_attempt_held(pg) -> None:
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    first = await live_run(store, "reclaimed", worker="worker-1")
    assert await leases.claim(first, ONLY) == "ws://pool-1/"

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second'"
            " WHERE run_id = $1",
            uuid.UUID(first.run_id),
        )
    claim = await store.claim_next(worker_id="worker-2")
    assert claim is not None and claim.execution.fencing_epoch == first.fencing_epoch + 1
    second = BrowserHolder(first.owner_id, first.run_id, "worker-2", claim.execution.fencing_epoch)

    # The attempt that died can claim nothing, and the new one takes what the old one held.
    assert await leases.claim(first, ONLY) is None
    assert await leases.claim(second, ONLY) == "ws://pool-1/"
    assert await holder_of(pool, "ws://pool-1/") == second.run_id


async def test_a_stale_fencing_epoch_cannot_claim(pg) -> None:
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    holder = await live_run(store, "stale")
    stale = BrowserHolder(
        holder.owner_id, holder.run_id, holder.worker_id, holder.fencing_epoch - 1
    )

    assert await leases.claim(stale, ONLY) is None
    assert await holder_of(pool, "ws://pool-1/") is None


async def test_a_finished_run_frees_its_endpoint_without_releasing_it(pg) -> None:
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    finished = await live_run(store, "finished")
    waiting = await live_run(store, "waiting")
    assert await leases.claim(finished, ONLY) == "ws://pool-1/"
    assert await leases.claim(waiting, ONLY) is None

    await store.finish_success(
        owner_id=finished.owner_id,
        run_id=finished.run_id,
        worker_id=finished.worker_id,
        fencing_epoch=finished.fencing_epoch,
        result={"answer": "done"},
    )

    assert await leases.claim(waiting, ONLY) == "ws://pool-1/"


async def test_of_two_concurrent_claims_of_the_only_endpoint_exactly_one_wins(pg) -> None:
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    holders = [await live_run(store, f"racer-{index}") for index in range(4)]

    claimed = await asyncio.gather(*(leases.claim(holder, ONLY) for holder in holders))

    assert sorted(endpoint for endpoint in claimed if endpoint is not None) == ["ws://pool-1/"]
    winner = holders[claimed.index("ws://pool-1/")]
    assert await holder_of(pool, "ws://pool-1/") == winner.run_id


class _OneConnection:
    """A pool of the single connection a test set up, so a store can be given a session to hold."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        yield self._connection


async def test_a_claim_that_read_an_endpoint_free_does_not_take_it_from_a_claim_that_won_it(
    pg,
) -> None:
    """Two claims of one free endpoint cannot both succeed, however they interleave.

    The held-up claim reads the endpoint free and is stopped before it locks the row; the
    other claim takes the endpoint and commits meanwhile. A statement that judged freedom
    by a join made before the lock would keep its stale verdict after the lock and take the
    endpoint from the Run that won it. The stop is a row policy for one role, whose function
    waits on an advisory lock this test holds, so the interleaving is the test's to choose.
    """
    store, pool = pg
    leases = PGAgentBrowserLeaseStore(pool=pool)
    await leases.register_endpoints(ONLY)
    held_up = await live_run(store, "held-up")
    winner = await live_run(store, "winner")
    role = f"held_up_{uuid.uuid4().hex[:8]}"
    claiming: asyncio.Task[str | None] | None = None
    async with pool.acquire() as door, pool.acquire() as held_connection:
        try:
            await door.execute(
                f"""
                CREATE ROLE {role} NOLOGIN;
                GRANT SELECT ON dlightrag_runs TO {role};
                GRANT SELECT, UPDATE ON dlightrag_agent_browser_leases TO {role};
                CREATE FUNCTION wait_at_the_door() RETURNS boolean LANGUAGE plpgsql AS $$
                BEGIN
                    PERFORM pg_advisory_lock_shared(7);
                    PERFORM pg_advisory_unlock_shared(7);
                    RETURN true;
                END $$;
                ALTER TABLE dlightrag_agent_browser_leases ENABLE ROW LEVEL SECURITY;
                CREATE POLICY held_up ON dlightrag_agent_browser_leases
                    FOR ALL TO {role} USING (wait_at_the_door()) WITH CHECK (true)
                """
            )
            await held_connection.execute(f"SET ROLE {role}")
            await door.execute("SELECT pg_advisory_lock(7)")

            async def stopped_at_the_door() -> bool:
                return bool(
                    await door.fetchval(
                        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
                        " AND database = (SELECT oid FROM pg_database"
                        " WHERE datname = current_database())"
                    )
                )

            claiming = asyncio.create_task(
                PGAgentBrowserLeaseStore(pool=_OneConnection(held_connection)).claim(held_up, ONLY)
            )
            await eventually(stopped_at_the_door)
            assert await leases.claim(winner, ONLY) == "ws://pool-1/"
            await door.execute("SELECT pg_advisory_unlock(7)")

            assert await asyncio.wait_for(claiming, 10) is None
            assert await holder_of(pool, "ws://pool-1/") == winner.run_id
        finally:
            await door.execute("SELECT pg_advisory_unlock_all()")
            if claiming is not None:
                await asyncio.gather(claiming, return_exceptions=True)
            await held_connection.execute("RESET ROLE")
            await door.execute(f"DROP OWNED BY {role}; DROP ROLE IF EXISTS {role}")


# -- the Compose provider ------------------------------------------------------------------


async def test_a_javascript_only_page_renders_its_script_text_through_the_egress_proxy(pg) -> None:
    store, pool = pg
    holder = await live_run(store, "renders")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        browser = RunAgentBrowser(provider, holder, settings())

        page = await browser.render("http://js.example/")

        assert '<div id="app">quote from script</div>' in page.html.decode()
        assert (page.requested_url, page.final_url, page.status) == (
            "http://js.example/",
            "http://js.example/",
            200,
        )
        # The browser asked the egress proxy for the page, once; nothing reached it directly.
        (request,) = proxy.fetched("http://js.example/")
        assert request.method == "GET"
        # The Run holds its browser while it may render again, and gives it back at settlement.
        assert await holder_of(pool, server.endpoint) == holder.run_id
        await browser.aclose()
        assert await holder_of(pool, server.endpoint) is None


async def test_a_page_that_answers_an_error_status_or_a_download_is_not_a_page(pg) -> None:
    store, pool = pg
    holder = await live_run(store, "fails")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        browser = RunAgentBrowser(provider, holder, settings())

        with pytest.raises(AgentBrowserError) as forbidden:
            await browser.render("http://forbidden.example/")
        with pytest.raises(AgentBrowserError) as download:
            await browser.render("http://files.example/report.csv")
        with pytest.raises(AgentBrowserError) as missing:
            # The proxy answers its own 404 for a page it holds none of.
            await browser.render("http://missing.example/")

        assert forbidden.value.reason == "http_status"
        assert forbidden.value.public_message == "The page answered HTTP 403 to the Agent Browser."
        assert download.value.reason == "download"
        assert missing.value.reason == "http_status"
        # A failed render leaves the browser usable for the next one.
        assert (await browser.render("http://js.example/")).status == 200
        await browser.aclose()


async def test_a_busy_pool_makes_a_render_wait_and_then_report_busy(pg) -> None:
    store, pool = pg
    holding = await live_run(store, "holding")
    waiting = await live_run(store, "waiting")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        held = await provider.lease(holding, wait_seconds=0)

        started = time.monotonic()
        with pytest.raises(AgentBrowserError) as busy:
            await provider.lease(waiting, wait_seconds=0.5)

        assert busy.value.reason == "busy"
        assert time.monotonic() - started >= 0.5
        await held.aclose()


async def test_a_waiting_lease_takes_the_browser_the_moment_its_holder_gives_it_back(pg) -> None:
    store, pool = pg
    holding = await live_run(store, "holding")
    waiting = await live_run(store, "waiting")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        held = await provider.lease(holding, wait_seconds=0)

        taking = asyncio.create_task(provider.lease(waiting, wait_seconds=30))
        await asyncio.sleep(0.8)
        assert not taking.done()
        await held.aclose()
        taken = await asyncio.wait_for(taking, 30)

        assert await holder_of(pool, server.endpoint) == waiting.run_id
        await taken.aclose()


async def test_an_endpoint_nothing_answers_on_is_skipped_and_an_empty_pool_is_unreachable(
    pg, caplog
) -> None:
    store, pool = pg
    holder = await live_run(store, "dead-pool")
    dead = "ws://127.0.0.1:1/"
    async with AsyncExitStack() as stack:
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, dead, proxy=proxy))
        with caplog.at_level(logging.ERROR, logger=COMPOSE_LOGGER):
            with pytest.raises(AgentBrowserError) as unreachable:
                await provider.lease(holder, wait_seconds=5)

        assert unreachable.value.reason == "unreachable"
        # The endpoint it could not reach is not left claimed, and the log names it and
        # the failure's type, never a page.
        assert await holder_of(pool, dead) is None
        assert [
            record.getMessage() for record in caplog.records if record.name == COMPOSE_LOGGER
        ] == [f"Agent Browser connect failed (Error): endpoint={dead}"]

        live = await stack.enter_async_context(run_server())
        mixed, _ = await stack.enter_async_context(pool_of(pool, dead, live.endpoint, proxy=proxy))
        browser = RunAgentBrowser(mixed, holder, settings())
        assert (await browser.render("http://js.example/")).status == 200
        assert await holder_of(pool, live.endpoint) == holder.run_id
        await browser.aclose()


async def test_a_pool_container_that_dies_disconnects_the_render_and_the_next_one_leases_afresh(
    pg,
) -> None:
    store, pool = pg
    holder = await live_run(store, "survivor")
    async with AsyncExitStack() as stack:
        first = await stack.enter_async_context(run_server())
        second = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(
            pool_of(pool, first.endpoint, second.endpoint, proxy=proxy)
        )
        browser = RunAgentBrowser(provider, holder, settings())
        await browser.render("http://js.example/")
        leased = first if await holder_of(pool, first.endpoint) == holder.run_id else second

        await leased.stop()
        with pytest.raises(AgentBrowserError) as dropped:
            await browser.render("http://js.example/")

        assert dropped.value.reason == "disconnected"
        assert await holder_of(pool, leased.endpoint) is None
        assert (await browser.render("http://js.example/")).status == 200
        await browser.aclose()


# -- idle release --------------------------------------------------------------------------


async def test_a_run_gives_its_browser_back_when_idle_and_leases_again_for_its_next_render(
    pg,
) -> None:
    store, pool = pg
    holder = await live_run(store, "idle")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        browser = RunAgentBrowser(provider, holder, settings(idle=0.6))

        await browser.render("http://js.example/")
        assert await holder_of(pool, server.endpoint) == holder.run_id

        async def free() -> bool:
            return await holder_of(pool, server.endpoint) is None

        await eventually(free)
        await browser.render("http://js.example/")
        assert await holder_of(pool, server.endpoint) == holder.run_id
        assert len(proxy.fetched("http://js.example/")) == 2
        await browser.aclose()
        assert await holder_of(pool, server.endpoint) is None


async def test_renders_that_follow_each_other_never_leave_the_browser_idle(pg) -> None:
    store, pool = pg
    holder = await live_run(store, "back-to-back")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        browser = RunAgentBrowser(provider, holder, settings(idle=0.5))

        # Each render starts the idle time over, so five of them span more than it.
        for _ in range(5):
            await browser.render("http://js.example/")
            await asyncio.sleep(0.2)
            assert await holder_of(pool, server.endpoint) == holder.run_id
        await browser.aclose()


async def test_with_no_idle_time_a_run_gives_its_browser_back_after_each_render(pg) -> None:
    store, pool = pg
    holder = await live_run(store, "immediate")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        browser = RunAgentBrowser(provider, holder, settings(idle=0))

        for _ in range(2):
            await browser.render("http://js.example/")

            async def free() -> bool:
                return await holder_of(pool, server.endpoint) is None

            await eventually(free)
        assert len(proxy.fetched("http://js.example/")) == 2
        await browser.aclose()


async def test_closing_a_run_cancels_its_pending_release_and_frees_the_browser_at_once(pg) -> None:
    store, pool = pg
    holder = await live_run(store, "closing")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        browser = RunAgentBrowser(provider, holder, settings(idle=600))
        await browser.render("http://js.example/")

        await browser.aclose()
        await browser.aclose()

        assert await holder_of(pool, server.endpoint) is None
        with pytest.raises(AgentBrowserError) as closed:
            await browser.render("http://js.example/")
        assert closed.value.reason == "not_configured"


# -- Chromium's sandbox ----------------------------------------------------------------------


async def test_an_endpoint_that_cannot_sandbox_runs_unsandboxed_and_is_remembered(
    pg, caplog
) -> None:
    store, pool = pg
    first_run = await live_run(store, "sandbox-1")
    second_run = await live_run(store, "sandbox-2")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        refusal = await stack.enter_async_context(sandbox_refusal(server.endpoint))
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, refusal.endpoint, proxy=proxy))

        with caplog.at_level(logging.WARNING, logger=COMPOSE_LOGGER):
            first = RunAgentBrowser(provider, first_run, settings())
            assert (await first.render("http://js.example/")).status == 200
            # The sandbox is asked for first; the refusal is answered by an unsandboxed connect.
            assert refusal.asked == [True, False]
            assert first.sandbox == "unavailable"
            await first.aclose()

            second = RunAgentBrowser(provider, second_run, settings())
            assert (await second.render("http://js.example/")).status == 200
            # The endpoint is remembered: no sandboxed attempt is made again.
            assert refusal.asked == [True, False, False]
            assert second.sandbox == "unavailable"
            await second.aclose()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert refusal.endpoint in warnings[0].getMessage()
        assert "sandbox" in warnings[0].getMessage()


async def test_a_sandboxed_connect_nothing_answers_is_not_taken_for_a_missing_sandbox(
    pg, caplog
) -> None:
    store, pool = pg
    holder = await live_run(store, "unanswered")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        silent = await stack.enter_async_context(sandbox_refusal(server.endpoint, answers=False))
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(
            pool_of(pool, silent.endpoint, proxy=proxy, connect_timeout=1.0)
        )

        for asked in ([True], [True, True]):
            with caplog.at_level(logging.WARNING, logger=COMPOSE_LOGGER):
                with pytest.raises(AgentBrowserError) as unreachable:
                    await provider.lease(holder, wait_seconds=5)

            # It fails as any connect nobody answers does. The endpoint is not retried without
            # the sandbox, so it is not remembered as unable to run it, and it is not left claimed.
            assert unreachable.value.reason == "unreachable"
            assert silent.asked == asked
            assert await holder_of(pool, silent.endpoint) is None
        failure = f"Agent Browser connect failed (TimeoutError): endpoint={silent.endpoint}"
        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (logging.ERROR, failure)
        ] * 2


async def test_an_endpoint_that_can_sandbox_does_and_asks_nothing_else_of_it(pg, caplog) -> None:
    store, pool = pg
    holder = await live_run(store, "sandboxed")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))

        with caplog.at_level(logging.WARNING, logger=COMPOSE_LOGGER):
            browser = RunAgentBrowser(provider, holder, settings())
            await browser.render("http://js.example/")

        assert browser.sandbox == "chromium"
        assert caplog.records == []
        await browser.aclose()


async def test_a_playwright_driver_that_died_is_replaced_by_the_next_lease(pg) -> None:
    store, pool = pg
    before = await live_run(store, "driver-before")
    after = await live_run(store, "driver-after")
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES))
        provider, _ = await stack.enter_async_context(pool_of(pool, server.endpoint, proxy=proxy))
        first = RunAgentBrowser(provider, before, settings())
        await first.render("http://js.example/")
        await first.aclose()

        # The driver is the node process this process started; the run-server's is not its child.
        drivers = subprocess.run(
            ["pgrep", "-P", str(os.getpid()), "-f", "playwright/driver/node"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert len(drivers) == 1
        os.kill(int(drivers[0]), signal.SIGKILL)
        await asyncio.sleep(0.5)

        second = RunAgentBrowser(provider, after, settings())
        assert (await second.render("http://js.example/")).status == 200
        await second.aclose()
