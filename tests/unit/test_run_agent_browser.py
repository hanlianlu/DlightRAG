# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""How one Research Run leases, shares, and gives back its Agent Browser."""

from __future__ import annotations

import asyncio

import pytest

from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    AgentBrowserSettings,
    BrowserHolder,
    RunAgentBrowser,
    browser_failure,
)
from tests.support.agent_browser import FakeLease, FakeProvider

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 3)


def settings(*, idle: float = 600.0) -> AgentBrowserSettings:
    return AgentBrowserSettings(
        lease_wait_seconds=7.0,
        navigation_timeout_seconds=30.0,
        settle_timeout_seconds=5.0,
        idle_release_seconds=idle,
        max_page_bytes=1000,
    )


async def test_a_run_leases_nothing_until_it_renders_and_then_shares_one_browser() -> None:
    lease = FakeLease()
    provider = FakeProvider(lease)
    browser = RunAgentBrowser(provider, HOLDER, settings())
    assert provider.leased == 0

    pages = await asyncio.gather(*(browser.render(f"http://p{n}.example/") for n in range(3)))

    assert [page.final_url for page in pages] == [f"http://p{n}.example/" for n in range(3)]
    # One lease for the Run, claimed under its own claim and with the configured wait.
    assert (provider.holders, provider.waits) == ([HOLDER], [7.0])
    assert sorted(lease.rendered) == [f"http://p{n}.example/" for n in range(3)]
    await browser.aclose()


async def test_no_more_than_four_pages_render_at_once() -> None:
    gate = asyncio.Event()
    lease = FakeLease(gate=gate)
    browser = RunAgentBrowser(FakeProvider(lease), HOLDER, settings())

    renders = [asyncio.create_task(browser.render(f"http://p{n}.example/")) for n in range(6)]
    await asyncio.sleep(0.05)
    assert (lease.active, len(lease.rendered)) == (4, 4)

    gate.set()
    await asyncio.gather(*renders)
    assert (lease.peak, len(lease.rendered)) == (4, 6)
    await browser.aclose()


async def test_a_browser_that_disconnects_is_closed_and_the_next_render_leases_afresh() -> None:
    dead = FakeLease(failure=browser_failure("disconnected"))
    fresh = FakeLease(sandbox="unavailable")
    provider = FakeProvider(dead, fresh)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    with pytest.raises(AgentBrowserError) as dropped:
        await browser.render("http://one.example/")
    assert dropped.value.reason == "disconnected"
    assert dead.closed == 1

    assert (await browser.render("http://two.example/")).final_url == "http://two.example/"
    assert (provider.leased, fresh.rendered) == (2, ["http://two.example/"])
    # What the Run records is how its browser was first leased.
    assert browser.sandbox == "chromium"
    await browser.aclose()
    assert (dead.closed, fresh.closed) == (1, 1)


async def test_other_render_failures_leave_the_browser_in_place() -> None:
    lease = FakeLease(failure=browser_failure("timeout", seconds=30))
    provider = FakeProvider(lease)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    for _ in range(2):
        with pytest.raises(AgentBrowserError) as failed:
            await browser.render("http://slow.example/")
        assert failed.value.reason == "timeout"

    assert (provider.leased, lease.closed) == (1, 0)
    await browser.aclose()


async def test_a_busy_pool_fails_the_render_and_the_next_render_tries_the_pool_again() -> None:
    lease = FakeLease()
    provider = FakeProvider(browser_failure("busy"), lease)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    with pytest.raises(AgentBrowserError) as busy:
        await browser.render("http://a.example/")
    assert busy.value.reason == "busy"
    assert browser.sandbox is None

    assert (await browser.render("http://a.example/")).status == 200
    assert provider.leased == 2
    await browser.aclose()


async def test_an_idle_browser_is_given_back_and_the_next_render_leases_another() -> None:
    first, second = FakeLease(), FakeLease()
    provider = FakeProvider(first, second)
    browser = RunAgentBrowser(provider, HOLDER, settings(idle=0.05))

    await browser.render("http://a.example/")
    assert first.closed == 0
    await asyncio.sleep(0.3)
    assert first.closed == 1

    await browser.render("http://b.example/")
    assert (provider.leased, second.rendered) == (2, ["http://b.example/"])
    await browser.aclose()
    assert second.closed == 1


async def test_a_browser_with_a_render_in_flight_is_not_idle() -> None:
    gate = asyncio.Event()
    lease = FakeLease(gate=gate)
    browser = RunAgentBrowser(FakeProvider(lease), HOLDER, settings(idle=0.05))

    rendering = asyncio.create_task(browser.render("http://slow.example/"))
    await asyncio.sleep(0.3)
    assert lease.closed == 0

    gate.set()
    await rendering
    await asyncio.sleep(0.3)
    assert lease.closed == 1
    await browser.aclose()


async def test_settlement_gives_the_browser_back_once_and_ends_the_idle_timer() -> None:
    lease = FakeLease()
    provider = FakeProvider(lease)
    browser = RunAgentBrowser(provider, HOLDER, settings(idle=0.05))
    await browser.render("http://a.example/")

    await browser.aclose()
    await browser.aclose()
    await asyncio.sleep(0.2)

    assert lease.closed == 1
    with pytest.raises(AgentBrowserError) as closed:
        await browser.render("http://b.example/")
    assert closed.value.reason == "not_configured"
    assert provider.leased == 1


async def test_a_run_that_never_rendered_settles_without_touching_the_pool() -> None:
    provider = FakeProvider()

    await RunAgentBrowser(provider, HOLDER, settings()).aclose()

    assert provider.leased == 0
