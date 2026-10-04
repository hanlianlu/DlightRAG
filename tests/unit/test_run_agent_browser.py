# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""How one Research Run leases, shares, and gives back its Agent Browser."""

from __future__ import annotations

import asyncio

import pytest

from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    AgentBrowserSettings,
    AgentPage,
    BrowserHolder,
    PageLimits,
    RunAgentBrowser,
    browser_failure,
    page_failure,
)
from tests.support.agent_browser import FakeLease, FakeProvider, browser_settings

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 3)


def settings(*, idle: float = 600.0) -> AgentBrowserSettings:
    return browser_settings(wait=7.0, idle=idle)


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
    fresh = FakeLease()
    provider = FakeProvider(dead, fresh)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    with pytest.raises(AgentBrowserError) as dropped:
        await browser.render("http://one.example/")
    assert dropped.value.reason == "disconnected"
    assert dead.closed == 1

    assert (await browser.render("http://two.example/")).final_url == "http://two.example/"
    assert (provider.leased, fresh.rendered) == (2, ["http://two.example/"])
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


# -- Agent Sessions' pages ------------------------------------------------------------------


async def open_page(browser: RunAgentBrowser, scope: str, url: str = "http://a.example/") -> None:
    await browser.with_page(scope, lambda page: page.navigate(url), open_page=True)


async def test_an_agent_session_with_no_page_leases_nothing_and_says_why() -> None:
    provider = FakeProvider()
    browser = RunAgentBrowser(provider, HOLDER, settings())

    with pytest.raises(AgentBrowserError) as unopened:
        await browser.with_page("child", lambda page: page.navigate("http://a.example/"))

    assert unopened.value.reason == "no_page"
    assert 'browser(action="navigate"' in unopened.value.public_message
    assert (provider.leased, browser.current_url("child")) == (0, None)
    await browser.aclose()


async def test_agent_pages_open_on_the_runs_one_browser_under_its_limits() -> None:
    lease = FakeLease()
    provider = FakeProvider(lease)
    browser = RunAgentBrowser(
        provider, HOLDER, browser_settings(navigation=7, settle=2, action=3, depth=9)
    )

    await open_page(browser, "parent", "http://parent.example/")
    await open_page(browser, "child", "http://child.example/")

    assert (provider.leased, len(lease.pages)) == (1, 2)
    assert lease.limits == [PageLimits(7, 3, 2, 9, 1024 * 1024)] * 2
    assert (browser.current_url("parent"), browser.current_url("child")) == (
        "http://parent.example/",
        "http://child.example/",
    )
    await browser.aclose()
    assert [page.closed for page in lease.pages] == [1, 1]
    assert lease.closed == 1


async def test_an_agent_sessions_later_calls_reach_the_page_it_opened() -> None:
    lease = FakeLease()
    browser = RunAgentBrowser(FakeProvider(lease), HOLDER, settings())
    await open_page(browser, "parent", "http://one.example/")

    await browser.with_page("parent", lambda page: page.navigate("http://two.example/"))

    assert lease.pages[0].visited == ["http://one.example/", "http://two.example/"]
    assert len(lease.pages) == 1
    await browser.aclose()


async def test_a_run_keeps_its_browser_while_any_agent_page_is_open() -> None:
    first, second = FakeLease(), FakeLease()
    provider = FakeProvider(first, second)
    browser = RunAgentBrowser(provider, HOLDER, settings(idle=0.05))
    await open_page(browser, "parent")
    await open_page(browser, "child")

    await browser.close_page("child")
    await asyncio.sleep(0.3)
    assert (first.closed, first.pages[1].closed) == (0, 1)

    await browser.close_page("parent")
    await asyncio.sleep(0.3)
    assert first.closed == 1

    await open_page(browser, "parent")
    assert (provider.leased, len(second.pages)) == (2, 1)
    await browser.aclose()


async def test_a_render_in_a_browser_with_a_page_does_not_start_the_idle_clock() -> None:
    lease = FakeLease()
    browser = RunAgentBrowser(FakeProvider(lease), HOLDER, settings(idle=0.05))
    await open_page(browser, "parent")

    await browser.render("http://r.example/")
    await asyncio.sleep(0.3)

    assert lease.closed == 0
    await browser.aclose()


async def test_closing_an_agent_page_leaves_the_others_and_the_lease() -> None:
    lease = FakeLease()
    browser = RunAgentBrowser(FakeProvider(lease), HOLDER, settings())
    await open_page(browser, "parent")
    await open_page(browser, "child")

    await browser.close_page("child")
    await browser.close_page("child")
    await browser.close_page("never-opened")

    assert [page.closed for page in lease.pages] == [0, 1]
    assert (browser.current_url("parent"), browser.current_url("child")) == (
        "http://a.example/",
        None,
    )
    with pytest.raises(AgentBrowserError) as gone:
        await browser.with_page("child", lambda page: page.navigate("http://a.example/"))
    assert gone.value.reason == "no_page"
    assert lease.closed == 0
    await browser.aclose()


async def test_a_disconnect_gives_the_browser_back_and_loses_every_page_that_lived_on_it() -> None:
    dead = FakeLease(call_failure=page_failure("disconnected"))
    fresh = FakeLease()
    provider = FakeProvider(dead, fresh)
    browser = RunAgentBrowser(provider, HOLDER, settings())
    with pytest.raises(AgentBrowserError) as dropped:
        await open_page(browser, "parent")
    assert dropped.value.reason == "disconnected"
    assert dead.closed == 1

    # The scope that saw it die starts over on a browser leased afresh.
    await open_page(browser, "parent")
    assert (provider.leased, len(fresh.pages)) == (2, 1)
    await browser.aclose()


async def test_the_other_sessions_of_a_browser_that_disconnected_are_told_their_page_was_lost() -> (
    None
):
    lease = FakeLease()
    provider = FakeProvider(lease, FakeLease())
    browser = RunAgentBrowser(provider, HOLDER, settings())
    await open_page(browser, "parent")
    await open_page(browser, "child")
    lease.pages[0].failure = page_failure("disconnected")

    with pytest.raises(AgentBrowserError) as dropped:
        await browser.with_page("parent", lambda page: page.navigate("http://b.example/"))
    assert dropped.value.reason == "disconnected"
    assert lease.closed == 1

    for scope in ("child", "parent"):
        with pytest.raises(AgentBrowserError) as lost:
            await browser.with_page(scope, lambda page: page.navigate("http://b.example/"))
        assert lost.value.reason == "page_lost"
    assert provider.leased == 1

    # Opening a page again is how an Agent Session recovers, and it clears only its own loss.
    await open_page(browser, "child")
    assert provider.leased == 2
    with pytest.raises(AgentBrowserError) as still_lost:
        await browser.with_page("parent", lambda page: page.navigate("http://b.example/"))
    assert still_lost.value.reason == "page_lost"
    await browser.aclose()


async def test_a_late_disconnect_report_from_a_browser_the_run_replaced_changes_nothing() -> None:
    old, fresh = FakeLease(), FakeLease()
    provider = FakeProvider(old, fresh)
    browser = RunAgentBrowser(provider, HOLDER, settings())
    await open_page(browser, "slow")
    await open_page(browser, "quick")
    release = asyncio.Event()

    async def disconnects_late(page: AgentPage) -> None:
        await release.wait()
        raise page_failure("disconnected")

    late = asyncio.create_task(browser.with_page("slow", disconnects_late))
    await asyncio.sleep(0)
    old.pages[1].failure = page_failure("disconnected")
    with pytest.raises(AgentBrowserError):
        await browser.with_page("quick", lambda page: page.navigate("http://b.example/"))
    await open_page(browser, "after")
    assert (old.closed, provider.leased) == (1, 2)

    release.set()
    with pytest.raises(AgentBrowserError):
        await late

    assert fresh.closed == 0
    assert browser.current_url("after") == "http://a.example/"
    await browser.aclose()


async def test_a_busy_pool_means_no_page_was_opened_and_the_next_try_asks_the_pool_again() -> None:
    lease = FakeLease()
    provider = FakeProvider(browser_failure("busy"), lease)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    with pytest.raises(AgentBrowserError) as busy:
        await open_page(browser, "parent")
    assert busy.value.reason == "busy"
    assert busy.value.public_message == (
        "Every Agent Browser is in use by other Runs, so no page was opened. Try again later."
    )

    await open_page(browser, "parent")
    assert (provider.leased, len(lease.pages)) == (2, 1)
    await browser.aclose()


async def test_a_page_that_fails_to_open_leaves_a_browser_with_no_page_to_rest() -> None:
    lease = FakeLease(
        opening_failure=page_failure("action_failed", action="open", target="a page", detail="x")
    )
    browser = RunAgentBrowser(FakeProvider(lease), HOLDER, settings(idle=0.05))

    with pytest.raises(AgentBrowserError) as failed:
        await open_page(browser, "parent")
    assert failed.value.reason == "action_failed"
    assert lease.closed == 0

    await asyncio.sleep(0.3)
    assert lease.closed == 1
    await browser.aclose()


async def test_a_browser_found_disconnected_while_opening_a_page_is_given_back() -> None:
    dead = FakeLease(opening_failure=page_failure("disconnected"))
    fresh = FakeLease()
    provider = FakeProvider(dead, fresh)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    with pytest.raises(AgentBrowserError) as dropped:
        await open_page(browser, "parent")
    assert (dropped.value.reason, dead.closed) == ("disconnected", 1)

    await open_page(browser, "parent")
    assert (provider.leased, len(fresh.pages)) == (2, 1)
    await browser.aclose()


async def test_a_run_that_settled_opens_no_page() -> None:
    lease = FakeLease()
    provider = FakeProvider(lease)
    browser = RunAgentBrowser(provider, HOLDER, settings())
    await open_page(browser, "parent")

    await browser.aclose()
    await browser.aclose()

    assert (lease.closed, lease.pages[0].closed) == (1, 1)
    with pytest.raises(AgentBrowserError) as closed:
        await open_page(browser, "child")
    assert closed.value.reason == "not_configured"
    assert provider.leased == 1
