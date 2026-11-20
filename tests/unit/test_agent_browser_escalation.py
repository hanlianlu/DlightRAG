# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The headed browser a Run leases: what it carries, and why nothing counts it.

Headed is a property of the lease, not a one-shot action: the Run gives back its headless
browser, leases a headed one from the same pool, and the page it was working on moves to it
with its cookies, its web storage, and its URL. Once the Run holds a headed browser, asking
again is asking for the lease it has. What the move cannot carry is the DOM, and the model
reads that rather than a page that looks the way it did.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    AgentBrowserSettings,
    BrowserHolder,
    PageObservation,
    SessionState,
    StorageState,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from dlightrag.engine.answer.tools.browser import BrowserToolHost
from tests.support.agent_browser import FakeLease, FakeProvider, browser_settings

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 3)
URL = "https://shop.example/cart"
CART = "https://shop.example/cart"
SESSION = SessionState(
    url=URL,
    cookies=({"name": "sid", "value": "abc", "domain": "shop.example", "path": "/"},),
    origins={"https://shop.example": StorageState(local={"token": "t1"}, session={"step": "2"})},
)


def settings(**kwargs: object) -> AgentBrowserSettings:
    return browser_settings(headed=True, **kwargs)  # pyright: ignore[reportArgumentType]


async def test_a_run_asks_for_a_headed_browser_and_the_session_moves_with_it() -> None:
    headless, headed = FakeLease(session=SESSION), FakeLease(session=SESSION)
    provider = FakeProvider(headless, headed)
    browser = RunAgentBrowser(provider, HOLDER, settings())
    await browser.with_page("session-1", _navigate, open_page=True)

    await browser.with_page("session-1", _navigate, headed=True)

    # One browser at a time: the headless lease went back before the headed one was taken.
    assert (provider.headed, len(provider.holders)) == ([False, True], 2)
    assert (headless.closed, headless.pages[0].closed) == (1, 1)
    # The session is read from the browser being left and given to the one replacing it.
    assert headed.restored == [SESSION]
    assert headless.pages[0].restored == []
    await browser.aclose()


async def test_the_page_the_run_was_on_is_reopened_on_the_headed_browser() -> None:
    headless = FakeLease(session=SESSION)
    headed = FakeLease(session=SESSION)
    browser = RunAgentBrowser(FakeProvider(headless, headed), HOLDER, settings())
    await browser.with_page("session-1", _navigate, open_page=True)

    observation = await browser.with_page("session-1", _observe, headed=True)

    # The call that asked for the headed browser acts on the page that browser opened at
    # the URL the headless one was on.
    assert observation.page is not None and observation.page.url == URL
    assert headed.pages[-1].visited == [URL]
    await browser.aclose()


async def test_a_headed_lease_stays_headed_however_often_the_agent_asks() -> None:
    headless, headed, spare = FakeLease(session=SESSION), FakeLease(session=SESSION), FakeLease()
    provider = FakeProvider(headless, headed, spare)
    browser = RunAgentBrowser(provider, HOLDER, settings())
    await browser.with_page("session-1", _navigate, open_page=True)

    await browser.with_page("session-1", _navigate, headed=True)
    assert browser.headed is True

    # The Run holds a headed browser, so asking again is the lease it has: no move, no
    # refusal, and no second lease for a counter to have spent.
    assert await browser.with_page("session-1", _navigate, headed=True) == URL
    assert await browser.with_page("session-1", _observe, headed=True) is not None

    assert (browser.headed, provider.headed) == (True, [False, True])
    assert provider.leased == 2
    await browser.aclose()


async def test_a_run_whose_first_lease_is_headed_never_leases_a_second_one() -> None:
    headed, spare = FakeLease(session=SESSION), FakeLease()
    provider = FakeProvider(headed, spare)
    browser = RunAgentBrowser(provider, HOLDER, settings())

    await browser.with_page("session-1", _navigate, open_page=True, headed=True)
    await browser.with_page("session-1", _navigate, headed=True)

    assert (provider.headed, provider.leased) == ([True], 1)
    assert browser.headed is True
    await browser.aclose()


async def test_a_deployment_that_leases_no_headed_browser_refuses_a_headed_lease() -> None:
    provider = FakeProvider(FakeLease(), may_go_headed=False)
    browser = RunAgentBrowser(provider, HOLDER, browser_settings())

    await browser.with_page("session-1", _navigate, open_page=True)
    assert browser.headed is False

    with pytest.raises(AgentBrowserError) as refused:
        await browser.with_page("session-1", _navigate, headed=True)
    assert refused.value.reason == "not_configured"
    with pytest.raises(AgentBrowserError) as refused:
        await provider.lease(HOLDER, wait_seconds=0, headed=True)
    assert refused.value.reason == "not_configured"
    # Only the headless browser the Session already had was ever leased.
    assert (provider.headed, browser.headed) == ([False], False)
    await browser.aclose()


async def test_a_session_a_browser_will_not_give_up_does_not_hold_the_move_open() -> None:
    headless = FakeLease(session=SESSION, capture_fails=True)
    headed = FakeLease()
    browser = RunAgentBrowser(FakeProvider(headless, headed), HOLDER, settings())
    await browser.with_page("session-1", _navigate, open_page=True)

    await browser.with_page("session-1", _navigate, headed=True)

    # The browser is still left, and the headed one is leased on the session as far as it
    # could be read: a move that cannot read the session is still a move.
    assert headed.restored == [SessionState(url="")]
    assert (browser.headed, headless.closed) == (True, 1)
    await browser.aclose()


def test_the_browser_tool_takes_no_switch_over_a_refusal() -> None:
    assert "escalate" not in {field.name for field in fields(BrowserToolHost)}


async def _navigate(page: object) -> str:
    observation: PageObservation = await page.navigate(CART)  # pyright: ignore[reportAttributeAccessIssue]
    assert observation.page is not None
    return observation.page.url


async def _observe(page: object) -> PageObservation:
    return await page.navigate(CART)  # pyright: ignore[reportAttributeAccessIssue]
