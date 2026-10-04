# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A leased browser that stops answering is given up, not reused by its Run."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from dlightrag.adapters.agent_browser import leased_browser, page
from dlightrag.adapters.agent_browser.leased_browser import PlaywrightLeasedBrowser
from dlightrag.adapters.agent_browser.page import PlaywrightAgentPage
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    BrowserHolder,
    FilledPasswords,
    PageLimits,
    RunAgentBrowser,
)
from tests.support.agent_browser import FakeLease, FakeProvider, browser_settings

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
PAGE = "http://slow.example/"


class _Page:
    """A page that loads at once and serializes to a paragraph."""

    url = PAGE
    main_frame = object()

    def on(self, event: str, handler: object) -> None:
        pass

    async def goto(self, url: str, **_options: object) -> None:
        pass

    async def wait_for_load_state(self, state: str, **_options: object) -> None:
        pass

    async def content(self) -> str:
        return "<p>rendered</p>"


class _Context:
    """A context whose close never returns, as one in a wedged browser does."""

    async def new_page(self) -> _Page:
        return _Page()

    async def close(self) -> None:
        await asyncio.Event().wait()


class _WedgedBrowser:
    def __init__(self) -> None:
        self.closed = 0

    async def new_context(self, **_options: object) -> _Context:
        return _Context()

    def is_connected(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed += 1


async def test_a_browser_whose_context_will_not_close_is_given_up_though_its_render_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leased_browser, "CLOSE_SECONDS", 0.05)
    released: list[str] = []

    async def release() -> None:
        released.append("wedged")

    wedged = _WedgedBrowser()
    healthy = FakeLease()
    provider = FakeProvider(PlaywrightLeasedBrowser(cast(Any, wedged), release), healthy)
    browser = RunAgentBrowser(provider, HOLDER, browser_settings())

    with pytest.raises(AgentBrowserError) as lost:
        # A close with no limit would hold this render for good, and the test with it.
        async with asyncio.timeout(5):
            await browser.render(PAGE)

    # The page itself rendered; what is lost is a browser that cannot be trusted again.
    assert lost.value.reason == "disconnected"
    assert (wedged.closed, released) == (1, ["wedged"])

    assert (await browser.render(PAGE)).final_url == "http://slow.example/"
    assert (provider.leased, healthy.rendered) == (2, [PAGE])
    await browser.aclose()


class _DeafBrowser(_WedgedBrowser):
    """A browser that takes a request for a context and never answers it."""

    async def new_context(self, **_options: object) -> _Context:
        await asyncio.Event().wait()
        raise AssertionError("a deaf browser answers nothing")


async def test_a_browser_that_will_not_open_a_page_is_given_up_and_leased_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(leased_browser, "CLOSE_SECONDS", 0.05)
    released: list[str] = []

    async def release() -> None:
        released.append("deaf")

    deaf = _DeafBrowser()
    healthy = FakeLease()
    provider = FakeProvider(PlaywrightLeasedBrowser(cast(Any, deaf), release), healthy)
    browser = RunAgentBrowser(provider, HOLDER, browser_settings())

    with pytest.raises(AgentBrowserError) as lost:
        # An open with no limit would hold the Run's browser lock, and every page and render
        # waiting for it, for good.
        async with asyncio.timeout(5):
            await browser.with_page("parent", lambda page: page.navigate(PAGE), open_page=True)

    assert lost.value.reason == "disconnected"
    assert (deaf.closed, released) == (1, ["deaf"])

    await browser.with_page("parent", lambda page: page.navigate(PAGE), open_page=True)
    assert (provider.leased, len(healthy.pages)) == (2, 1)
    await browser.aclose()


async def test_an_agent_page_that_will_not_close_does_not_hold_the_run_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(page, "CLOSE_SECONDS", 0.05)
    context = _Context()
    agent_page = PlaywrightAgentPage(
        cast(Any, _WedgedBrowser()),
        cast(Any, context),
        cast(Any, await context.new_page()),
        PageLimits(5, 5, 0, 12, 1024),
        FilledPasswords(),
    )

    # A close with no limit would hold the Run's settlement open for good.
    async with asyncio.timeout(5):
        await agent_page.aclose()
