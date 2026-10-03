# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A leased browser that stops answering is given up, not reused by its Run."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from dlightrag.adapters.agent_browser import playwright_session
from dlightrag.adapters.agent_browser.playwright_session import PlaywrightLeasedBrowser
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    BrowserHolder,
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
    monkeypatch.setattr(playwright_session, "_CLOSE_SECONDS", 0.05)
    released: list[str] = []

    async def release() -> None:
        released.append("wedged")

    wedged = _WedgedBrowser()
    healthy = FakeLease()
    provider = FakeProvider(
        PlaywrightLeasedBrowser(cast(Any, wedged), release, sandbox="chromium"), healthy
    )
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
