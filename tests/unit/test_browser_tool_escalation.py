# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The ``browser`` tool's answer to a page that refused it, and to the agent asking for a
browser with a display.

A call a bot filter refused is reported as the refusal and stops there: the Agent decides,
and a ``navigate`` with ``headed=true`` is the one move it has. A call that makes that move
says so in its answer, because the page it acts on afterwards is a different browser.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    AgentPage,
    BrowserHolder,
    PageLimits,
    SessionState,
    browser_failure,
)
from dlightrag.engine.answer.agent_browser.passwords import (
    FilledPasswords,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.browser import (
    HEADED,
    BrowserToolHost,
    browser_tool,
)
from tests.support.agent_browser import (
    FakeLease,
    FakePage,
    FakeProvider,
    browser_settings,
    idle_accounts,
)
from tests.support.dns import public_dns
from tests.support.resources import call

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
URL = "https://blocked.example/"
BLOCKED = browser_failure("blocked", marker="whaleguard")


class _BlockingPage(FakePage):
    """A page whose first navigation answers with the refusal a bot filter gives."""

    def __init__(self, *, refusal: AgentBrowserError | None) -> None:
        super().__init__(session=SessionState(url=URL))
        self._refusal = refusal
        self.refusals = 0

    async def navigate(self, url: str):
        # The page records where it is even when the answer is a refusal: the browser did
        # navigate, and it is that URL the refusal was read on.
        if self._refusal is not None and self.refusals == 0:
            self.refusals += 1
            self.visited.append(url)
            raise self._refusal
        return await super().navigate(url)


class _BlockingLease(FakeLease):
    """A lease whose first page may be refused once; every page after it reads."""

    def __init__(self, *, refusal: AgentBrowserError | None) -> None:
        super().__init__(session=SessionState(url=URL))
        self._refusal = refusal

    async def open_page(self, limits: PageLimits, passwords: FilledPasswords) -> AgentPage:
        page = _BlockingPage(refusal=self._refusal)
        self._refusal = None
        self.pages.append(page)
        self.limits.append(limits)
        self.passwords.append(passwords)
        # The doubles are what the tool drives; the protocol they stand in for is not
        # implemented member by member here.
        return cast(AgentPage, page)


def host(provider: FakeProvider) -> BrowserToolHost:
    """The tool's host as the executor composes it: a browser with a headed member."""
    return BrowserToolHost(
        RunAgentBrowser(provider, HOLDER, browser_settings(headed=True)),
        ResourceRegistry(),
        _reads_nothing,
        idle_accounts(),
    )


async def _reads_nothing(request: Any, runtime: Any) -> Any:
    raise AssertionError("the browser tool reads no Resource here")


def _tool(host: BrowserToolHost) -> Any:
    return browser_tool(
        host,
        environment=None,
        scheduler=None,  # pyright: ignore[reportArgumentType]
        spill=None,
        image_preparer=None,
        child=False,
    )


async def _navigate(host: BrowserToolHost, **args: Any) -> Any:
    return await call(_tool(host), action="navigate", url=URL, **args)


@pytest.fixture(autouse=True)
def _public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """``blocked.example`` is a public host, as the URL check the tool makes requires."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


async def test_a_refused_call_is_reported_as_the_refusal_and_nothing_moves() -> None:
    headless = _BlockingLease(refusal=BLOCKED)
    provider = FakeProvider(headless, FakeLease())

    result = await _navigate(host(provider))

    # The refusal is the whole answer: the platform does not lease anything for the agent.
    assert result.is_error is True
    assert BLOCKED.public_message in result.text_content
    assert provider.headed == [False]


async def test_the_agent_moves_to_a_display_with_headed_true_and_the_answer_says_so() -> None:
    headless = _BlockingLease(refusal=None)
    provider = FakeProvider(headless, FakeLease())
    tool_host = host(provider)

    first = await _navigate(tool_host)
    result = await _navigate(tool_host, headed=True)

    # The Run held a headless browser, so the call moved it and the note says which browser
    # the next call drives.
    assert (first.is_error, result.is_error) == (False, False)
    assert provider.headed == [False, True]
    assert HEADED in result.text_content
    assert "cookies and web storage moved with it" in result.text_content


async def test_a_run_whose_first_browser_is_headed_reads_no_move_note() -> None:
    provider = FakeProvider(FakeLease())

    result = await _navigate(host(provider), headed=True)

    # Nothing moved: the Run leased a browser with a display as its first browser.
    assert result.is_error is False
    assert provider.headed == [True]
    assert HEADED not in result.text_content


async def test_a_headed_call_on_an_already_headed_lease_is_not_refused() -> None:
    provider = FakeProvider(FakeLease(), FakeLease())
    tool_host = host(provider)

    first = await _navigate(tool_host, headed=True)
    second = await _navigate(tool_host, headed=True)

    # The second call is a lease that is already headed, so it is made where the Run is and
    # no note claims a move that did not happen.
    assert (first.is_error, second.is_error) == (False, False)
    assert provider.headed == [True]
    assert HEADED not in second.text_content


async def test_a_deployment_with_no_headed_member_refuses_headed_true() -> None:
    host_without_headed = BrowserToolHost(
        RunAgentBrowser(FakeProvider(FakeLease()), HOLDER, browser_settings()),
        ResourceRegistry(),
        _reads_nothing,
        idle_accounts(),
    )

    result = await _navigate(host_without_headed, headed=True)

    assert result.is_error is True
    assert browser_failure("not_configured").public_message in result.text_content
