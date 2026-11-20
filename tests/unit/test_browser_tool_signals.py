# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The facts the ``browser`` tool repeats besides the call: how much it cut, and whether
the page it read had gone quiet.

Each one is a signal the agent acts on — wait, scroll, or snapshot again — rather than a
sentence it has to guess from a page that looks empty.
"""

from __future__ import annotations

from typing import Any

import pytest

from dlightrag.engine.answer.agent_browser.contracts import (
    BrowserHolder,
    DownloadRefusal,
    FoundElements,
    PageEvents,
    PageState,
    SessionState,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.browser import DROVE, BrowserToolHost, browser_tool
from tests.support.agent_browser import (
    FakeLease,
    FakeLivePage,
    FakePage,
    FakeProvider,
    browser_settings,
    idle_accounts,
)
from tests.support.dns import public_dns
from tests.support.resources import call

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
URL = "https://shop.example/"


def host(page: FakePage, *, live: FakeLivePage | None = None, **settings: Any) -> BrowserToolHost:
    provider = FakeProvider(FakeLease(page=page), live=live)
    return BrowserToolHost(
        RunAgentBrowser(provider, HOLDER, browser_settings(**settings)),
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


@pytest.fixture(autouse=True)
def _public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """``shop.example`` is a public host, as the URL check the tool makes requires."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


async def test_a_page_that_had_not_settled_says_so_and_what_to_do() -> None:
    page = FakePage(
        session=SessionState(url=URL), events=PageEvents(settled=False), snapshot="- text"
    )

    result = await call(_tool(host(page, settle=5.0)), action="navigate", url=URL)

    assert result.is_error is False
    assert "The page had not gone quiet within 5 seconds of this call" in result.text_content
    assert "still loading, so something may be missing" in result.text_content
    assert "Wait, scroll, or snapshot again" in result.text_content
    # The snapshot is still what the call answers with: the signal frames it, never replaces it.
    assert result.text_content.rstrip().endswith("- text")


async def test_the_report_holds_the_refusal_bound_and_says_how_many_it_left_out() -> None:
    page = FakePage(
        session=SessionState(url=URL),
        events=PageEvents(
            refused_downloads=(
                DownloadRefusal("invoice.pdf", "too_large"),
                DownloadRefusal("report.csv", "limit"),
            ),
            omitted_refusals=17,
        ),
        snapshot="- text",
    )

    result = await call(_tool(host(page)), action="navigate", url=URL)

    assert "The download invoice.pdf was not admitted: it exceeds 1048576 bytes." in (
        result.text_content
    )
    assert "The download report.csv was not admitted: one call admits at most 4 downloads." in (
        result.text_content
    )
    # What the bound left out is counted, never a list that reads as all of them.
    assert "(17 more refused download(s) omitted)" in result.text_content


async def test_the_first_call_after_a_person_drove_the_page_says_so_and_asks_it_once() -> None:
    live = FakeLivePage()
    page = FakePage(session=SessionState(url=URL), snapshot="- text")
    tool = _tool(host(page, live=live))

    before = await call(tool, action="navigate", url=URL)
    live.control.take()
    live.control.release()
    after = await call(tool, action="navigate", url=URL)
    again = await call(tool, action="navigate", url=URL)

    # Nobody had driven it, then somebody had: the first call that learns it says so, and
    # the page is the same page — the seat is the agent's again.
    assert DROVE not in before.text_content
    assert DROVE in after.text_content
    assert "snapshot before acting" in after.text_content
    assert DROVE not in again.text_content


async def test_a_call_a_person_refused_does_not_spend_the_notice() -> None:
    live = FakeLivePage()
    page = FakePage(session=SessionState(url=URL), snapshot="- text")
    tool = _tool(host(page, live=live))
    await call(tool, action="navigate", url=URL)
    live.control.take()

    refused = await call(tool, action="navigate", url=URL)
    live.control.release()
    after = await call(tool, action="navigate", url=URL)

    # The call was refused while the person held the seat; the notice is still there for the
    # first call that is actually made on the page afterwards.
    assert refused.is_error is True
    assert DROVE in after.text_content


async def test_a_call_that_failed_does_not_spend_the_notice_either() -> None:
    live = FakeLivePage()
    page = FakePage(session=SessionState(url=URL), snapshot="- text")
    tool = _tool(host(page, live=live))
    await call(tool, action="navigate", url=URL)
    live.control.take()
    live.control.release()

    refused = await call(tool, action="wait", seconds=31)
    after = await call(tool, action="navigate", url=URL)

    # A call that answered with an error read nothing of the page, so it spends nothing: the
    # notice is still there for the first call that succeeds after the seat came back.
    assert refused.is_error is True
    assert DROVE not in refused.text_content
    assert DROVE in after.text_content


class _FindingPage(FakePage):
    """A page that answers a find with the limit and the total it was asked for."""

    def __init__(self, *, settled: bool = True) -> None:
        super().__init__(session=SessionState(url=URL))
        self.limits: list[int] = []
        self.settled = settled

    async def find(self, query: str, *, limit: int) -> FoundElements:
        self.limits.append(limit)
        return FoundElements(
            page=PageState(URL, "Shop"),
            events=PageEvents(settled=self.settled),
            lines=tuple(f'line "{query}" {number}' for number in range(min(limit, 3))),
            total=120,
        )


async def test_find_honours_the_limit_the_call_asks_for() -> None:
    page = _FindingPage()
    tool = _tool(host(page))
    await call(tool, action="navigate", url=URL)

    asked = await call(tool, action="find", query="lamp", limit=100)
    await call(tool, action="find", query="lamp")

    assert page.limits == [100, 30]
    assert "120 element(s) match" in asked.text_content


async def test_a_read_that_had_not_settled_says_so_with_what_to_do() -> None:
    page = _FindingPage(settled=False)
    tool = _tool(host(page, settle=5.0))
    await call(tool, action="navigate", url=URL)

    found = await call(tool, action="find", query="lamp")

    # A find is itself a read of the page, so it carries the same signal a snapshot does.
    assert "The page had not gone quiet within 5 seconds of this call" in found.text_content
    assert "snapshot again" in found.text_content


async def test_a_wait_past_one_call_s_stretch_is_refused_in_the_surface_s_words() -> None:
    page = FakePage(session=SessionState(url=URL))
    result = await call(_tool(host(page)), action="wait", seconds=31)

    # It is a sentence the model can act on, not a validation dump, and nothing was waited for.
    assert result.is_error is True
    assert "A single wait is at most 30 seconds" in result.text_content
    assert "Call wait again" in result.text_content
    assert page.visited == [] and page.waits == []


async def test_a_find_past_one_call_s_lines_is_refused_in_the_surface_s_words() -> None:
    page = _FindingPage()
    tool = _tool(host(page))

    refused = await call(tool, action="find", query="lamp", limit=101)

    assert refused.is_error is True
    assert "A single find answers with at most 100 matching lines" in refused.text_content
    # Nothing was searched: the refusal comes before the page is touched at all.
    assert page.limits == []


async def test_a_page_that_settled_or_was_not_waited_for_says_nothing() -> None:
    settled = FakePage(
        session=SessionState(url=URL), events=PageEvents(settled=True), snapshot="- text"
    )
    unmeasured = FakePage(
        session=SessionState(url=URL), events=PageEvents(settled=None), snapshot="- text"
    )

    measured = await call(_tool(host(settled, settle=5.0)), action="navigate", url=URL)
    not_waited = await call(_tool(host(unmeasured, settle=0.0)), action="navigate", url=URL)

    assert "had not gone quiet" not in measured.text_content
    assert "had not gone quiet" not in not_waited.text_content
