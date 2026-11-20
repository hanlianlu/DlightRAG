# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the ``browser`` tool says about a page's dialogs, and who answers them.

Our code answers none of them by itself: a dialog the call did not answer leaves the page
stopped and travels in the answer, the call that carries a verdict is the one that answers,
and the report is bounded with the count of what the bound left out.
"""

from __future__ import annotations

from typing import Any

import pytest

from dlightrag.engine.answer.agent_browser.contracts import (
    MAX_DIALOGS_PER_CALL,
    BrowserHolder,
    PageDialog,
    PageEvents,
    SessionState,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.browser import BrowserToolHost, browser_tool
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
URL = "https://shop.example/"
WAITING = PageDialog("confirm", "Delete this project?", "waiting")
ANSWERED = PageDialog("confirm", "Delete this project?", "accepted")


def _dialog_page(
    dialogs: tuple[PageDialog, ...], *, snapshot: str | None = None, omitted: int = 0
) -> FakePage:
    """A page whose navigation answers with the dialogs a test gives it."""
    return FakePage(
        session=SessionState(url=URL),
        events=PageEvents(dialogs=dialogs, omitted_dialogs=omitted),
        snapshot=snapshot,
    )


def host(page: FakePage) -> BrowserToolHost:
    provider = FakeProvider(FakeLease(page=page))
    return BrowserToolHost(
        RunAgentBrowser(provider, HOLDER, browser_settings()),
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


async def test_a_dialog_nobody_answered_is_reported_with_what_answers_it() -> None:
    page = _dialog_page((WAITING,))
    result = await call(_tool(host(page)), action="navigate", url=URL)

    assert result.is_error is False
    assert 'The page showed a confirm dialog: "Delete this project?"' in result.text_content
    assert "it is waiting for an answer" in result.text_content
    assert 'dialog="accept"' in result.text_content and 'dialog="dismiss"' in result.text_content
    # Nothing answered it: the page is where the dialog left it.
    assert page.verdicts == [None]


async def test_a_prompt_reports_the_default_that_accepting_takes() -> None:
    page = _dialog_page((PageDialog("prompt", "Your name?", "waiting", "Ada"),))
    result = await call(_tool(host(page)), action="navigate", url=URL)

    assert 'Its default text is "Ada"; accepting takes that as the answer.' in result.text_content


async def test_the_call_that_carries_a_verdict_says_the_dialog_was_answered() -> None:
    page = _dialog_page((ANSWERED,), snapshot="- button [ref=e1]")
    result = await call(_tool(host(page)), action="navigate", url=URL, dialog="accept")

    assert result.is_error is False
    assert (
        'The page showed a confirm dialog: "Delete this project?" (accepted).'
        in result.text_content
    )
    # The verdict reaches the page for the call that carries it, and only for that call.
    assert page.verdicts == ["accept"]


async def test_the_report_holds_the_bound_and_says_how_many_dialogs_it_left_out() -> None:
    page = _dialog_page(
        (PageDialog("alert", "question 0", "accepted"), WAITING),
        snapshot="- button [ref=e1]",
        omitted=17,
    )
    result = await call(_tool(host(page)), action="navigate", url=URL, dialog="accept")

    assert 'The page showed a alert dialog: "question 0" (accepted).' in result.text_content
    # The dialog the page waits on is reported whatever the bound left out of the rest...
    assert 'The page showed a confirm dialog: "Delete this project?"' in result.text_content
    assert "it is waiting for an answer" in result.text_content
    # ...and what the bound did leave out is counted, never a list that reads as all of them.
    assert "(17 more dialog(s) omitted)" in result.text_content


def test_the_dialog_bound_is_the_ports_and_it_is_never_a_silent_cut() -> None:
    from dlightrag.adapters.agent_browser import page as page_module

    # A five-dialog cap came back once already, and it hid the sixth behind nothing at all.
    # The bound is the port's, and the observation counts what it left out.
    assert page_module.MAX_DIALOGS_PER_CALL == MAX_DIALOGS_PER_CALL
    assert MAX_DIALOGS_PER_CALL > 5
