# Copyright 2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A call whose own action closes its page reports what is active, not a failure.

It is also where a dialog is the agent's to answer: the page is stopped by one nobody has
answered, each page's dialog is that page's own, the report is bounded and says how many the
bound left out, and the call that carries a verdict is the one that answers. Every read says
whether the page had gone quiet before it was read, and a call's download refusals are bounded
the same way the dialogs are, with the count of what the bound left out. A ``textbox`` line
states what its field takes, beside its ref, while another role's line reads as the driver
wrote it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from dlightrag.adapters.agent_browser.page import PlaywrightAgentPage
from dlightrag.engine.answer.agent_browser.contracts import (
    MAX_DIALOGS_PER_CALL,
    MAX_DOWNLOADS_PER_CALL,
    MAX_REFUSALS_PER_CALL,
    AgentBrowserError,
    PageLimits,
)
from dlightrag.engine.answer.agent_browser.passwords import (
    FilledPasswords,
)
from tests.support.agent_browser import client_connection

BLOCKED = "Page.click: Timeout 5000ms exceeded."


class _Download:
    """A download as the driver hands it over: it saves the bytes a test gives it."""

    def __init__(self, name: str, content: bytes = b"data") -> None:
        self.suggested_filename = name
        self.url = f"http://shop.example/{name}"
        self.content = content
        self.saved = 0
        self.discarded = 0
        self.deleted = 0

    async def save_as(self, path: Path) -> None:
        path.write_bytes(self.content)
        self.saved += 1

    async def cancel(self) -> None:
        self.discarded += 1

    async def delete(self) -> None:
        self.deleted += 1


class _Dialog:
    """A dialog as the driver hands it over: it is answered only when something answers it."""

    def __init__(self, kind: str = "confirm", message: str = "Sure?", default: str = "") -> None:
        self.type = kind
        self.message = message
        self.default_value = default
        self.answers: list[str] = []

    async def accept(self) -> None:
        self.answers.append("accept")

    async def dismiss(self) -> None:
        self.answers.append("dismiss")


class _Response:
    """One navigation answer, as the page reports it to the response handler."""

    def __init__(self, status: int, tab: _Tab) -> None:
        self.status = status
        self.frame = tab.main_frame

        class _Request:
            @staticmethod
            def is_navigation_request() -> bool:
                return True

        self.request = _Request()


class _Control:
    """The element a ref names: clicking it closes its tab, or shows what the tab holds."""

    def __init__(self, tab: _Tab, ref: str) -> None:
        self._tab = tab
        self._ref = ref

    async def count(self) -> int:
        return 1

    async def evaluate(self, _script: str, **_options: Any) -> list[str]:
        """What the field states it accepts, as the page's own script would answer."""
        return self._tab.constraints.get(self._ref, [])

    async def fill(self, text: str, **_options: Any) -> None:
        self._tab.typed.append(("fill", text))

    async def press_sequentially(self, text: str, **_options: Any) -> None:
        self._tab.typed.append(("keys", text))

    async def click(self, **_options: Any) -> None:
        self._tab.clicks += 1
        if self._tab.status is not None and "response" in self._tab.handlers:
            self._tab.handlers["response"](_Response(self._tab.status, self._tab))
        if self._tab.closes_on_click:
            self._tab.closed = True
            raise PlaywrightError("Target page, context or browser has been closed")
        # Chromium shows each dialog the script raises, and a dialog nothing answers stops
        # the action until the driver gives up on it.
        for dialog in self._tab.dialogs:
            await self._tab.handlers["dialog"](dialog)
            if not dialog.answers:
                raise PlaywrightTimeoutError(BLOCKED)
        self._tab.dialogs = [dialog for dialog in self._tab.dialogs if not dialog.answers]

    async def press(self, key: str, **_options: Any) -> None:
        pass


class _Tab:
    """A page that loads at once, snapshots to one Close control, and holds what a test sets."""

    main_frame = object()

    def __init__(
        self,
        url: str,
        *,
        closes_on_click: bool = False,
        dialogs: list[_Dialog] | None = None,
        status: int | None = None,
        body: str = "",
        title: str = "",
        snapshot: str = '- button "Close" [ref=e1]',
    ) -> None:
        self.url = url
        self.closes_on_click = closes_on_click
        self.dialogs = dialogs if dialogs is not None else []
        self.status = status
        self.body = body
        self.headline = title
        self.snapshot = snapshot
        #: What each ref's element states it accepts, as the page's own script answers for it.
        self.constraints: dict[str, list[str]] = {}
        self.closed = False
        self.clicks = 0
        self.typed: list[tuple[str, str]] = []
        self.handlers: dict[str, Any] = {}
        self.keyboard = self

    async def content(self) -> str:
        return self.body

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def is_closed(self) -> bool:
        return self.closed

    async def wait_for_load_state(self, state: str, **_options: Any) -> None:
        pass

    async def title(self) -> str:
        return self.headline or ("Details" if "details" in self.url else "Results")

    async def aria_snapshot(self, **_options: Any) -> str:
        if any(not dialog.answers for dialog in self.dialogs):
            # A page stopped by a dialog answers no read of what it holds.
            raise AssertionError("a stopped page must not be snapshotted")
        return self.snapshot

    async def screenshot(self, **_options: Any) -> bytes:
        return b"png"

    def get_by_ref(self, ref: str) -> _Control:
        """The documented 1.64 lookup for a snapshot ref; ``locator`` is not used for refs."""
        return _Control(self, ref)

    async def press(self, key: str) -> None:
        pass


class _Browser:
    _impl_obj = client_connection()

    def is_connected(self) -> bool:
        return True


class _LostBrowser(_Browser):
    """A browser whose connection closed, which Playwright announces as disconnected later."""

    _impl_obj = client_connection(closed=True)


class _Context:
    async def close(self) -> None:
        pass


class _SlowTab(_Tab):
    """A page whose long-poll never lets it go quiet, as an SPA with a live socket does."""

    async def wait_for_load_state(self, state: str, **_options: Any) -> None:
        if state == "networkidle":
            raise PlaywrightTimeoutError("Timeout 5000ms exceeded.")


class _UnreadableTab(_Tab):
    """A page whose snapshot fails, as a call that met dialogs and never got to report them."""

    async def aria_snapshot(self, **_options: Any) -> str:
        raise PlaywrightTimeoutError(BLOCKED)


def _agent_page(
    page: _Tab, *, settle: float = 0.0, browser: _Browser | None = None
) -> PlaywrightAgentPage:
    return PlaywrightAgentPage(
        cast(Any, browser or _Browser()),
        cast(Any, _Context()),
        cast(Any, page),
        PageLimits(5, 5, settle, 12, 1024),
        FilledPasswords(),
    )


async def test_a_click_that_closes_its_own_page_returns_to_the_page_before_it() -> None:
    """A Close control closes its tab as it lands, leaving the driver to read a page that
    is gone: a slow machine sees an error where a fast one sees a plain result. The call
    answers with what is active now, which is the page before it."""
    results = _Tab("http://shop.example/results?q=lamp")
    details = _Tab("http://shop.example/details", closes_on_click=True)
    agent_page = _agent_page(results)
    results.handlers["popup"](details)
    await agent_page.press("Tab", ref=None)

    closed = await agent_page.click("e1")

    assert closed.events.returned is True
    assert closed.page is not None
    assert closed.page.url == results.url


async def test_a_click_that_closes_the_last_page_reports_no_page() -> None:
    """One page closing is the call's outcome too: nothing is active until the next
    navigate, rather than a failure the model has to interpret."""
    only = _Tab("http://shop.example/details", closes_on_click=True)
    agent_page = _agent_page(only)

    closed = await agent_page.click("e1")

    assert closed.events.closed is True
    assert closed.page is None


async def test_a_page_closed_on_its_own_is_a_closed_page_and_a_lost_connection_is_a_disconnect() -> (
    None
):
    """A call on pages that are all closed is a closed page while the browser's connection is
    open, and a disconnect once it has closed, though the browser still reads as connected."""
    closed = _Tab("http://shop.example/results")
    closed.closed = True

    with pytest.raises(AgentBrowserError) as gone:
        await _agent_page(closed).snapshot()
    with pytest.raises(AgentBrowserError) as lost:
        await _agent_page(closed, browser=_LostBrowser()).snapshot()

    assert (gone.value.reason, lost.value.reason) == ("page_closed", "disconnected")


async def test_a_call_that_fails_as_its_connection_closes_is_a_disconnect_not_a_closed_page() -> (
    None
):
    """The page closes with the connection and the call fails with the driver's closed-target
    error, as it does when a closed page's call does; only the connection tells them apart."""
    tab = _Tab("http://shop.example/details", closes_on_click=True)

    with pytest.raises(AgentBrowserError) as lost:
        await _agent_page(tab, browser=_LostBrowser()).click("e1")

    assert lost.value.reason == "disconnected"


async def test_a_page_that_never_goes_quiet_is_read_and_says_it_had_not_settled() -> None:
    tab = _SlowTab("http://shop.example/feed")

    read = await _agent_page(tab, settle=5.0).click("e1")

    # The read happened, and the call says it was taken while the page was still loading.
    assert read.events.settled is False
    assert read.snapshot == '- button "Close" [ref=e1]'


async def test_a_page_that_goes_quiet_settles_and_one_that_was_not_waited_for_claims_nothing() -> (
    None
):
    quiet = await _agent_page(_Tab("http://shop.example/results"), settle=5.0).click("e1")
    unwaited = await _agent_page(_Tab("http://shop.example/results")).click("e1")

    assert quiet.events.settled is True
    # A deployment that waits for no quiet reads the page without claiming either way.
    assert unwaited.events.settled is None


async def test_type_empties_the_field_and_types_the_text_as_keystrokes() -> None:
    tab = _Tab("http://shop.example/search")

    await _agent_page(tab).type_text("e1", "lamp", submit=False)

    # Filling sets a value in one step and dispatches no key, which a suggestion widget
    # listens for; what the call does is clear the field and type, so the widget answers.
    assert tab.typed == [("fill", ""), ("keys", "lamp")]


async def test_a_dialog_nobody_answered_stops_the_page_and_is_reported() -> None:
    dialog = _Dialog()
    tab = _Tab("http://shop.example/results", dialogs=[dialog])
    agent_page = _agent_page(tab)

    stopped = await agent_page.click("e1")

    # Our code neither accepted nor dismissed it, and the answer the call gives is the
    # dialog itself rather than an empty page.
    assert dialog.answers == []
    assert [(shown.kind, shown.message, shown.answer) for shown in stopped.events.dialogs] == [
        ("confirm", "Sure?", "waiting")
    ]
    assert stopped.snapshot is None


async def test_a_page_that_stays_stopped_is_reported_again_by_every_call() -> None:
    dialog = _Dialog()
    tab = _Tab("http://shop.example/results", dialogs=[dialog])
    agent_page = _agent_page(tab)
    await agent_page.click("e1")

    again = await agent_page.click("e1")

    # The second call does nothing: the page is where the dialog left it, and the model is
    # told that rather than left to think the click landed twice.
    assert tab.clicks == 1
    assert [shown.answer for shown in again.events.dialogs] == ["waiting"]
    assert again.snapshot is None


async def test_the_call_that_carries_a_verdict_answers_the_waiting_dialog_and_goes_on() -> None:
    dialog = _Dialog()
    tab = _Tab("http://shop.example/results", dialogs=[dialog])
    agent_page = _agent_page(tab)
    await agent_page.click("e1")

    await agent_page.answer_dialogs("accept")
    answered = await agent_page.snapshot()

    assert dialog.answers == ["accept"]
    assert [(shown.message, shown.answer) for shown in answered.events.dialogs] == [
        ("Sure?", "accepted")
    ]
    assert answered.snapshot == '- button "Close" [ref=e1]'


async def test_a_dialog_that_arrives_during_the_call_is_answered_by_the_verdict() -> None:
    dialog = _Dialog("prompt", "Your name?", default="Ada")
    tab = _Tab("http://shop.example/results", dialogs=[dialog])
    agent_page = _agent_page(tab)

    await agent_page.answer_dialogs("dismiss")
    acted = await agent_page.click("e1")

    assert dialog.answers == ["dismiss"]
    shown = acted.events.dialogs[0]
    assert (shown.answer, shown.default) == ("dismissed", "Ada")


async def test_every_dialog_one_action_raises_is_reported_up_to_the_bound_it_counts() -> None:
    dialogs = [_Dialog(message=f"question {number}") for number in range(MAX_DIALOGS_PER_CALL + 4)]
    tab = _Tab("http://shop.example/results", dialogs=dialogs)
    agent_page = _agent_page(tab)

    await agent_page.answer_dialogs("accept")
    acted = await agent_page.click("e1")

    # The bound is the report's, and what it left out is counted rather than hidden.
    assert [shown.message for shown in acted.events.dialogs] == [
        f"question {number}" for number in range(MAX_DIALOGS_PER_CALL)
    ]
    assert acted.events.omitted_dialogs == 4


async def test_every_refused_download_is_reported_up_to_the_bound_it_counts() -> None:
    started = MAX_DOWNLOADS_PER_CALL + MAX_REFUSALS_PER_CALL + 3
    tab = _Tab("http://shop.example/results")
    agent_page = _agent_page(tab)
    for number in range(started):
        tab.handlers["download"](_Download(f"file-{number}.pdf"))

    acted = await agent_page.click("e1")

    # Files are admitted up to the call's own bound, and every download past it is refused;
    # the refusals the model reads are bounded too, and the bound counts what it left out.
    assert len(acted.events.downloads) == MAX_DOWNLOADS_PER_CALL
    assert len(acted.events.refused_downloads) == MAX_REFUSALS_PER_CALL
    assert acted.events.omitted_refusals == started - MAX_DOWNLOADS_PER_CALL - MAX_REFUSALS_PER_CALL
    assert {refusal.reason for refusal in acted.events.refused_downloads} == {"limit"}


async def test_the_dialog_a_page_is_stopped_on_is_reported_past_the_bound_too() -> None:
    dialogs = [_Dialog(message=f"question {number}") for number in range(MAX_DIALOGS_PER_CALL)]
    waiting = _Dialog("confirm", "Still waiting?")
    tab = _UnreadableTab("http://shop.example/results", dialogs=dialogs)
    agent_page = _agent_page(tab)
    await agent_page.answer_dialogs("accept")
    with pytest.raises(AgentBrowserError):
        # The call met a full report's worth of dialogs and failed before it could hand them over.
        await agent_page.click("e1")

    tab.dialogs = [waiting]
    await agent_page.answer_dialogs(None)
    stopped = await agent_page.click("e1")

    # The dialog nobody answered stops the page, and it is reported whatever the bound holds.
    assert [shown.answer for shown in stopped.events.dialogs] == [
        *["accepted"] * MAX_DIALOGS_PER_CALL,
        "waiting",
    ]
    assert stopped.events.omitted_dialogs == 0


async def test_each_page_s_dialog_waits_for_its_own_answer_and_the_verdict_answers_both() -> None:
    second = _Dialog("prompt", "Your name?", "Ada")
    results = _Tab("http://shop.example/results", dialogs=[_Dialog("confirm", "Delete?")])
    details = _Tab("http://shop.example/details")
    agent_page = _agent_page(results)
    results.handlers["popup"](details)

    await agent_page.click("e1")
    await agent_page.press("Tab", ref=None)
    details.dialogs = [second]
    stopped = await agent_page.click("e1")

    # A popup's dialog does not overwrite the first page's: both are waiting, both are reported.
    assert [(shown.message, shown.answer) for shown in stopped.events.dialogs] == [
        ("Delete?", "waiting"),
        ("Your name?", "waiting"),
    ]
    await agent_page.answer_dialogs("accept")

    # One verdict answers every page's dialog, and neither page's is lost to the other's.
    assert results.dialogs[0].answers == ["accept"]
    assert details.dialogs[0].answers == ["accept"]
    answered = await agent_page.snapshot()
    assert [(shown.message, shown.answer) for shown in answered.events.dialogs] == [
        ("Delete?", "accepted"),
        ("Your name?", "accepted"),
    ]


async def test_every_read_says_whether_the_page_had_gone_quiet_before_it_was_read() -> None:
    reads = {
        "snapshot": lambda page: page.snapshot(),
        "find": lambda page: page.find("Close", limit=30),
        "screenshot": lambda page: page.screenshot(full_page=False),
        "capture": lambda page: page.capture(),
    }
    for name, read in reads.items():
        quiet = await read(_agent_page(_Tab("http://shop.example/feed"), settle=5.0))
        loading = await read(_agent_page(_SlowTab("http://shop.example/feed"), settle=5.0))
        unwaited = await read(_agent_page(_Tab("http://shop.example/feed"), settle=0.0))

        # A read is the remedy the not-settled note names, so it says whether waiting helped.
        assert quiet.events.settled is True, name
        assert loading.events.settled is False, name
        assert unwaited.events.settled is None, name


async def test_a_textbox_line_states_what_its_field_takes_and_another_role_does_not() -> None:
    """A field's own constraints are written after the ref of its ``textbox`` line, which is the
    role the fields DlightRAG fills are printed under; a line of another role is left as the
    driver wrote it."""
    tab = _Tab(
        "http://shop.example/signup",
        snapshot="\n".join(
            (
                '- textbox "Password" [ref=e1]',
                '- searchbox "Find" [ref=e2]',
                '- textbox "In a frame" [ref=f2e7]',
                '- button "Join" [ref=e3]',
            )
        ),
    )
    tab.constraints = {
        "e1": ["minlength 12", "maxlength 20", "pattern [A-Za-z0-9._-]{12,20}", "required"],
        "e2": ["maxlength 8"],
        "f2e7": ["maxlength 8"],
    }

    observed = await _agent_page(tab).snapshot()

    assert observed.snapshot == "\n".join(
        (
            '- textbox "Password" [ref=e1] (minlength 12, maxlength 20, '
            "pattern [A-Za-z0-9._-]{12,20}, required)",
            '- searchbox "Find" [ref=e2]',
            '- textbox "In a frame" [ref=f2e7] (maxlength 8)',
            '- button "Join" [ref=e3]',
        )
    )


async def test_every_text_a_sentence_repeats_says_when_it_was_cut() -> None:
    long_title = "T" * 260
    long_line = "- text " + "x" * 400
    dialog = _Dialog("alert", "m" * 260)

    class _LongTab(_Tab):
        async def aria_snapshot(self, **_options: Any) -> str:
            return long_line

    tab = _LongTab("http://shop.example/results", dialogs=[dialog], title=long_title)
    agent_page = _agent_page(tab, settle=5.0)

    stopped = await agent_page.click("e1")
    await agent_page.answer_dialogs("accept")
    found = await agent_page.find("x", limit=30)

    # A title, a find line, and a dialog's words each carry the cut and its count.
    assert stopped.page is not None
    assert stopped.page.title.endswith(" … [200 of 260 characters]")
    assert stopped.events.dialogs[0].message.endswith(" … [200 of 260 characters]")
    assert found.lines[0].endswith(" … [300 of 407 characters]")


async def test_a_proxy_refusal_is_named_as_the_deployment_boundary_not_a_bot_filter() -> None:
    tab = _Tab(
        "http://shop.example:8443/",
        status=403,
        body="<html><body><h1>ERROR</h1><pre>Access Denied</pre>"
        "<address>Generated by squid/6.6</address></body></html>",
    )
    agent_page = _agent_page(tab)

    try:
        await agent_page.click("e1")
    except AgentBrowserError as refused:
        assert refused.reason == "egress_refused"
        assert "egress proxy refused the connection" in refused.public_message
    else:  # pragma: no cover - the refusal is raised
        raise AssertionError("a proxy's own refusal page was read as the site")


async def test_a_read_while_the_page_waits_names_the_dialog_it_waits_on() -> None:
    dialog = _Dialog("prompt", "Your name?", default="Ada")
    tab = _Tab("http://shop.example/results", dialogs=[dialog])
    agent_page = _agent_page(tab)
    await agent_page.click("e1")

    try:
        await agent_page.find("Close", limit=30)
    except AgentBrowserError as refused:
        assert refused.reason == "dialog_waiting"
        assert 'a prompt dialog nobody has answered: "Your name?"' in refused.public_message
        assert 'Its default text is "Ada".' in refused.public_message
    else:  # pragma: no cover - the read is refused
        raise AssertionError("a stopped page answered a find")
