# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Agent Page: an Agent Session's browser context, and what it does on its active page."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from playwright.async_api import (
    Browser,
    BrowserContext,
    Dialog,
    Download,
    FilePayload,
    Locator,
    Page,
    Response,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from dlightrag.adapters.agent_browser.driver import (
    CLOSE_SECONDS,
    browser_lost,
    milliseconds,
    navigation_status,
    network_failure,
    page_content,
    page_origin,
    proxy_page_refusal,
    starts_download,
)
from dlightrag.engine.answer.agent_browser.accounts import (
    account_site,
)
from dlightrag.engine.answer.agent_browser.block import (
    refused_on_navigation,
)
from dlightrag.engine.answer.agent_browser.contracts import (
    MAX_DIALOGS_PER_CALL,
    MAX_DOWNLOADS_PER_CALL,
    MAX_REFUSALS_PER_CALL,
    REF_PATTERN,
    AgentBrowserError,
    AgentBrowserFailure,
    CredentialFill,
    CredentialForm,
    DialogVerdict,
    DownloadedFile,
    DownloadRefusal,
    FoundElements,
    PageCapture,
    PageDialog,
    PageEvents,
    PageLimits,
    PageObservation,
    PageScreenshot,
    PageState,
    SessionState,
    StorageState,
    UploadFile,
    browser_failure,
    page_failure,
    refusal_failure,
)
from dlightrag.engine.answer.agent_browser.elision import (
    elide,
)
from dlightrag.engine.answer.agent_browser.passwords import (
    FilledPasswords,
)

logger = logging.getLogger(__name__)

#: A popup's page event, or a link's download, reaches the client a few milliseconds after
#: the click that caused it returns (25 ms measured). A call that may have caused one waits
#: this long for it before it reports what the pages did.
_EVENT_GRACE_SECONDS = 0.2
_DOWNLOAD_POLL_SECONDS = 0.1
_DOWNLOAD_DIRECTORY_PREFIX = "dlightrag-browser-download-"
_TITLE_CHARS = 200
#: How much of a dialog's message, or of the text a wait names, a sentence repeats.
_QUOTED_CHARS = 200
_FOUND_LINE_CHARS = 300
_DETAIL_CHARS = 300
#: One snapshot line that names a text field: the role such a field is printed under, and the
#: ref the next call names it by. What a field accepts is not in the snapshot, so it is read off
#: the element and written after the ref, before the value the line may go on to print.
_FIELD_LINE = re.compile(rf"^(?P<head>\s*- textbox\b.*?\[ref=(?P<ref>{REF_PATTERN})\])")
#: What one field states it accepts: the constraints a model choosing a value must know, and
#: the ones no snapshot prints. A field that states none answers with nothing.
_FIELD_CONSTRAINTS = """
(element) => {
  const constraints = [];
  if (element.minLength > 0) constraints.push(`minlength ${element.minLength}`);
  if (element.maxLength > 0) constraints.push(`maxlength ${element.maxLength}`);
  const pattern = element.getAttribute('pattern');
  if (pattern) constraints.push(`pattern ${pattern}`);
  if (element.required) constraints.push('required');
  return constraints;
}
"""
#: How much of what a field states one snapshot line repeats.
_CONSTRAINTS_CHARS = 200
#: A scroll moves the page by this share of the viewport's height.
_SCROLL_SHARE = 0.8
_DEFAULT_VIEWPORT_HEIGHT = 720
#: How long the restore script waits, on each document, for the origin it holds state for.
_RESTORE_WINDOW_SECONDS = 60_000
_RESTORE_SCRIPT = """
(origin, local, session, deadline) => {
  if (Date.now() > deadline || location.origin !== origin) return;
  for (const [store, entries] of [[localStorage, local], [sessionStorage, session]]) {
    for (const [key, value] of Object.entries(entries)) {
      try { store.setItem(key, value); } catch (error) {}
    }
  }
}
"""
_READ_STORAGE_SCRIPT = """
() => ({
  local: Object.fromEntries(Object.entries({...localStorage})),
  session: Object.fromEntries(Object.entries({...sessionStorage})),
})
"""


class PlaywrightAgentPage:
    """An anonymous context of the Run's browser with one active page.

    A popup or a new tab becomes the active page once the call that opened it has acted, or,
    when it opens later, when the next call that names no ref begins; a call that names a ref
    acts on the page the ref came from. A download, a dialog, and the answer of a navigation
    are noted as they happen and reported by the next call that returns.

    A dialog is the agent's to answer (ADR 0032), and each page's dialog is its own: one the
    call did not answer leaves that page stopped, and the call reports the dialog rather than
    completing an action nobody chose; the call that carries ``answer_dialogs``' verdict
    answers every dialog a page is stopped on, and goes on. Every dialog is reported, up to
    the bound that says what it left out, and none is answered by this class's own judgement.

    The passwords DlightRAG fills into its fields are redacted from every text it returns
    (ADR 0034), because the driver prints a filled value in a snapshot and in the page's own
    HTML and URL, and quotes it in the error of a failed fill.
    """

    def __init__(
        self,
        browser: Browser,
        context: BrowserContext,
        page: Page,
        limits: PageLimits,
        passwords: FilledPasswords,
    ) -> None:
        self._browser = browser
        self._context = context
        self._limits = limits
        self._passwords = passwords
        #: The open pages, the oldest first; the last is the active page.
        self._pages: list[Page] = [page]
        #: Popups opened since a call last returned, which no call has adopted yet.
        self._popups: list[Page] = []
        self._downloads: list[Download] = []
        #: The dialogs the pages showed since a call last returned, at most the bound.
        self._dialogs: list[PageDialog] = []
        #: How many dialogs the bound left out of that list.
        self._omitted_dialogs = 0
        #: The dialog each page is stopped on, which no call has answered; empty when none is.
        self._waiting: dict[Page, Dialog] = {}
        #: How each of those is reported, for as long as it stays unanswered.
        self._waiting_dialog: dict[Page, PageDialog] = {}
        #: What the call now running answers dialogs with, which no call decided for itself.
        self._verdict: DialogVerdict | None = None
        #: The last main-frame navigation answer of each page during the current call.
        self._statuses: dict[Page, int] = {}
        self._watch(page)

    @classmethod
    async def open(
        cls,
        browser: Browser,
        context: BrowserContext,
        limits: PageLimits,
        passwords: FilledPasswords,
    ) -> PlaywrightAgentPage:
        """Start an Agent Page in ``context``, which it owns from then on, with a blank page."""
        try:
            page = await context.new_page()
        except BaseException:
            await _close_context(context)
            raise
        return cls(browser, context, page, limits, passwords)

    def current_url(self) -> str | None:
        pages = self._open_pages()
        return self._passwords.redact(pages[-1].url) if pages else None

    def active_page(self) -> Page | None:
        """The page this Agent Page drives now, for a live session to show; None when none is open."""
        pages = self._open_pages()
        return pages[-1] if pages else None

    async def answer_dialogs(self, verdict: DialogVerdict | None) -> None:
        """Say what the pages' dialogs are answered with for the call that follows.

        A verdict answers every dialog a page of this context is stopped on, and every dialog
        the call meets; None leaves them waiting, which is the default. The verdict holds
        until the next call states its own, because the tool states one for every call it
        makes. Each page's waiting dialog is its own, so one page's answer never spends
        another's.
        """
        self._verdict = verdict
        if verdict is None:
            return
        for page in list(self._waiting_dialogs()):
            dialog = self._waiting.pop(page)
            self._waiting_dialog.pop(page, None)
            await self._answer(dialog, verdict)
            # The call that answers a dialog is the one that reads it as answered; the call
            # that met it read it as waiting, and it travels until this one.
            self._note_dialog(self._shown(dialog, _verdict_answer(verdict)))

    def _waiting_on(self, page: Page) -> bool:
        """Whether a dialog nobody answered is stopping ``page``, so a call may not act on it."""
        return page in self._waiting

    async def _held(self, page: Page, *, adopted: bool = False) -> PageObservation | None:
        """What a call answers when a dialog nobody answered is stopping ``page``, or None.

        The page can act on nothing until a call answers the dialog, so the call does nothing
        and answers with the page the dialog left: its state, the dialog, and no snapshot.
        """
        if not self._waiting_on(page):
            return None
        return await self._after(page, adopted=adopted)

    def _refuse_held(self, page: Page) -> None:
        """Refuse a read of a page a dialog nobody answered is stopping."""
        if dialog := self._waiting.get(page):
            raise self._waiting_failure(dialog)

    async def navigate(self, url: str) -> PageObservation:
        page, adopted = await self._page_to_navigate()
        if held := await self._held(page, adopted=adopted):
            return held
        # A URL that answers with a file starts a download instead of loading a page.
        downloading = False
        try:
            await page.goto(url, wait_until="load", timeout=milliseconds(self._nav_seconds))
        except PlaywrightError as exc:
            if self._waiting_on(page):
                # The navigation stopped on a dialog nobody answered, which is its result.
                return await self._after(page, adopted=adopted)
            downloading = starts_download(exc)
            if not downloading:
                raise self._failure(
                    exc,
                    page,
                    "navigate to",
                    "the page",
                    timed_out=browser_failure("timeout", seconds=self._nav_seconds),
                ) from exc
        return await self._after(page, adopted=adopted, expecting_events=downloading)

    async def back(self) -> PageObservation:
        page, adopted = self._start()
        if held := await self._held(page, adopted=adopted):
            return held
        before = page.url
        try:
            response = await page.go_back(
                wait_until="load", timeout=milliseconds(self._nav_seconds)
            )
        except PlaywrightError as exc:
            if self._waiting_on(page):
                return await self._after(page, adopted=adopted)
            raise self._failure(
                exc,
                page,
                "go back from",
                "the page",
                timed_out=browser_failure("timeout", seconds=self._nav_seconds),
            ) from exc
        # A step back inside the same document answers nothing, and still moved the page.
        if response is None and page.url == before:
            raise page_failure("no_history")
        return await self._after(page, adopted=adopted)

    async def snapshot(self) -> PageObservation:
        page, events = self._observing()
        if self._waiting_on(page):
            # The page is stopped, and what this call observed is still what it reports.
            return await self._observation(page, events)
        settled = await self._quiet(page)
        return await self._observation(page, events, settled=settled)

    async def find(self, query: str, *, limit: int) -> FoundElements:
        page, events = self._observing()
        self._refuse_held(page)
        settled = await self._quiet(page)
        tree = await self._tree(page, depth=None, where="the page")
        needle = query.casefold()
        matches = [line.strip() for line in tree.splitlines() if needle in line.casefold()]
        return FoundElements(
            page=await self._state(page, "the page"),
            events=await self._deliver(events, page, settled=settled),
            lines=tuple(elide(line, _FOUND_LINE_CHARS) for line in matches[:limit]),
            total=len(matches),
        )

    async def wait(
        self, *, text: str | None, text_gone: str | None, seconds: float | None
    ) -> PageObservation:
        page, adopted = self._start()
        if held := await self._held(page, adopted=adopted):
            return held
        if seconds is not None:
            await asyncio.sleep(seconds)
        else:
            appears = text is not None
            awaited = (text if appears else text_gone) or ""
            try:
                await page.get_by_text(awaited).first.wait_for(
                    state="visible" if appears else "hidden",
                    timeout=milliseconds(self._nav_seconds),
                )
            except PlaywrightError as exc:
                if self._waiting_on(page):
                    return await self._after(page, adopted=adopted)
                raise self._failure(
                    exc,
                    page,
                    "wait for",
                    "the text",
                    timed_out=page_failure(
                        "wait_timeout",
                        text=elide(" ".join(awaited.split()), _QUOTED_CHARS),
                        change="appear" if appears else "disappear",
                        seconds=self._nav_seconds,
                    ),
                ) from exc
        return await self._after(page, adopted=adopted)

    async def click(self, ref: str) -> PageObservation:
        page = self._begin()
        if held := await self._held(page):
            return held
        element = await self._element(page, ref)
        try:
            await element.click(timeout=self._action_ms)
        except PlaywrightError as exc:
            return await self._action_failed(
                exc, page, "click", f"element {ref}", ref=ref, verb="clicked", expecting_events=True
            )
        return await self._after(page, expecting_events=True)

    async def type_text(self, ref: str, text: str, *, submit: bool) -> PageObservation:
        page = self._begin()
        if held := await self._held(page):
            return held
        element = await self._element(page, ref)
        try:
            # The field is emptied and the text is typed a key at a time: filling sets the
            # value in one step and dispatches no keystroke, which a suggestion or autocomplete
            # widget answers to and a page listening for keys never sees.
            await element.fill("", timeout=self._action_ms)
            await element.press_sequentially(text, timeout=self._action_ms)
            if submit:
                await element.press("Enter", timeout=self._action_ms)
        except PlaywrightError as exc:
            return await self._action_failed(
                exc,
                page,
                "type into",
                f"element {ref}",
                ref=ref,
                verb="filled",
                expecting_events=submit,
            )
        return await self._after(page, expecting_events=submit)

    async def select(self, ref: str, values: tuple[str, ...]) -> PageObservation:
        page = self._begin()
        if held := await self._held(page):
            return held
        element = await self._element(page, ref)
        try:
            await element.select_option(list(values), timeout=self._action_ms)
        except PlaywrightError as exc:
            return await self._action_failed(
                exc,
                page,
                "select an option of",
                f"element {ref}",
                ref=ref,
                verb="selected",
                expecting_events=True,
            )
        return await self._after(page, expecting_events=True)

    async def press(self, key: str, *, ref: str | None) -> PageObservation:
        page, adopted = self._start(ref)
        if held := await self._held(page, adopted=adopted):
            return held
        element = None if ref is None else await self._element(page, ref)
        try:
            if element is None:
                await page.keyboard.press(key)
            else:
                await element.press(key, timeout=self._action_ms)
        except PlaywrightError as exc:
            if "Unknown key" in str(exc):
                raise page_failure("invalid_key", key=key) from exc
            return await self._action_failed(
                exc,
                page,
                "press a key on",
                "the page" if ref is None else f"element {ref}",
                ref=ref,
                verb="pressed",
                adopted=adopted,
                expecting_events=True,
            )
        return await self._after(page, adopted=adopted, expecting_events=True)

    async def scroll(self, *, direction: Literal["up", "down"], ref: str | None) -> PageObservation:
        page, adopted = self._start(ref)
        if held := await self._held(page, adopted=adopted):
            return held
        element = None if ref is None else await self._element(page, ref)
        viewport = page.viewport_size
        height = viewport["height"] if viewport else _DEFAULT_VIEWPORT_HEIGHT
        distance = _SCROLL_SHARE * height * (1 if direction == "down" else -1)
        try:
            if element is not None:
                # The wheel scrolls what lies under the pointer, so aim the pointer at the element.
                await element.hover(timeout=self._action_ms)
            await page.mouse.wheel(0, distance)
        except PlaywrightError as exc:
            return await self._action_failed(
                exc,
                page,
                "scroll",
                "the page" if ref is None else f"element {ref}",
                ref=ref,
                verb="scrolled",
                adopted=adopted,
            )
        return await self._after(page, adopted=adopted)

    async def upload(self, ref: str, files: tuple[UploadFile, ...]) -> PageObservation:
        page = self._begin()
        if held := await self._held(page):
            return held
        element = await self._element(page, ref)
        payload: list[FilePayload] = [
            {"name": file.name, "mimeType": file.mime_type, "buffer": file.content}
            for file in files
        ]
        try:
            if await element.evaluate(
                "e => e instanceof HTMLInputElement && e.type === 'file'", timeout=self._action_ms
            ):
                await element.set_input_files(payload, timeout=self._action_ms)
            else:
                # A button can open the file chooser of an input the page keeps out of sight.
                try:
                    async with page.expect_file_chooser(timeout=self._action_ms) as chooser:
                        await element.click(timeout=self._action_ms)
                    await (await chooser.value).set_files(payload, timeout=self._action_ms)
                except PlaywrightTimeoutError:
                    raise page_failure("not_file_input", ref=ref) from None
        except PlaywrightError as exc:
            return await self._action_failed(
                exc,
                page,
                "upload files to",
                f"element {ref}",
                ref=ref,
                verb="given files",
                expecting_events=True,
            )
        return await self._after(page, expecting_events=True)

    async def screenshot(self, *, full_page: bool) -> PageScreenshot:
        page, events = self._observing()
        self._refuse_held(page)
        settled = await self._quiet(page)
        if self._passwords and await self._holds_a_password(page):
            raise page_failure("password_held")
        try:
            png = await page.screenshot(
                type="png",
                full_page=full_page,
                timeout=milliseconds(self._nav_seconds),
                animations="disabled",
                caret="hide",
            )
        except PlaywrightError as exc:
            raise self._failure(exc, page, "take a screenshot of", "the page") from exc
        return PageScreenshot(
            page=await self._state(page, "the page"),
            events=await self._deliver(events, page, settled=settled),
            png=png,
        )

    async def capture(self) -> PageCapture:
        page, events = self._observing()
        self._refuse_held(page)
        settled = await self._quiet(page)
        try:
            html = await page_content(page, self._nav_seconds)
        except PlaywrightError as exc:
            raise self._failure(exc, page, "capture", "the page") from exc
        return PageCapture(
            page=await self._state(page, "the page"),
            events=await self._deliver(events, page, settled=settled),
            html=self._passwords.redact_bytes(html.encode("utf-8", errors="replace")),
        )

    async def credential_form(
        self,
        *,
        site: str,
        password_refs: tuple[str, ...],
        email_ref: str | None,
        username_ref: str | None,
    ) -> CredentialForm:
        page = self._begin()
        self._refuse_held(page)
        limits: list[int] = []
        email = username = ""
        try:
            for ref in password_refs:
                field = await self._field(page, ref, "password", site)
                limits.append(await field.evaluate("e => e.maxLength", timeout=self._action_ms))
            # What a text field holds is returned, so a page that shows a filled password in one
            # is redacted like every other text.
            if email_ref is not None:
                field = await self._field(page, email_ref, "email", site)
                email = self._passwords.redact(await field.input_value(timeout=self._action_ms))
            if username_ref is not None:
                field = await self._field(page, username_ref, "username", site)
                username = self._passwords.redact(await field.input_value(timeout=self._action_ms))
        except PlaywrightError as exc:
            raise self._failure(exc, page, "read", "the form's fields") from exc
        # A field with no maxlength reads -1.
        positive = [limit for limit in limits if limit > 0]
        return CredentialForm(min(positive, default=None), email, username)

    async def fill_credentials(
        self, fills: tuple[CredentialFill, ...], *, site: str
    ) -> PageObservation:
        page = self._begin()
        if held := await self._held(page):
            return held
        # Every ref is checked before anything is filled, so a bad one fills nothing.
        fields = [await self._field(page, fill.ref, fill.kind, site) for fill in fills]
        # A password is known to the redaction before it reaches the page.
        for fill in fills:
            if fill.kind == "password":
                self._passwords.add(fill.value)
        filled: list[Locator] = []
        for fill, field in zip(fills, fields, strict=True):
            failure = await self._fill_secretly(field, fill.value.get_secret_value())
            if failure is not None:
                await self._clear([*filled, field])
                raise page_failure(
                    failure, ref=fill.ref, verb="filled", seconds=self._limits.action_timeout
                )
            filled.append(field)
        return await self._after(page)

    async def clear_fields(self, refs: tuple[str, ...], *, site: str) -> None:
        page = self._begin()
        if self._waiting_on(page):
            return
        fields = [(await self._input(page, ref, site))[0] for ref in refs]
        await self._clear(fields)

    async def aclose(self) -> None:
        """Close the context and every page in it; a context that does not answer is given up."""
        await _close_context(self._context)

    async def capture_session(self) -> SessionState:
        """Where the active page is, and the context's cookies and each origin's storage.

        What the context holds is read here rather than from the lease, because the context
        is this page's; a browser that does not answer a cookie read carries none.
        """
        origins: dict[str, StorageState] = {}
        for page in self._open_pages():
            origin = page_origin(page.url)
            if origin is None or origin in origins:
                continue
            origins[origin] = await self._storage(page)
        try:
            cookies = await self._context.cookies()
        except PlaywrightError:
            cookies = []
        pages = self._open_pages()
        return SessionState(
            url=self._passwords.redact(pages[-1].url) if pages else "",
            cookies=tuple(cookies),
            origins=origins,
        )

    async def restore_session(self, session: SessionState) -> None:
        """Put ``session``'s cookies and storage into this page's context, before it navigates.

        The script is installed once per origin and waits for a document of its origin, so
        the state is there before that document's own scripts run; a document of any other
        origin is left alone, and the wait ends when the context is about to be gone anyway.
        """
        if session.cookies:
            try:
                await self._context.add_cookies(cast(Any, list(session.cookies)))
            except PlaywrightError:
                # A cookie the context refuses costs that cookie; the move is worth
                # more than it, and the Run is told nothing it could act on.
                logger.warning("Failed to restore an Agent Browser cookie", exc_info=True)
        if not session.origins:
            return
        deadline = int(time.time() * 1000 + _RESTORE_WINDOW_SECONDS)
        for origin, state in session.origins.items():
            await self._context.add_init_script(
                script=f"({_RESTORE_SCRIPT})({_js(origin)}, {_js(state.local)}, "
                f"{_js(state.session)}, {deadline});"
            )

    async def _storage(self, page: Page) -> StorageState:
        """One origin's web storage as its page holds it now."""
        try:
            read = await page.evaluate(_READ_STORAGE_SCRIPT)
        except PlaywrightError:
            return StorageState()
        return StorageState(local=dict(read["local"]), session=dict(read["session"]))

    # -- the pages ------------------------------------------------------------------------

    def _watch(self, page: Page) -> None:
        """Note what a page does besides the calls: its popups, downloads, dialogs, answers."""
        page.on("popup", self._on_popup)
        page.on("download", lambda download: self._downloads.append(download))
        page.on("dialog", lambda dialog: self._on_dialog(page, dialog))
        page.on("response", lambda response: self._on_response(page, response))

    def _on_popup(self, popup: Page) -> None:
        self._popups.append(popup)
        self._watch(popup)

    def _on_response(self, page: Page, response: Response) -> None:
        if (status := navigation_status(page, response)) is not None:
            self._statuses[page] = status

    async def _on_dialog(self, page: Page, dialog: Dialog) -> None:
        """Note a dialog a page showed, and answer it only where the call was told to.

        A dialog nobody has answered stops the page that showed it, which is the state the
        call that met it reports: what the page does with it is the agent's judgement, not
        this class's.
        """
        if self._verdict is None:
            # A dialog nobody has answered is reported by every call until one answers it, so
            # it is kept as what its page waits on, and none of it is the bound's to drop.
            self._waiting[page] = dialog
            self._waiting_dialog[page] = self._shown(dialog, "waiting")
            return
        self._note_dialog(self._shown(dialog, _verdict_answer(self._verdict)))
        await self._answer(dialog, self._verdict)

    async def _answer(self, dialog: Dialog, verdict: DialogVerdict) -> None:
        """Tell the dialog what the call decided; a page that closed with it open says nothing."""
        try:
            await (dialog.accept() if verdict == "accept" else dialog.dismiss())
        except PlaywrightError:
            return

    def _shown(
        self, dialog: Dialog, answer: Literal["accepted", "dismissed", "waiting"]
    ) -> PageDialog:
        """One dialog as the model reads it: its kind, its words, and a prompt's default."""
        default = dialog.default_value if dialog.type == "prompt" else ""
        return PageDialog(
            kind=dialog.type,
            message=self._quoted(dialog.message),
            answer=answer,
            default=self._quoted(default or ""),
        )

    def _quoted(self, text: str) -> str:
        """One line of page text a sentence repeats, cut and marked: no control character."""
        return elide(" ".join(self._passwords.redact(text).split()), _QUOTED_CHARS)

    def _waiting_failure(self, dialog: Dialog) -> AgentBrowserError:
        """The sentence a read is refused with while nobody has answered ``dialog``."""
        default = self._quoted(dialog.default_value or "") if dialog.type == "prompt" else ""
        return page_failure(
            "dialog_waiting",
            kind=dialog.type,
            message=self._quoted(dialog.message),
            detail=f' Its default text is "{default}".' if default else "",
        )

    def _note_dialog(self, shown: PageDialog) -> None:
        """Keep one dialog for the next call that returns, or count what the bound left out.

        A page can raise dialogs without end, and one that answers its own leaves nothing on
        it: what the model reads is bounded, and ``PageEvents.omitted_dialogs`` says how many
        the bound left out rather than a list that reads as the whole of them.
        """
        if len(self._dialogs) < MAX_DIALOGS_PER_CALL:
            self._dialogs.append(shown)
        else:
            self._omitted_dialogs += 1

    def _open_pages(self) -> list[Page]:
        return [page for page in self._pages if not page.is_closed()]

    def _begin(self) -> Page:
        """The active page a call starts on. The answers noted for the last call are forgotten."""
        self._statuses.clear()
        self._pages = self._open_pages()
        if not self._pages:
            # Every page closes with a browser that is gone.
            raise page_failure("disconnected" if browser_lost(self._browser) else "page_closed")
        return self._pages[-1]

    def _start(self, ref: str | None = None) -> tuple[Page, bool]:
        """The page a call starts on, and whether it began on a popup it adopted.

        A call that names a ref acts on the page the ref came from. Any other call acts on
        the newest page, a popup that opened since the last call included.
        """
        if ref is not None:
            return self._begin(), False
        page, events = self._observing()
        return page, events.new_page

    async def _page_to_navigate(self) -> tuple[Page, bool]:
        """The newest page, a popup that opened since the last call included, or a fresh one
        when none is open."""
        self._statuses.clear()
        adopted = self._take_popups()
        if not self._pages:
            try:
                page = await self._context.new_page()
            except PlaywrightError as exc:
                raise self._failure(exc, None, "open", "a page") from exc
            self._pages.append(page)
            self._watch(page)
        return self._pages[-1], adopted

    def _observing(self) -> tuple[Page, PageEvents]:
        """The page a call that names no ref acts on or observes, and what changed since.

        A popup that opened since the last call is that page: nothing was asked of the old
        one, so there is no ref to keep acting on it.
        """
        events = self._adopt(self._begin())
        return self._pages[-1], events

    def _take_popups(self) -> bool:
        """Make the popups that opened since the last call pages of this one, the newest
        active, and say whether there were any."""
        popups = [popup for popup in self._popups if not popup.is_closed()]
        self._popups.clear()
        self._pages = [*self._open_pages(), *popups]
        return bool(popups)

    def _adopt(self, acted: Page) -> PageEvents:
        """Make the newest popup the active page, or note that ``acted`` closed."""
        if self._take_popups():
            return PageEvents(new_page=True)
        if acted.is_closed():
            return PageEvents(returned=True) if self._pages else PageEvents(closed=True)
        return PageEvents()

    # -- what a call returns -------------------------------------------------------------------

    async def _after(
        self, acted: Page, *, adopted: bool = False, expecting_events: bool = False
    ) -> PageObservation:
        """What an acting call left, once the pages it touched have settled.

        ``expecting_events`` is for a call that may have opened a popup or started a
        download, which the pages announce a moment after the action returns. The page the
        call acted on settles before popups are adopted, so one it opened while that page
        loaded is reported by this call. ``adopted`` says the call began on a popup. A page a
        dialog nobody answered is stopping is not waited for: it cannot settle until the
        dialog is answered, and the call reports the dialog instead.
        """
        if expecting_events:
            await asyncio.sleep(_EVENT_GRACE_SECONDS)
        deadline = self._settle_deadline()
        settled: bool | None = None
        if not self._waiting_on(acted):
            settled = await self._settle(acted, deadline)
        events = self._adopt(acted)
        if adopted and not (events.returned or events.closed):
            events = replace(events, new_page=True)
        if events.new_page and not self._waiting_on(self._pages[-1]):
            # A popup is still loading, and may close itself while it does.
            popup = self._pages[-1]
            popup_settled = await self._settle(popup, deadline)
            settled = popup_settled if settled is None else settled and popup_settled
            if popup.is_closed():
                events = self._adopt(popup)
        if events.closed:
            # Every page closes with a browser that is gone.
            if browser_lost(self._browser):
                raise page_failure("disconnected")
            return PageObservation(None, await self._deliver(events, None), None)
        page = self._pages[-1]
        if acted:
            # A call that changed the page may have landed on a refusal: the page a bot
            # filter answers is named here rather than reported as a page that read empty.
            await self._refused(page, events)
        return await self._observation(page, events, acted=True, settled=settled)

    async def _refused(self, page: Page, events: PageEvents) -> None:
        """Raise the refusal the page the call landed on is: a bot filter's, or our own proxy's."""
        status = self._statuses.get(page)
        if status is None or status < 400:
            return
        try:
            html = (await page_content(page, self._nav_seconds)).encode("utf-8", errors="replace")
        except PlaywrightError:
            # The page did not answer a read of what it is; the call reports as it would.
            return
        if (proxy := proxy_page_refusal(status, html)) is not None:
            raise proxy
        if block := refused_on_navigation(status, html):
            raise refusal_failure(block)

    async def _observation(
        self,
        page: Page,
        events: PageEvents,
        *,
        acted: bool = False,
        settled: bool | None = None,
    ) -> PageObservation:
        """The page's state and snapshot, and what the pages did around the call.

        After an action ``acted`` says in a failure that the action itself completed. A page a
        dialog nobody answered is stopping has no snapshot to read: the read itself would
        wait for the dialog, and nothing in the page can have changed since it opened.
        """
        where = "the page (the action itself completed)" if acted else "the page"
        state = await self._state(page, where)
        if self._waiting_on(page):
            return PageObservation(state, await self._deliver(events, page, settled=settled), None)
        tree = await self._tree(page, depth=self._limits.snapshot_depth, where=where)
        return PageObservation(state, await self._deliver(events, page, settled=settled), tree)

    def _settle_deadline(self) -> float:
        return time.monotonic() + self._limits.settle_timeout

    async def _quiet(self, page: Page) -> bool | None:
        """Whether ``page`` went quiet within this call's settle wait, before it is read.

        Every call that returns a view of the page measures this, so the agent can tell
        whether waiting helped rather than guess it from a page that looks empty.
        """
        return await self._settle(page, self._settle_deadline())

    async def _settle(self, page: Page, deadline: float) -> bool | None:
        """Wait until ``deadline`` for the page to load and go quiet, and say whether it did.

        A page that never goes quiet is read as it stands, and the call says so rather than
        presenting it as settled: a partial read taken for the whole page is how a working
        site comes to look empty. None is a deployment that waits for no quiet at all, which
        claims neither. A page that closes, or a browser that disconnects, while it is
        waited for is reported by what reads the page next.
        """
        if self._limits.settle_timeout <= 0:
            return None
        for state in ("domcontentloaded", "networkidle"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                await page.wait_for_load_state(state, timeout=milliseconds(remaining))
            except PlaywrightError:
                return False
        return True

    async def _state(self, page: Page, where: str) -> PageState:
        try:
            title = await page.title()
        except PlaywrightError as exc:
            raise self._failure(exc, page, "read the title of", where) from exc
        return PageState(
            url=self._passwords.redact(page.url),
            title=elide(" ".join(self._passwords.redact(title).split()), _TITLE_CHARS),
        )

    async def _tree(self, page: Page, *, depth: int | None, where: str) -> str:
        """The page's accessibility snapshot, which prints a filled password input's value and
        each field's own constraints beside its ref."""
        try:
            tree = await page.aria_snapshot(mode="ai", depth=depth, timeout=self._action_ms)
        except PlaywrightError as exc:
            raise self._failure(exc, page, "take a snapshot of", where) from exc
        return self._passwords.redact(await self._with_constraints(page, tree))

    async def _with_constraints(self, page: Page, tree: str) -> str:
        """``tree`` with what each text field states it accepts written beside the ref naming it.

        A snapshot says what a field is called and what it holds, never what it takes, and a
        model that has to decide a value — how long a generated password may be, what pattern
        the site demands — can only decide from what the page itself states. So every ``textbox``
        line carries its ``minlength``, ``maxlength``, ``pattern`` and ``required`` after its
        ref, and a field that states none reads as it always did. Reading a constraint never
        fails the call: an element the page has since replaced keeps the line it had.
        """
        lines = tree.splitlines()
        for number, line in enumerate(lines):
            if (match := _FIELD_LINE.match(line)) is None:
                continue
            if constraints := await self._constraints(page, match["ref"]):
                placed = f"{match['head']} ({constraints})"
                lines[number] = line.replace(match["head"], placed, 1)
        return "\n".join(lines)

    async def _constraints(self, page: Page, ref: str) -> str:
        """What the field ``ref`` names states it accepts, as one parenthesis reads, or empty."""
        field = page.get_by_ref(ref)
        try:
            constraints = await field.evaluate(_FIELD_CONSTRAINTS, timeout=self._action_ms)
        except PlaywrightError:
            return ""
        # A snapshot line is one line, whatever a page wrote into an attribute.
        return elide(" ".join(", ".join(constraints).split()), _CONSTRAINTS_CHARS)

    async def _element(self, page: Page, ref: str) -> Locator:
        """The element a ref names in the page's latest snapshot."""
        element = page.get_by_ref(ref)
        try:
            found = await element.count()
        except PlaywrightError as exc:
            raise self._failure(exc, page, "look for", f"element {ref}") from exc
        if found == 0:
            raise page_failure("stale_ref", ref=ref)
        return element

    async def _field(
        self, page: Page, ref: str, kind: Literal["email", "username", "password"], site: str
    ) -> Locator:
        """The field a ref names, once it is known to be of ``kind`` and in a frame of ``site``."""
        field, seen = await self._input(page, ref, site)
        if kind == "password":
            if seen != "password":
                raise page_failure("not_password_field", ref=ref)
        elif seen not in ("text", "email"):
            raise page_failure("not_text_field", ref=ref)
        return field

    async def _input(self, page: Page, ref: str, site: str) -> tuple[Locator, str]:
        """The element a ref names and its input type, once it is known to be in a frame of ``site``.

        A password is only ever filled into a field of the site it belongs to, and the frame's
        own address decides that, not the page's: a ref inside an iframe names the iframe.
        """
        field = await self._element(page, ref)
        try:
            handle = await field.element_handle(timeout=self._action_ms)
            frame = await handle.owner_frame()
            seen = await field.evaluate(
                "e => e instanceof HTMLInputElement ? e.type : ''", timeout=self._action_ms
            )
        except PlaywrightError as exc:
            raise self._failure(
                exc, page, "look at", f"element {ref}", ref=ref, verb="checked"
            ) from exc
        if account_site(frame.url if frame else "") != site:
            raise page_failure("wrong_site", ref=ref, site=site)
        return field, seen

    async def _fill_secretly(self, field: Locator, value: str) -> AgentBrowserFailure | None:
        """Fill ``value`` into the field and read it back; how that failed, or None.

        The driver's error for a failed fill quotes the value in its call log, so none is kept,
        chained or logged here, and the verdict is taken outside the handler that caught it.
        """
        echoed: str | None = None
        failed = False
        try:
            await field.fill(value, timeout=self._action_ms)
            echoed = await field.input_value(timeout=self._action_ms)
        except asyncio.CancelledError:
            raise
        except Exception:
            failed = True
        if failed:
            return "disconnected" if browser_lost(self._browser) else "not_actionable"
        return None if echoed == value else "field_rejected"

    async def _clear(self, fields: Sequence[Locator]) -> None:
        """Empty fields a failed call had filled, so no half-filled form is left to submit."""
        for field in fields:
            with suppress(PlaywrightError):
                await field.fill("", timeout=self._action_ms)

    async def _holds_a_password(self, page: Page) -> bool:
        """Whether a frame of the page holds a filled password in a field or in its visible text,
        or cannot be read to say.

        A browser draws a password field as dots, but that is the browser's decision and the
        page's to change: a script can turn the field into a text field. So every input and
        textarea is read, password fields included, along with the frame's visible text, and
        compared here, so no password is ever sent into a page. A filled form is screenshotted
        after it is submitted, not before.
        """
        try:
            async with asyncio.timeout(self._limits.action_timeout):
                for frame in page.frames:
                    values = await frame.locator("input, textarea").evaluate_all(
                        "els => els.map(e => e.value)"
                    )
                    text = await frame.locator("body").inner_text()
                    if any(self._passwords.found_in(value) for value in values):
                        return True
                    if self._passwords.found_in(text):
                        return True
        except Exception:
            return True
        return False

    def _waiting_dialogs(self) -> dict[Page, PageDialog]:
        """The dialog each open page is stopped on, by page; one dies with the page it stopped."""
        for page in [page for page in self._waiting_dialog if page.is_closed()]:
            self._waiting.pop(page, None)
            self._waiting_dialog.pop(page, None)
        return self._waiting_dialog

    def _take_dialogs(self) -> tuple[tuple[PageDialog, ...], int]:
        """The bound's dialogs, then the dialog each page is stopped on, and what the bound cut.

        A waiting dialog travels with every call until a call answers it: what stopped the
        page is why this call did nothing, and the model reads that rather than a page that
        silently stayed put. It is never one the bound left out, whatever else was.
        """
        dialogs = self._dialogs
        omitted, self._omitted_dialogs = self._omitted_dialogs, 0
        self._dialogs = []
        dialogs.extend(self._waiting_dialogs().values())
        return tuple(dialogs), omitted

    async def _deliver(
        self, events: PageEvents, page: Page | None, *, settled: bool | None = None
    ) -> PageEvents:
        """Add what the pages did besides the call: the files they downloaded, the dialogs
        they showed, the status their navigation answered, and whether they went quiet.

        A page can start downloads without end, so the refusals are bounded like the dialogs:
        what the model reads is at most ``MAX_REFUSALS_PER_CALL`` of them, with the count of
        the rest in ``PageEvents.omitted_refusals``.
        """
        downloads, self._downloads = self._downloads, []
        dialogs, omitted = self._take_dialogs()
        saved: list[DownloadedFile] = []
        refused: list[DownloadRefusal] = []
        omitted_refusals = 0
        for index, download in enumerate(downloads):
            if index >= MAX_DOWNLOADS_PER_CALL:
                await _discard(download)
                refusal = DownloadRefusal(self._filename(download), "limit")
            else:
                outcome = await self._save(download)
                if isinstance(outcome, DownloadedFile):
                    saved.append(outcome)
                    continue
                refusal = outcome
            if len(refused) < MAX_REFUSALS_PER_CALL:
                refused.append(refusal)
            else:
                omitted_refusals += 1
        return replace(
            events,
            downloads=tuple(saved),
            refused_downloads=tuple(refused),
            omitted_refusals=omitted_refusals,
            dialogs=dialogs,
            omitted_dialogs=omitted,
            http_status=None if page is None else self._statuses.get(page),
            settled=settled,
        )

    async def _save(self, download: Download) -> DownloadedFile | DownloadRefusal:
        """Copy a download over the protocol, stopping once it is over the limit.

        The copy lives in a directory of its own for as long as this takes, and the pool's
        copy is deleted whatever the outcome. A copy this process cannot make, write, or read
        is a refusal like any other, so the call's other downloads still count.
        """
        name = self._filename(download)
        try:
            directory = Path(tempfile.mkdtemp(prefix=_DOWNLOAD_DIRECTORY_PREFIX))
        except OSError:
            await _discard(download)
            return DownloadRefusal(name, "failed")
        target = directory / "download"
        copy = asyncio.ensure_future(download.save_as(target))
        try:
            try:
                async with asyncio.timeout(self._nav_seconds):
                    while not copy.done():
                        if _size(target) > self._limits.max_download_bytes:
                            return DownloadRefusal(name, "too_large")
                        await asyncio.wait({copy}, timeout=_DOWNLOAD_POLL_SECONDS)
                    copy.result()
                if _size(target) > self._limits.max_download_bytes:
                    return DownloadRefusal(name, "too_large")
                return DownloadedFile(
                    self._passwords.redact(download.url),
                    name,
                    await asyncio.to_thread(
                        lambda: self._passwords.redact_bytes(target.read_bytes())
                    ),
                )
            except TimeoutError:
                return DownloadRefusal(name, "timeout")
            except PlaywrightError, OSError:
                return DownloadRefusal(name, "failed")
        finally:
            copy.cancel()
            await asyncio.gather(copy, return_exceptions=True)
            shutil.rmtree(directory, ignore_errors=True)
            await _discard(download)

    def _filename(self, download: Download) -> str:
        return self._passwords.redact(download.suggested_filename)

    # -- failures ----------------------------------------------------------------------------

    @property
    def _nav_seconds(self) -> float:
        return self._limits.navigation_timeout

    @property
    def _action_ms(self) -> float:
        return milliseconds(self._limits.action_timeout)

    async def _action_failed(
        self,
        exc: PlaywrightError,
        page: Page,
        action: str,
        target: str,
        *,
        ref: str | None = None,
        verb: str = "",
        adopted: bool = False,
        expecting_events: bool = False,
    ) -> PageObservation:
        """The driver error of a call whose own action closed its page.

        A control that closes its page (a Close button, a popup that closes itself) lands
        the action and leaves the driver reading a page that is gone, which a slow machine
        sees as an error where a fast one sees a plain result. What the call left is what
        an earlier call left, so it answers like any other call: the page that is active
        again, or none. An action that stopped on a dialog nobody answered is the same
        kind of result, and answers with the dialog the page is waiting on.
        """
        if self._waiting_on(page):
            return await self._after(page, adopted=adopted, expecting_events=expecting_events)
        if page.is_closed() and not browser_lost(self._browser):
            return await self._after(page, adopted=adopted, expecting_events=expecting_events)
        raise self._failure(exc, page, action, target, ref=ref, verb=verb) from exc

    def _failure(
        self,
        exc: PlaywrightError,
        page: Page | None,
        action: str,
        target: str,
        *,
        timed_out: AgentBrowserError | None = None,
        ref: str | None = None,
        verb: str = "",
    ) -> AgentBrowserError:
        """The sentence the model reads for a driver failure, without the driver's own text.

        A browser that is gone loses every page and a page that is gone loses its call.
        Running out of time means what the call says it means: ``timed_out`` where it has
        an answer, and for an element action that the element ``ref`` was not ``verb``.
        """
        if browser_lost(self._browser):
            logger.warning("Agent Browser disconnected during a call (%s)", type(exc).__name__)
            return page_failure("disconnected")
        if page is not None and page.is_closed():
            return page_failure("page_closed")
        if isinstance(exc, PlaywrightTimeoutError):
            if timed_out is not None:
                return timed_out
            if ref is not None:
                return page_failure(
                    "not_actionable", ref=ref, verb=verb, seconds=self._limits.action_timeout
                )
        elif (named := network_failure(exc)) is not None:
            return named
        logger.warning("Agent Browser call failed (%s): %s", type(exc).__name__, action)
        return page_failure("action_failed", action=action, target=target, detail=self._detail(exc))

    def _detail(self, exc: PlaywrightError) -> str:
        """The first line of what the driver said, without its call log, and without a password
        a URL or a message in it may quote."""
        head = str(exc).split("Call log:", 1)[0].strip()
        first = self._passwords.redact(head.splitlines()[0]) if head else type(exc).__name__
        return elide(first.rstrip("."), _DETAIL_CHARS)


def _verdict_answer(answer: DialogVerdict) -> Literal["accepted", "dismissed"]:
    """How one dialog the call answered with ``answer`` is reported to the model."""
    return "accepted" if answer == "accept" else "dismissed"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _js(value: object) -> str:
    """A JSON literal for the restore script, which no value of a page can break out of."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


async def _discard(download: Download) -> None:
    """Free the pool's copy of a download, and stop it if it is still coming."""
    with suppress(PlaywrightError):
        await download.cancel()
    with suppress(PlaywrightError):
        await download.delete()


async def _close_context(context: BrowserContext) -> None:
    try:
        async with asyncio.timeout(CLOSE_SECONDS):
            await context.close()
    except Exception:
        logger.warning("Failed to close an Agent Browser context", exc_info=True)


__all__ = ["PlaywrightAgentPage"]
