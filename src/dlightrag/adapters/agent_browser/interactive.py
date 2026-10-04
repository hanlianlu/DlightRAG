# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Agent Session's browser context: its pages, and what it does on the active one."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Literal

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
    NETWORK_ERROR,
    milliseconds,
    page_content,
)
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    DownloadedFile,
    DownloadRefusal,
    FoundElements,
    InteractiveLimits,
    PageCapture,
    PageEvents,
    PageObservation,
    PageScreenshot,
    PageState,
    UploadFile,
    browser_failure,
    interactive_failure,
)

logger = logging.getLogger(__name__)

#: A popup's page event, or a link's download, reaches the client a few milliseconds after
#: the click that caused it returns (25 ms measured). A call that may have caused one waits
#: this long for it before it reports what the pages did.
_EVENT_GRACE_SECONDS = 0.2
_MAX_DOWNLOADS = 4
_MAX_DIALOGS = 5
_DOWNLOAD_POLL_SECONDS = 0.1
_DOWNLOAD_DIRECTORY_PREFIX = "dlightrag-browser-download-"
_TITLE_CHARS = 200
#: How much of a dialog's message, or of the text a wait names, a sentence repeats.
_QUOTED_CHARS = 200
_FOUND_LINE_CHARS = 300
_DETAIL_CHARS = 300
#: A scroll moves the page by this share of the viewport's height.
_SCROLL_SHARE = 0.8
_DEFAULT_VIEWPORT_HEIGHT = 720


class PlaywrightBrowserSession:
    """An anonymous context of the Run's browser with one active page.

    A popup or a new tab becomes the active page once the call that opened it has acted,
    so a ref always acts on the page it came from. A download, a dialog, and the answer of
    a navigation are noted as they happen and reported by the next call that returns.
    """

    def __init__(
        self, browser: Browser, context: BrowserContext, page: Page, limits: InteractiveLimits
    ) -> None:
        self._browser = browser
        self._context = context
        self._limits = limits
        #: The open pages, the oldest first; the last is the active page.
        self._pages: list[Page] = [page]
        #: Popups opened since a call last returned, which no call has adopted yet.
        self._popups: list[Page] = []
        self._downloads: list[Download] = []
        self._dialogs: list[str] = []
        #: The last main-frame navigation answer of each page during the current call.
        self._statuses: dict[Page, int] = {}
        self._watch(page)

    @classmethod
    async def open(
        cls, browser: Browser, context: BrowserContext, limits: InteractiveLimits
    ) -> PlaywrightBrowserSession:
        """Start a session in ``context``, which it owns from then on, with a blank page."""
        try:
            page = await context.new_page()
        except BaseException:
            await _close_context(context)
            raise
        return cls(browser, context, page, limits)

    def current_url(self) -> str | None:
        pages = self._open_pages()
        return pages[-1].url if pages else None

    async def navigate(self, url: str) -> PageObservation:
        page = await self._page_to_navigate()
        # A URL that answers with a file starts a download instead of loading a page.
        downloading = False
        try:
            await page.goto(url, wait_until="load", timeout=milliseconds(self._nav_seconds))
        except PlaywrightError as exc:
            downloading = "Download is starting" in str(exc)
            if not downloading:
                raise self._failure(
                    exc,
                    page,
                    "navigate to",
                    "the page",
                    timed_out=browser_failure("timeout", seconds=self._nav_seconds),
                ) from exc
        return await self._after(page, expecting_events=downloading)

    async def back(self) -> PageObservation:
        page = self._begin()
        before = page.url
        try:
            response = await page.go_back(
                wait_until="load", timeout=milliseconds(self._nav_seconds)
            )
        except PlaywrightError as exc:
            raise self._failure(
                exc,
                page,
                "go back from",
                "the page",
                timed_out=browser_failure("timeout", seconds=self._nav_seconds),
            ) from exc
        # A step back inside the same document answers nothing, and still moved the page.
        if response is None and page.url == before:
            raise interactive_failure("no_history")
        return await self._after(page)

    async def snapshot(self) -> PageObservation:
        page, events = self._observing()
        return await self._observation(page, events, settle=events.new_page)

    async def find(self, query: str, *, limit: int) -> FoundElements:
        page, events = self._observing()
        tree = await self._tree(page, depth=None, where="the page")
        needle = query.casefold()
        matches = [line.strip() for line in tree.splitlines() if needle in line.casefold()]
        return FoundElements(
            page=await self._state(page, "the page"),
            events=await self._deliver(events, page),
            lines=tuple(line[:_FOUND_LINE_CHARS] for line in matches[:limit]),
            total=len(matches),
        )

    async def wait(
        self, *, text: str | None, text_gone: str | None, seconds: float | None
    ) -> PageObservation:
        page = self._begin()
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
                raise self._failure(
                    exc,
                    page,
                    "wait for",
                    "the text",
                    timed_out=interactive_failure(
                        "wait_timeout",
                        text=" ".join(awaited.split())[:_QUOTED_CHARS],
                        change="appear" if appears else "disappear",
                        seconds=self._nav_seconds,
                    ),
                ) from exc
        return await self._after(page)

    async def click(self, ref: str) -> PageObservation:
        page = self._begin()
        element = await self._element(page, ref)
        try:
            await element.click(timeout=self._action_ms)
        except PlaywrightError as exc:
            raise self._failure(
                exc, page, "click", f"element {ref}", ref=ref, verb="clicked"
            ) from exc
        return await self._after(page, expecting_events=True)

    async def type_text(self, ref: str, text: str, *, submit: bool) -> PageObservation:
        page = self._begin()
        element = await self._element(page, ref)
        try:
            await element.fill(text, timeout=self._action_ms)
            if submit:
                await element.press("Enter", timeout=self._action_ms)
        except PlaywrightError as exc:
            raise self._failure(
                exc, page, "type into", f"element {ref}", ref=ref, verb="filled"
            ) from exc
        return await self._after(page, expecting_events=submit)

    async def select(self, ref: str, values: tuple[str, ...]) -> PageObservation:
        page = self._begin()
        element = await self._element(page, ref)
        try:
            await element.select_option(list(values), timeout=self._action_ms)
        except PlaywrightError as exc:
            raise self._failure(
                exc, page, "select an option of", f"element {ref}", ref=ref, verb="selected"
            ) from exc
        return await self._after(page, expecting_events=True)

    async def press(self, key: str, *, ref: str | None) -> PageObservation:
        page = self._begin()
        element = None if ref is None else await self._element(page, ref)
        try:
            if element is None:
                await page.keyboard.press(key)
            else:
                await element.press(key, timeout=self._action_ms)
        except PlaywrightError as exc:
            if "Unknown key" in str(exc):
                raise interactive_failure("invalid_key", key=key) from exc
            raise self._failure(
                exc,
                page,
                "press a key on",
                "the page" if ref is None else f"element {ref}",
                ref=ref,
                verb="pressed",
            ) from exc
        return await self._after(page, expecting_events=True)

    async def scroll(self, *, direction: Literal["up", "down"], ref: str | None) -> PageObservation:
        page = self._begin()
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
            raise self._failure(
                exc,
                page,
                "scroll",
                "the page" if ref is None else f"element {ref}",
                ref=ref,
                verb="scrolled",
            ) from exc
        return await self._after(page)

    async def upload(self, ref: str, files: tuple[UploadFile, ...]) -> PageObservation:
        page = self._begin()
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
                    raise interactive_failure("not_file_input", ref=ref) from None
        except PlaywrightError as exc:
            raise self._failure(
                exc, page, "upload files to", f"element {ref}", ref=ref, verb="given files"
            ) from exc
        return await self._after(page, expecting_events=True)

    async def screenshot(self, *, full_page: bool) -> PageScreenshot:
        page, events = self._observing()
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
            events=await self._deliver(events, page),
            png=png,
        )

    async def capture(self) -> PageCapture:
        page, events = self._observing()
        try:
            html = await page_content(page, self._nav_seconds)
        except PlaywrightError as exc:
            raise self._failure(exc, page, "capture", "the page") from exc
        return PageCapture(
            page=await self._state(page, "the page"),
            events=await self._deliver(events, page),
            html=html.encode("utf-8", errors="replace"),
        )

    async def aclose(self) -> None:
        await _close_context(self._context)

    # -- the pages ------------------------------------------------------------------------

    def _watch(self, page: Page) -> None:
        """Note what a page does besides the calls: its popups, downloads, dialogs, answers."""
        page.on("popup", self._on_popup)
        page.on("download", lambda download: self._downloads.append(download))
        page.on("dialog", self._on_dialog)
        page.on("response", lambda response: self._on_response(page, response))

    def _on_popup(self, popup: Page) -> None:
        self._popups.append(popup)
        self._watch(popup)

    def _on_response(self, page: Page, response: Response) -> None:
        if response.request.is_navigation_request() and response.frame == page.main_frame:
            self._statuses[page] = response.status

    async def _on_dialog(self, dialog: Dialog) -> None:
        """Answer a dialog at once, because its page waits for it, and keep a line about it."""
        # A prompt asks for text the model never gave. Every other dialog completes what a
        # call set in motion, and refusing it would silently undo that.
        accepted = dialog.type != "prompt"
        message = " ".join(dialog.message.split())[:_QUOTED_CHARS]
        try:
            await (dialog.accept() if accepted else dialog.dismiss())
        except PlaywrightError:
            # The page closed with its dialog open.
            return
        verdict = "accepted" if accepted else "dismissed"
        self._dialogs.append(f'The page showed a {dialog.type} dialog: "{message}" ({verdict}).')

    def _open_pages(self) -> list[Page]:
        return [page for page in self._pages if not page.is_closed()]

    def _begin(self) -> Page:
        """The active page a call starts on. The answers noted for the last call are forgotten."""
        self._statuses.clear()
        self._pages = self._open_pages()
        if not self._pages:
            # Every page closes with a browser that is gone.
            raise interactive_failure(
                "page_closed" if self._browser.is_connected() else "disconnected"
            )
        return self._pages[-1]

    async def _page_to_navigate(self) -> Page:
        """The active page, or a fresh one when the last page closed."""
        self._statuses.clear()
        self._pages = self._open_pages()
        if self._pages:
            return self._pages[-1]
        try:
            page = await self._context.new_page()
        except PlaywrightError as exc:
            raise self._failure(exc, None, "open", "a page") from exc
        self._pages.append(page)
        self._watch(page)
        return page

    def _observing(self) -> tuple[Page, PageEvents]:
        """The page a call that only looks at the page observes, and what changed since.

        A popup that opened since the last call is that page: nothing was asked of the old
        one, so there is no ref to keep acting on it.
        """
        events = self._adopt(self._begin())
        return self._pages[-1], events

    def _adopt(self, acted: Page) -> PageEvents:
        """Make the newest popup the active page, or note that ``acted`` closed."""
        popups = [popup for popup in self._popups if not popup.is_closed()]
        self._popups.clear()
        self._pages = [*self._open_pages(), *popups]
        if popups:
            return PageEvents(new_page=True)
        if acted.is_closed():
            return PageEvents(returned=True) if self._pages else PageEvents(closed=True)
        return PageEvents()

    # -- what a call returns -------------------------------------------------------------------

    async def _after(self, acted: Page, *, expecting_events: bool = False) -> PageObservation:
        """What an acting call left, once its page has settled.

        ``expecting_events`` is for a call that may have opened a popup or started a
        download, which the pages announce a moment after the action returns.
        """
        if expecting_events:
            await asyncio.sleep(_EVENT_GRACE_SECONDS)
        events = self._adopt(acted)
        if events.closed:
            return PageObservation(None, await self._deliver(events, None), None)
        return await self._observation(self._pages[-1], events, settle=True, acted=True)

    async def _observation(
        self, page: Page, events: PageEvents, *, settle: bool, acted: bool = False
    ) -> PageObservation:
        """The page's state and snapshot, and what the pages did around the call.

        A page the call opened is still loading, so ``settle`` waits for it first. After an
        action ``acted`` says in a failure that the action itself completed.
        """
        if settle:
            await self._settle(page)
        where = "the page (the action itself completed)" if acted else "the page"
        state = await self._state(page, where)
        tree = await self._tree(page, depth=self._limits.snapshot_depth, where=where)
        return PageObservation(state, await self._deliver(events, page), tree)

    async def _settle(self, page: Page) -> None:
        """Wait, within one budget, for the page to load and go quiet.

        A page that never does is read as it stands.
        """
        deadline = time.monotonic() + self._limits.settle_timeout
        for state in ("domcontentloaded", "networkidle"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            with suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state(state, timeout=milliseconds(remaining))

    async def _state(self, page: Page, where: str) -> PageState:
        try:
            title = await page.title()
        except PlaywrightError as exc:
            raise self._failure(exc, page, "read the title of", where) from exc
        return PageState(url=page.url, title=" ".join(title.split())[:_TITLE_CHARS])

    async def _tree(self, page: Page, *, depth: int | None, where: str) -> str:
        try:
            return await page.aria_snapshot(mode="ai", depth=depth, timeout=self._action_ms)
        except PlaywrightError as exc:
            raise self._failure(exc, page, "take a snapshot of", where) from exc

    async def _element(self, page: Page, ref: str) -> Locator:
        """The element a ref names in the page's latest snapshot."""
        element = page.locator(f"aria-ref={ref}")
        try:
            found = await element.count()
        except PlaywrightError as exc:
            raise self._failure(exc, page, "look for", f"element {ref}") from exc
        if found == 0:
            raise interactive_failure("stale_ref", ref=ref)
        return element

    async def _deliver(self, events: PageEvents, page: Page | None) -> PageEvents:
        """Add what the pages did besides the call: the files they downloaded, the dialogs
        they showed, and the status their navigation answered."""
        downloads, self._downloads = self._downloads, []
        dialogs, self._dialogs = self._dialogs[:_MAX_DIALOGS], []
        saved: list[DownloadedFile] = []
        refused: list[DownloadRefusal] = []
        for index, download in enumerate(downloads):
            if index >= _MAX_DOWNLOADS:
                await _discard(download)
                refused.append(DownloadRefusal(download.suggested_filename, "limit"))
                continue
            outcome = await self._save(download)
            if isinstance(outcome, DownloadedFile):
                saved.append(outcome)
            else:
                refused.append(outcome)
        return replace(
            events,
            downloads=tuple(saved),
            refused_downloads=tuple(refused),
            dialogs=tuple(dialogs),
            http_status=None if page is None else self._statuses.get(page),
        )

    async def _save(self, download: Download) -> DownloadedFile | DownloadRefusal:
        """Copy a download over the protocol, stopping once it is over the limit.

        The copy lives in a directory of its own for as long as this takes, and the pool's
        copy is deleted whatever the outcome.
        """
        name = download.suggested_filename
        directory = Path(tempfile.mkdtemp(prefix=_DOWNLOAD_DIRECTORY_PREFIX))
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
            except TimeoutError:
                return DownloadRefusal(name, "timeout")
            except PlaywrightError:
                return DownloadRefusal(name, "failed")
            if _size(target) > self._limits.max_download_bytes:
                return DownloadRefusal(name, "too_large")
            return DownloadedFile(download.url, name, await asyncio.to_thread(target.read_bytes))
        finally:
            copy.cancel()
            await asyncio.gather(copy, return_exceptions=True)
            shutil.rmtree(directory, ignore_errors=True)
            await _discard(download)

    # -- failures ----------------------------------------------------------------------------

    @property
    def _nav_seconds(self) -> float:
        return self._limits.navigation_timeout

    @property
    def _action_ms(self) -> float:
        return milliseconds(self._limits.action_timeout)

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
        if not self._browser.is_connected():
            logger.warning("Agent Browser disconnected during a call (%s)", type(exc).__name__)
            return interactive_failure("disconnected")
        if page is not None and page.is_closed():
            return interactive_failure("page_closed")
        if isinstance(exc, PlaywrightTimeoutError):
            if timed_out is not None:
                return timed_out
            if ref is not None:
                return interactive_failure(
                    "not_actionable", ref=ref, verb=verb, seconds=self._limits.action_timeout
                )
        elif token := NETWORK_ERROR.search(str(exc)):
            return browser_failure("navigation_failed", detail=token.group())
        logger.warning("Agent Browser call failed (%s): %s", type(exc).__name__, action)
        return interactive_failure(
            "action_failed", action=action, target=target, detail=_detail(exc)
        )


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _detail(exc: PlaywrightError) -> str:
    """The first line of what the driver said, without its call log."""
    head = str(exc).split("Call log:", 1)[0].strip()
    first = head.splitlines()[0] if head else type(exc).__name__
    return first[:_DETAIL_CHARS].rstrip(".")


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


__all__ = ["PlaywrightBrowserSession"]
