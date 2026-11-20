# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One leased Playwright browser: the renders it gives and the Agent Pages it hosts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from playwright.async_api import Browser, BrowserContext, Page, Response
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from dlightrag.adapters.agent_browser.driver import (
    CLOSE_SECONDS,
    browser_lost,
    milliseconds,
    navigation_status,
    network_failure,
    page_content,
    proxy_page_refusal,
    starts_download,
)
from dlightrag.adapters.agent_browser.page import PlaywrightAgentPage
from dlightrag.engine.answer.agent_browser.block import (
    refused_on_navigation,
)
from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    AgentPage,
    PageLimits,
    RenderedPage,
    SessionState,
    browser_failure,
    page_failure,
    refusal_failure,
)
from dlightrag.engine.answer.agent_browser.passwords import (
    FilledPasswords,
)

logger = logging.getLogger(__name__)


async def new_agent_context(browser: Browser, *, accept_downloads: bool) -> BrowserContext:
    """Make an anonymous context, the one way an Agent Browser context is made.

    It passes no proxy of its own, so the context inherits the launch proxy that is the
    pool's only way out, and the connection never exposes the application's network to
    the browser (ADR 0032). It starts empty and lets no page register a worker; it keeps
    downloads only where an Agent Page reads them.
    """
    return await browser.new_context(accept_downloads=accept_downloads, service_workers="block")


class PlaywrightLeasedBrowser:
    """A Run's browser: each render is a temporary context, each Agent Session one of its own."""

    def __init__(
        self, browser: Browser, release: Callable[[], Awaitable[None]], *, headless: bool = True
    ) -> None:
        self._browser = browser
        self._release = release
        self._headless = headless
        self._closed = False
        #: The Agent Page opened last, which a live session watches; it ends with its context.
        self._current: PlaywrightAgentPage | None = None

    @property
    def headless(self) -> bool:
        """Whether this browser runs without a display, as the lease that was taken asked."""
        return self._headless

    async def render(
        self, url: str, *, navigation_timeout: float, settle_timeout: float
    ) -> RenderedPage:
        # An anonymous context starts empty and ends with the render: no cookie or storage
        # survives it, and it neither keeps downloads nor lets a page register a worker.
        try:
            context = await new_agent_context(self._browser, accept_downloads=False)
        except PlaywrightError as exc:
            raise self._failure(exc, navigation_timeout) from exc
        try:
            page = await context.new_page()
            statuses: list[int] = []
            page.on("response", lambda response: _note_navigation(page, response, statuses))
            await page.goto(url, wait_until="load", timeout=milliseconds(navigation_timeout))
            settled = True
            if settle_timeout > 0:
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=milliseconds(settle_timeout)
                    )
                except PlaywrightTimeoutError:
                    # A page that never goes quiet is still read as it stands, and the
                    # rendering says it had not settled so a partial read is not taken for
                    # the whole page.
                    settled = False
            html = await page_content(page, navigation_timeout)
            status = statuses[-1] if statuses else None
            body = html.encode("utf-8", errors="replace")
            if (proxy := proxy_page_refusal(status, body)) is not None:
                # Our own egress boundary is named before any page's refusal: the proxy's
                # error page is not the site, and it is not a bot filter's either.
                raise proxy
            if (block := refused_on_navigation(status, body)) is not None:
                # A refusal is named before anything is read: a filter's own page is not a page.
                raise refusal_failure(block)
            # A page that answered 403, 404, 429 or 503 is a page: its status travels on the
            # representation and the caller reads the body it answered with. The interactive
            # path shows the same page with the same status.
            return RenderedPage(
                requested_url=url,
                final_url=page.url,
                html=body,
                status=status,
                headless=self._headless,
                settled=settled,
            )
        except AgentBrowserError:
            raise
        except PlaywrightError as exc:
            raise self._failure(exc, navigation_timeout) from exc
        finally:
            try:
                async with asyncio.timeout(CLOSE_SECONDS):
                    await context.close()
            except PlaywrightError:
                # The context went with its browser, or the browser is gone.
                pass
            except TimeoutError:
                # A browser that does not answer a close is wedged, and the Run must not
                # lease it again for its next render, whatever this one found.
                logger.warning("Failed to close an Agent Browser context in time")
                raise browser_failure("disconnected") from None

    async def open_page(self, limits: PageLimits, passwords: FilledPasswords) -> AgentPage:
        """Open an Agent Session's Agent Page, which lives until it is closed."""
        if browser_lost(self._browser):
            raise page_failure("disconnected")
        try:
            # The Run holds its browser's lock while this waits, so a browser that does not
            # answer must not hold every other page, render, and settlement with it.
            async with asyncio.timeout(CLOSE_SECONDS):
                context = await new_agent_context(self._browser, accept_downloads=True)
                page = await PlaywrightAgentPage.open(self._browser, context, limits, passwords)
                self._current = page
                return page
        except TimeoutError:
            # A browser that does not answer is wedged, and the Run must not lease it again.
            logger.warning("Failed to open an Agent Browser page in time")
            raise page_failure("disconnected") from None
        except PlaywrightError as exc:
            if browser_lost(self._browser):
                raise page_failure("disconnected") from exc
            logger.warning("Agent Browser failed to open a page (%s)", type(exc).__name__)
            raise page_failure(
                "action_failed", action="open", target="a page", detail=type(exc).__name__
            ) from exc

    def live_page(self) -> Page | None:
        """The page this browser is driving for its Run now, for a live session to show.

        It is the Agent Page opened last, which is the one the Run is working on; a page
        whose context has closed is no page at all.
        """
        page = self._current
        return None if page is None else page.active_page()

    async def capture_session(self) -> SessionState:
        """The session of the page this browser drives, cookies and all; none when it drives none.

        The cookies are the context's, which the page reaches; the URL and the web storage
        are the page's own. A browser driving no page has nothing a Run was working on.
        """
        page = self._current
        return SessionState(url="") if page is None else await page.capture_session()

    async def restore_session(self, session: SessionState) -> None:
        """Put the carried-over cookies and storage into the page this browser is opening.

        Both land before any navigation the new browser makes, so the page's own scripts see
        them. A cookie is put back the way it was: an ``httpOnly`` one stays out of scripts.
        """
        if self._current is not None:
            await self._current.restore_session(session)

    async def aclose(self) -> None:
        """Disconnect the browser, then release its lease; failures are logged, not raised."""
        if self._closed:
            return
        self._closed = True
        try:
            async with asyncio.timeout(CLOSE_SECONDS):
                await self._browser.close()
        except Exception:
            logger.warning("Failed to disconnect an Agent Browser", exc_info=True)
        finally:
            try:
                await self._release()
            except Exception:
                logger.warning("Failed to release an Agent Browser lease", exc_info=True)

    def _failure(self, exc: PlaywrightError, navigation_timeout: float) -> AgentBrowserError:
        """The model-safe reason a render failed, without the driver's own text."""
        if browser_lost(self._browser):
            return browser_failure("disconnected")
        if isinstance(exc, PlaywrightTimeoutError):
            return browser_failure("timeout", seconds=navigation_timeout)
        if starts_download(exc):
            return browser_failure("download")
        return network_failure(exc) or browser_failure("navigation_failed")


def _note_navigation(page: Page, response: Response, statuses: list[int]) -> None:
    """Record each main-frame navigation answer; the last one is the page's status."""
    if (status := navigation_status(page, response)) is not None:
        statuses.append(status)


__all__ = ["PlaywrightLeasedBrowser"]
