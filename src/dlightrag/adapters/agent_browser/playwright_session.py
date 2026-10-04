# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One leased Playwright browser: the page renders it gives and the sessions it hosts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from playwright.async_api import Browser, BrowserContext, Page, Response
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from dlightrag.adapters.agent_browser.driver import (
    CLOSE_SECONDS,
    NETWORK_ERROR,
    milliseconds,
    page_content,
)
from dlightrag.adapters.agent_browser.interactive import PlaywrightBrowserSession
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    BrowserSession,
    InteractiveLimits,
    RenderedPage,
    browser_failure,
    interactive_failure,
)

logger = logging.getLogger(__name__)


async def new_agent_context(browser: Browser, *, accept_downloads: bool) -> BrowserContext:
    """Make an anonymous context, the one way an Agent Browser context is made.

    It passes no proxy of its own, so the context inherits the launch proxy that is the
    pool's only way out, and the connection never exposes the application's network to
    the browser (ADR 0032). It starts empty and lets no page register a worker; it keeps
    downloads only where a session reads them.
    """
    return await browser.new_context(accept_downloads=accept_downloads, service_workers="block")


class PlaywrightLeasedBrowser:
    """A Run's browser: each render is a temporary context, each Agent Session one of its own."""

    def __init__(self, browser: Browser, release: Callable[[], Awaitable[None]]) -> None:
        self._browser = browser
        self._release = release
        self._closed = False

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
            if settle_timeout > 0:
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=milliseconds(settle_timeout)
                    )
                except PlaywrightTimeoutError:
                    # A page that never goes quiet is still read as it stands.
                    pass
            html = await page_content(page, navigation_timeout)
            status = statuses[-1] if statuses else None
            if status is not None and status >= 400:
                raise browser_failure("http_status", status=status)
            return RenderedPage(
                requested_url=url,
                final_url=page.url,
                html=html.encode("utf-8", errors="replace"),
                status=status,
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

    async def open_session(self, limits: InteractiveLimits) -> BrowserSession:
        """Open an Agent Session's context, which lives until the session closes."""
        if not self._browser.is_connected():
            raise interactive_failure("disconnected")
        try:
            context = await new_agent_context(self._browser, accept_downloads=True)
            return await PlaywrightBrowserSession.open(self._browser, context, limits)
        except PlaywrightError as exc:
            if not self._browser.is_connected():
                raise interactive_failure("disconnected") from exc
            logger.warning("Agent Browser failed to open a page (%s)", type(exc).__name__)
            raise interactive_failure(
                "action_failed", action="open", target="a page", detail=type(exc).__name__
            ) from exc

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
        if not self._browser.is_connected():
            return browser_failure("disconnected")
        if isinstance(exc, PlaywrightTimeoutError):
            return browser_failure("timeout", seconds=navigation_timeout)
        text = str(exc)
        if "Download is starting" in text:
            return browser_failure("download")
        token = NETWORK_ERROR.search(text)
        return browser_failure("navigation_failed", detail=token.group() if token else None)


def _note_navigation(page: Page, response: Response, statuses: list[int]) -> None:
    """Record each main-frame navigation answer; the last one is the page's status."""
    if response.request.is_navigation_request() and response.frame == page.main_frame:
        statuses.append(response.status)


__all__ = ["PlaywrightLeasedBrowser", "new_agent_context"]
