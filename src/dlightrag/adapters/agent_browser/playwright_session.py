# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One leased Playwright browser, and the page renders it gives."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

from playwright.async_api import Browser, Page, Response
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    BrowserSandbox,
    RenderedPage,
    browser_failure,
)

logger = logging.getLogger(__name__)

#: The only part of a navigation failure the model reads: Chromium's network error token.
_NETWORK_ERROR = re.compile(r"net::ERR_[A-Z0-9_]+")
#: Disconnecting a browser that does not answer must not hold a Run's settlement open.
_CLOSE_SECONDS = 10.0


class PlaywrightLeasedBrowser:
    """A Run's browser: each render is a temporary anonymous context of its own."""

    def __init__(
        self,
        browser: Browser,
        release: Callable[[], Awaitable[None]],
        *,
        sandbox: BrowserSandbox,
    ) -> None:
        self._browser = browser
        self._release = release
        self._sandbox: BrowserSandbox = sandbox
        self._closed = False

    @property
    def sandbox(self) -> BrowserSandbox:
        return self._sandbox

    async def render(
        self, url: str, *, navigation_timeout: float, settle_timeout: float, max_bytes: int
    ) -> RenderedPage:
        # An anonymous context starts empty and ends with the render: no cookie or storage
        # survives it, and it neither keeps downloads nor lets a page register a worker.
        try:
            context = await self._browser.new_context(
                accept_downloads=False, service_workers="block"
            )
        except PlaywrightError as exc:
            raise self._failure(exc, navigation_timeout) from exc
        try:
            page = await context.new_page()
            statuses: list[int] = []
            page.on("response", lambda response: _note_navigation(page, response, statuses))
            await page.goto(url, wait_until="load", timeout=_milliseconds(navigation_timeout))
            if settle_timeout > 0:
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=_milliseconds(settle_timeout)
                    )
                except PlaywrightTimeoutError:
                    # A page that never goes quiet is still read as it stands.
                    pass
            html = await _content(page, navigation_timeout)
            status = statuses[-1] if statuses else None
            if status is not None and status >= 400:
                raise browser_failure("http_status", status=status)
            encoded = html.encode("utf-8", errors="replace")
            if len(encoded) > max_bytes:
                raise browser_failure("too_large", limit=max_bytes)
            return RenderedPage(requested_url=url, final_url=page.url, html=encoded, status=status)
        except AgentBrowserError:
            raise
        except PlaywrightError as exc:
            raise self._failure(exc, navigation_timeout) from exc
        finally:
            try:
                await context.close()
            except PlaywrightError:
                # The context went with its browser, or the browser is gone.
                pass

    async def aclose(self) -> None:
        """Disconnect the browser, then release its lease; failures are logged, not raised."""
        if self._closed:
            return
        self._closed = True
        try:
            async with asyncio.timeout(_CLOSE_SECONDS):
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
        token = _NETWORK_ERROR.search(text)
        return browser_failure("navigation_failed", detail=token.group() if token else None)


def _milliseconds(seconds: float) -> float:
    """Playwright's timeouts are milliseconds, and 0 would mean no limit at all."""
    return max(1.0, seconds * 1000)


def _note_navigation(page: Page, response: Response, statuses: list[int]) -> None:
    """Record each main-frame navigation answer; the last one is the page's status."""
    if response.request.is_navigation_request() and response.frame == page.main_frame:
        statuses.append(response.status)


async def _content(page: Page, navigation_timeout: float) -> str:
    """The serialized DOM; a navigation in progress is waited out once."""
    try:
        return await page.content()
    except PlaywrightError:
        await page.wait_for_load_state("load", timeout=_milliseconds(navigation_timeout))
        return await page.content()


__all__ = ["PlaywrightLeasedBrowser"]
