# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Research Run's use of its Agent Browser: lease on first need, release when idle.

A Run leases its browser the first time a Rendered Read renders or an Agent Session
opens its Agent Page, shares it between its Parent and Child Sessions, and gives it back
when it settles, or after it has sat unused for the configured time (ADR 0032). The
browser is unused when no Agent Page is open and no render is in flight, so a Run keeps
its browser for as long as any Agent Page is open, and the idle clock starts only when
none is. The lease itself lives and expires with the Run's own lease, so a Run that
dies frees its browser without this class running.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import cast

from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    AgentBrowserSettings,
    AgentPage,
    BrowserHolder,
    BrowserProvider,
    LeasedBrowser,
    PageLimits,
    RenderedPage,
    browser_failure,
    page_failure,
)

logger = logging.getLogger(__name__)

#: Renders one Run runs at a time in its browser; each one is a context of its own.
_CONCURRENT_RENDERS = 4


class RunAgentBrowser:
    """The browser one Run leases lazily and shares across its Agent Sessions."""

    def __init__(
        self,
        provider: BrowserProvider,
        holder: BrowserHolder,
        settings: AgentBrowserSettings,
    ) -> None:
        self._provider = provider
        self._holder = holder
        self._settings = settings
        self._limits = PageLimits(
            navigation_timeout=settings.navigation_timeout_seconds,
            action_timeout=settings.action_timeout_seconds,
            settle_timeout=settings.settle_timeout_seconds,
            snapshot_depth=settings.snapshot_depth,
            max_download_bytes=settings.max_download_bytes,
        )
        self._lease: LeasedBrowser | None = None
        #: Each Agent Session's page, by the scope its tool calls run in. Only the
        #: current lease has any, so the lease lock guards the two together.
        self._pages: dict[str, AgentPage] = {}
        #: The scopes whose page died with a browser that disconnected.
        self._lost: set[str] = set()
        self._lease_lock = asyncio.Lock()
        self._renders = asyncio.Semaphore(_CONCURRENT_RENDERS)
        self._in_flight = 0
        self._idle: asyncio.Task[None] | None = None
        self._closed = False

    async def render(self, url: str) -> RenderedPage:
        """Render one page in a temporary context of this Run's browser."""
        if self._closed:
            raise browser_failure("not_configured")
        self._cancel_idle()
        self._in_flight += 1
        try:
            async with self._renders:
                lease = await self._leased()
                try:
                    return await lease.render(
                        url,
                        navigation_timeout=self._settings.navigation_timeout_seconds,
                        settle_timeout=self._settings.settle_timeout_seconds,
                    )
                except AgentBrowserError as exc:
                    if exc.reason == "disconnected":
                        # A dead browser is not reused: the next render leases a fresh one.
                        await self._discard(lease)
                    raise
        finally:
            self._in_flight -= 1
            self._rest()

    @property
    def limits(self) -> PageLimits:
        """The bounds every page of this Run works within."""
        return self._limits

    def current_url(self, scope: str) -> str | None:
        """Where the Agent Session's active page is, or None when it has no page."""
        page = self._pages.get(scope)
        return None if page is None else page.current_url()

    async def with_page[T](
        self,
        scope: str,
        call: Callable[[AgentPage], Awaitable[T]],
        *,
        open_page: bool = False,
    ) -> T:
        """Run ``call`` on the Agent Session's page, opening it first when ``open_page``.

        Without a page and without ``open_page`` it leases nothing and says why there is
        none: the browser that held it disconnected, or the Session never opened one.
        """
        page = self._pages.get(scope)
        if page is not None:
            # A page lives exactly as long as the lease it was opened on, so with no
            # await between the two reads the lease is the page's.
            lease = cast(LeasedBrowser, self._lease)
        elif open_page:
            lease, page = await self._open_page(scope)
        else:
            raise page_failure("page_lost" if scope in self._lost else "no_page")
        try:
            return await call(page)
        except AgentBrowserError as exc:
            if exc.reason == "disconnected":
                await self._discard(lease)
            raise

    async def close_page(self, scope: str) -> None:
        """Close the Agent Session's page when its Session ends; the browser stays leased."""
        async with self._lease_lock:
            page = self._pages.pop(scope, None)
            self._lost.discard(scope)
        if page is not None:
            await page.aclose()
        self._rest()

    async def aclose(self) -> None:
        """Close the pages, then disconnect and release the browser; settlement calls this
        once, any time after."""
        if self._closed:
            return
        self._closed = True
        idle, self._idle = self._idle, None
        if idle is not None:
            idle.cancel()
            await asyncio.gather(idle, return_exceptions=True)
        async with self._lease_lock:
            pages = list(self._pages.values())
            self._pages.clear()
            lease, self._lease = self._lease, None
            await asyncio.gather(*(page.aclose() for page in pages))
            if lease is not None:
                await lease.aclose()

    async def _leased(self) -> LeasedBrowser:
        async with self._lease_lock:
            return await self._leased_locked()

    async def _leased_locked(self) -> LeasedBrowser:
        if self._closed:
            raise browser_failure("not_configured")
        if self._lease is None:
            self._lease = await self._provider.lease(
                self._holder, wait_seconds=self._settings.lease_wait_seconds
            )
        return self._lease

    async def _open_page(self, scope: str) -> tuple[LeasedBrowser, AgentPage]:
        """Lease the browser if the Run holds none, and register the scope's page on it.

        Leasing and registering are one critical section, so a page is never registered on
        a browser the Run has since given back.
        """
        async with self._lease_lock:
            self._cancel_idle()
            try:
                try:
                    lease = await self._leased_locked()
                except AgentBrowserError as exc:
                    # The pool's sentences say a page was not rendered; here none was opened.
                    raise page_failure(exc.reason) from exc
                try:
                    page = await lease.open_page(self._limits)
                except AgentBrowserError as exc:
                    if exc.reason == "disconnected":
                        await self._discard_locked(lease)
                    raise
                self._pages[scope] = page
                self._lost.discard(scope)
                return lease, page
            finally:
                # A browser with no page open rests, and one with a page does not.
                self._rest()

    async def _discard(self, lease: LeasedBrowser) -> None:
        async with self._lease_lock:
            await self._discard_locked(lease)

    async def _discard_locked(self, lease: LeasedBrowser) -> None:
        """Give a browser that disconnected back, and lose every page that lived on it.

        A late report from a browser the Run already replaced finds the lease gone and
        does nothing. The lease is closed under the lock, so nothing can lease the
        endpoint while it is being given back.
        """
        if self._lease is not lease:
            return
        self._lease = None
        self._lost.update(self._pages)
        self._pages.clear()
        await lease.aclose()

    def _cancel_idle(self) -> None:
        idle, self._idle = self._idle, None
        if idle is not None:
            idle.cancel()

    def _unused(self) -> bool:
        return not self._in_flight and not self._pages

    def _rest(self) -> None:
        """Start the idle timer once no page is open and no render is in flight."""
        if self._closed or not self._unused() or self._lease is None:
            return
        self._cancel_idle()
        self._idle = asyncio.create_task(self._release_when_idle(), name="agent-browser-idle")

    async def _release_when_idle(self) -> None:
        await asyncio.sleep(self._settings.idle_release_seconds)
        # Once the release has begun it runs to its end: a render or a page that arrives
        # meanwhile waits on the lease lock, so it can never lease the endpoint this one is
        # still giving back and then lose it to the release.
        try:
            await asyncio.shield(self._release_if_idle())
        except Exception:
            logger.warning("Failed to release an idle Agent Browser", exc_info=True)

    async def _release_if_idle(self) -> None:
        async with self._lease_lock:
            if not self._unused() or self._lease is None:
                return
            lease, self._lease = self._lease, None
            await lease.aclose()


__all__ = ["RunAgentBrowser"]
