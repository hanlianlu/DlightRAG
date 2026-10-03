# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Research Run's use of its Agent Browser: lease on first need, release when idle.

A Run leases its browser the first time a Rendered Read needs it, shares it between
its Parent and Child Sessions, and gives it back after it has sat unused for the
configured time or when the Run settles, whichever comes first (ADR 0032). The lease
itself lives and expires with the Run's own lease, so a Run that dies frees its browser
without this class running.
"""

from __future__ import annotations

import asyncio
import logging

from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    AgentBrowserSettings,
    BrowserHolder,
    BrowserProvider,
    BrowserSandbox,
    LeasedBrowser,
    RenderedPage,
    browser_failure,
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
        self._lease: LeasedBrowser | None = None
        self._lease_lock = asyncio.Lock()
        self._renders = asyncio.Semaphore(_CONCURRENT_RENDERS)
        self._in_flight = 0
        self._idle: asyncio.Task[None] | None = None
        self._sandbox: BrowserSandbox | None = None
        self._closed = False

    @property
    def sandbox(self) -> BrowserSandbox | None:
        """Whether Chromium was sandboxed when this Run first leased its browser."""
        return self._sandbox

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
                        max_bytes=self._settings.max_page_bytes,
                    )
                except AgentBrowserError as exc:
                    if exc.reason == "disconnected":
                        # A dead browser is not reused: the next render leases a fresh one.
                        await self._discard(lease)
                    raise
        finally:
            self._in_flight -= 1
            self._rest()

    async def aclose(self) -> None:
        """Disconnect and release the browser; settlement calls this once, any time after."""
        if self._closed:
            return
        self._closed = True
        idle, self._idle = self._idle, None
        if idle is not None:
            idle.cancel()
            await asyncio.gather(idle, return_exceptions=True)
        async with self._lease_lock:
            lease, self._lease = self._lease, None
            if lease is not None:
                await lease.aclose()

    async def _leased(self) -> LeasedBrowser:
        async with self._lease_lock:
            if self._closed:
                raise browser_failure("not_configured")
            if self._lease is None:
                self._lease = await self._provider.lease(
                    self._holder, wait_seconds=self._settings.lease_wait_seconds
                )
                if self._sandbox is None:
                    self._sandbox = self._lease.sandbox
            return self._lease

    async def _discard(self, lease: LeasedBrowser) -> None:
        async with self._lease_lock:
            if self._lease is lease:
                self._lease = None
            await lease.aclose()

    def _cancel_idle(self) -> None:
        idle, self._idle = self._idle, None
        if idle is not None:
            idle.cancel()

    def _rest(self) -> None:
        """Start the idle timer once no render is left in flight."""
        if self._closed or self._in_flight or self._lease is None:
            return
        self._cancel_idle()
        self._idle = asyncio.create_task(self._release_when_idle(), name="agent-browser-idle")

    async def _release_when_idle(self) -> None:
        await asyncio.sleep(self._settings.idle_release_seconds)
        # Once the release has begun it runs to its end: a render that arrives meanwhile
        # waits on the lease lock, so it can never lease the endpoint this one is still
        # giving back and then lose it to the release.
        try:
            await asyncio.shield(self._release_if_idle())
        except Exception:
            logger.warning("Failed to release an idle Agent Browser", exc_info=True)

    async def _release_if_idle(self) -> None:
        async with self._lease_lock:
            if self._in_flight or self._lease is None:
                return
            lease, self._lease = self._lease, None
            await lease.aclose()


__all__ = ["RunAgentBrowser"]
