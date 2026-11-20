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
from dataclasses import dataclass
from typing import cast

from dlightrag.engine.answer.agent_browser.accounts import RunAgentAccounts
from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    AgentBrowserSettings,
    AgentPage,
    BrowserHolder,
    BrowserProvider,
    LeasedBrowser,
    PageLimits,
    RenderedPage,
    SessionState,
    browser_failure,
    page_failure,
)
from dlightrag.engine.answer.agent_browser.passwords import FilledPasswords

logger = logging.getLogger(__name__)

#: Renders one Run runs at a time in its browser; each one is a context of its own.
_CONCURRENT_RENDERS = 4
#: How long moving this Run to a headed browser may take: the wait for a free one and the
#: connect it makes, with room for the lease store between them.
_HEADED_SECONDS = 45.0


class RunAgentBrowser:
    """The browser one Run leases lazily and shares across its Agent Sessions.

    A Run keeps its browser while any Agent Session has a page open; the idle clock starts
    when none has and no render is in flight.
    """

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
        #: The passwords filled into each Agent Session's pages, by scope.
        self._passwords: dict[str, FilledPasswords] = {}
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

    @property
    def headed(self) -> bool | None:
        """Whether the browser this Run holds now runs with a display, or None holding none.

        Headed is a property of the lease, not an action: the Run keeps the headed browser it
        asked for, and a call that asks again reads that lease rather than a spent cap. What
        the deployment offers is the settings' question, not this one's.
        """
        lease = self._lease
        return None if lease is None else not lease.headless

    def taken_over(self) -> bool:
        """Whether a person holds this Run's browser now, so its ``browser`` calls are refused.

        A Run with no browser leased here has nothing for anybody to hold, so it answers no
        and the call goes on to lease one. The live surface answers for its own seat.
        """
        live = self._provider.page_for(self._holder)
        return live is not None and live.taken_over()

    def take_drive_notice(self) -> bool:
        """Whether a person has driven this Run's browser since this was last asked.

        Asked of the lease's own seat, so a Run that leases a fresh browser is not told about
        a person who drove the one it gave back. Asking answers it once.
        """
        live = self._provider.page_for(self._holder)
        return live is not None and live.take_drive_notice()

    def current_url(self, scope: str) -> str | None:
        """Where the Agent Session's active page is, or None when it has no page."""
        page = self._pages.get(scope)
        return None if page is None else page.current_url()

    def filled_passwords(self, scope: str) -> FilledPasswords:
        """The passwords filled into the Agent Session's pages in this Run, which every text
        of theirs, and every mail the Session reads, passes through.

        The set outlives the Session's page, a disconnect and a Child's drive, and goes with the
        Run. It never leases a browser.
        """
        if (passwords := self._passwords.get(scope)) is None:
            passwords = self._passwords[scope] = FilledPasswords()
        return passwords

    async def with_page[T](
        self,
        scope: str,
        call: Callable[[AgentPage], Awaitable[T]],
        *,
        open_page: bool = False,
        headed: bool = False,
    ) -> T:
        """Run ``call`` on the Agent Session's page, opening it first when ``open_page``.

        Without a page and without ``open_page`` it leases nothing and says why there is
        none: the browser that held it disconnected, or the Session never opened one.
        ``headed`` is the agent asking for a browser with a display, which the Run leases in
        place of a headless one, session and all; a Run that already holds a headed browser
        is asking for the lease it has, and a deployment that leases none refuses it rather
        than answering it from the browser the Run already has.
        """
        if headed and not self._settings.headed:
            raise browser_failure("not_configured")
        page = self._pages.get(scope)
        if page is not None and headed and self.headed is False:
            # The Session has a page on a browser that is not headed, and the call asks for
            # one that is: it moves, and the page it acts on is the one the new browser got.
            page = await self._headed_page(scope)
        if page is not None:
            # A page lives exactly as long as the lease it was opened on, so with no
            # await between the two reads the lease is the page's.
            lease = cast(LeasedBrowser, self._lease)
        elif open_page:
            lease, page = await self._open_page(scope, headed=headed)
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
        self._passwords.clear()
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

    async def _leased(self, *, headed: bool = False) -> LeasedBrowser:
        async with self._lease_lock:
            return await self._leased_locked(headed=headed)

    async def _leased_locked(self, *, headed: bool = False) -> LeasedBrowser:
        if self._closed:
            raise browser_failure("not_configured")
        if self._lease is None:
            self._lease = await self._provider.lease(
                self._holder, wait_seconds=self._settings.lease_wait_seconds, headed=headed
            )
        return self._lease

    async def _headed_page(self, scope: str) -> AgentPage:
        """Move the Run to a headed browser and answer the page it opened for ``scope``."""
        async with self._lease_lock:
            self._cancel_idle()
            try:
                session = await self._move_to_headed(scope)
                page = await self._open_on_headed(scope, session)
                self._pages[scope] = page
                self._lost.discard(scope)
                return page
            finally:
                self._rest()

    async def _move_to_headed(self, scope: str) -> SessionState:
        """Read the Session's session, then give its headless browser back.

        Called under the lease lock by the one way a Run reaches a headed browser: the agent
        asking for one. The page of ``scope`` is closed, and the lease being left is closed
        even when the one that follows fails: a Run holds one browser, and its session is
        what the new one opens on.
        """
        lease = self._lease
        page = self._pages.pop(scope, None)
        session = SessionState(url="")
        if page is not None and lease is not None:
            try:
                session = await lease.capture_session()
            except AgentBrowserError:
                # A browser that will not say what it held is still a browser to leave.
                logger.warning("Failed to read an Agent Browser session before moving it")
        if page is not None:
            await page.aclose()
        self._lease = None
        if lease is not None:
            await lease.aclose()
        return session

    async def _open_on_headed(self, scope: str, session: SessionState) -> AgentPage:
        """Open a fresh page for ``scope`` on a headed browser, on the session carried over."""
        try:
            async with asyncio.timeout(_HEADED_SECONDS):
                headed = await self._leased_locked(headed=True)
                page = await headed.open_page(self._limits, self.filled_passwords(scope))
                await headed.restore_session(session)
                return page
        except BaseException:
            # The old browser is gone and no headed one took its place: the Session has no
            # page, and its next call says so rather than the refusal this was recovering from.
            self._lost.add(scope)
            raise

    async def _open_page(
        self, scope: str, *, headed: bool = False
    ) -> tuple[LeasedBrowser, AgentPage]:
        """Lease the browser if the Run holds none, and register the scope's page on it.

        Leasing and registering are one critical section, so a page is never registered on
        a browser the Run has since given back. Asking for a browser with a display while a
        headless one is leased moves the Run to one first, session and all; a Run that
        already holds a headed browser is asking for the lease it has. A deployment that
        leases no headed browser refuses the request rather than answering it headless.
        """
        async with self._lease_lock:
            self._cancel_idle()
            try:
                if headed and self.headed is False:
                    session = await self._move_to_headed(scope)
                else:
                    session = None
                try:
                    lease = await self._leased_locked(headed=headed)
                except AgentBrowserError as exc:
                    # The pool's sentences say a page was not rendered; here none was opened.
                    raise page_failure(exc.reason) from exc
                try:
                    page = await lease.open_page(self._limits, self.filled_passwords(scope))
                    if session is not None:
                        await lease.restore_session(session)
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


@dataclass(frozen=True, slots=True)
class RunBrowsing:
    """What one Research Run browses with: its Agent Browser, and the Agent Accounts it signs in
    with on the pages of that browser. A deployment that configures a browser composes both, so
    a Run has both or neither."""

    browser: RunAgentBrowser
    accounts: RunAgentAccounts


__all__ = ["RunAgentBrowser", "RunBrowsing"]
