# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Compose Agent Browser: a pool of Playwright run-servers, leased per Run.

Each pool container runs ``playwright run-server --max-clients 1``, which launches a
fresh browser for a connection and closes it with that connection. This provider
claims one endpoint for a Run in PostgreSQL, connects to it with the egress proxy in
the launch options, and gives the Run a ``PlaywrightLeasedBrowser`` (ADR 0032). It
never passes ``expose_network``, which would route browser traffic back out through
the application's own network.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from functools import partial
from typing import Protocol

from playwright.async_api import Browser, Playwright, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from dlightrag.adapters.agent_browser.playwright_session import PlaywrightLeasedBrowser
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    BrowserHolder,
    BrowserSandbox,
    LeasedBrowser,
    browser_failure,
)

logger = logging.getLogger(__name__)

#: How often a lease waiting for a free browser looks again.
_POLL_SECONDS = 0.5
#: Stopping the driver must not hold the process's shutdown open.
_STOP_SECONDS = 10.0


class BrowserLeases(Protocol):
    """The shared record of which Run holds each endpoint."""

    async def register_endpoints(self, endpoints: Sequence[str]) -> None: ...

    async def claim(
        self, holder: BrowserHolder, endpoints: Sequence[str], exclude: Sequence[str] = ()
    ) -> str | None: ...

    async def release(self, holder: BrowserHolder, endpoint: str) -> None: ...


def launch_options_header(proxy: str, *, sandbox: bool = True) -> str:
    """The ``x-playwright-launch-options`` value that launches a headless, proxied Chromium.

    The server honors ``chromiumSandbox`` only because it runs with ``--unsafe``, and
    launches Chromium with ``--no-sandbox`` unless it is true.
    """
    options: dict[str, object] = {"headless": True}
    if sandbox:
        options["chromiumSandbox"] = True
    options["proxy"] = {"server": proxy}
    return json.dumps(options, separators=(",", ":"))


def _store_failure(exc: Exception) -> AgentBrowserError:
    """The failure a lease store that cannot be reached becomes: an unreachable pool."""
    logger.error("Agent Browser lease store failed (%s)", type(exc).__name__)
    return browser_failure("unreachable")


class ComposeBrowserProvider:
    """Leases a Run one browser of the Compose pool."""

    def __init__(
        self,
        *,
        endpoints: Sequence[str],
        egress_proxy: str,
        connect_timeout_seconds: float,
        leases: BrowserLeases,
    ) -> None:
        self._endpoints = tuple(endpoints)
        self._egress_proxy = egress_proxy
        self._connect_timeout_ms = max(1.0, connect_timeout_seconds * 1000)
        self._leases = leases
        self._driver: Playwright | None = None
        self._driver_lock = asyncio.Lock()
        self._registered = False
        self._register_lock = asyncio.Lock()
        #: Endpoints whose Chromium cannot start its sandbox, learned this process and
        #: kept for its lifetime (ADR 0024 degrades the same way for Landlock).
        self._unsandboxed: set[str] = set()
        self._closed = False

    async def lease(self, holder: BrowserHolder, *, wait_seconds: float) -> LeasedBrowser:
        """Claim an endpoint for ``holder`` and connect to it, waiting up to ``wait_seconds``.

        Failing, it raises an ``AgentBrowserError`` and nothing else. A lease store that
        cannot be reached leaves the pool as unreachable as members that are down do.
        """
        if self._closed:
            raise browser_failure("not_configured")
        await self._register_endpoints()
        deadline = time.monotonic() + wait_seconds
        excluded: list[str] = []
        while True:
            if len(excluded) == len(self._endpoints):
                raise browser_failure("unreachable")
            endpoint = await self._claim(holder, excluded)
            if endpoint is not None:
                try:
                    browser, sandbox = await self._open(endpoint)
                except Exception as exc:
                    await self._give_back(holder, endpoint)
                    excluded.append(endpoint)
                    logger.error(
                        "Agent Browser connect failed (%s): endpoint=%s",
                        type(exc).__name__,
                        endpoint,
                    )
                    continue
                except BaseException:
                    # Cancelled mid-connect: the endpoint must not stay claimed for a Run
                    # that never got its browser.
                    await asyncio.shield(self._leases.release(holder, endpoint))
                    raise
                return PlaywrightLeasedBrowser(
                    browser, partial(self._leases.release, holder, endpoint), sandbox=sandbox
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise browser_failure("busy")
            await asyncio.sleep(min(_POLL_SECONDS, remaining))

    async def aclose(self) -> None:
        """Stop the driver; browsers still leased go with it and their Runs see them close."""
        self._closed = True
        async with self._driver_lock:
            driver, self._driver = self._driver, None
        if driver is None:
            return
        try:
            await asyncio.wait_for(driver.stop(), _STOP_SECONDS)
        except Exception:
            logger.warning("Failed to stop the Playwright driver", exc_info=True)

    async def _register_endpoints(self) -> None:
        async with self._register_lock:
            if not self._registered:
                try:
                    await self._leases.register_endpoints(self._endpoints)
                except Exception as exc:
                    raise _store_failure(exc) from exc
                self._registered = True

    async def _claim(self, holder: BrowserHolder, excluded: Sequence[str]) -> str | None:
        try:
            return await self._leases.claim(holder, self._endpoints, excluded)
        except Exception as exc:
            raise _store_failure(exc) from exc

    async def _give_back(self, holder: BrowserHolder, endpoint: str) -> None:
        """Release an endpoint that was claimed and never got a browser.

        A release that fails is logged and the lease goes on to the next endpoint: the
        row is free anyway once the Run's own lease expires.
        """
        try:
            await self._leases.release(holder, endpoint)
        except Exception:
            logger.warning("Failed to release an Agent Browser lease", exc_info=True)

    async def _open(self, endpoint: str) -> tuple[Browser, BrowserSandbox]:
        """Connect with Chromium's sandbox, or without it where the endpoint cannot run it.

        A sandboxed connect that fails and an immediate unsandboxed one that succeeds
        say the endpoint cannot start the sandbox, whatever the driver's words were:
        that is recorded once, and its later leases connect unsandboxed directly. A connect
        that fails with Playwright's TimeoutError says nothing about the sandbox, only that
        the endpoint did not answer in time, so it fails as any connect does and is never
        retried without the sandbox.
        """
        if endpoint in self._unsandboxed:
            return await self._connect(endpoint, sandbox=False), "unavailable"
        try:
            return await self._connect(endpoint, sandbox=True), "chromium"
        except PlaywrightTimeoutError:
            raise
        except Exception:
            browser = await self._connect(endpoint, sandbox=False)
        self._unsandboxed.add(endpoint)
        logger.warning(
            "Agent Browser endpoint %s cannot start Chromium's sandbox, so its browsers run "
            "without it; the container's user namespaces and seccomp profile decide",
            endpoint,
        )
        return browser, "unavailable"

    async def _connect(self, endpoint: str, *, sandbox: bool) -> Browser:
        headers = {
            "x-playwright-launch-options": launch_options_header(
                self._egress_proxy, sandbox=sandbox
            )
        }
        for attempt in (1, 2):
            driver = await self._started()
            try:
                return await driver.chromium.connect(
                    endpoint, timeout=self._connect_timeout_ms, headers=headers
                )
            except Exception:
                # A driver process that has died connects nowhere again: start another
                # and try once more, as nothing about the endpoint failed.
                if attempt == 2 or not await self._replace_if_ended(driver):
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _started(self) -> Playwright:
        async with self._driver_lock:
            if self._closed:
                raise browser_failure("not_configured")
            if self._driver is None:
                self._driver = await async_playwright().start()
            return self._driver

    async def _replace_if_ended(self, driver: Playwright) -> bool:
        """Forget ``driver`` when the driver process behind it has died, and say so."""
        # Playwright offers no public query for this, and a dead driver raises a bare
        # Exception; its transport has failed once this future is done. 1.63.0 is pinned
        # (ADR 0032) and a test kills the driver, so an upgrade that moves this fails there.
        if not driver._impl_obj._connection._transport.on_error_future.done():
            return False
        async with self._driver_lock:
            if self._driver is driver:
                self._driver = None
        try:
            # Stopping it makes the browsers other Runs hold through it fail at once
            # instead of waiting on a driver that will never answer.
            await asyncio.wait_for(driver.stop(), _STOP_SECONDS)
        except Exception:
            logger.warning("Failed to stop a dead Playwright driver", exc_info=True)
        return True


__all__ = ["BrowserLeases", "ComposeBrowserProvider", "launch_options_header"]
