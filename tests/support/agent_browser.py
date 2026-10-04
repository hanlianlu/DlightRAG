# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Doubles for the Agent Browser: a run-server to connect to, the web it renders, and a renderer.

The pool is real where it can be: ``run_server`` is the Playwright server a pool container
runs, driving a real Chromium, so a test observes the same launch options, connections and
requests production sends. What stands in for the outside world is the proxy a launch is
given: ``web_proxy`` carries a handful of canned ``http://*.example`` pages, as Squid would
carry the public Web. ``LaunchRecorder`` is a pool member that refuses every connection and
keeps what each one asked to launch, and ``RecordingRenderer`` is the Agent Browser as a Run's
Resource Registry sees it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    AgentBrowserSettings,
    BrowserHolder,
    InteractiveLimits,
    LeasedBrowser,
    PageEvents,
    PageObservation,
    PageState,
    RenderedPage,
)

_HEAD_LIMIT = 64 * 1024


class RunServer:
    """One ``playwright run-server --unsafe`` on loopback."""

    def __init__(self, process: asyncio.subprocess.Process, endpoint: str) -> None:
        self._process = process
        self.endpoint = endpoint

    async def stop(self) -> None:
        """Kill the server and the node process it runs; the container dying."""
        with suppress(ProcessLookupError):
            os.killpg(self._process.pid, signal.SIGKILL)
        await self._process.wait()


@asynccontextmanager
async def run_server() -> AsyncIterator[RunServer]:
    """Start a real run-server, as a pool container does, and yield it."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "playwright",
        "run-server",
        "--port",
        "0",
        "--host",
        "127.0.0.1",
        "--unsafe",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Its own process group, so stopping it stops the node process behind the CLI.
        start_new_session=True,
    )
    server = RunServer(process, "")
    drain: asyncio.Task[None] | None = None
    try:
        assert process.stdout is not None
        line = await asyncio.wait_for(process.stdout.readline(), 60)
        listening = re.search(r"Listening on (ws://\S+)", line.decode())
        assert listening is not None, line
        server.endpoint = listening.group(1)
        # A server that logs more than the pipe holds would stall, so the rest is read away.
        drain = asyncio.create_task(_drain(process.stdout))
        yield server
    finally:
        if drain is not None:
            drain.cancel()
        await server.stop()


async def _drain(stream: asyncio.StreamReader) -> None:
    while await stream.read(65536):
        pass


@dataclass(frozen=True, slots=True)
class Served:
    """One canned page of the web a ``web_proxy`` carries.

    A page with a ``hold`` is answered only once the event is set, so a test can observe
    the world while the browser waits for it.
    """

    body: str | bytes
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    hold: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class ProxiedRequest:
    method: str
    target: str
    headers: Mapping[str, str]


class WebProxy:
    """A loopback HTTP proxy that answers ``GET http://host/path`` from a table of pages.

    It records every request the browser makes through it. A target it holds no page for
    is a 404, and ``CONNECT`` is refused: the pages are plain HTTP.
    """

    def __init__(self, pages: Mapping[str, Served]) -> None:
        self._pages = dict(pages)
        self.requests: list[ProxiedRequest] = []
        self.url = ""

    def fetched(self, url: str) -> list[ProxiedRequest]:
        """The requests the browser made for exactly ``url``."""
        return [request for request in self.requests if request.target == url]

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            method, target, headers = _parse_head(head)
            self.requests.append(ProxiedRequest(method, target, headers))
            page = self._pages.get(target) if method == "GET" else None
            if method == "CONNECT":
                writer.write(_response(403, b"", {}))
            elif page is None:
                writer.write(_response(404, b"not found", {"content-type": "text/plain"}))
            else:
                if page.hold is not None:
                    await page.hold.wait()
                body = page.body.encode() if isinstance(page.body, str) else page.body
                headers = {"content-type": "text/html; charset=utf-8", **page.headers}
                writer.write(_response(page.status, body, headers))
            await writer.drain()
        except asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError:
            pass
        finally:
            writer.close()


@asynccontextmanager
async def web_proxy(pages: Mapping[str, Served]) -> AsyncIterator[WebProxy]:
    """Serve ``pages`` (by absolute URL) as the web the browser reaches through its proxy."""
    proxy = WebProxy(pages)
    server = await asyncio.start_server(proxy.serve, "127.0.0.1", 0, limit=_HEAD_LIMIT)
    proxy.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        yield proxy
    finally:
        # A page still held would keep its connection, and so the server, open.
        for page in pages.values():
            if page.hold is not None:
                page.hold.set()
        server.close()
        await server.wait_closed()


class LaunchRecorder:
    """A pool member that refuses every connection and keeps the launch it asked for.

    The refusal stands for a member that cannot honor the launch, a host that cannot start the
    sandbox included. ``launches`` holds the decoded ``x-playwright-launch-options`` of each
    connection, in order, so a test sees what a provider sends and whether it sends anything
    after being refused.
    """

    def __init__(self) -> None:
        self.launches: list[dict[str, object]] = []
        self.endpoint = ""

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            _, _, headers = _parse_head(head)
            self.launches.append(json.loads(headers["x-playwright-launch-options"]))
            writer.write(_response(500, b"Chromium sandboxing failed", {}))
            await writer.drain()
        except asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError:
            pass
        finally:
            writer.close()


@asynccontextmanager
async def launch_recorder() -> AsyncIterator[LaunchRecorder]:
    """Serve a pool member that records the launch of each connection and refuses it."""
    recorder = LaunchRecorder()
    server = await asyncio.start_server(recorder.serve, "127.0.0.1", 0, limit=_HEAD_LIMIT)
    recorder.endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/"
    try:
        yield recorder
    finally:
        server.close()
        await server.wait_closed()


def _parse_head(head: bytes) -> tuple[str, str, dict[str, str]]:
    lines = head.decode("latin-1").split("\r\n")
    method, target, _ = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()
    return method, target, headers


def _response(status: int, body: bytes, headers: Mapping[str, str]) -> bytes:
    lines = [f"HTTP/1.1 {status} Test", *(f"{name}: {value}" for name, value in headers.items())]
    lines += [f"content-length: {len(body)}", "connection: close", "", ""]
    return "\r\n".join(lines).encode("latin-1") + body


def browser_settings(
    *,
    wait: float = 1.0,
    navigation: float = 5.0,
    settle: float = 0.0,
    action: float = 5.0,
    depth: int = 12,
    download_bytes: int = 1024 * 1024,
    idle: float = 600.0,
) -> AgentBrowserSettings:
    """The Agent Browser settings of a test: a one-member pool, and the bounds a test varies."""
    return AgentBrowserSettings(
        endpoints=("ws://pool-1/",),
        egress_proxy="http://egress:3128",
        chromium_sandbox=True,
        connect_timeout_seconds=15.0,
        lease_wait_seconds=wait,
        navigation_timeout_seconds=navigation,
        settle_timeout_seconds=settle,
        action_timeout_seconds=action,
        snapshot_depth=depth,
        max_download_bytes=download_bytes,
        idle_release_seconds=idle,
    )


type Page = str | bytes | RenderedPage | AgentBrowserError


class RecordingRenderer:
    """The Agent Browser as a Run's Resource Registry sees it: pages by URL, every render kept.

    A page is HTML text, a ready ``RenderedPage``, or an ``AgentBrowserError`` the render
    raises. A list of them answers a URL's renders in turn, the last one for every render
    after. A URL it holds no page for is a mistake in the test.
    """

    def __init__(self, pages: Mapping[str, Page | list[Page]]) -> None:
        self._pages = {
            url: list(page) if isinstance(page, list) else [page] for url, page in pages.items()
        }
        self.calls: list[str] = []

    async def __call__(self, url: str) -> RenderedPage:
        self.calls.append(url)
        queue = self._pages[url]
        page = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(page, AgentBrowserError):
            raise page
        if isinstance(page, RenderedPage):
            return page
        html = page.encode() if isinstance(page, str) else page
        return RenderedPage(requested_url=url, final_url=url, html=html, status=200)


class FakeSession:
    """An Agent Session's page as its Run sees it: it can be called, asked where it is, and closed.

    A ``failure`` is what every call raises, as a browser that disconnected would.
    """

    def __init__(self, *, failure: AgentBrowserError | None = None) -> None:
        self.failure = failure
        self.visited: list[str] = []
        self.closed = 0

    def current_url(self) -> str | None:
        return self.visited[-1] if self.visited else None

    async def navigate(self, url: str) -> PageObservation:
        if self.failure is not None:
            raise self.failure
        self.visited.append(url)
        return PageObservation(PageState(url, ""), PageEvents(), "")

    async def aclose(self) -> None:
        self.closed += 1


class FakeLease:
    """A leased browser that renders from a table, hosts fake sessions, and records its use.

    ``gate`` holds every render until it is set, so a test can have renders in flight, and
    ``peak`` is the most that were. ``session_failure`` is what every session it opens
    raises from its calls, and ``opening_failure`` what opening one raises.
    """

    def __init__(
        self,
        *,
        failure: AgentBrowserError | None = None,
        gate: asyncio.Event | None = None,
        session_failure: AgentBrowserError | None = None,
        opening_failure: AgentBrowserError | None = None,
    ) -> None:
        self.failure = failure
        self.gate = gate
        self.session_failure = session_failure
        self.opening_failure = opening_failure
        self.rendered: list[str] = []
        self.sessions: list[FakeSession] = []
        self.limits: list[InteractiveLimits] = []
        self.closed = 0
        self.active = 0
        self.peak = 0

    async def render(
        self, url: str, *, navigation_timeout: float, settle_timeout: float
    ) -> RenderedPage:
        self.rendered.append(url)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.failure is not None:
                raise self.failure
            return RenderedPage(url, url, f"<p>{url}</p>".encode(), 200)
        finally:
            self.active -= 1

    async def open_session(self, limits: InteractiveLimits) -> Any:
        if self.opening_failure is not None:
            raise self.opening_failure
        self.limits.append(limits)
        session = FakeSession(failure=self.session_failure)
        self.sessions.append(session)
        return session

    async def aclose(self) -> None:
        self.closed += 1


class FakeLeases:
    """The lease store as the pool provider sees it, with operations that can be made to fail.

    A store whose database is down raises from every operation it is told to fail. A claim
    that works takes the first endpoint not excluded, and every claim and release is
    recorded.
    """

    def __init__(self, *failing: Literal["register_endpoints", "claim", "release"]) -> None:
        self._failing = set(failing)
        self.claimed: list[str] = []
        self.released: list[str] = []

    def _operation(self, name: str) -> None:
        if name in self._failing:
            raise ConnectionError("the database is down")

    async def register_endpoints(self, endpoints: Sequence[str]) -> None:
        self._operation("register_endpoints")

    async def claim(
        self, holder: BrowserHolder, endpoints: Sequence[str], exclude: Sequence[str] = ()
    ) -> str | None:
        self._operation("claim")
        endpoint = next((endpoint for endpoint in endpoints if endpoint not in exclude), None)
        if endpoint is not None:
            self.claimed.append(endpoint)
        return endpoint

    async def release(self, holder: BrowserHolder, endpoint: str) -> None:
        self.released.append(endpoint)
        self._operation("release")


class FakeProvider:
    """A pool that hands out leased browsers in turn, or fails a lease it is told to."""

    def __init__(self, *outcomes: LeasedBrowser | AgentBrowserError) -> None:
        self._outcomes = list(outcomes)
        self.holders: list[BrowserHolder] = []
        self.waits: list[float] = []
        self.closed = False

    @property
    def leased(self) -> int:
        return len(self.holders)

    async def lease(self, holder: BrowserHolder, *, wait_seconds: float) -> LeasedBrowser:
        self.holders.append(holder)
        self.waits.append(wait_seconds)
        outcome = self._outcomes.pop(0) if self._outcomes else FakeLease()
        if isinstance(outcome, AgentBrowserError):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True


__all__ = [
    "FakeLease",
    "FakeLeases",
    "FakeProvider",
    "FakeSession",
    "LaunchRecorder",
    "ProxiedRequest",
    "RecordingRenderer",
    "RunServer",
    "Served",
    "WebProxy",
    "browser_settings",
    "launch_recorder",
    "run_server",
    "web_proxy",
]
