# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Doubles for the Agent Browser: a run-server to connect to, the web it renders, and a renderer.

The pool is real where it can be: ``run_server`` is the Playwright server a pool container
runs, driving a real Chromium, so a test observes the same launch options, connections and
requests production sends. What stands in for the outside world is the proxy a launch is
given: ``web_proxy`` carries a handful of canned ``http://*.example`` pages, as Squid would
carry the public Web. ``SandboxRefusal`` is a container that cannot start Chromium's sandbox,
and ``RecordingRenderer`` is the Agent Browser as a Run's Resource Registry sees it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    BrowserHolder,
    BrowserSandbox,
    LeasedBrowser,
    RenderedPage,
)

_HEAD_LIMIT = 64 * 1024


class RunServer:
    """One ``playwright run-server --max-clients 1 --unsafe`` on loopback."""

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
        "--max-clients",
        "1",
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
    """One canned page of the web a ``web_proxy`` carries."""

    body: str | bytes
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


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
        server.close()
        await server.wait_closed()


class SandboxRefusal:
    """A pool container that cannot start Chromium's sandbox, in front of a real run-server.

    It forwards every connection that does not ask for the sandbox and refuses, at the
    WebSocket upgrade, every one that does. ``asked`` records what each connection asked.
    """

    def __init__(self, upstream: str) -> None:
        self._upstream = upstream
        self.asked: list[bool] = []
        self.endpoint = ""

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            _, _, headers = _parse_head(head)
            options = json.loads(headers.get("x-playwright-launch-options", "{}"))
            sandboxed = options.get("chromiumSandbox") is True
            self.asked.append(sandboxed)
            if sandboxed:
                writer.write(_response(500, b"Chromium sandboxing failed", {}))
                await writer.drain()
                return
            host, _, port = self._upstream.removeprefix("ws://").rstrip("/").partition(":")
            upstream_reader, upstream_writer = await asyncio.open_connection(host, int(port))
            upstream_writer.write(head)
            await upstream_writer.drain()
            await asyncio.gather(
                _pipe(reader, upstream_writer),
                _pipe(upstream_reader, writer),
            )
        except asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError:
            pass
        finally:
            writer.close()
            if upstream_writer is not None:
                upstream_writer.close()


@asynccontextmanager
async def sandbox_refusal(upstream: str) -> AsyncIterator[SandboxRefusal]:
    """Front the run-server at ``upstream`` with a container that refuses the sandbox."""
    refusal = SandboxRefusal(upstream)
    server = await asyncio.start_server(refusal.serve, "127.0.0.1", 0, limit=_HEAD_LIMIT)
    refusal.endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/"
    try:
        yield refusal
    finally:
        server.close()
        await server.wait_closed()


async def _pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
    try:
        while chunk := await source.read(65536):
            sink.write(chunk)
            await sink.drain()
    except ConnectionError:
        pass
    finally:
        sink.close()


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


class FakeLease:
    """A leased browser that renders from a table and records how it was used.

    ``gate`` holds every render until it is set, so a test can have renders in flight, and
    ``peak`` is the most that were.
    """

    def __init__(
        self,
        *,
        sandbox: BrowserSandbox = "chromium",
        failure: AgentBrowserError | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.sandbox: BrowserSandbox = sandbox
        self.failure = failure
        self.gate = gate
        self.rendered: list[str] = []
        self.closed = 0
        self.active = 0
        self.peak = 0

    async def render(
        self, url: str, *, navigation_timeout: float, settle_timeout: float, max_bytes: int
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

    async def aclose(self) -> None:
        self.closed += 1


class FakeProvider:
    """A pool that hands out ``FakeLease`` objects in turn, or fails a lease it is told to."""

    def __init__(self, *outcomes: FakeLease | AgentBrowserError) -> None:
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
    "FakeProvider",
    "ProxiedRequest",
    "RecordingRenderer",
    "RunServer",
    "SandboxRefusal",
    "Served",
    "WebProxy",
    "run_server",
    "sandbox_refusal",
    "web_proxy",
]
