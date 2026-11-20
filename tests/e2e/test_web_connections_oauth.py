# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The MCP authorization journey on one page, end to end.

A reader creates a Connection in Settings, clicks Authorize, is sent to the provider's consent
page, comes back through the callback, and watches the card report what happened. Everything is
real - the page, the HTTP, the Connections service, its PostgreSQL store, and the MCP SDK's own
OAuth client - except the provider itself: its metadata and token endpoints answer from a fake
authorization server, and its consent page is played by the browser's route interceptor. Nothing
here reaches the network.

Three things this locks that no unit test can: the click sends the browser to the provider, the
return lands on Settings -> Connections with the marker consumed, and the card a reader comes
back to reports the authorization that just happened rather than falling back to no credential.
"""

from __future__ import annotations

import asyncio
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Locator, Page, Route, expect

from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_connection_authorization_pg import cipher
from tests.integration.test_connections_pg import FakeMcp
from tests.support.application_double import application_double
from tests.support.dns import public_dns
from tests.unit.test_connection_oauth import FakeAuthorizationServer

ENDPOINT = "https://mcp.example/mcp"
LABEL = "Fixture server"

# What the page needs to boot. None of it is the journey under test: the shell's own reads are
# answered by the browser, so the one thing that runs on the real server is Connections.
BOOTSTRAP = {
    "contract_version": 6,
    "workspaces": [
        {
            "workspace": "default",
            "display_name": "Default",
            "embedding_model": "fixture",
            "changes": [],
        }
    ],
    "workspaces_next_cursor": None,
    "primary_workspace": "default",
    "active_workspaces": ["default"],
    "known_workspaces": ["default"],
    "default_workspace": "default",
    "answer_attachments": {
        "count_limit": 6,
        "image_max_bytes": 1024,
        "document_max_bytes": 2048,
        "extensions": ["md"],
        "image_capability": "unknown",
        "image_limit": 0,
        "accept": "",
    },
    "active_html_preview_enabled": True,
    "agent_effort": {"levels": ["low", "high", "max"], "default": "high"},
    "answer_feedback_enabled": True,
}


def _install_shell_routes(page: Page) -> None:
    """Answer the shell's boot reads; everything under /web/api/connections stays the server's."""

    def bootstrap(route: Route) -> None:
        route.fulfill(json=BOOTSTRAP)

    def conversations(route: Route) -> None:
        route.fulfill(json={"items": [], "next_cursor": None})

    def memory(route: Route) -> None:
        if route.request.url.endswith("/settings"):
            route.fulfill(json={"enabled": False, "active_count": None})
            return
        route.fulfill(json={"memories": [], "next_cursor": None})

    def agent_accounts(route: Route) -> None:
        route.fulfill(
            json={
                "available": True,
                "registration": {"allowed": True, "enabled": True},
                "accounts": [],
            }
        )

    page.route("**/web/api/bootstrap", bootstrap)
    page.route("**/web/api/conversations**", conversations)
    page.route("**/web/api/memory**", memory)
    page.route("**/web/api/agent-accounts**", agent_accounts)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Deployment(NamedTuple):
    """One running deployment: where it answers, and what its provider was asked."""

    base_url: str
    provider: FakeAuthorizationServer


def _wait_until_serving(base_url: str) -> None:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/web/", timeout=0.5):
                return
        except OSError, urllib.error.URLError:
            time.sleep(0.05)
    raise AssertionError("the E2E Connections server did not become ready")


@contextmanager
def _connections_deployment() -> Iterator[Deployment]:
    """One real server whose Connections service is real, on a database of its own."""
    import httpx2
    import uvicorn

    from dlightrag.adapters.http.server import create_app
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.adapters.postgres.connections import PGConnectionsStore
    from dlightrag.application.config.loading import get_config, set_config
    from dlightrag.application.config.sections import DlightragConfig
    from dlightrag.application.connections.policy import ConnectionPolicy
    from dlightrag.application.connections.service import Connections

    port = _free_port()
    ready = threading.Event()
    stop = threading.Event()
    failures: list[BaseException] = []
    previous_config = get_config()
    provider = FakeAuthorizationServer()

    async def serve() -> None:
        async with isolated_run_runtime("e2e_conn_oauth") as (_runs, pool):
            store = PGConnectionsStore(pool=pool)
            await store.initialize(validate_only=False)
            connections = Connections(
                store=store,
                mcp=FakeMcp(),
                cipher=cipher(),
                policy=ConnectionPolicy(),
                oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(provider)),
            )
            config = DlightragConfig(
                _env_file=None,
                deployment={"working_dir": tempfile.mkdtemp(prefix="dlightrag-e2e-oauth-")},
                models={
                    "chat": {
                        "roles": {
                            role: {"model": "fixture-model"}
                            for role in ("extract", "query", "keyword", "vlm")
                        }
                    }
                },
                access={"auth_mode": "none"},
            )
            set_config(config)
            application = application_double(config, connections=connections)
            # The application is bound by the app's lifespan, so the patch must cover startup
            # and the whole serving period, not just the factory call.
            with patch(
                "dlightrag.adapters.http.server.create_application",
                AsyncMock(return_value=application),
            ):
                app = create_app()
                server = uvicorn.Server(
                    uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
                )
                serving = asyncio.create_task(server.serve())
                ready.set()
                try:
                    while not stop.is_set():
                        await asyncio.sleep(0.05)
                finally:
                    server.should_exit = True
                    await serving
            await connections.aclose()

    def run() -> None:
        try:
            asyncio.run(serve())
        except BaseException as exc:  # surfaced to the test, not swallowed by the thread
            failures.append(exc)
            ready.set()

    thread = threading.Thread(target=run, daemon=True, name="e2e-connections")
    thread.start()
    try:
        assert ready.wait(timeout=60), "the E2E Connections server never started"
        if failures:
            raise failures[0]
        base_url = f"http://127.0.0.1:{port}"
        _wait_until_serving(base_url)
        yield Deployment(base_url, provider)
    finally:
        stop.set()
        thread.join(timeout=60)
        set_config(previous_config)
    if failures:
        raise failures[0]


def _postgres_available() -> bool:
    """Ask PostgreSQL, on a thread of its own: the test's own thread may hold an event loop."""
    from tests.support.pg import postgres_available

    answer: list[bool] = []

    def probe() -> None:
        answer.append(asyncio.run(postgres_available()))

    probe_thread = threading.Thread(target=probe, daemon=True)
    probe_thread.start()
    probe_thread.join(timeout=30)
    return bool(answer and answer[0])


@pytest.fixture(scope="module")
def connections_deployment() -> Iterator[Deployment]:
    if not _postgres_available():
        pytest.skip("PostgreSQL not available")
    with _connections_deployment() as deployment:
        yield deployment


def _connections(page: Page, base_url: str) -> list[dict[str, Any]]:
    response = page.request.get(f"{base_url}/web/api/connections/mcp")
    assert response.ok, response.text()
    return list(response.json()["connections"])


def _row_reports_oauth(card: Locator) -> bool:
    """Whether the card's own row has read the finished authorization, not the test's own reads."""
    text = card.text_content() or ""
    return "OAuth" in text and "Authorization in progress" not in text


def _wait_for(page: Page, predicate: Callable[[], bool], message: str, timeout: float = 30) -> None:
    """Wait for what the page reports out of band: a route it took, a document it loaded.

    The wait goes through Playwright: a plain sleep would hold the driver, and the route
    handler that has to answer the navigation would never run.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        page.wait_for_timeout(50)
    raise AssertionError(message)


def _await_connection(
    page: Page, base_url: str, connection_id: str, predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    """Wait for the owner's projection to report one Connection a way the test expects."""
    deadline = time.monotonic() + 30
    seen: dict[str, Any] = {}
    while time.monotonic() < deadline:
        for connection in _connections(page, base_url):
            if connection["connection_id"] == connection_id:
                seen = connection
                if predicate(connection):
                    return connection
        page.wait_for_timeout(100)
    raise AssertionError(f"the Connection never reached the expected state: {seen}")


@pytest.mark.e2e
def test_a_reader_authorizes_a_connection_and_the_card_says_so(
    page: Page, connections_deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = connections_deployment.base_url
    provider = connections_deployment.provider
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    # The provider's consent page: it agrees at once and sends the browser back with a code,
    # exactly as a provider's redirect does - a real navigation, a real callback request.
    consented: list[str] = []

    def consent(route: Route) -> None:
        consented.append(route.request.url)
        query = parse_qs(urlparse(route.request.url).query)
        # The provider checks the exchange against the request it consented to, PKCE included.
        provider.authorization = query
        state = query["state"][0]
        route.fulfill(
            status=303,
            headers={
                "Location": (
                    f"{base_url}/web/oauth/connections/mcp/callback"
                    f"?code=test-code&state={state}&iss=https://as.example"
                )
            },
        )

    # One document load per return: the provider redirects the browser, and it lands back here.
    loads: list[int] = []
    page.on("load", lambda _: loads.append(1))
    callbacks: list[tuple[int, str]] = []
    page.on(
        "response",
        lambda r: (
            callbacks.append((r.status, r.headers.get("location", "")))
            if "connections/mcp/callback" in r.url
            else None
        ),
    )
    page.route("https://as.example/authorize*", consent)
    _install_shell_routes(page)

    page.goto(f"{base_url}/web/")
    dialog = page.get_by_role("dialog", name="Settings")
    page.get_by_role("button", name="Settings", exact=True).click()
    dialog.wait_for()
    dialog.get_by_role("button", name="Connections", exact=True).click()

    # A reader's own Connection: the create form, filled and submitted from the page.
    dialog.get_by_role("button", name="Add MCP connection").click()
    dialog.locator("[data-new-label]").fill(LABEL)
    dialog.locator("[data-new-endpoint]").fill(ENDPOINT)
    dialog.get_by_role("button", name="Add connection").click()
    card = dialog.locator("article").filter(has_text=LABEL)
    card.wait_for()
    connection_id = _connections(page, base_url)[0]["connection_id"]

    # The reader opens the card they just made, picks the credential it needs, and authorizes:
    # one click, and the browser is on its way to the provider.
    card.locator("[data-card]").click()
    card.get_by_role("button", name="OAuth", exact=True).click()
    card.get_by_role("button", name="Authorize with OAuth").click()
    _wait_for(
        page,
        lambda: len(consented) == 1,
        "the browser never reached the provider's consent page",
    )

    # The provider sends the browser back to the callback, which answers with the fixed return:
    # a fresh page, which the marker opens on Settings -> Connections.
    _wait_for(page, lambda: len(loads) >= 2, "the browser never came back to the app")
    dialog = page.get_by_role("dialog", name="Settings")
    dialog.wait_for()
    expect(dialog.locator("nav [aria-current='page']")).to_contain_text("Connections")

    # The callback was accepted, not refused: nothing asked the reader to start again.
    assert callbacks and callbacks[0][0] == 303
    assert "authorization=restart" not in callbacks[0][1]

    # The exchange and publication happen in the worker; the card reports them.
    authorized = _await_connection(
        page, base_url, connection_id, lambda item: item["authorization_status"] == "succeeded"
    )
    assert [request.url.path for request in provider.requests].count("/token") == 1
    assert authorized["authentication"] == "oauth"
    # The reader acts on what the page shows, so the test waits for the page to have read it.
    _wait_for(
        page,
        lambda: _row_reports_oauth(card),
        "the card never reported the authorization it finished",
    )

    # Turning it on is the reader's own standing authorization, and then it is ready to use.
    card.get_by_role("switch").click()
    consent_dialog = page.locator("#connections-consent")
    consent_dialog.wait_for(state="visible")
    consent_dialog.locator("button[value=enable]").click()
    enabled = _await_connection(page, base_url, connection_id, lambda item: item["enabled"])
    assert enabled["status"] == "ready"
