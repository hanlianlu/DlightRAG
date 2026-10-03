# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""How the Agent Browser pool fails: only an Agent Browser error leaves a lease, and a read goes on."""

from __future__ import annotations

import logging
from typing import Literal

import pytest

from dlightrag.adapters.agent_browser.pool import PooledBrowserProvider
from dlightrag.engine.answer.agent_browser import AgentBrowserError, BrowserHolder, RunAgentBrowser
from dlightrag.engine.answer.resources.registry import AgentBrowserRender, ResourceRegistry
from tests.support.agent_browser import FakeLeases, browser_settings
from tests.support.dns import public_dns
from tests.support.resources import call, tools

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
LOGGER = "dlightrag.adapters.agent_browser.pool"
PAGE = "https://spa.example.com/app.html"


def pool(leases: FakeLeases, *endpoints: str) -> PooledBrowserProvider:
    return PooledBrowserProvider(
        endpoints=endpoints or ("ws://pool-1/", "ws://pool-2/"),
        egress_proxy="http://egress:3128",
        connect_timeout_seconds=1.0,
        leases=leases,
    )


@pytest.mark.parametrize("failing", ["register_endpoints", "claim"])
async def test_a_lease_store_that_cannot_be_reached_leaves_the_pool_unreachable(
    failing: Literal["register_endpoints", "claim"], caplog: pytest.LogCaptureFixture
) -> None:
    provider = pool(FakeLeases(failing))

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        with pytest.raises(AgentBrowserError) as unreachable:
            await provider.lease(HOLDER, wait_seconds=0)

    assert unreachable.value.reason == "unreachable"
    # The log names the failure's type, and the model reads nothing the store said.
    assert [record.getMessage() for record in caplog.records] == [
        "Agent Browser lease store failed (ConnectionError)"
    ]
    assert "database" not in unreachable.value.public_message
    await provider.aclose()


async def test_an_endpoint_nothing_answers_on_is_skipped_even_if_giving_it_back_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    leases = FakeLeases("release")
    provider = pool(leases, "ws://127.0.0.1:1/")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(AgentBrowserError) as unreachable:
            await provider.lease(HOLDER, wait_seconds=5)

    # The failed release is only logged: the Run's own lease frees the row when it ends.
    assert unreachable.value.reason == "unreachable"
    assert leases.released == ["ws://127.0.0.1:1/"]
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.WARNING, "Failed to release an Agent Browser lease"),
        (logging.ERROR, "Agent Browser connect failed (Error): endpoint=ws://127.0.0.1:1/"),
    ]
    await provider.aclose()


async def test_a_read_goes_on_without_the_lease_store_and_says_the_browser_was_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    async def blocked(url: str, **_kwargs: object) -> object:
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", blocked)
    provider = pool(FakeLeases("claim"))
    browser = RunAgentBrowser(provider, HOLDER, browser_settings(wait=0))
    async with ResourceRegistry(
        extract_chain=(AgentBrowserRender(),), page_renderer=browser.render
    ) as registry:
        read, _ = tools(registry)

        result = await call(read, url=PAGE)

    # The browser is the last step of the chain, and the Tool call survives its failure.
    assert result.is_error is False
    assert "extraction_status=unavailable" in result.text_content
    assert "Agent Browser: The Agent Browser is unreachable" in result.text_content
    await browser.aclose()
    await provider.aclose()
