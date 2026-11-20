# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the pool sends for a headed lease, and what it refuses to send one for.

The launch options are the whole of the mechanism: the same member launches a headed
Chromium from the same image, so what a lease asks for over the wire is what a test reads.
"""

from __future__ import annotations

import pytest

from dlightrag.adapters.agent_browser.pool import PooledBrowserProvider
from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    BrowserHolder,
)
from tests.support.agent_browser import FakeLeases, launch_recorder

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
OTHER = BrowserHolder("owner", "22222222-2222-2222-2222-222222222222", "worker-2", 1)
EGRESS = "http://egress:3128"


def pool(
    leases: FakeLeases, *endpoints: str, headed: tuple[str, ...] = ()
) -> PooledBrowserProvider:
    return PooledBrowserProvider(
        endpoints=endpoints,
        headed_endpoints=headed,
        egress_proxy=EGRESS,
        chromium_sandbox=True,
        connect_timeout_seconds=1.0,
        leases=leases,
    )


@pytest.mark.parametrize(
    ("headed_option", "headless"),
    [pytest.param(False, True, id="headless"), pytest.param(True, False, id="headed")],
)
async def test_a_lease_asks_for_a_display_only_when_it_wants_one(
    headed_option: bool, headless: bool
) -> None:
    leases = FakeLeases()
    async with launch_recorder() as member:
        provider = pool(leases, member.endpoint, headed=(member.endpoint,))
        try:
            with pytest.raises(AgentBrowserError):
                await provider.lease(HOLDER, wait_seconds=0, headed=headed_option)
        finally:
            await provider.aclose()

    assert member.launches == [
        {"headless": headless, "chromiumSandbox": True, "proxy": {"server": EGRESS}}
    ]


async def test_a_headed_lease_claims_only_a_headed_member() -> None:
    leases = FakeLeases()
    provider = pool(leases, "ws://plain/", "ws://displayed/", headed=("ws://displayed/",))
    provider._connect = _refuse  # pyright: ignore[reportAttributeAccessIssue]
    try:
        with pytest.raises(AgentBrowserError):
            await provider.lease(HOLDER, wait_seconds=0, headed=True)
    finally:
        await provider.aclose()

    # A headless member is never asked to run with a display, and never claimed for one.
    assert leases.claimed == ["ws://displayed/"]


async def test_a_deployment_with_no_headed_member_refuses_a_headed_lease() -> None:
    leases = FakeLeases()
    provider = pool(leases, "ws://pool-1/")
    try:
        with pytest.raises(AgentBrowserError) as refused:
            await provider.lease(HOLDER, wait_seconds=0, headed=True)
        assert refused.value.reason == "not_configured"
        # Nothing was claimed, and nothing was asked of a member.
        assert leases.claimed == []
    finally:
        await provider.aclose()


async def test_a_run_that_asks_for_the_display_twice_leases_it_twice() -> None:
    leases = FakeLeases()
    provider = pool(leases, "ws://plain/", "ws://displayed/", headed=("ws://displayed/",))
    provider._connect = _refuse  # pyright: ignore[reportAttributeAccessIssue]
    try:
        for _ in range(2):
            with pytest.raises(AgentBrowserError):
                await provider.lease(HOLDER, wait_seconds=0, headed=True)
    finally:
        await provider.aclose()

    # No Run has a headed lease to spend: each lease is the Run's own browser, and the pool
    # keeps no record of what it leased before.
    assert leases.claimed == ["ws://displayed/", "ws://displayed/"]


async def _refuse(endpoint: str, *, headed: bool) -> None:
    """A connect that fails, as a member that cannot honor the launch does."""
    raise ConnectionError(f"no browser at {endpoint}")
