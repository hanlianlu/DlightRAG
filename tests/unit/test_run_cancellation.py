# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run cancellation wakes through the notification hub; authority stays in the rescan."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from dlightrag.adapters.postgres.core import _notifications
from dlightrag.adapters.postgres.core._channels import RUN_CANCEL_CHANNEL
from dlightrag.adapters.postgres.core._notifications import ChannelWatcher, PGNotificationHub
from dlightrag.adapters.postgres.runtime._child import cancellation_notify_key
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from tests.unit.test_pg_notifications import ListenEndpoint, until


class _Store(PGRunStore):
    """A run store whose authoritative cancel-pending read is a list this test controls."""

    def __init__(self, hub: PGNotificationHub) -> None:
        super().__init__(notifications=hub)
        self.pending: dict[str, list[tuple[str, str]]] = {}
        self.failures = 0  # how many rescans, first to last, fail before reading
        self.rescans = 0

    async def iter_cancel_pending(
        self, *, worker_id: str, page_size: int = 200
    ) -> AsyncIterator[tuple[str, str]]:
        self.rescans += 1
        if self.failures:
            self.failures -= 1
            raise ConnectionResetError("cancel-pending read failed")
        for item in list(self.pending.get(worker_id, ())):
            yield item


def _listener(store: _Store, on_cancel: Callable[[str, str], Awaitable[None]]) -> ChannelWatcher:
    return store.build_cancellation_listener(worker_id="worker-1", on_cancel=on_cancel)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.01)


async def test_a_wake_digest_alone_signals_nothing_without_the_authoritative_rescan() -> None:
    endpoint = ListenEndpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    store = _Store(hub)
    cancelled: list[tuple[str, str]] = []

    async def on_cancel(owner_id: str, run_id: str) -> None:
        cancelled.append((owner_id, run_id))

    listener = _listener(store, on_cancel)
    await listener.start()
    await asyncio.wait_for(listener.ready.wait(), timeout=2)
    endpoint.opened[0].notify(RUN_CANCEL_CHANNEL, cancellation_notify_key("o", "r1"))
    await until(lambda: store.rescans == 2)
    assert cancelled == []

    store.pending["worker-1"] = [("o", "r1")]
    store.pending["worker-2"] = [("o", "r2")]
    endpoint.opened[0].notify(RUN_CANCEL_CHANNEL, cancellation_notify_key("o", "r1"))
    await until(lambda: cancelled == [("o", "r1")])

    await listener.aclose()
    await hub.aclose()


async def test_readiness_waits_for_a_rescan_and_every_signal_it_sends_to_succeed() -> None:
    endpoint = ListenEndpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    store = _Store(hub)
    store.failures = 1
    store.pending["worker-1"] = [("o", "r1")]
    attempts: list[tuple[str, str]] = []

    async def on_cancel(owner_id: str, run_id: str) -> None:
        attempts.append((owner_id, run_id))
        if len(attempts) == 1:
            raise RuntimeError("local signal unavailable")

    listener = _listener(store, on_cancel)
    await listener.start()
    await until(lambda: len(attempts) == 1)

    assert not listener.ready.is_set()
    await asyncio.wait_for(listener.ready.wait(), timeout=2)
    assert store.rescans == 3  # the failed read, the failed signal, and the one that held
    assert attempts == [("o", "r1"), ("o", "r1")]
    await listener.aclose()
    await hub.aclose()


async def test_a_cancel_requested_while_disconnected_is_found_once_listening_again() -> None:
    endpoint = ListenEndpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    store = _Store(hub)
    cancelled: list[tuple[str, str]] = []

    async def on_cancel(owner_id: str, run_id: str) -> None:
        cancelled.append((owner_id, run_id))

    listener = _listener(store, on_cancel)
    await listener.start()
    await asyncio.wait_for(listener.ready.wait(), timeout=2)
    endpoint.opened[0].terminate()
    store.pending["worker-1"] = [("o", "missed")]  # its NOTIFY reached no connection

    await until(lambda: cancelled == [("o", "missed")])
    assert len(endpoint.opened) == 2
    await listener.aclose()
    await hub.aclose()


def test_the_wake_digest_names_one_run_of_one_owner() -> None:
    digest = cancellation_notify_key("owner", "run")

    assert digest == cancellation_notify_key("owner", "run")
    assert len(digest) == 64
    assert digest != cancellation_notify_key("owner", "other-run")
    assert cancellation_notify_key("ab", "c") != cancellation_notify_key("a", "bc")
