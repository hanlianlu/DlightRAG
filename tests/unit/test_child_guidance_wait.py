# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A waiting child re-reads its guidance even while the notification hub cannot connect."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dlightrag.adapters.postgres.core._notifications import PGNotificationHub
from dlightrag.adapters.postgres.runtime import _child
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from tests.unit.test_pg_notifications import ListenEndpoint, until


class _Store(PGRunStore):
    """A run store whose one guidance row is a status this test changes."""

    def __init__(self, hub: PGNotificationHub) -> None:
        super().__init__(notifications=hub)
        self.status = "pending"
        self.reads = 0

    async def load_child_guidance(
        self, *, owner_id: str, run_id: str, request_id: str
    ) -> dict[str, Any] | None:
        self.reads += 1
        return {
            "status": self.status,
            "reply": "Use the official report." if self.status == "replied" else None,
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        }


async def test_a_reply_reaches_a_waiting_child_while_the_hub_cannot_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_child, "_GUIDANCE_REREAD_SECONDS", 0.05)
    endpoint = ListenEndpoint(failures=1_000_000)  # the hub never gets a connection
    hub = PGNotificationHub(connect=endpoint.connect)
    store = _Store(hub)
    waiter = asyncio.create_task(
        store.wait_for_child_guidance(
            owner_id="owner",
            run_id=str(uuid.uuid7()),
            request_id=str(uuid.uuid7()),
            timeout_seconds=3600,
        )
    )
    await until(lambda: store.reads == 1)

    store.status = "replied"  # its NOTIFY reached no connection

    row = await asyncio.wait_for(waiter, timeout=2)
    assert row is not None and row["reply"] == "Use the official report."
    await hub.aclose()
