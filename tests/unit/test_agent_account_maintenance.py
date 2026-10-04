# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A writer re-seals Agent Account envelopes after a key rotation, and a failing store never stops it."""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence

import pytest
from pydantic import SecretStr

from dlightrag.application import agent_accounts
from dlightrag.application.agent_accounts import AgentAccountMaintenance
from dlightrag.engine.answer.agent_browser import (
    ACCOUNT_LABEL,
    StoredAgentAccount,
    generate_password,
)
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.support.agent_browser import MemoryAccountStore

KEYS = {
    "test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE=",
    "next": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI=",
}


def ring(active: str, *keys: str) -> CredentialCipher:
    return CredentialCipher(
        SecretStr(json.dumps({"active": active, "keys": {key: KEYS[key] for key in keys}}))
    )


async def until(condition: Callable[[], bool], seconds: float = 5) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def _a_pass_every_few_milliseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_accounts, "_MAINTENANCE_SECONDS", 0.01)


def sealed_under_test_key() -> tuple[MemoryAccountStore, StoredAgentAccount]:
    key_id, envelope = ring("test", "test").seal(
        generate_password(), label=ACCOUNT_LABEL, binding=("owner", "shop.example", "id")
    )
    row = StoredAgentAccount("owner", "shop.example", "id", "a@x.example", None, key_id, envelope)
    store = MemoryAccountStore()
    store.rows[("owner", "shop.example")] = row
    return store, row


async def test_a_pass_moves_the_envelopes_of_a_retired_key_to_the_active_one() -> None:
    store, row = sealed_under_test_key()
    maintenance = AgentAccountMaintenance(store=store, cipher=ring("next", "test", "next"))

    resealed = await maintenance.maintain()

    assert resealed == 1
    moved = store.rows[("owner", "shop.example")]
    assert (moved.key_id, moved.account_id) == ("next", row.account_id)
    assert await maintenance.maintain() == 0


async def test_the_loop_passes_now_and_again_and_stops_when_closed() -> None:
    store, _ = sealed_under_test_key()
    asked = 0
    original = store.sealed_under

    async def counting(*, key_ids: Sequence[str], limit: int) -> tuple[StoredAgentAccount, ...]:
        nonlocal asked
        asked += 1
        return await original(key_ids=key_ids, limit=limit)

    store.sealed_under = counting  # type: ignore[method-assign]
    maintenance = AgentAccountMaintenance(store=store, cipher=ring("next", "test", "next"))

    maintenance.start()
    maintenance.start()
    await until(lambda: asked >= 3 and store.rows[("owner", "shop.example")].key_id == "next")
    await maintenance.aclose()
    after_close = asked
    await asyncio.sleep(0.1)

    assert asked == after_close
    await maintenance.aclose()


async def test_a_store_that_fails_is_retried_and_only_the_kind_of_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, _ = sealed_under_test_key()
    failures = 0
    original = store.sealed_under

    async def down(*, key_ids: Sequence[str], limit: int) -> tuple[StoredAgentAccount, ...]:
        nonlocal failures
        failures += 1
        if failures <= 2:
            raise ConnectionError("password authentication failed for user dlightrag at db:5432")
        return await original(key_ids=key_ids, limit=limit)

    store.sealed_under = down  # type: ignore[method-assign]
    caplog.set_level(logging.DEBUG)
    maintenance = AgentAccountMaintenance(store=store, cipher=ring("next", "test", "next"))

    maintenance.start()
    await until(lambda: store.rows[("owner", "shop.example")].key_id == "next")
    await maintenance.aclose()

    assert failures >= 3
    assert "Agent Account maintenance failed (ConnectionError)" in caplog.text
    assert "authentication" not in caplog.text and "db:5432" not in caplog.text
