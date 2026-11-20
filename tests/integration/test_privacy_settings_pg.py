# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Conversation Contribution on PostgreSQL: an owner with no row contributes nothing, the moment a
permission began is decided inside the statement that writes it, and a flag and a moment that
disagree cannot be kept.

A permission is read by whatever later takes an owner's conversations, so these tests hold the
store to what keeps that honest: the moment is the database's own, it moves only when a permission
begins, two writers never leave a mixture of two states, and nothing but a turn-on gives a
permission.

Every test owns a scratch database; skipped without PostgreSQL.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import (
    PGRunStore,
)
from dlightrag.application.privacy import PrivacySettings
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_TABLE = "dlightrag_privacy_settings"
_OFF = PrivacySettings(contribute=False, since=None)


@pytest.fixture
async def runtime() -> AsyncIterator[tuple[PGRunStore, Any]]:
    await skip_without_postgres()
    async with isolated_run_runtime("privacy") as created:
        yield created


@pytest.fixture
def store(runtime: tuple[PGRunStore, Any]) -> PGRunStore:
    return runtime[0]


@pytest.fixture
def pool(runtime: tuple[PGRunStore, Any]) -> Any:
    return runtime[1]


async def _rows(pool: Any) -> int:
    async with pool.acquire() as conn:
        return int(await conn.fetchval(f"SELECT count(*) FROM {_TABLE}"))


async def _clock(pool: Any) -> datetime.datetime:
    """The database's own clock, which a permission's moment is taken from."""
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT clock_timestamp()")


async def _turn(store: PGRunStore, owner: str, contribute: bool) -> PrivacySettings:
    return await store.write_privacy_settings(owner_id=owner, contribute=contribute)


async def test_an_owner_who_never_chose_contributes_nothing_and_has_no_row(
    store: PGRunStore, pool: Any
) -> None:
    assert await store.read_privacy_settings(owner_id="alice") == _OFF
    assert await _rows(pool) == 0

    # Turning it off for an owner who never chose leaves them contributing nothing.
    assert await _turn(store, "alice", False) == _OFF
    assert await store.read_privacy_settings(owner_id="alice") == _OFF


async def test_turning_it_on_begins_the_permission_at_the_databases_own_clock(
    store: PGRunStore, pool: Any
) -> None:
    before = await _clock(pool)
    on = await _turn(store, "alice", True)
    after = await _clock(pool)

    assert on.contribute is True
    assert on.since is not None and before <= on.since <= after
    assert await store.read_privacy_settings(owner_id="alice") == on


async def test_turning_it_on_while_it_is_on_keeps_the_moment_it_began(store: PGRunStore) -> None:
    first = await _turn(store, "alice", True)
    await asyncio.sleep(0.05)

    again = await _turn(store, "alice", True)

    assert again == first
    assert await store.read_privacy_settings(owner_id="alice") == first


async def test_turning_it_off_ends_the_permission_and_clears_its_moment(
    store: PGRunStore, pool: Any
) -> None:
    await _turn(store, "alice", True)

    off = await _turn(store, "alice", False)

    assert off == _OFF
    assert await store.read_privacy_settings(owner_id="alice") == _OFF
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT contribute, contribute_since FROM {_TABLE}")
    assert (row["contribute"], row["contribute_since"]) == (False, None)
    # Turning it off again changes nothing.
    assert await _turn(store, "alice", False) == _OFF
    assert await _rows(pool) == 1


async def test_turning_it_on_again_begins_a_new_permission_after_the_old_one(
    store: PGRunStore,
) -> None:
    first = await _turn(store, "alice", True)
    await _turn(store, "alice", False)
    await asyncio.sleep(0.05)

    second = await _turn(store, "alice", True)

    assert first.since is not None and second.since is not None
    assert second.since > first.since, (
        "the new permission begins later and covers nothing begun before it"
    )
    assert await store.read_privacy_settings(owner_id="alice") == second


async def test_each_owner_has_their_own_permission(store: PGRunStore, pool: Any) -> None:
    alice = await _turn(store, "alice", True)
    assert await store.read_privacy_settings(owner_id="bob") == _OFF

    bob = await _turn(store, "bob", True)
    await _turn(store, "alice", False)

    assert await store.read_privacy_settings(owner_id="alice") == _OFF
    assert await store.read_privacy_settings(owner_id="bob") == bob
    assert alice.contribute and bob.contribute
    assert await _rows(pool) == 2


async def test_the_table_refuses_a_flag_and_a_moment_that_disagree(pool: Any) -> None:
    """The permission is its moment, so a row that says on without one, or off with one, is no row."""
    now = datetime.datetime.now(datetime.UTC)
    insert = f"INSERT INTO {_TABLE} (owner_id, contribute, contribute_since) VALUES ($1, $2, $3)"
    check = f"{_TABLE}_since_check"

    async with pool.acquire() as conn:
        for contribute, moment in ((True, None), (False, now)):
            with pytest.raises(asyncpg.CheckViolationError, match=check):
                await conn.execute(insert, "alice", contribute, moment)
        await conn.execute(insert, "alice", False, None)
        # A row that is whole cannot be made to disagree afterwards either.
        for change in ("contribute = TRUE", "contribute_since = NOW()"):
            with pytest.raises(asyncpg.CheckViolationError, match=check):
                await conn.execute(f"UPDATE {_TABLE} SET {change} WHERE owner_id = 'alice'")

    assert await _rows(pool) == 1


async def test_turn_ons_at_once_agree_on_one_moment(store: PGRunStore, pool: Any) -> None:
    """Two tabs, or a page and its retry: whichever lands first begins the permission."""
    writers = 12
    async with contextlib.AsyncExitStack() as stack:
        # A connection for each writer is already open, so they really do run side by side.
        for _ in range(writers):
            await stack.enter_async_context(pool.acquire())
    results = await asyncio.gather(*(_turn(store, "alice", True) for _ in range(writers)))

    assert all(result.contribute for result in results)
    assert len({result.since for result in results}) == 1
    assert await _rows(pool) == 1


async def _wait_until_blocked(pool: Any, task: asyncio.Task[Any]) -> None:
    """Return once a statement on the privacy table is waiting for another transaction's lock."""
    async with asyncio.timeout(8):
        while True:
            async with pool.acquire() as observer:
                blocked = await observer.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                    f" AND wait_event_type = 'Lock' AND query LIKE '%{_TABLE}%'"
                )
            if blocked:
                return
            assert not task.done(), "the write did not wait for the other transaction"
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("contribute", [True, False])
async def test_a_write_that_waits_on_another_decides_against_what_that_one_left(
    store: PGRunStore, pool: Any, contribute: bool
) -> None:
    """The moment is decided from the row as it is when the write gets to it, not as it was read.

    The owner's permission begins in a transaction that has not committed yet. A write that meets
    it waits, and once it commits a turn-on finds the permission already given and keeps its
    moment, where a turn-off ends it. Nothing the writer saw before waiting survives the wait.
    """
    began = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
    async with pool.acquire() as held:
        tx = held.transaction()
        await tx.start()
        await held.execute(
            f"INSERT INTO {_TABLE} (owner_id, contribute, contribute_since) VALUES ($1, TRUE, $2)",
            "alice",
            began,
        )
        task = asyncio.create_task(_turn(store, "alice", contribute))
        try:
            await _wait_until_blocked(pool, task)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await tx.rollback()
            raise
        await tx.commit()
        written = await asyncio.wait_for(task, 8)

    assert written == (PrivacySettings(contribute=True, since=began) if contribute else _OFF)
    assert await store.read_privacy_settings(owner_id="alice") == written


async def test_a_turn_on_that_waits_behind_a_turn_off_is_dated_after_it_committed(
    store: PGRunStore, pool: Any
) -> None:
    """`NOW()` is when a transaction began, so a turn-on that waited for a turn-off to commit would
    be dated before it, and would cover conversations begun under the permission just withdrawn."""
    await _turn(store, "alice", True)
    async with pool.acquire() as held:
        tx = held.transaction()
        await tx.start()
        await held.execute(
            f"UPDATE {_TABLE} SET contribute = FALSE, contribute_since = NULL"
            " WHERE owner_id = 'alice'"
        )
        task = asyncio.create_task(_turn(store, "alice", True))
        try:
            await _wait_until_blocked(pool, task)
            await asyncio.sleep(0.2)
            # Read inside the transaction, so it is no later than the instant the turn-off commits.
            withdrawn_by = await held.fetchval("SELECT clock_timestamp()")
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await tx.rollback()
            raise
        await tx.commit()
        on = await asyncio.wait_for(task, 8)

    assert on.contribute is True and on.since is not None
    assert on.since >= withdrawn_by, "the new permission begins after the one withdrawn ended"
    async with pool.acquire() as conn:
        updated_at = await conn.fetchval(f"SELECT updated_at FROM {_TABLE}")
    assert updated_at >= on.since
