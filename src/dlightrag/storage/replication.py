# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL physical-replication helpers for read-after-write barriers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig


async def wait_for_current_wal_replay(
    config: DlightragConfig,
    *,
    timeout: float,
    poll_interval: float = 0.1,
) -> str:
    """Wait until the configured replica has replayed the current primary WAL LSN.

    Returns the LSN used as the barrier. If the replica endpoint resolves to a
    writable primary (local-dev fallback), the barrier is already satisfied.
    """
    primary_conn = await asyncpg.connect(**config.pg_connection_kwargs("primary"))
    try:
        lsn = await primary_conn.fetchval("SELECT pg_current_wal_lsn()::text")
    finally:
        await primary_conn.close()

    replica_conn = await asyncpg.connect(**config.pg_connection_kwargs("replica"))
    try:
        in_recovery = await replica_conn.fetchval("SELECT pg_is_in_recovery()")
        if not in_recovery:
            return str(lsn)

        deadline = time.monotonic() + timeout
        while True:
            caught_up = await replica_conn.fetchval(
                "SELECT COALESCE(pg_last_wal_replay_lsn() >= $1::pg_lsn, false)",
                lsn,
            )
            if caught_up:
                return str(lsn)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Replica did not replay WAL LSN {lsn} within {timeout:.1f}s")
            await asyncio.sleep(poll_interval)
    finally:
        await replica_conn.close()
