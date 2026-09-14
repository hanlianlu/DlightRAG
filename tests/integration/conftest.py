# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration-suite safety net: no scratch database outlives the session that created it."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

import asyncpg
import pytest

from tests.support.pg import PG_CONN_KWARGS, drop_database

logger = logging.getLogger(__name__)

_SCRATCH = "select datname from pg_database where datname like 'dlightrag\\_%'"
_BUSY = "select 1 from pg_stat_activity where datname = $1 limit 1"


async def _scratch_databases() -> set[str]:
    """The scratch databases that exist right now, or nothing when no server answers."""
    try:
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
    except Exception:
        return set()
    try:
        rows = await connection.fetch(_SCRATCH)
        return {str(row["datname"]) for row in rows}
    finally:
        await connection.close()


async def _sweep(before: set[str]) -> list[str]:
    """Drop what this session left behind, skipping anything another run still holds open."""
    swept: list[str] = []
    for database in sorted(await _scratch_databases() - before):
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
        try:
            busy = await connection.fetchval(_BUSY, database)
        finally:
            await connection.close()
        if busy:
            continue
        await drop_database(database)
        swept.append(database)
    return swept


@pytest.fixture(scope="session", autouse=True)
def _no_scratch_database_outlives_the_session() -> Iterator[None]:
    """Sweep the scratch databases this session created, so a suite that fails to clean up its own
    database cannot accumulate debris.

    Only databases that appeared after the session started are considered, and one with a live
    backend is left alone because another run may own it. The fixture is synchronous on purpose:
    an async session fixture's own teardown is the thing that cannot be trusted here.
    """
    before = asyncio.run(_scratch_databases())
    yield
    swept = asyncio.run(_sweep(before))
    if swept:
        logger.warning("swept scratch databases left behind: %s", ", ".join(swept))
