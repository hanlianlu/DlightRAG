# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cross-process per-workspace write gate for promotion cutover.

An application-level fence check alone cannot drain in-flight LightRAG writes
that run on other pools and other processes. This module is the second half of
the protocol:

* every workspace write runs on one dedicated connection that holds a session
  advisory *shared* lock for the entire critical write (an ingest job, one
  delete/reset/metadata call);
* promotion first sets the durable registry write fence, then takes the
  advisory lock *exclusively*. The exclusive request drains every current
  shared holder (in-flight writes) and blocks later shared requests until the
  cutover transaction has committed;
* before any blocking lock or capacity wait, a durable fence *preflight* makes
  a fenced workspace fail immediately (HTTP 409), and after acquiring the
  shared lock every writer re-checks the fence again so a writer that raced
  the fence request never writes into the partition copy.

Capacity and connection hygiene: gate connections are *dedicated* (never taken
from the shared domain pool — a long ingest job would starve every other
domain operation under small ``pool_max_size``), and the number of concurrent
gated writes is bounded by a process-wide semaphore sized from the configured
domain ``pool_max_size``. The connection-budget sanity check accounts for
``lightrag + domain + gate`` connections per process. On success a connection
is closed normally after the advisory lock is released; if the unlock itself
fails the connection is *terminated* so a session lock can never leak.

The lock key is derived in SQL from the workspace text (``hashtextextended``),
so every process agrees on it without shipping identifiers.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.engine.rag.workspace.ports import WorkspaceWriteFencedError

logger = logging.getLogger(__name__)

# How long the exclusive request polls before giving up. Promotion retries the
# whole attempt afterwards, so this is a cancellation/backstop bound, not a
# correctness constraint.
EXCLUSIVE_LOCK_TIMEOUT_SECONDS = 300.0
EXCLUSIVE_LOCK_POLL_SECONDS = 0.5

_TRY_SHARED = "SELECT pg_try_advisory_lock_shared(hashtextextended($1, 0))"
_TRY_EXCLUSIVE = "SELECT pg_try_advisory_lock(hashtextextended($1, 0))"
_RELEASE_SHARED = "SELECT pg_advisory_unlock_shared(hashtextextended($1, 0))"
_RELEASE_EXCLUSIVE = "SELECT pg_advisory_unlock(hashtextextended($1, 0))"

# The durable fence is the registry row; an active fence blocks every writer.
# A stale 'promoting' state (crashed worker, fence timestamp already expired,
# leftover NOT VALID exclusion proofs) is treated as a conservative active
# fence with a small bounded retry window until a reclaimed worker cleans up.
STALE_PROMOTING_RETRY_SECONDS = 5.0

_FENCE_STATE = (
    "SELECT COALESCE(write_fence_until, to_timestamp(0)) AS write_fence_until, "
    "promotion_state "
    "FROM dlightrag_workspace_meta WHERE workspace = $1"
)

# Process-wide gate capacity: concurrent gated writes are bounded by the
# configured domain pool max size, so gate sessions cannot grow unboundedly.
# The semaphore is created lazily per event loop and re-created when the
# binding or the loop changes (test processes rebind the pool frequently).
_gate_semaphore: asyncio.Semaphore | None = None
_gate_semaphore_loop: asyncio.AbstractEventLoop | None = None
_gate_semaphore_capacity: int | None = None


def _gate_capacity() -> int:
    config = pg_pool._active_config()
    try:
        size = int(config.storage.postgres.pool_max_size)
    except Exception:
        size = 10
    return max(1, size)


def _current_gate_semaphore() -> asyncio.Semaphore:
    """Return the process-wide gate semaphore bound to the running loop."""
    global _gate_semaphore, _gate_semaphore_loop, _gate_semaphore_capacity
    loop = asyncio.get_running_loop()
    capacity = _gate_capacity()
    if (
        _gate_semaphore is None
        or _gate_semaphore_loop is not loop
        or _gate_semaphore_capacity != capacity
    ):
        _gate_semaphore = asyncio.Semaphore(capacity)
        _gate_semaphore_loop = loop
        _gate_semaphore_capacity = capacity
    return _gate_semaphore


async def _active_fence_seconds(conn: Any, workspace: str) -> float:
    """Return the remaining fence duration for one workspace, or 0.0.

    A registry row whose promotion state is still 'promoting' keeps blocking
    writers even after its fence timestamp expired (crash between the
    committed exclusion proofs and the cutover): the leftover NOT VALID
    checks must be removed by a reclaimed worker before writes are safe, so
    this reports the small bounded retry window in that interval.
    """
    row = await conn.fetchrow(_FENCE_STATE, workspace)
    if row is None:
        return 0.0
    fence_until = row["write_fence_until"]
    remaining = 0.0
    if fence_until is not None:
        remaining = float(
            await conn.fetchval(
                "SELECT GREATEST(0.0, EXTRACT(EPOCH FROM ($1::timestamptz - NOW())))",
                fence_until,
            )
            or 0.0
        )
    if remaining <= 0 and str(row.get("promotion_state") or "") == "promoting":
        return STALE_PROMOTING_RETRY_SECONDS
    return remaining


async def _acquire_connection() -> Any:
    """Open one dedicated gate connection, never borrowed from the pool."""
    import asyncpg

    config = pg_pool._active_config()
    return await asyncpg.connect(**config.pg_connection_kwargs())


async def _release_connection(conn: Any, *, hard: bool) -> None:
    """Release a dedicated gate connection; terminate when unsure of locks."""
    try:
        if hard:
            conn.terminate()
        else:
            await conn.close()
    except Exception:
        logger.debug("Gate connection release failed", exc_info=True)
        try:
            conn.terminate()
        except Exception:
            logger.warning("Gate connection termination failed", exc_info=True)


async def _unlock(conn: Any, workspace: str, *, exclusive: bool) -> None:
    statement = _RELEASE_EXCLUSIVE if exclusive else _RELEASE_SHARED
    await conn.fetchval(statement, workspace)


async def _preflight_fence(workspace: str) -> None:
    """Refuse an active fence before any blocking lock or gate-capacity wait.

    Synchronous admin writes must fail immediately (409) while a promotion
    holds the exclusive lock — they must not queue behind it and then return
    200 after the cutover. The preflight uses the already-bounded domain pool,
    rather than opening an unbounded wave of dedicated sessions ahead of the
    gate semaphore. The post-lock recheck below still covers the race where
    the fence lands after this preflight.
    """
    remaining = await pg_pool.run(lambda conn: _active_fence_seconds(conn, workspace))
    if remaining > 0:
        raise WorkspaceWriteFencedError(
            workspace=workspace,
            retry_after_seconds=remaining,
        )


@asynccontextmanager
async def workspace_write_gate(
    workspace: str,
    *,
    exclusive: bool = False,
) -> AsyncIterator[Any]:
    """Hold the workspace gate for one critical write (shared) or cutover (exclusive).

    Shared acquisition preflights the durable fence before any blocking wait
    and re-checks it after the shared lock is granted, raising
    ``WorkspaceWriteFencedError`` on either refusal, so a fenced workspace
    fails promptly and a writer that raced the fence request never writes
    into the copy window.
    """
    if not exclusive:
        await _preflight_fence(workspace)
    semaphore = _current_gate_semaphore()
    async with semaphore:
        conn = await _acquire_connection()
        released = False
        try:
            if exclusive:
                acquired = await conn.fetchval(_TRY_EXCLUSIVE, workspace)
                deadline = asyncio.get_running_loop().time() + EXCLUSIVE_LOCK_TIMEOUT_SECONDS
                while not acquired:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for the workspace write gate to drain"
                        )
                    await asyncio.sleep(EXCLUSIVE_LOCK_POLL_SECONDS)
                    acquired = await conn.fetchval(_TRY_EXCLUSIVE, workspace)
            else:
                # Never block behind a promotion after the preflight: the
                # fence can land while this caller waits for gate capacity.
                # A non-blocking shared attempt followed by another durable
                # fence read preserves prompt retryable refusal in that race.
                acquired = await conn.fetchval(_TRY_SHARED, workspace)
                deadline = asyncio.get_running_loop().time() + EXCLUSIVE_LOCK_TIMEOUT_SECONDS
                while not acquired:
                    remaining = await _active_fence_seconds(conn, workspace)
                    if remaining > 0:
                        raise WorkspaceWriteFencedError(
                            workspace=workspace,
                            retry_after_seconds=remaining,
                        )
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("timed out waiting for the workspace write gate")
                    await asyncio.sleep(EXCLUSIVE_LOCK_POLL_SECONDS)
                    acquired = await conn.fetchval(_TRY_SHARED, workspace)
                try:
                    remaining = await _active_fence_seconds(conn, workspace)
                except BaseException:
                    await _unlock(conn, workspace, exclusive=False)
                    raise
                if remaining > 0:
                    await _unlock(conn, workspace, exclusive=False)
                    raise WorkspaceWriteFencedError(
                        workspace=workspace,
                        retry_after_seconds=remaining,
                    )
            try:
                yield conn
            finally:
                unlock_failed = False
                try:
                    await _unlock(conn, workspace, exclusive=exclusive)
                except Exception:
                    unlock_failed = True
                    raise
                finally:
                    # Hard-close (terminate) whenever the unlock did not
                    # complete: a session advisory lock must never follow the
                    # connection.
                    await _release_connection(conn, hard=unlock_failed)
                    released = True
        finally:
            if not released:
                # Lock acquisition failed part-way: never leak a held lock.
                try:
                    conn.terminate()
                except Exception:
                    logger.warning("Gate connection termination failed", exc_info=True)


__all__ = [
    "EXCLUSIVE_LOCK_TIMEOUT_SECONDS",
    "STALE_PROMOTING_RETRY_SECONDS",
    "workspace_write_gate",
]
