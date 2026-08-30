# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the cross-process per-workspace write gate."""

import datetime
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus import workspace_write_gate as gate_module
from dlightrag.engine.rag.workspace.ports import WorkspaceWriteFencedError


class _GateConn:
    def __init__(
        self,
        *,
        fence_until: datetime.datetime | None = None,
        exclusive_attempts: int = 1,
        shared_acquired: bool = True,
    ) -> None:
        self.fence_until = fence_until
        self.exclusive_attempts = exclusive_attempts
        self.shared_acquired = shared_acquired
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.calls.append((query, args))
        if "pg_try_advisory_lock_shared" in query:
            return self.shared_acquired
        if "pg_try_advisory_lock" in query:
            self.exclusive_attempts -= 1
            return self.exclusive_attempts <= 0
        if "EXTRACT(EPOCH FROM" in query:
            assert self.fence_until is not None
            remaining = (self.fence_until - datetime.datetime.now(datetime.UTC)).total_seconds()
            return max(0.0, remaining)
        return None

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.calls.append((query, args))
        if "dlightrag_workspace_meta" in query:
            if self.fence_until is None:
                return None
            return {"write_fence_until": self.fence_until}
        return None

    async def close(self) -> None:
        self.closed = True


async def test_shared_gate_passes_unfenced_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _GateConn()
    monkeypatch.setattr(gate_module.pg_pool, "run", _run_with(conn))
    monkeypatch.setattr(gate_module, "_acquire_connection", _acquire(conn))

    entered = False
    async with gate_module.workspace_write_gate("ws", exclusive=False):
        entered = True
        assert any("pg_try_advisory_lock_shared" in query for query, _ in conn.calls)

    assert entered is True
    assert any("pg_advisory_unlock_shared" in query for query, _ in conn.calls)
    assert conn.closed is True


async def test_shared_gate_refuses_an_active_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _GateConn(
        fence_until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=30)
    )
    monkeypatch.setattr(gate_module.pg_pool, "run", _run_with(conn))
    monkeypatch.setattr(gate_module, "_acquire_connection", _acquire(conn))

    with pytest.raises(WorkspaceWriteFencedError) as excinfo:
        async with gate_module.workspace_write_gate("ws", exclusive=False):
            pytest.fail("gate must not open under an active fence")

    assert 25.0 <= excinfo.value.retry_after_seconds <= 30.1
    assert "ws" in str(excinfo.value)
    # The preflight refuses BEFORE any blocking lock: no shared advisory lock
    # was ever taken, so a synchronous admin write fails promptly (409) even
    # while promotion holds the exclusive lock.
    assert not any("pg_try_advisory_lock_shared" in query for query, _ in conn.calls)


async def test_shared_gate_preflights_the_fence_before_any_blocking_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _GateConn(
        fence_until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=30)
    )
    monkeypatch.setattr(gate_module.pg_pool, "run", _run_with(conn))
    monkeypatch.setattr(gate_module, "_acquire_connection", _acquire(conn))

    with pytest.raises(WorkspaceWriteFencedError):
        async with gate_module.workspace_write_gate("ws", exclusive=False):
            pytest.fail("gate must not open under an active fence")

    # The durable fence read ran first; the blocking shared lock never ran.
    lock_index = next(
        (index for index, (query, _) in enumerate(conn.calls) if "advisory_lock" in query),
        None,
    )
    fence_index = next(
        (
            index
            for index, (query, _) in enumerate(conn.calls)
            if "dlightrag_workspace_meta" in query
        ),
        None,
    )
    assert fence_index is not None and fence_index < (lock_index or 10**9)
    assert lock_index is None


async def test_shared_gate_refuses_when_fence_lands_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_conn = _GateConn()
    gate_conn = _GateConn(
        fence_until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=30),
        shared_acquired=False,
    )
    monkeypatch.setattr(gate_module.pg_pool, "run", _run_with(preflight_conn))
    monkeypatch.setattr(gate_module, "_acquire_connection", _acquire(gate_conn))

    with pytest.raises(WorkspaceWriteFencedError):
        async with gate_module.workspace_write_gate("ws", exclusive=False):
            pytest.fail("gate must refuse a fence that lands after preflight")

    assert any("pg_try_advisory_lock_shared" in query for query, _ in gate_conn.calls)
    assert any("dlightrag_workspace_meta" in query for query, _ in gate_conn.calls)
    assert not any("pg_advisory_unlock_shared" in query for query, _ in gate_conn.calls)


async def test_exclusive_gate_polls_try_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _GateConn(exclusive_attempts=2)
    monkeypatch.setattr(gate_module, "_acquire_connection", _acquire(conn))
    monkeypatch.setattr(gate_module, "EXCLUSIVE_LOCK_POLL_SECONDS", 0.01)

    async with gate_module.workspace_write_gate("ws", exclusive=True) as inner:
        assert inner is conn

    try_calls = [args for query, args in conn.calls if "pg_try_advisory_lock" in query]
    assert len(try_calls) == 2
    assert any("pg_advisory_unlock(" in query for query, _ in conn.calls)
    # Exclusive acquisition never re-checks the durable fence (the worker does
    # that explicitly after the lock is held).
    assert not any("dlightrag_workspace_meta" in query for query, _ in conn.calls)


def _acquire(conn: Any) -> Any:  # noqa: ANN001, ANN401
    async def acquire() -> Any:
        return conn

    return acquire


def _run_with(conn: Any) -> Any:  # noqa: ANN001, ANN401
    async def run(operation: Any) -> Any:  # noqa: ANN401
        return await operation(conn)

    return run


async def test_gate_semaphore_bounds_concurrent_gated_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    entered = 0
    max_seen = 0
    release = asyncio.Event()
    first_in = asyncio.Event()

    class _CounterConn:
        def __init__(self) -> None:
            self.closed = False

        async def fetchrow(self, query: str, *args: Any) -> Any:
            return None  # unfenced

        async def fetchval(self, query: str, *args: Any) -> Any:
            if "pg_try_advisory_lock_shared" in query:
                return True
            return None

        async def close(self) -> None:
            self.closed = True

    def _acquire(conn: Any) -> Any:  # noqa: ANN001, ANN401
        async def acquire() -> Any:
            return conn

        return acquire

    conn = _CounterConn()
    monkeypatch.setattr(gate_module.pg_pool, "run", _run_with(conn))
    monkeypatch.setattr(gate_module, "_acquire_connection", _acquire(conn))
    monkeypatch.setattr(gate_module, "_gate_capacity", lambda: 1)

    async def first() -> None:
        nonlocal entered, max_seen
        async with gate_module.workspace_write_gate("ws", exclusive=False):
            entered += 1
            max_seen = max(max_seen, entered)
            first_in.set()
            await release.wait()
            entered -= 1

    async def second() -> None:
        nonlocal entered, max_seen
        async with gate_module.workspace_write_gate("ws", exclusive=False):
            entered += 1
            max_seen = max(max_seen, entered)
            entered -= 1

    first_task = asyncio.create_task(first())
    await first_in.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0.05)
    assert second_task.done() is False  # blocked on the capacity semaphore
    assert max_seen == 1
    release.set()
    await asyncio.wait_for(first_task, timeout=5)
    await asyncio.wait_for(second_task, timeout=5)
    assert max_seen == 1  # capacity never exceeded
