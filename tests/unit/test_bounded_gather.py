# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for bounded_gather concurrency utility."""

from __future__ import annotations

import asyncio

import pytest

from dlightrag.utils.concurrency import bounded_gather


class TestBoundedGather:
    """Verify bounded_gather correctness and concurrency control."""

    async def test_all_coroutines_complete(self) -> None:
        """All results returned in order."""

        async def double(x: int) -> int:
            return x * 2

        coros = [double(i) for i in range(5)]
        results = await bounded_gather(coros, max_concurrent=3)

        assert results == [0, 2, 4, 6, 8]

    async def test_respects_concurrency_limit(self) -> None:
        """At most max_concurrent tasks run simultaneously."""
        peak = 0
        current = 0
        lock = asyncio.Lock()

        async def track() -> str:
            nonlocal peak, current
            async with lock:
                current += 1
                peak = max(peak, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return "done"

        coros = [track() for _ in range(10)]
        results = await bounded_gather(coros, max_concurrent=3)

        assert len(results) == 10
        assert peak <= 3

    async def test_single_concurrency(self) -> None:
        """max_concurrent=1 processes sequentially."""
        order: list[int] = []

        async def append(i: int) -> int:
            order.append(i)
            await asyncio.sleep(0.01)
            return i

        coros = [append(i) for i in range(5)]
        results = await bounded_gather(coros, max_concurrent=1)

        assert results == [0, 1, 2, 3, 4]
        assert order == [0, 1, 2, 3, 4]

    async def test_partial_failures_collected(self) -> None:
        """Failures don't stop other tasks; exceptions appear in results."""

        async def maybe_fail(i: int) -> int:
            if i == 2:
                raise ValueError(f"fail-{i}")
            return i

        coros = [maybe_fail(i) for i in range(5)]
        results = await bounded_gather(coros, max_concurrent=3)

        assert results[0] == 0
        assert results[1] == 1
        assert isinstance(results[2], ValueError)
        assert str(results[2]) == "fail-2"
        assert results[3] == 3
        assert results[4] == 4

    async def test_empty_input(self) -> None:
        """Empty coroutine list returns empty results."""
        results = await bounded_gather([], max_concurrent=4)
        assert results == []

    async def test_single_item(self) -> None:
        """Single coroutine works correctly."""

        async def one() -> str:
            return "ok"

        results = await bounded_gather([one()], max_concurrent=4)
        assert results == ["ok"]

    async def test_max_concurrent_larger_than_tasks(self) -> None:
        """More workers than tasks still works."""

        async def val(x: int) -> int:
            return x

        coros = [val(i) for i in range(3)]
        results = await bounded_gather(coros, max_concurrent=10)
        assert results == [0, 1, 2]

    async def test_error_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Failed tasks are logged with their index."""

        async def fail() -> None:
            raise RuntimeError("boom")

        with caplog.at_level("WARNING"):
            results = await bounded_gather([fail()], max_concurrent=2, task_name="test-op")

        assert isinstance(results[0], RuntimeError)
        assert "boom" in caplog.text
        assert "test-op" in caplog.text
