# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Access scheduler: overlapping writes wait; disjoint reads overlap; bash serializes."""

import asyncio

import pytest
from dlightrag_agent.environment import AccessScheduler, AllAccess, PathAccess


@pytest.mark.asyncio
async def test_overlapping_write_waits_for_the_first() -> None:
    scheduler = AccessScheduler()
    order: list[str] = []
    started = asyncio.Event()

    async def first() -> None:
        async with scheduler.hold(PathAccess(path="/ws/a", kind="write")):
            started.set()
            order.append("first-start")
            await asyncio.sleep(0.02)
            order.append("first-end")

    async def second() -> None:
        await started.wait()
        async with scheduler.hold(PathAccess(path="/ws/a", kind="write")):
            order.append("second")

    await asyncio.gather(first(), second())
    assert order == ["first-start", "first-end", "second"]


@pytest.mark.asyncio
async def test_disjoint_reads_overlap() -> None:
    scheduler = AccessScheduler()
    overlapping = asyncio.Event()
    holding = asyncio.Event()

    async def reader(name: str) -> None:
        async with scheduler.hold(PathAccess(path=f"/ws/{name}", kind="read")):
            if name == "a":
                holding.set()
                await overlapping.wait()
            else:
                await holding.wait()
                overlapping.set()

    await asyncio.wait_for(asyncio.gather(reader("a"), reader("b")), timeout=1)


@pytest.mark.asyncio
async def test_bash_serializes_the_batch() -> None:
    scheduler = AccessScheduler()
    order: list[str] = []
    started = asyncio.Event()

    async def bash() -> None:
        async with scheduler.hold(AllAccess()):
            started.set()
            order.append("bash")
            await asyncio.sleep(0.02)

    async def reader() -> None:
        await started.wait()
        async with scheduler.hold(PathAccess(path="/ws/a", kind="read")):
            order.append("read")

    await asyncio.gather(bash(), reader())
    assert order == ["bash", "read"]
