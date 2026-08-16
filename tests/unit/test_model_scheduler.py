# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for process-wide fair model-call admission."""

import asyncio

import pytest
from dlightrag_ai.scheduler import ModelScheduler, model_call_scope


async def test_waiting_runs_are_admitted_round_robin() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def operation(label: str, *, block: bool = False) -> str:
        order.append(label)
        if block:
            first_started.set()
            await release_first.wait()
        return label

    async def scheduled(owner: str, label: str, *, block: bool = False) -> str:
        with model_call_scope(owner):
            return await scheduler.run(lambda: operation(label, block=block))

    first = asyncio.create_task(scheduled("run-a", "a1", block=True))
    await first_started.wait()
    same_run = asyncio.create_task(scheduled("run-a", "a2"))
    other_run = asyncio.create_task(scheduled("run-b", "b1"))
    await asyncio.sleep(0)

    release_first.set()
    assert await asyncio.gather(first, same_run, other_run) == ["a1", "a2", "b1"]
    assert order == ["a1", "b1", "a2"]

    # A newly freed slot rotates to the next waiting owner. The first queued
    # run keeps FIFO ordering within its own queue.
    first_started.clear()
    release_first.clear()
    first = asyncio.create_task(scheduled("run-a", "a3", block=True))
    await first_started.wait()
    a4 = asyncio.create_task(scheduled("run-a", "a4"))
    b2 = asyncio.create_task(scheduled("run-b", "b2"))
    a5 = asyncio.create_task(scheduled("run-a", "a5"))
    await asyncio.sleep(0)
    release_first.set()

    assert await asyncio.gather(first, a4, b2, a5) == ["a3", "a4", "b2", "a5"]
    assert order[-4:] == ["a3", "b2", "a4", "a5"]


async def test_stream_holds_slot_until_closed_and_closes_source() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    source_closed = asyncio.Event()
    second_started = asyncio.Event()

    async def source():
        try:
            yield "first"
            yield "second"
        finally:
            source_closed.set()

    with model_call_scope("run-a"):
        stream = scheduler.stream(source)
        assert await anext(stream) == "first"

    async def second() -> str:
        second_started.set()
        return "done"

    async def run_second() -> str:
        with model_call_scope("run-b"):
            return await scheduler.run(second)

    waiting = asyncio.create_task(run_second())
    await asyncio.sleep(0)
    assert not second_started.is_set()

    await stream.aclose()

    assert source_closed.is_set()
    assert await waiting == "done"


async def test_cancellation_after_grant_releases_slot() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    queued: asyncio.Task[str]

    async def first_operation() -> str:
        first_started.set()
        await release_first.wait()
        return "first"

    async def run_first() -> str:
        with model_call_scope("run-a"):
            result = await scheduler.run(first_operation)
        queued.cancel()
        return result

    async def queued_operation() -> str:
        return "queued"

    async def run_queued() -> str:
        with model_call_scope("run-b"):
            return await scheduler.run(queued_operation)

    first = asyncio.create_task(run_first())
    await first_started.wait()
    queued = asyncio.create_task(run_queued())
    await asyncio.sleep(0)
    release_first.set()

    assert await first == "first"
    assert isinstance(
        (await asyncio.gather(queued, return_exceptions=True))[0], asyncio.CancelledError
    )

    async def final_operation() -> str:
        return "final"

    assert await asyncio.wait_for(scheduler.run(final_operation), timeout=0.1) == "final"


async def test_global_limit_applies_across_owners() -> None:
    scheduler = ModelScheduler(max_concurrency=2)
    release = asyncio.Event()
    two_started = asyncio.Event()
    active = 0
    max_active = 0

    async def operation() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
        finally:
            active -= 1

    async def scheduled(owner: str) -> None:
        with model_call_scope(owner):
            await scheduler.run(operation)

    tasks = [asyncio.create_task(scheduled(f"run-{index}")) for index in range(5)]
    await two_started.wait()
    await asyncio.sleep(0)

    assert max_active == 2
    release.set()
    await asyncio.gather(*tasks)
    assert active == 0


async def test_queued_cancellation_preserves_later_owner_work() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocker() -> None:
        started.set()
        await release.wait()

    holder = asyncio.create_task(scheduler.run(blocker))
    await started.wait()

    async def value(label: str) -> str:
        return label

    with model_call_scope("run-b"):
        cancelled = asyncio.create_task(scheduler.run(lambda: value("cancelled")))
        survivor = asyncio.create_task(scheduler.run(lambda: value("survivor")))
    await asyncio.sleep(0)
    cancelled.cancel()
    assert isinstance(
        (await asyncio.gather(cancelled, return_exceptions=True))[0],
        asyncio.CancelledError,
    )

    release.set()
    await holder
    assert await survivor == "survivor"


async def test_cancelled_waiter_reaped_before_resume_does_not_release_another_slot() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()
    survivor_started = asyncio.Event()
    release_survivor = asyncio.Event()
    final_started = asyncio.Event()

    async def holder_operation() -> None:
        holder_started.set()
        await release_holder.wait()

    async def cancelled_operation() -> None:
        raise AssertionError("cancelled waiter must never start")

    async def survivor_operation() -> None:
        survivor_started.set()
        await release_survivor.wait()

    async def final_operation() -> None:
        final_started.set()

    holder = asyncio.create_task(scheduler.run(holder_operation))
    await holder_started.wait()
    with model_call_scope("run-b"):
        cancelled = asyncio.create_task(scheduler.run(cancelled_operation))
    with model_call_scope("run-c"):
        survivor = asyncio.create_task(scheduler.run(survivor_operation))
    await asyncio.sleep(0)

    # Queue the holder's wakeup before cancellation. Its release therefore
    # reaps the cancelled future and grants run-c before run-b resumes cleanup.
    release_holder.set()
    cancelled.cancel()
    await survivor_started.wait()
    final = asyncio.create_task(scheduler.run(final_operation))
    await asyncio.sleep(0)
    assert not final_started.is_set()

    release_survivor.set()
    await holder
    assert isinstance(
        (await asyncio.gather(cancelled, return_exceptions=True))[0],
        asyncio.CancelledError,
    )
    await survivor
    await final
    assert final_started.is_set()


async def test_failure_and_active_cancellation_release_slots() -> None:
    scheduler = ModelScheduler(max_concurrency=1)

    async def fail() -> None:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await scheduler.run(fail)

    active_started = asyncio.Event()

    async def block() -> None:
        active_started.set()
        await asyncio.Event().wait()

    active = asyncio.create_task(scheduler.run(block))
    await active_started.wait()
    active.cancel()
    assert isinstance(
        (await asyncio.gather(active, return_exceptions=True))[0],
        asyncio.CancelledError,
    )

    async def final() -> str:
        return "released"

    assert await scheduler.run(final) == "released"
