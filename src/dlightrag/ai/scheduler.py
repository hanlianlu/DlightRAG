# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Process-wide fair admission for model provider requests."""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Hashable
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")

_model_call_owner: ContextVar[Hashable | None] = ContextVar("model_call_owner", default=None)


@contextmanager
def model_call_scope(owner: Hashable) -> Generator[None]:
    """Bind nested model requests to one fair-scheduling owner."""
    token = _model_call_owner.set(owner)
    try:
        yield
    finally:
        _model_call_owner.reset(token)


class ModelScheduler:
    """Bound global model concurrency while round-robining waiting owners.

    One scheduler belongs to one asyncio event loop. Scheduled operations call
    providers directly; recursively scheduling from inside an admitted
    operation would wait on its own capacity.
    """

    def __init__(self, *, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._max_concurrency = int(max_concurrency)
        self._active = 0
        self._waiters: dict[Hashable, deque[asyncio.Future[None]]] = {}
        self._owners: deque[Hashable] = deque()
        self._last_owner: Hashable | None = None

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one logical provider request under fair process admission."""
        await self._acquire(self._owner())
        try:
            return await operation()
        finally:
            self._release()

    def stream(self, factory: Callable[[], AsyncGenerator[T]]) -> AsyncGenerator[T]:
        """Schedule one provider stream until exhaustion, close, or failure."""
        owner = self._owner()

        async def _scheduled() -> AsyncGenerator[T]:
            await self._acquire(owner)
            try:
                async with aclosing(factory()) as source:
                    async for item in source:
                        yield item
            finally:
                self._release()

        return _scheduled()

    def _owner(self) -> Hashable:
        owner = _model_call_owner.get()
        if owner is not None:
            return owner
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("model scheduling requires an active asyncio task")
        return task

    async def _acquire(self, owner: Hashable) -> None:
        if self._active < self._max_concurrency and not self._owners:
            self._active += 1
            self._last_owner = owner
            return
        waiter = asyncio.get_running_loop().create_future()
        queue = self._waiters.get(owner)
        if queue is None:
            queue = deque()
            self._waiters[owner] = queue
            self._owners.append(owner)
        queue.append(waiter)
        self._grant_waiters()
        try:
            await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                self._release()
            else:
                self._remove_waiter(owner, waiter)
                self._grant_waiters()
            raise

    def _release(self) -> None:
        if self._active < 1:
            raise RuntimeError("model scheduler released an unowned slot")
        self._active -= 1
        self._grant_waiters()

    def _grant_waiters(self) -> None:
        while self._active < self._max_concurrency and self._owners:
            if len(self._owners) > 1 and self._owners[0] == self._last_owner:
                self._owners.rotate(-1)
            owner = self._owners.popleft()
            queue = self._waiters[owner]
            while queue and queue[0].done():
                queue.popleft()
            if not queue:
                del self._waiters[owner]
                continue
            waiter = queue.popleft()
            if queue:
                self._owners.append(owner)
            else:
                del self._waiters[owner]
            self._active += 1
            self._last_owner = owner
            waiter.set_result(None)

    def _remove_waiter(self, owner: Hashable, waiter: asyncio.Future[None]) -> None:
        queue = self._waiters.get(owner)
        if queue is None:
            return
        try:
            queue.remove(waiter)
        except ValueError:
            return
        if queue:
            return
        del self._waiters[owner]
        try:
            self._owners.remove(owner)
        except ValueError:
            pass


__all__ = ["ModelScheduler", "model_call_scope"]
