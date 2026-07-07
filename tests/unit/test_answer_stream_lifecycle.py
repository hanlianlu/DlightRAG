# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Regression tests for answer-stream lifecycle (semaphore release + idle timeout).

Guards the SSE-disconnect leak: when a client disconnects mid-stream the route's
``finally`` must close the token iterator, which releases the bounded
answer-stream slot and cancels the upstream LLM connection.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.core.servicemanager import _ScopedAnswerStream


class _FakeRawStream:
    """Minimal async iterator with an ``aclose`` hook, like the LLM stream."""

    def __init__(self, chunks: list[str], *, hang: bool = False) -> None:
        self._chunks = chunks
        self._index = 0
        self.hang = hang
        self.closed = False

    def __aiter__(self) -> _FakeRawStream:
        return self

    async def __anext__(self) -> str:
        if self.hang:
            await asyncio.Event().wait()  # never resolves
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self) -> None:
        self.closed = True


async def test_aclose_releases_semaphore_on_early_stop() -> None:
    """The core disconnect regression: partial consume + aclose frees the slot."""
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    assert sem.locked()

    raw = _FakeRawStream(["a", "b", "c"])
    stream = _ScopedAnswerStream(raw, sem)

    # Consume one token, then stop early (client disconnected mid-stream).
    first = await stream.__anext__()
    assert first == "a"

    # This is what the route's `finally` runs.
    await aclose_answer_stream(stream)

    assert not sem.locked(), "semaphore permit leaked on early stop"
    assert raw.closed, "upstream stream was not cancelled"


async def test_aclose_is_idempotent_after_full_consume() -> None:
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    raw = _FakeRawStream(["x"])
    stream = _ScopedAnswerStream(raw, sem)

    tokens = [chunk async for chunk in stream]
    assert tokens == ["x"]
    assert not sem.locked()  # released on StopAsyncIteration

    await aclose_answer_stream(stream)  # must not release twice / raise
    assert not sem.locked()


async def test_iter_answer_tokens_times_out_when_idle() -> None:
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    stream = _ScopedAnswerStream(_FakeRawStream([], hang=True), sem)

    with pytest.raises(TimeoutError):
        async for _ in iter_answer_tokens(stream, idle_timeout=0.05):
            pass


async def test_iter_answer_tokens_passthrough_str_and_none() -> None:
    assert [c async for c in iter_answer_tokens("hello", idle_timeout=1.0)] == ["hello"]
    assert [c async for c in iter_answer_tokens(None, idle_timeout=1.0)] == []


async def test_iter_answer_tokens_yields_all_chunks() -> None:
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    stream = _ScopedAnswerStream(_FakeRawStream(["a", "b", "c"]), sem)
    collected: list[str] = []
    token_iter: AsyncIterator[str] = iter_answer_tokens(stream, idle_timeout=1.0)
    async for chunk in token_iter:
        collected.append(chunk)
    assert collected == ["a", "b", "c"]


async def test_aclose_answer_stream_noop_for_none_and_str() -> None:
    await aclose_answer_stream(None)
    await aclose_answer_stream("plain string")
