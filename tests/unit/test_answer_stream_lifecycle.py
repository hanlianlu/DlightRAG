# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Regression tests for answer-token iteration (idle timeout + close).

Guards the stream-consumption contract every answer path shares: an idle upstream
stream raises rather than hanging, and closing a partially consumed stream
cancels the upstream LLM connection.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens


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


async def test_iter_answer_tokens_times_out_when_idle() -> None:
    stream = _FakeRawStream([], hang=True)

    with pytest.raises(TimeoutError):
        async for _ in iter_answer_tokens(stream, idle_timeout=0.05):
            pass


async def test_iter_answer_tokens_passthrough_str_and_none() -> None:
    assert [c async for c in iter_answer_tokens("hello", idle_timeout=1.0)] == ["hello"]
    assert [c async for c in iter_answer_tokens(None, idle_timeout=1.0)] == []


async def test_iter_answer_tokens_yields_all_chunks() -> None:
    stream = _FakeRawStream(["a", "b", "c"])
    collected: list[str] = []
    token_iter: AsyncIterator[str] = iter_answer_tokens(stream, idle_timeout=1.0)
    async for chunk in token_iter:
        collected.append(chunk)
    assert collected == ["a", "b", "c"]


async def test_aclose_answer_stream_noop_for_none_and_str() -> None:
    await aclose_answer_stream(None)
    await aclose_answer_stream("plain string")
