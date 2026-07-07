# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Streaming answer wrapper with post-stream citation validation."""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable
from typing import Any, cast

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.parser import clean_invalid_citations, strip_generated_references_section

logger = logging.getLogger(__name__)


class AnswerStream(AsyncIterator[str]):
    """Async iterator that passes through tokens and cleans citations post-stream.

    Yields all tokens as-is for real-time display. After the stream ends,
    ``self.answer`` contains the cleaned answer text (invalid citations
    referencing non-existent chunks/docs are removed).

    When an ``indexer`` is provided (from the answer engine), invalid
    citations are cleaned from the final answer. This ensures the streaming
    path produces the same citation quality as the non-streaming path.
    """

    def __init__(
        self,
        raw_iterator: AsyncIterator[str],
        *,
        indexer: CitationIndexer | None = None,
    ) -> None:
        self._raw = raw_iterator
        self._indexer = indexer
        self._parts: list[str] = []
        self._gen = self._iterate()
        self.answer: str = ""

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        """Cancel the underlying LLM stream to stop wasting tokens on disconnect."""
        aclose = getattr(self._raw, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await cast(Awaitable[Any], result)

    async def _iterate(self) -> AsyncIterator[str]:  # type: ignore[override]
        async for chunk in self._raw:
            self._parts.append(chunk)
            yield chunk

        full = "".join(self._parts)
        full = strip_generated_references_section(full)

        if self._indexer is not None:
            full = clean_invalid_citations(self._indexer, full)

        self.answer = full
        logger.info(
            "[AnswerStream] Post-stream: answer_len=%d, validated=%s",
            len(self.answer),
            self._indexer is not None,
        )


async def iter_answer_tokens(
    token_iter: AsyncIterator[str] | str | None,
    *,
    idle_timeout: float,
) -> AsyncIterator[str]:
    """Yield answer tokens with a per-token inactivity deadline.

    Each token fetch is bounded by ``idle_timeout`` seconds so a stalled upstream
    LLM raises ``TimeoutError`` instead of hanging the request forever. ``str`` and
    ``None`` iterators are passed through. Closing the underlying stream is the
    caller's responsibility (see :func:`aclose_answer_stream`).
    """
    if token_iter is None:
        return
    if isinstance(token_iter, str):
        yield token_iter
        return
    aiterator = token_iter.__aiter__()
    while True:
        try:
            async with asyncio.timeout(idle_timeout):
                chunk = await aiterator.__anext__()
        except StopAsyncIteration:
            break
        yield chunk


async def aclose_answer_stream(token_iter: object) -> None:
    """Close an answer token iterator if it supports ``aclose``.

    Releases the bounded answer-stream slot and cancels the upstream LLM
    connection. No-op for ``None``/``str`` iterators.
    """
    aclose = getattr(token_iter, "aclose", None)
    if aclose is not None:
        await aclose()


__all__ = ["AnswerStream", "aclose_answer_stream", "iter_answer_tokens"]
