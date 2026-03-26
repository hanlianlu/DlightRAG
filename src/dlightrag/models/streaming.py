# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Streaming answer wrapper with post-stream citation validation."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from dlightrag.citations.parser import clean_invalid_citations

if TYPE_CHECKING:
    from dlightrag.citations.indexer import CitationIndexer

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

    async def _iterate(self) -> AsyncIterator[str]:  # type: ignore[override]
        async for chunk in self._raw:
            self._parts.append(chunk)
            yield chunk

        full = "".join(self._parts)

        if self._indexer is not None:
            full = clean_invalid_citations(self._indexer, full)

        self.answer = full
        logger.info(
            "[AnswerStream] Post-stream: answer_len=%d, validated=%s",
            len(self.answer),
            self._indexer is not None,
        )


__all__ = ["AnswerStream"]
