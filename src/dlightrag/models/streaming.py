# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Streaming answer wrapper with post-stream reference extraction."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from dlightrag.citations.parser import extract_references
from dlightrag.models.schemas import Reference

logger = logging.getLogger(__name__)


class AnswerStream(AsyncIterator[str]):
    """Async iterator that passes through tokens and extracts references post-stream.

    Yields all tokens as-is for real-time display. After the stream ends,
    ``self.references`` contains extracted references and ``self.answer``
    contains the cleaned answer text (JSON blocks / reference sections stripped).

    Consumers should use ``self.answer`` (not locally accumulated text) for
    post-stream processing like CitationProcessor.
    """

    def __init__(self, raw_iterator: AsyncIterator[str]) -> None:
        self._raw = raw_iterator
        self._parts: list[str] = []
        self._gen = self._iterate()
        self.references: list[Reference] = []
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
        self.answer, self.references = extract_references(full)
        logger.info(
            "[AnswerStream] Post-stream extraction: refs=%d, answer_len=%d",
            len(self.references),
            len(self.answer),
        )


__all__ = ["AnswerStream"]
