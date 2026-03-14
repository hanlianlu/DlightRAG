# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Streaming JSON parser for structured answer output."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from enum import Enum, auto

from dlightrag.models.schemas import Reference, StructuredAnswer

logger = logging.getLogger(__name__)


class _State(Enum):
    BEFORE_ANSWER = auto()
    INSIDE_ANSWER = auto()
    AFTER_ANSWER = auto()
    PASSTHROUGH = auto()
    DONE = auto()


class StreamingAnswerParser:
    """State machine that incrementally parses StructuredAnswer JSON from a token stream.

    Expects JSON of the form: {"answer": "...", "references": [...]}
    Emits answer text tokens in real-time. After stream ends, call finish()
    to get the complete StructuredAnswer with parsed references.

    If the stream doesn't look like JSON, switches to passthrough mode —
    all text is emitted as-is and finish() returns StructuredAnswer with
    empty references.
    """

    _ANSWER_PREFIX = '"answer"'

    def __init__(self) -> None:
        self._state = _State.BEFORE_ANSWER
        self._buffer = ""
        self._answer_parts: list[str] = []
        self._full_json = ""
        self._passthrough_parts: list[str] = []
        self._pending_escape = False

    def feed(self, chunk: str) -> str:
        """Feed a token chunk. Returns answer text to emit (may be empty)."""
        self._full_json += chunk

        if self._state == _State.PASSTHROUGH:
            self._passthrough_parts.append(chunk)
            return chunk

        if self._state == _State.DONE or self._state == _State.AFTER_ANSWER:
            return ""

        if self._state == _State.BEFORE_ANSWER:
            return self._handle_before_answer(chunk)

        if self._state == _State.INSIDE_ANSWER:
            return self._handle_inside_answer(chunk)

        return ""

    def finish(self) -> StructuredAnswer:
        """Call after stream ends. Returns parsed result or degraded fallback."""
        logger.info(
            "[StreamParser] finish: state=%s full_json_len=%d",
            self._state.name,
            len(self._full_json),
        )

        if self._state == _State.PASSTHROUGH:
            full = "".join(self._passthrough_parts)
            logger.info(
                "[StreamParser] finish: PASSTHROUGH mode, returning raw text "
                "(len=%d) with empty refs. first200=%s",
                len(full),
                repr(full[:200]),
            )
            return StructuredAnswer(answer=full, references=[])

        # Try to parse the full accumulated JSON
        try:
            data = json.loads(self._full_json)
            answer = data.get("answer", "")
            raw_refs = data.get("references", [])
            if not isinstance(raw_refs, list):
                raw_refs = []
            refs = []
            for r in raw_refs:
                try:
                    refs.append(Reference.model_validate(r))
                except Exception:
                    pass
            logger.info(
                "[StreamParser] finish: JSON parse OK, refs=%d answer_len=%d",
                len(refs),
                len(answer),
            )
            return StructuredAnswer(answer=answer, references=refs)
        except (json.JSONDecodeError, Exception) as exc:
            answer = "".join(self._answer_parts) or self._full_json
            logger.warning(
                "[StreamParser] finish: JSON parse FAILED (%s: %s) "
                "full_json_first200=%s — returning empty refs",
                type(exc).__name__,
                exc,
                repr(self._full_json[:200]),
            )
            return StructuredAnswer(answer=answer, references=[])

    def _handle_before_answer(self, chunk: str) -> str:
        """Accumulate until we find the answer value start, then switch state."""
        self._buffer += chunk

        # Check if this doesn't look like JSON at all
        stripped = self._buffer.lstrip()
        if stripped and not stripped.startswith("{"):
            logger.info(
                "[StreamParser] switching to PASSTHROUGH: buffer doesn't start "
                "with '{', starts with: %s",
                repr(stripped[:50]),
            )
            self._state = _State.PASSTHROUGH
            self._passthrough_parts.append(self._buffer)
            return self._buffer

        # Look for the "answer" key
        if self._ANSWER_PREFIX not in self._buffer:
            return ""

        # Find the start of the answer string value
        idx = self._buffer.find(self._ANSWER_PREFIX)
        after_key = self._buffer[idx + len(self._ANSWER_PREFIX) :]
        colon_pos = after_key.find(":")
        if colon_pos == -1:
            return ""
        after_colon = after_key[colon_pos + 1 :].lstrip()
        if not after_colon:
            return ""
        if after_colon[0] != '"':
            self._state = _State.PASSTHROUGH
            self._passthrough_parts.append(self._buffer)
            return self._buffer

        # Everything after the opening quote is answer content
        self._state = _State.INSIDE_ANSWER
        remaining = after_colon[1:]
        if remaining:
            return self._handle_inside_answer(remaining)
        return ""

    def _handle_inside_answer(self, chunk: str) -> str:
        """Extract answer text, handling JSON string escapes."""
        emitted = []
        i = 0
        text = chunk

        # Handle pending escape from previous chunk boundary
        if self._pending_escape and text:
            self._pending_escape = False
            esc = text[0]
            if esc == "n":
                emitted.append("\n")
            elif esc == "t":
                emitted.append("\t")
            elif esc == '"':
                emitted.append('"')
            elif esc == "\\":
                emitted.append("\\")
            elif esc == "/":
                emitted.append("/")
            else:
                emitted.append(esc)
            i = 1

        while i < len(text):
            ch = text[i]
            if ch == "\\":
                if i + 1 < len(text):
                    esc = text[i + 1]
                    if esc == "n":
                        emitted.append("\n")
                    elif esc == "t":
                        emitted.append("\t")
                    elif esc == '"':
                        emitted.append('"')
                    elif esc == "\\":
                        emitted.append("\\")
                    elif esc == "/":
                        emitted.append("/")
                    else:
                        emitted.append(esc)
                    i += 2
                    continue
                else:
                    # Incomplete escape at chunk boundary
                    self._pending_escape = True
                    break
            elif ch == '"':
                # End of answer string
                self._state = _State.AFTER_ANSWER
                result = "".join(emitted)
                self._answer_parts.append(result)
                return result
            else:
                emitted.append(ch)
            i += 1

        result = "".join(emitted)
        self._answer_parts.append(result)
        return result


class AnswerStream(AsyncIterator[str]):
    """Async iterator that wraps a raw token stream with JSON parsing.

    After iteration completes, ``self.references`` contains the parsed
    references. Consumers that don't care about structured output can
    iterate normally — it's a regular ``AsyncIterator[str]``.

    If iteration is interrupted (e.g., client disconnect),
    ``self.references`` remains the default empty list.
    """

    def __init__(
        self,
        raw_iterator: AsyncIterator[str],
        parser: StreamingAnswerParser,
    ) -> None:
        self._raw = raw_iterator
        self._parser = parser
        self.references: list[Reference] = []
        self._gen = self._iterate()

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._gen.__anext__()

    async def _iterate(self) -> AsyncIterator[str]:  # type: ignore[override]
        async for chunk in self._raw:
            text = self._parser.feed(chunk)
            if text:
                yield text
        result = self._parser.finish()
        self.references = result.references
