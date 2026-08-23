# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Memory-bounded streaming output with optional durable full-result staging."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Protocol

from dlightrag.agent.environment.local import ProcessChunk
from dlightrag.agent.tools.contracts import CommittedOutput


class OutputStage(Protocol):
    """One uncommitted full-output staging object."""

    def append(self, data: bytes) -> None: ...

    async def commit(self) -> CommittedOutput: ...

    def discard(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolOutputSnapshot:
    """One bounded model view plus full-output continuation metadata."""

    text: str
    total_bytes: int
    total_lines: int
    truncated: bool
    receipt: CommittedOutput | None = None


class StreamingToolOutput:
    """Decode process chunks, stage the full stream, and retain a bounded tail."""

    def __init__(
        self,
        *,
        stage: OutputStage | None,
        max_bytes: int,
        max_lines: int,
    ) -> None:
        if max_bytes < 1 or max_lines < 1:
            raise ValueError("streaming output bounds must be positive")
        self._stage = stage
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._decoders = {
            "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        self._tail = ""
        self._total_bytes = 0
        self._newlines = 0
        self._has_text = False
        self._ends_with_newline = False
        self._finished = False

    def append(self, chunk: ProcessChunk) -> ToolOutputSnapshot:
        if self._finished:
            raise RuntimeError("streaming output is already finished")
        text = self._decoders[chunk.stream].decode(chunk.data, final=False)
        self._append_text(text)
        return self.snapshot()

    def snapshot(self) -> ToolOutputSnapshot:
        return ToolOutputSnapshot(
            text=self._tail,
            total_bytes=self._total_bytes,
            total_lines=self._total_lines,
            truncated=self._is_truncated,
        )

    def abort(self) -> None:
        """Discard staging synchronously; safe on cancellation paths."""
        if self._finished:
            return
        self._finished = True
        if self._stage is not None:
            self._stage.discard()

    async def finish(self) -> ToolOutputSnapshot:
        if self._finished:
            raise RuntimeError("streaming output is already finished")
        for decoder in self._decoders.values():
            self._append_text(decoder.decode(b"", final=True))
        self._finished = True
        receipt: CommittedOutput | None = None
        if self._is_truncated:
            if self._stage is not None:
                receipt = await self._stage.commit()
        elif self._stage is not None:
            self._stage.discard()
        return ToolOutputSnapshot(
            text=self._tail,
            total_bytes=self._total_bytes,
            total_lines=self._total_lines,
            truncated=self._is_truncated,
            receipt=receipt,
        )

    @property
    def _total_lines(self) -> int:
        if not self._has_text:
            return 0
        return self._newlines + (0 if self._ends_with_newline else 1)

    @property
    def _is_truncated(self) -> bool:
        return self._total_bytes > self._max_bytes or self._total_lines > self._max_lines

    def _append_text(self, text: str) -> None:
        if not text:
            return
        data = text.encode("utf-8")
        if self._stage is not None:
            self._stage.append(data)
        self._total_bytes += len(data)
        self._newlines += text.count("\n")
        self._has_text = True
        self._ends_with_newline = text.endswith("\n")
        self._tail = _bounded_complete_line_tail(
            self._tail + text,
            max_bytes=self._max_bytes,
            max_lines=self._max_lines,
        )


def _bounded_complete_line_tail(text: str, *, max_bytes: int, max_lines: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    while lines and len("".join(lines).encode("utf-8")) > max_bytes:
        lines.pop(0)
    return "".join(lines)


__all__ = ["OutputStage", "StreamingToolOutput", "ToolOutputSnapshot"]
