# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral telemetry contracts and the standalone no-op adapter."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol


def safe_log_text(value: object, *, max_length: int = 240) -> str:
    """Return a bounded single-line string for telemetry and log fields."""
    text = str(value).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return f"{text[: max_length - 3]}..."


class Observation(Protocol):
    """One active operation that accepts neutral updates."""

    def update(self, **kwargs: Any) -> None: ...


class Telemetry(Protocol):
    """Create observations without coupling a core package to a telemetry SDK."""

    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Any | None = None,
        session_id: str | None = None,
    ) -> AbstractAsyncContextManager[Observation]: ...


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        del kwargs


class NoopTelemetry:
    """Standalone telemetry adapter that records nothing."""

    @asynccontextmanager
    async def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Any | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Observation]:
        del name, as_type, input, metadata, session_id
        yield _NoopObservation()


NOOP_TELEMETRY = NoopTelemetry()

__all__ = [
    "NOOP_TELEMETRY",
    "NoopTelemetry",
    "Observation",
    "Telemetry",
    "safe_log_text",
]
