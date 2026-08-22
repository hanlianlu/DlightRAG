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


def bounded_telemetry_text(value: object, *, max_length: int = 4000) -> str:
    """Bound one telemetry string while preserving its original line structure."""
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}... [truncated {len(text) - max_length} chars]"


def telemetry_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound model-message text and remove inline image bytes before telemetry."""
    summarized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            safe_content: Any = bounded_telemetry_text(content)
        elif isinstance(content, list):
            safe_content = []
            for block in content:
                if isinstance(block, str):
                    safe_content.append(bounded_telemetry_text(block))
                elif isinstance(block, dict) and block.get("type") == "text":
                    safe_content.append(
                        {
                            **block,
                            "text": bounded_telemetry_text(block.get("text", "")),
                        }
                    )
                elif isinstance(block, dict) and block.get("type") == "image_url":
                    safe_content.append({"type": "image_url", "image_url": "[image omitted]"})
                else:
                    safe_content.append(block)
        else:
            safe_content = content
        summarized.append({**message, "content": safe_content})
    return summarized


def telemetry_error_message(telemetry: Telemetry, exc: BaseException) -> str:
    """Return raw error text only when the injected privacy policy permits it."""
    return str(exc) if telemetry.capture_sensitive_data else type(exc).__name__


class Observation(Protocol):
    """One active operation that accepts neutral updates."""

    def update(self, **kwargs: Any) -> None: ...


class Telemetry(Protocol):
    """Create observations without coupling a core package to a telemetry SDK."""

    @property
    def capture_sensitive_data(self) -> bool: ...

    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Any | None = None,
        session_id: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Observation]: ...


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        del kwargs


class NoopTelemetry:
    """Standalone telemetry adapter that records nothing."""

    capture_sensitive_data = False

    @asynccontextmanager
    async def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Any | None = None,
        session_id: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AsyncIterator[Observation]:
        del name, as_type, input, metadata, session_id, model, model_parameters
        yield _NoopObservation()


NOOP_TELEMETRY = NoopTelemetry()

__all__ = [
    "NOOP_TELEMETRY",
    "NoopTelemetry",
    "Observation",
    "Telemetry",
    "bounded_telemetry_text",
    "safe_log_text",
    "telemetry_error_message",
    "telemetry_messages",
]
