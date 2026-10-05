# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral telemetry contracts, the span vocabulary, and the no-op adapter.

The vocabulary is the observability contract. A span name is an API for
reviewers, dashboards, and evaluators, so names are closed, stable, and
verb-first, and never carry a model id, workspace, run, or other dynamic
value: those belong in ``model``, ``metadata``, or attribution. One unit of
work owns exactly one root observation, and everything it does nests inside
it: :meth:`Telemetry.trace` attributes a trace (conversation session and user)
for every observation opened in its scope, so no call site threads ids. Startup
capability probes and sweeps are not units of work and open no observation.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    nullcontext,
)
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

type SpanName = Literal[
    "call-rerank-model",
    "embed-text",
    "execute-agent-tool",
    "generate-agent-turn",
    "generate-answer",
    "generate-completion",
    "compact-session",
    "highlight-sources",
    "ingest-documents",
    "plan-retrieval",
    "rerank-passages",
    "retrieve-context",
    "run-answer",
    "run-retrieval",
]

type SpanType = Literal[
    "agent",
    "chain",
    "embedding",
    "generation",
    "retriever",
    "span",
    "tool",
]

SPAN_TYPES: Final[Mapping[SpanName, SpanType]] = MappingProxyType(
    {
        "call-rerank-model": "span",
        "embed-text": "embedding",
        "execute-agent-tool": "tool",
        "generate-agent-turn": "generation",
        "generate-answer": "chain",
        "generate-completion": "generation",
        "compact-session": "generation",
        "highlight-sources": "chain",
        "ingest-documents": "chain",
        "plan-retrieval": "chain",
        "rerank-passages": "span",
        "retrieve-context": "retriever",
        "run-answer": "agent",
        "run-retrieval": "chain",
    }
)


#: Name fragments whose values are secrets, in settings fields, provider options
#: such as request headers, and telemetry payloads. One list, so a secret hidden
#: from a settings dump is hidden from an exported trace too.
SECRET_KEY_PATTERNS: tuple[str, ...] = (
    "api_key",
    "api-key",
    "api_secret",
    "api_token",
    "authorization",
    "secret",
    "verification_key",
    "password",
    "connection_string",
    "milvus_uri",
    "account_key",
    "sas_token",
    "token",
)


def is_secret_key(key: object) -> bool:
    """Whether a field or header name holds a secret."""
    normalized = str(key).lower()
    return any(pattern in normalized for pattern in SECRET_KEY_PATTERNS)


def hides_secret_value(key: object, value: object) -> bool:
    """Whether a value under the secret name ``key`` must be hidden.

    Anything that can carry the secret hides unless it is empty: text, bytes,
    containers, models and other objects alike, so a value type nobody listed
    cannot slip out. A flag stays readable, since one bit carries no credential.
    A number stays readable only as a count under a name that merely contains
    "token" (``max_tokens``, ``chunk_token_size``); under any other secret name,
    or a name that ends in "token" (``otp_token``), a number can be the
    credential itself.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return not _names_a_token_count(key)
    try:
        return bool(value)
    except Exception:  # a truth value that cannot be read still hides
        return True


def _names_a_token_count(key: object) -> bool:
    normalized = str(key).lower()
    matched = {pattern for pattern in SECRET_KEY_PATTERNS if pattern in normalized}
    return matched == {"token"} and not normalized.endswith("token")


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

    def trace(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AbstractContextManager[None]: ...

    def observe(
        self,
        name: SpanName,
        *,
        input: Any | None = None,
        metadata: Any | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Observation]: ...


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        del kwargs


class NoopTelemetry:
    """Standalone telemetry adapter that records nothing."""

    capture_sensitive_data = False

    def trace(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AbstractContextManager[None]:
        del session_id, user_id
        return nullcontext()

    @asynccontextmanager
    async def observe(
        self,
        name: SpanName,
        *,
        input: Any | None = None,
        metadata: Any | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AsyncIterator[Observation]:
        del name, input, metadata, model, model_parameters
        yield _NoopObservation()


NOOP_TELEMETRY = NoopTelemetry()

__all__ = [
    "NOOP_TELEMETRY",
    "SECRET_KEY_PATTERNS",
    "SPAN_TYPES",
    "NoopTelemetry",
    "Observation",
    "SpanName",
    "SpanType",
    "Telemetry",
    "bounded_telemetry_text",
    "hides_secret_value",
    "is_secret_key",
    "safe_log_text",
    "telemetry_error_message",
    "telemetry_messages",
]
