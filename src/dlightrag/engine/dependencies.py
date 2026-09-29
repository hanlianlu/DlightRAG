# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Explicit, dependency-neutral classification for retryable interruptions.

Only failures that cross a typed dependency boundary, or a small set of known
client-library transport/status surfaces, are retryable.  Configuration,
authentication, schema, input, context-capacity, and unknown failures remain
non-retryable.
"""

from __future__ import annotations

import ssl
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Literal

import httpx

# The OpenAI and Anthropic SDKs ship their own httpx fork. They wrap a failed
# request, but an error while reading a streamed response reaches the caller as
# the fork's own transport exception.
_HTTP_CLIENTS: list[ModuleType] = [httpx]
try:
    import httpx2
except ImportError:  # pragma: no cover - installed with the provider SDKs
    pass
else:
    _HTTP_CLIENTS.append(httpx2)

type DependencyComponent = Literal["corpus_storage", "parser", "providers"]


class TransientDependencyError(RuntimeError):
    """A named external dependency is temporarily unavailable."""

    def __init__(self, component: DependencyComponent, message: str) -> None:
        self.component: DependencyComponent = component
        super().__init__(message)


class ProviderUnavailableError(TransientDependencyError):
    """A model provider request failed for an explicitly transient reason."""

    def __init__(self) -> None:
        super().__init__("providers", "Model provider is temporarily unavailable")


class ParserUnavailableError(TransientDependencyError):
    """The configured document parser is temporarily unavailable.

    A durable deferral on this error must be bounded. The verdict comes from the
    parser's connection or status alone, so a document that crashes or exhausts
    the parser service (an out-of-memory kill, for example) reads as a refused
    connection or a 5xx on every attempt.
    """

    def __init__(self) -> None:
        super().__init__("parser", "Document parser is temporarily unavailable")


#: A Run that dependency outages defer this many times fails instead of deferring
#: again. The count spans every component and belongs to the one Run: other and
#: new Runs keep their own, and each deferral keeps its exponential backoff.
MAX_DEPENDENCY_DEFERRALS = 10
_DEFERRALS_KEY = "dependency_deferrals"

_COMPONENT_NAMES: dict[DependencyComponent, str] = {
    "corpus_storage": "Corpus storage",
    "parser": "The document parser",
    "providers": "The model provider",
}


class DependencyRetriesExhausted(RuntimeError):
    """A Run spent its dependency deferrals; it fails with this public error."""

    kind = "dependency_unavailable"

    def __init__(self, component: DependencyComponent) -> None:
        self.component: DependencyComponent = component
        self.public_message = (
            f"{_COMPONENT_NAMES[component]} stayed unavailable through "
            f"{MAX_DEPENDENCY_DEFERRALS} retries, so this Run stopped. Try it again later."
        )
        super().__init__(self.public_message)


_AUTH_STATUS_CODES = frozenset({401, 403})
# 520-524 are an edge proxy (Cloudflare) reporting its origin failed, down,
# unreachable, or slow; 529 is an overloaded provider (Anthropic). 409 is left
# out on purpose: it reports a conflict with the target's current state, not an
# unavailable dependency, and a durable Run would otherwise defer on a conflict
# that resending the same request cannot resolve. The OpenAI and Anthropic SDKs
# still retry 409 within their own bounded budget.
_RETRYABLE_STATUS_CODES = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
)
# A refused or reset connection, a timed-out socket operation, or a server or
# proxy that dropped or refused the connection. An unsupported URL scheme or a
# local protocol violation is configuration or a client bug and stays
# non-retryable.
_TRANSIENT_HTTPX_ERRORS = tuple(
    error
    for client in _HTTP_CLIENTS
    for error in (
        client.TimeoutException,
        client.NetworkError,
        client.RemoteProtocolError,
        client.ProxyError,
    )
)
_HTTP_STATUS_ERRORS = tuple(client.HTTPStatusError for client in _HTTP_CLIENTS)
# OpenSSL reasons for a TLS protocol mismatch, such as an https URL for a
# plain-HTTP service (WRONG_VERSION_NUMBER) or no TLS version both sides accept.
_TLS_PROTOCOL_MISMATCH_REASONS = frozenset(
    {
        "NO_PROTOCOLS_AVAILABLE",
        "TLSV1_ALERT_PROTOCOL_VERSION",
        "UNKNOWN_PROTOCOL",  # OpenSSL 1.0's reason, kept for older builds
        "UNSUPPORTED_PROTOCOL",
        "WRONG_VERSION_NUMBER",
    }
)
_PROVIDER_MODULE_PREFIXES = ("openai", "anthropic", "google.genai", "google.api_core")
_STORAGE_MODULE_PREFIXES = ("asyncpg", "pymilvus", "grpc")
_AUTH_NAME_MARKERS = ("authentication", "unauthorized", "permissiondenied", "forbidden")
_TRANSIENT_PROVIDER_CLASS_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "DeadlineExceeded",
        "InternalServerError",
        "OverloadedError",
        "RateLimitError",
        "ServerError",
        "ServiceUnavailable",
        "ServiceUnavailableError",
        "TooManyRequests",
    }
)
# An error event inside an Anthropic stream arrives on the 200 response, so only
# its documented error type says the provider was overloaded, rate limited,
# timed out, or failed internally.
_TRANSIENT_ANTHROPIC_ERROR_TYPES = frozenset(
    {"api_error", "overloaded_error", "rate_limit_error", "timeout_error"}
)
_TRANSIENT_STORAGE_TEXT = (
    "broken pipe",
    "closed channel",
    "connection refused",
    "connection reset",
    "deadline exceeded",
    "fail connecting to server",
    "failed to connect",
    "ping timeout",
    "server unavailable",
    "temporarily unavailable",
)
_NON_RETRYABLE_TEXT = (
    "authentication",
    "context length",
    "context window",
    "credential",
    "forbidden",
    "invalid api key",
    "invalid input",
    "password authentication failed",
    "permission denied",
    "prompt is too long",
    "schema",
    "too many tokens",
    "unauthorized",
    "unsupported",
)


def classify_transient_dependency(
    exc: BaseException,
    *,
    component_hint: DependencyComponent | None = None,
) -> DependencyComponent | None:
    """Return the interrupted component only for an explicit transient failure.

    The whole cause chain is considered because adapters add useful domain
    context while retaining the original client exception.  Non-retryable
    markers win across the chain so an authentication or deterministic request
    rejection can never become retryable merely because a wrapper is present.
    """

    chain = tuple(_exception_chain(exc))
    if any(_is_non_retryable(item) for item in chain):
        return None
    # A client transport or provider connection failure caused by a
    # misconfigured endpoint is not an outage. Typed boundaries decide for
    # themselves, so a wrapper such as CorpusUnavailableError still defers.
    misconfigured = any(_is_misconfigured_endpoint(item) for item in chain)
    if not misconfigured and _is_transient_aiohttp_failure(exc):
        return component_hint or "providers"
    for item in chain:
        if isinstance(item, TransientDependencyError):
            return item.component
        if isinstance(item, TimeoutError | ConnectionError) and component_hint is not None:
            return component_hint
        if isinstance(item, _TRANSIENT_HTTPX_ERRORS):
            if misconfigured:
                continue
            return component_hint or "providers"
        if isinstance(item, _HTTP_STATUS_ERRORS):
            if _status_code(item) in _RETRYABLE_STATUS_CODES:
                return component_hint or "providers"
            continue
        module = type(item).__module__
        name = type(item).__name__
        status = _status_code(item)
        if module.startswith(_PROVIDER_MODULE_PREFIXES) and not misconfigured:
            if (
                status in _RETRYABLE_STATUS_CODES
                or name in _TRANSIENT_PROVIDER_CLASS_NAMES
                or (
                    module.startswith("anthropic")
                    and _anthropic_error_type(item) in _TRANSIENT_ANTHROPIC_ERROR_TYPES
                )
            ):
                return "providers"
        if module.startswith(_STORAGE_MODULE_PREFIXES):
            if status in _RETRYABLE_STATUS_CODES or any(
                marker in str(item).lower() for marker in _TRANSIENT_STORAGE_TEXT
            ):
                return "corpus_storage"
    return None


def is_transient_request_failure(exc: BaseException, *, text_vetoes: bool = True) -> bool:
    """Return whether one HTTP request failed for an explicitly transient reason.

    This is the request-level half of :func:`classify_transient_dependency` for
    code that retries or names a failed request itself (the embedding client and
    the document-parser transport boundary), so it agrees with durable deferral:
    a transient transport error or retryable status anywhere in the cause chain,
    and no non-retryable marker or misconfigured endpoint anywhere in it.

    ``text_vetoes=False`` skips the message-text markers and keeps only the
    status, exception-type, and endpoint vetoes, for a boundary whose error
    text carries caller-controlled content (a parser names the user's file).
    """

    chain = tuple(_exception_chain(exc))
    if any(
        _is_non_retryable(item, text=text_vetoes) or _is_misconfigured_endpoint(item)
        for item in chain
    ):
        return False
    return _is_transient_aiohttp_failure(exc) or any(
        isinstance(item, _TRANSIENT_HTTPX_ERRORS) or _status_code(item) in _RETRYABLE_STATUS_CODES
        for item in chain
    )


def next_dependency_retry(
    checkpoint: Mapping[str, Any] | None,
    component: DependencyComponent,
    *,
    base_seconds: int = 5,
    max_seconds: int = 60,
    outage: bool = True,
) -> tuple[dict[str, Any], int]:
    """Build one bounded, secret-free durable retry checkpoint and delay.

    An outage counts toward the Run's ``MAX_DEPENDENCY_DEFERRALS``, and the one past
    them raises ``DependencyRetriesExhausted`` instead. A wait that is no outage
    (``outage=False``), such as a write fence, keeps its backoff and counts nothing.
    """

    key = (
        "corpus_unavailable_attempt"
        if component == "corpus_storage"
        else f"{component}_unavailable_attempt"
    )
    attempt = _checkpoint_count(checkpoint, key) + 1
    # Bound the persisted counter as well as exponentiation.  Once the delay is
    # capped, a larger counter carries no scheduling information.
    capped_attempt = min(attempt, 32)
    delay = min(max_seconds, base_seconds * (2 ** (min(capped_attempt, 8) - 1)))
    retry: dict[str, Any] = {key: capped_attempt}
    deferrals = _checkpoint_count(checkpoint, _DEFERRALS_KEY) + int(outage)
    if deferrals > MAX_DEPENDENCY_DEFERRALS:
        raise DependencyRetriesExhausted(component)
    if deferrals:
        retry[_DEFERRALS_KEY] = deferrals
    return (retry, delay)


def _checkpoint_count(checkpoint: Mapping[str, Any] | None, key: str) -> int:
    value: Any = checkpoint.get(key) if isinstance(checkpoint, Mapping) else None
    try:
        return max(0, int(value or 0))
    except TypeError, ValueError:
        return 0


def dependency_component_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> DependencyComponent | None:
    """Read only this module's closed component vocabulary from a checkpoint."""

    if not isinstance(checkpoint, Mapping):
        return None
    for component, key in (
        ("corpus_storage", "corpus_unavailable_attempt"),
        ("parser", "parser_unavailable_attempt"),
        ("providers", "providers_unavailable_attempt"),
    ):
        if key in checkpoint:
            return component  # type: ignore[return-value]
    return None


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_non_retryable(exc: BaseException, *, text: bool = True) -> bool:
    status = _status_code(exc)
    if status in _AUTH_STATUS_CODES or (
        status is not None and 400 <= status < 500 and status not in _RETRYABLE_STATUS_CODES
    ):
        return True
    name = type(exc).__name__.lower()
    if any(marker in name for marker in _AUTH_NAME_MARKERS):
        return True
    if not text:
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _NON_RETRYABLE_TEXT)


def _is_misconfigured_endpoint(exc: BaseException) -> bool:
    """A TLS certificate that fails verification, or a TLS protocol mismatch.

    Both surface as a connection failure, yet resending cannot help. Only these
    named failures count as misconfiguration. Any other TLS error in a chain,
    such as the SSLWantReadError a peer resetting the handshake leaves behind,
    and any DNS failure (a stopped Compose service does not resolve until it
    restarts) are not, so the transport error carrying them decides whether
    the failure is transient.
    """

    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    return (
        isinstance(exc, ssl.SSLError)
        and getattr(exc, "reason", None) in _TLS_PROTOCOL_MISMATCH_REASONS
    )


def _is_transient_aiohttp_failure(exc: BaseException) -> bool:
    """Whether aiohttp itself raised a broken connection, response, or proxy 5xx.

    google-genai sends its async requests through aiohttp and raises aiohttp's
    errors unwrapped. Libraries that wrap them (aiobotocore for S3, azure-core
    for Azure Blob) own their failures, so an aiohttp error counts only when it
    is the exception raised. aiohttp is looked up, not imported: if it was never
    imported, no aiohttp error can be in the chain.
    """

    aiohttp: Any = sys.modules.get("aiohttp")
    parser_errors: Any = sys.modules.get("aiohttp.http_exceptions")
    if aiohttp is None or parser_errors is None:
        return False
    if isinstance(
        exc,
        aiohttp.ClientOSError  # every connector failure: refused, reset, DNS, proxy, TLS
        | aiohttp.ClientConnectionResetError
        | aiohttp.ServerDisconnectedError
        | aiohttp.ServerTimeoutError,
    ):
        return True
    if isinstance(exc, aiohttp.ClientPayloadError):
        # A body the peer cut short, not one the client could not decode.
        return any(
            isinstance(item, parser_errors.ContentLengthError | parser_errors.TransferEncodingError)
            for item in _exception_chain(exc)
        )
    if isinstance(exc, aiohttp.ClientResponseError):  # includes a proxy refusing CONNECT
        return _status_code(exc) in _RETRYABLE_STATUS_CODES
    return False


def _anthropic_error_type(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, Mapping) else None
    kind = error.get("type") if isinstance(error, Mapping) else None
    return kind if isinstance(kind, str) else None


def _status_code(exc: BaseException) -> int | None:
    aiohttp: Any = sys.modules.get("aiohttp")
    parser_errors: Any = sys.modules.get("aiohttp.http_exceptions")
    if parser_errors is not None and isinstance(exc, parser_errors.HttpProcessingError):
        return None  # an HTTP parser error code (400 for a truncated body), not a status
    if aiohttp is not None and isinstance(exc, aiohttp.ClientResponseError):
        status = getattr(exc, "status", None)  # .code is a deprecated alias
        return status if isinstance(status, int) and not isinstance(status, bool) else None
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raw_value = getattr(value, "value", None)
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            return raw_value
    return None


__all__ = [
    "MAX_DEPENDENCY_DEFERRALS",
    "DependencyComponent",
    "DependencyRetriesExhausted",
    "ParserUnavailableError",
    "ProviderUnavailableError",
    "TransientDependencyError",
    "classify_transient_dependency",
    "dependency_component_from_checkpoint",
    "is_transient_request_failure",
    "next_dependency_retry",
]
