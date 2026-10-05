# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Langfuse client lifecycle and process-wide tracing state."""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

from dlightrag.adapters.observability.masking import mask_langfuse_payload

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

_client: Any = None
_provider: TracerProvider | None = None
_trace_sensitive: bool = True


@contextmanager
def trace_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[None]:
    """Attribute the enclosing trace; every observation in scope inherits it."""
    session_id = _propagatable(session_id, field="session_id")
    user_id = _propagatable(user_id, field="user_id")
    if current_client() is None or (session_id is None and user_id is None):
        yield
        return
    stack = ExitStack()
    try:
        from langfuse import propagate_attributes

        stack.enter_context(propagate_attributes(session_id=session_id, user_id=user_id))
    except Exception:
        stack.close()
        logger.debug("Langfuse trace attribution failed (non-fatal)", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        stack.close()


def current_client() -> Any | None:
    """Return the process Langfuse client, if tracing is enabled."""
    return _client


def _propagatable(value: str | None, *, field: str) -> str | None:
    """Return a value Langfuse will keep; it silently drops the ones it will not."""
    if value is None:
        return None
    if not value.isascii() or not value or len(value) > 200:
        logger.warning("Dropping %s that Langfuse cannot attribute: %d chars", field, len(value))
        return None
    return value


def _package_release() -> str:
    """Default the Langfuse release to the running DlightRAG version.

    Read from installed metadata rather than the package root: importing
    ``dlightrag`` pulls the composition root and every transport with it.
    """
    try:
        from importlib.metadata import version

        return version("dlightrag")
    except Exception:
        return "unknown"


def install_client(
    client: Any | None,
    *,
    trace_sensitive: bool,
    provider: TracerProvider | None = None,
) -> None:
    """Install a client, its tracer provider, and the privacy policy as one state update."""
    global _client, _provider, _trace_sensitive
    _client = client
    _provider = provider
    _trace_sensitive = trace_sensitive


def init_tracing(config: Any) -> None:
    """Initialize Langfuse from its narrow settings, or disable it safely."""
    trace_sensitive = bool(getattr(config, "langfuse_trace_sensitive_data", True))
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        install_client(None, trace_sensitive=trace_sensitive)
        logger.info("Langfuse tracing disabled (keys missing in config)")
        return

    provider: TracerProvider | None = None
    try:
        from langfuse import Langfuse

        environment = getattr(config, "langfuse_environment", None)
        release = getattr(config, "langfuse_release", None) or _package_release()
        sample_rate = getattr(config, "langfuse_sample_rate", None)
        provider = _tracer_provider(
            environment=environment, release=release, sample_rate=sample_rate
        )
        kwargs: dict[str, Any] = {
            "public_key": config.langfuse_public_key,
            "secret_key": config.langfuse_secret_key,
            "base_url": config.langfuse_host,
            "mask": mask_langfuse_payload,
            "tracer_provider": provider,
        }
        optional_kwargs = {
            "environment": environment,
            "release": release,
            "sample_rate": sample_rate,
            "timeout": getattr(config, "langfuse_timeout", None),
            "flush_at": getattr(config, "langfuse_flush_at", None),
            "flush_interval": getattr(config, "langfuse_flush_interval", None),
        }
        kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})
        install_client(Langfuse(**kwargs), trace_sensitive=trace_sensitive, provider=provider)
        logger.info("Langfuse tracing enabled → %s", config.langfuse_host)
    except Exception:
        install_client(None, trace_sensitive=trace_sensitive)
        _stop_provider(provider)
        logger.warning(
            "Langfuse enabled but initialization failed. Falling back to tracing disabled.",
            exc_info=True,
        )


def _tracer_provider(
    *, environment: str | None, release: str, sample_rate: float | None
) -> TracerProvider:
    """Build the OpenTelemetry provider that only Langfuse uses.

    Left to itself the SDK registers its provider as the process-global one, so the
    spans other libraries open (FastAPI's request span, the MCP SDK's) become recording
    parents of an observation opened under them. Those parents are never exported, and
    the observation reaches Langfuse as a trace without a root. With a provider of its
    own, those spans do not record and every observation is a true root.

    It is built as the SDK builds its default one: the release and environment ride on
    the resource, and sampling is a trace-id ratio only below 1.
    """
    from langfuse import LangfuseOtelSpanAttributes
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    attributes = {LangfuseOtelSpanAttributes.RELEASE: release}
    if environment is not None:
        attributes[LangfuseOtelSpanAttributes.ENVIRONMENT] = environment
    return TracerProvider(
        resource=Resource.create(attributes),
        sampler=TraceIdRatioBased(sample_rate)
        if sample_rate is not None and sample_rate < 1
        else None,
    )


def trace_sensitive_enabled() -> bool:
    return _trace_sensitive


def shutdown_tracing() -> None:
    """Flush pending events, then stop the SDK's background resources and the provider.

    The SDK keeps one resource manager per public key for the life of the process, so
    tracing is not started again with the same key afterwards: that client would reuse
    the manager, and with it the provider stopped here.
    """
    global _client, _provider
    client, provider = _client, _provider
    _client = None
    _provider = None
    if client is None:
        return
    try:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            flush = getattr(client, "flush", None)
            if callable(flush):
                flush()
    except Exception:
        logger.debug("Langfuse shutdown failed (non-fatal)", exc_info=True)
    finally:
        _stop_provider(provider)


def _stop_provider(provider: TracerProvider | None) -> None:
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception:
        logger.debug("Langfuse tracer provider shutdown failed (non-fatal)", exc_info=True)


__all__ = [
    "current_client",
    "init_tracing",
    "install_client",
    "shutdown_tracing",
    "trace_sensitive_enabled",
]
