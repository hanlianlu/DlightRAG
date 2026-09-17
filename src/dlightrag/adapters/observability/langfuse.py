# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Langfuse client lifecycle and process-wide tracing state."""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from dlightrag.adapters.observability.masking import mask_langfuse_payload

logger = logging.getLogger(__name__)

_client: Any = None
_trace_sensitive: bool = True
_LANGFUSE_TRACER_SCOPE = "langfuse-sdk"


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


def install_client(client: Any | None, *, trace_sensitive: bool) -> None:
    """Install one client and privacy policy as an atomic process state update."""
    global _client, _trace_sensitive
    _client = client
    _trace_sensitive = trace_sensitive


def init_tracing(config: Any) -> None:
    """Initialize Langfuse from its narrow settings, or disable it safely."""
    trace_sensitive = bool(getattr(config, "langfuse_trace_sensitive_data", True))
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        install_client(None, trace_sensitive=trace_sensitive)
        logger.info("Langfuse tracing disabled (keys missing in config)")
        return

    try:
        from langfuse import Langfuse

        kwargs: dict[str, Any] = {
            "public_key": config.langfuse_public_key,
            "secret_key": config.langfuse_secret_key,
            "base_url": config.langfuse_host,
            "mask": mask_langfuse_payload,
        }
        optional_kwargs = {
            "environment": getattr(config, "langfuse_environment", None),
            "release": getattr(config, "langfuse_release", None) or _package_release(),
            "sample_rate": getattr(config, "langfuse_sample_rate", None),
            "timeout": getattr(config, "langfuse_timeout", None),
            "flush_at": getattr(config, "langfuse_flush_at", None),
            "flush_interval": getattr(config, "langfuse_flush_interval", None),
        }
        kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})
        if not getattr(config, "langfuse_export_external_spans", False):
            kwargs["should_export_span"] = _is_dlight_observation_span
        install_client(Langfuse(**kwargs), trace_sensitive=trace_sensitive)
        logger.info("Langfuse tracing enabled → %s", config.langfuse_host)
    except Exception:
        install_client(None, trace_sensitive=trace_sensitive)
        logger.warning(
            "Langfuse enabled but initialization failed. Falling back to tracing disabled.",
            exc_info=True,
        )


def trace_sensitive_enabled() -> bool:
    return _trace_sensitive


def shutdown_tracing() -> None:
    """Flush pending events and stop SDK background resources."""
    global _client
    client = _client
    _client = None
    if client is None:
        return
    try:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
    except Exception:
        logger.debug("Langfuse shutdown failed (non-fatal)", exc_info=True)


def _is_dlight_observation_span(span: Any) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None) == _LANGFUSE_TRACER_SCOPE


__all__ = [
    "current_client",
    "init_tracing",
    "install_client",
    "shutdown_tracing",
    "trace_sensitive_enabled",
]
