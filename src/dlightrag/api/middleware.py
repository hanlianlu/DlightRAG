# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request ID middleware for correlation tracing.

Assigns a unique request ID to each incoming request (from X-Request-Id
header or generated UUID4). The ID is stored in a contextvar for access
throughout the request lifecycle and included in the response headers.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Per-request ID (accessible from any async code in the request scope)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into each request for log correlation."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = rid
            return response
        finally:
            request_id_var.reset(token)


_REQUEST_ID_FACTORY_MARKER = "_dlightrag_request_id_factory"


def install_request_id_log_record_factory() -> None:
    """Ensure every log record has a request_id attribute."""
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, _REQUEST_ID_FACTORY_MARKER, False):
        return

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = current_factory(*args, **kwargs)
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get("") or "-"
        return record

    setattr(record_factory, _REQUEST_ID_FACTORY_MARKER, True)
    logging.setLogRecordFactory(record_factory)
