# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request ID middleware for correlation tracing.

Assigns a unique request ID to each incoming request (from X-Request-Id
header or generated UUID4). The ID is stored in a contextvar for access
throughout the request lifecycle and included in the response headers.
"""

import contextvars
import logging
import uuid
from typing import Any

from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dlightrag.api.models import ErrorDetail

# Per-request ID (accessible from any async code in the request scope)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class JsonBodyLimitMiddleware:
    """Keep an oversized JSON body out of memory.

    A declared Content-Length above the cap is refused; a chunked body that
    overruns it is cut short, so the route reads truncated JSON and answers 422
    rather than buffering whatever the client sends.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if "application/json" not in headers.get("content-type", "").lower():
            await self.app(scope, receive, send)
            return
        declared = headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            body = ErrorDetail(detail="Request body is too large", error_type="validation")
            await JSONResponse(status_code=413, content=body.model_dump())(scope, receive, send)
            return
        await self.app(scope, self._capped(receive), send)

    def _capped(self, receive: Receive) -> Receive:
        seen = 0

        async def capped() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] != "http.request":
                return message
            chunk = message.get("body", b"")
            if seen + len(chunk) > self._max_bytes:
                return {"type": "http.request", "body": chunk[: self._max_bytes - seen]}
            seen += len(chunk)
            return message

        return capped


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
