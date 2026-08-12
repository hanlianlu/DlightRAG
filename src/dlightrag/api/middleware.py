# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request ID middleware for correlation tracing.

Assigns a unique request ID to each incoming request (from X-Request-Id
header or generated UUID4). The ID is stored in a contextvar for access
throughout the request lifecycle and included in the response headers.
"""

import contextvars
import logging
import uuid
from collections.abc import Callable
from typing import Any

from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dlightrag.api.models import ErrorDetail

# Per-request ID (accessible from any async code in the request scope)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class _RequestBodyTooLarge(Exception):
    pass


def _split_body_too_large(
    exc: BaseException,
) -> tuple[BaseException | None, BaseException | None]:
    if isinstance(exc, _RequestBodyTooLarge):
        return exc, None
    if isinstance(exc, BaseExceptionGroup):
        return exc.split(_RequestBodyTooLarge)
    return None, exc


def _request_path(scope: Scope) -> str:
    path = str(scope.get("path") or "")
    root_path = str(scope.get("root_path") or "")
    if root_path and (path == root_path or path.startswith(root_path + "/")):
        return path[len(root_path) :] or "/"
    return path


class RequestBodyLimitMiddleware:
    """Keep oversized request bodies out of memory and temporary storage.

    Declared and chunked bodies return 413 when they exceed the active cap.
    Multipart routes use the general upload cap unless a tighter path-specific
    policy applies.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        multipart_path_max_bytes: dict[str, int] | None = None,
    ) -> None:
        self.app = app
        self._max_bytes = max_bytes
        self._multipart_path_max_bytes = dict(multipart_path_max_bytes or {})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        content_type = headers.get("content-type", "").lower()
        is_multipart = "multipart/form-data" in content_type
        path = _request_path(scope)
        max_bytes = self._max_bytes
        if is_multipart:
            max_bytes = self._multipart_path_max_bytes.get(path, self._max_bytes)
        declared = headers.get("content-length", "")
        if declared.isdecimal() and int(declared) > max_bytes:
            await self._send_too_large(scope, receive, send)
            return
        response_started = False
        overflowed = False
        replacement_sent = False

        def mark_overflow() -> None:
            nonlocal overflowed
            overflowed = True

        async def tracked_send(message: Message) -> None:
            nonlocal response_started, replacement_sent
            if replacement_sent:
                return
            if message["type"] == "http.response.start":
                if overflowed:
                    replacement_sent = True
                    await self._send_too_large(scope, receive, send)
                    return
                response_started = True
            await send(message)

        try:
            await self.app(
                scope,
                self._strictly_capped(receive, max_bytes, mark_overflow),
                tracked_send,
            )
        except BaseException as exc:
            matched, remainder = _split_body_too_large(exc)
            if response_started or replacement_sent or matched is None:
                raise
            if remainder is not None:
                raise remainder from exc
            await self._send_too_large(scope, receive, send)

    @staticmethod
    async def _send_too_large(scope: Scope, receive: Receive, send: Send) -> None:
        body = ErrorDetail(detail="Request body is too large", error_type="validation")
        await JSONResponse(status_code=413, content=body.model_dump())(scope, receive, send)

    @staticmethod
    def _strictly_capped(
        receive: Receive,
        max_bytes: int,
        mark_overflow: Callable[[], None],
    ) -> Receive:
        seen = 0

        async def capped() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] != "http.request":
                return message
            seen += len(message.get("body", b""))
            if seen > max_bytes:
                mark_overflow()
                raise _RequestBodyTooLarge
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
