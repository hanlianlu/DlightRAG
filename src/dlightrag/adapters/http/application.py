# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Access to the one Application instance bound by the HTTP lifespan."""

from __future__ import annotations

from typing import Any

from starlette.requests import HTTPConnection

from dlightrag.application.errors import ApplicationClosedError


def get_application(connection: HTTPConnection) -> Any:
    """Return the lifespan-bound Application; never compose a fallback service.

    The connection is a request or a WebSocket: both are bound to the same ASGI app, and
    the live preview's frames socket asks the same question of it a route does.
    """
    application = getattr(connection.app.state, "application", None)
    if application is None:
        raise ApplicationClosedError("Application is not bound to this HTTP transport")
    return application


__all__ = ["get_application"]
