# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Serve public HTTP reads from a handler, through the client the reads build.

Public HTTP acquisition builds one anonymous client per request. Tests keep that
client and its settings and replace only where its requests go, so they observe
exactly the requests production sends.
"""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dlightrag.engine import public_http

_AsyncClient = httpx.AsyncClient


def serve_public_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Answer every request public HTTP acquisition sends with ``handler``."""

    class _Served(_AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(public_http.httpx, "AsyncClient", _Served)
