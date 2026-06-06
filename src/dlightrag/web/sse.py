# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Server-sent event serialization helpers for the web UI."""

from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, payload: Any) -> str:
    """Return one JSON-encoded SSE event."""
    clean_event = event.replace("\r", "").replace("\n", "")
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {clean_event}\ndata: {data}\n\n"


__all__ = ["sse_event"]
