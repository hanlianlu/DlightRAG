# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Read LightRAG's document and deletion status the same way everywhere.

LightRAG reports a status as a plain string or a ``DocStatus`` enum, on a
mapping or an object, depending on the call. Every DlightRAG decision about an
upstream outcome compares the one normalized spelling this returns.
"""

from collections.abc import Mapping


def lightrag_status(value: object) -> str:
    """The lower-cased status of a LightRAG record or result; empty when absent."""
    raw = value.get("status") if isinstance(value, Mapping) else getattr(value, "status", None)
    return str(getattr(raw, "value", raw) or "").strip().lower()


__all__ = ["lightrag_status"]
