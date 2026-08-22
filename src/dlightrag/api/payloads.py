# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-payload coercion for the REST transport."""

from collections.abc import Mapping
from typing import Any

from dlightrag.rag.retrieval import MetadataFilter


def metadata_filter_from_payload(payload: Any | None) -> MetadataFilter | None:
    """Build a MetadataFilter from pydantic or plain-dict request payloads."""
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_none=True)
    elif isinstance(payload, Mapping):
        data = {key: value for key, value in payload.items() if value is not None}
    else:
        data = dict(payload)
    if not data:
        return None
    return MetadataFilter(**data)


__all__ = [
    "metadata_filter_from_payload",
]
