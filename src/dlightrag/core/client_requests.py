# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request projection for client-facing adapters."""

from collections.abc import Mapping, Sequence
from typing import Any

from dlightrag.core.client_contracts import dump_optional_list
from dlightrag.core.client_payloads import metadata_filter_from_payload


def _get(payload: Any, name: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name, default)
    return getattr(payload, name, default)


def query_kwargs_from_payload(payload: Any) -> dict[str, Any]:
    """Return query keyword arguments shared by retrieve/answer clients."""
    kwargs: dict[str, Any] = {}

    filters = metadata_filter_from_payload(_get(payload, "filters"))
    if filters is not None:
        kwargs["filters"] = filters

    bm25_query = _get(payload, "bm25_query")
    if bm25_query:
        kwargs["bm25_query"] = bm25_query

    query_images = _get(payload, "query_images")
    if query_images:
        kwargs["query_images"] = dump_optional_list(query_images)

    return kwargs


def query_image_blocks_from_urls(values: Sequence[str]) -> list[dict[str, Any]]:
    """Wrap CLI-supplied image URLs/data URIs as modern image_url content blocks."""
    return [{"type": "image_url", "image_url": {"url": value}} for value in values]


__all__ = [
    "query_image_blocks_from_urls",
    "query_kwargs_from_payload",
]
