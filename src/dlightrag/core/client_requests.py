# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request projection for client-facing adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dlightrag.core.client_contracts import dump_optional_list
from dlightrag.core.client_payloads import metadata_filter_from_payload


def _get(payload: Any, name: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name, default)
    return getattr(payload, name, default)


def query_kwargs_from_payload(
    payload: Any,
    *,
    include_multimodal_content: bool = True,
) -> dict[str, Any]:
    """Return manager keyword arguments shared by retrieve/answer clients."""
    kwargs: dict[str, Any] = {}

    filters = metadata_filter_from_payload(_get(payload, "filters"))
    if filters is not None:
        kwargs["filters"] = filters

    multimodal_content = _get(payload, "multimodal_content")
    if include_multimodal_content and multimodal_content:
        kwargs["multimodal_content"] = dump_optional_list(multimodal_content)

    query_images = _get(payload, "query_images")
    if query_images:
        kwargs["query_images"] = dump_optional_list(query_images)

    session_id = _get(payload, "session_id")
    if session_id:
        kwargs["session_id"] = session_id

    referenced_image_ids = _get(payload, "referenced_image_ids")
    if referenced_image_ids:
        kwargs["referenced_image_ids"] = referenced_image_ids

    return kwargs


def ingest_kwargs_from_payload(payload: Any) -> dict[str, Any]:
    """Return manager ingest kwargs from REST, MCP, or CLI payload objects."""
    source_type = _get(payload, "source_type")
    kwargs: dict[str, Any] = {}

    if source_type == "local":
        kwargs["path"] = _get(payload, "path")
    elif source_type == "azure_blob":
        kwargs["container_name"] = _get(payload, "container_name")
        blob_path = _get(payload, "blob_path")
        prefix = _get(payload, "prefix")
        if blob_path:
            kwargs["blob_path"] = blob_path
        if prefix is not None:
            kwargs["prefix"] = prefix
    elif source_type == "s3":
        kwargs["bucket"] = _get(payload, "bucket")
        key = _get(payload, "key")
        prefix = _get(payload, "prefix")
        if key:
            kwargs["key"] = key
        if prefix is not None:
            kwargs["prefix"] = prefix

    for name in ("replace", "title", "author", "metadata", "metadata_policy"):
        value = _get(payload, name)
        if value is not None:
            kwargs[name] = value

    return kwargs


def query_image_blocks_from_urls(values: Sequence[str]) -> list[dict[str, Any]]:
    """Wrap CLI-supplied image URLs/data URIs as modern image_url content blocks."""
    return [{"type": "image_url", "image_url": {"url": value}} for value in values]


__all__ = [
    "ingest_kwargs_from_payload",
    "query_image_blocks_from_urls",
    "query_kwargs_from_payload",
]
