# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral payload projection for API, MCP, and other clients."""

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from dlightrag.citations.source_builder import build_sources
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver

_PUBLIC_CHUNK_KEYS = (
    "reference_id",
    "file_path",
    "content",
    "page_idx",
    "bbox",
    "image_url",
    "thumbnail_url",
    "image_mime_type",
    "relevance_score",
    "metadata",
    "_workspace",
)


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


def project_contexts_for_client(
    contexts: RetrievalContexts,
    *,
    image_url_prefix: str | None = "/images",
) -> RetrievalContexts:
    """Return client-safe contexts without inline base64 image data."""
    chunks = [
        chunk
        for item in contexts.get("chunks", [])
        if (chunk := _project_chunk_context(item, image_url_prefix=image_url_prefix)) is not None
    ]
    return {
        "chunks": chunks,
        "entities": [dict(item) for item in contexts.get("entities", [])],
        "relationships": [dict(item) for item in contexts.get("relationships", [])],
    }


def _project_chunk_context(
    item: dict[str, Any],
    *,
    image_url_prefix: str | None,
) -> dict[str, Any] | None:
    row = dict(item)
    chunk_id = row.get("chunk_id") or row.get("id")
    if chunk_id is None:
        return None

    payload = {key: row[key] for key in _PUBLIC_CHUNK_KEYS if row.get(key) is not None}
    payload = {
        **{
            "chunk_id": str(chunk_id),
            "reference_id": "",
            "file_path": "",
            "content": "",
        },
        **payload,
    }
    image_data = row.get("image_data")
    if image_data and image_url_prefix and row.get("_workspace") and chunk_id:
        base_path = (
            f"{image_url_prefix.rstrip('/')}/"
            f"{quote(str(row['_workspace']), safe='')}/"
            f"{quote(str(chunk_id), safe='')}"
        )
        payload.setdefault("image_url", f"{base_path}?size=full")
        payload.setdefault("thumbnail_url", f"{base_path}?size=thumb")
    return payload


def retrieval_payload(
    result: RetrievalResult,
    *,
    source_url_resolver: SourceUrlResolver | None = None,
    image_url_prefix: str | None = "/images",
) -> dict[str, Any]:
    """Project retrieval results into a client-safe response dictionary."""
    contexts = project_contexts_for_client(result.contexts, image_url_prefix=image_url_prefix)
    sources = build_sources(
        contexts,
        source_url_resolver=source_url_resolver,
        image_url_prefix=image_url_prefix,
    )
    return {
        "answer": result.answer,
        "contexts": contexts,
        "sources": [source.model_dump() for source in sources],
        "trace": result.trace,
        "image_descriptions": result.image_descriptions,
        "current_image_ids": result.current_image_ids,
    }


def answer_payload(
    result: RetrievalResult,
    *,
    image_url_prefix: str | None = "/images",
) -> dict[str, Any]:
    """Project generated answer results into a client-safe response dictionary."""
    return {
        "answer": result.answer,
        "contexts": project_contexts_for_client(
            result.contexts,
            image_url_prefix=image_url_prefix,
        ),
        "references": [{"id": ref.id, "title": ref.title} for ref in result.references],
        "sources": [source.model_dump() for source in result.sources],
        "trace": result.trace,
        "image_descriptions": result.image_descriptions,
        "current_image_ids": result.current_image_ids,
    }


__all__ = [
    "answer_payload",
    "metadata_filter_from_payload",
    "project_contexts_for_client",
    "retrieval_payload",
]
