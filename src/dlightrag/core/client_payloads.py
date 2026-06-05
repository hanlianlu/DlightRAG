# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral payload projection for API, MCP, and other clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from dlightrag.citations.source_builder import build_sources
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult


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
    public: RetrievalContexts = {}
    for key, items in contexts.items():
        if not isinstance(items, list):
            continue
        public_items: list[dict[str, Any]] = []
        for item in items:
            row = dict(item)
            image_data = row.pop("image_data", None)
            if key == "chunks" and image_data and image_url_prefix:
                workspace = row.get("_workspace")
                chunk_id = row.get("chunk_id")
                if workspace and chunk_id:
                    base_path = (
                        f"{image_url_prefix.rstrip('/')}/"
                        f"{quote(str(workspace), safe='')}/"
                        f"{quote(str(chunk_id), safe='')}"
                    )
                    row.setdefault("image_url", f"{base_path}?size=full")
                    row.setdefault("thumbnail_url", f"{base_path}?size=thumb")
            public_items.append(row)
        public[key] = public_items
    return public


def retrieval_payload(
    result: RetrievalResult,
    *,
    path_resolver: PathResolver | None = None,
    image_url_prefix: str | None = "/images",
) -> dict[str, Any]:
    """Project retrieval results into a client-safe response dictionary."""
    contexts = project_contexts_for_client(result.contexts, image_url_prefix=image_url_prefix)
    sources = build_sources(
        contexts,
        path_resolver=path_resolver,
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
