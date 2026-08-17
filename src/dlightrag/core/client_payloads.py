# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral retrieval response assembly."""

from collections.abc import Mapping
from typing import Any

from dlightrag_rag.retrieval import MetadataFilter, RetrievalResult

from dlightrag.answer.citations.source_builder import build_sources
from dlightrag.answer.sources import (
    SourceDownloadLinkBuilder,
    project_contexts_for_client,
    project_source_payloads,
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


def retrieval_payload(
    result: RetrievalResult,
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_url_prefix: str | None = "/images",
) -> dict[str, Any]:
    """Project retrieval results into a client-safe response dictionary."""
    sources = build_sources(
        result.contexts,
        image_url_prefix=image_url_prefix,
    )
    source_payloads = project_source_payloads(
        sources,
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    contexts = project_contexts_for_client(
        result.contexts,
        image_url_prefix=image_url_prefix,
        visual_workspaces=visual_workspaces,
    )
    return {
        "contexts": contexts,
        "sources": [source.model_dump() for source in source_payloads],
        "trace": result.trace,
        "image_descriptions": result.image_descriptions,
    }


__all__ = [
    "metadata_filter_from_payload",
    "retrieval_payload",
]
