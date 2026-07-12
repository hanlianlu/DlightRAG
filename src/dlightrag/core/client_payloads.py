# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral payload projection for API, MCP, and other clients."""

import logging
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from dlightrag.citations.schemas import SourceReference, SourceReferencePayload
from dlightrag.citations.source_builder import build_sources
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.utils import log_safe

logger = logging.getLogger(__name__)


class SourceDownloadInvariantError(RuntimeError):
    """Raised when an HTTP adapter cannot safely project a source download."""


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
    payload["file_path"] = _display_file_name(payload["file_path"])
    if "metadata" in payload:
        metadata = payload["metadata"]
        if isinstance(metadata, Mapping):
            public_metadata = dict(metadata)
            public_metadata.pop("source_uri", None)
            public_metadata.pop("source_download_locator", None)
            payload["metadata"] = public_metadata
        else:
            payload.pop("metadata")
    if (
        image_url_prefix
        and row.get("_workspace")
        and chunk_id
        and (row.get("image_data") or _is_visual_chunk(row))
    ):
        base_path = (
            f"{image_url_prefix.rstrip('/')}/"
            f"{quote(str(row['_workspace']), safe='')}/"
            f"{quote(str(chunk_id), safe='')}"
        )
        payload.setdefault("image_url", f"{base_path}?size=full")
        payload.setdefault("thumbnail_url", f"{base_path}?size=thumb")
    return payload


def _is_visual_chunk(row: dict[str, Any]) -> bool:
    sidecar = row.get("sidecar")
    return isinstance(sidecar, dict) and sidecar.get("type") == "drawing"


def _display_file_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    candidate = value
    if "://" in value:
        try:
            candidate = unquote(urlsplit(value).path)
        except ValueError:
            candidate = unquote(value.split("?", 1)[0].split("#", 1)[0])
    if "\\" in candidate:
        return PureWindowsPath(candidate).name
    return Path(candidate.rstrip("/")).name


def project_source_payloads(
    sources: list[SourceReference],
    *,
    resolver: SourceDownloadLinkBuilder | None,
    downloadable_workspaces: set[str] | None = None,
) -> list[SourceReferencePayload]:
    """Convert internal sources into the strict public source contract."""
    projected: list[SourceReferencePayload] = []
    for source in sources:
        safe_source_id = log_safe(source.id)
        download_url = None
        can_download = (
            downloadable_workspaces is None or source.workspace in downloadable_workspaces
        )
        if resolver is not None and can_download:
            if not source.document_id:
                logger.info(
                    "source_download_projection_outcome",
                    extra={"outcome": "invalid", "source_id": safe_source_id},
                )
                raise SourceDownloadInvariantError(
                    f"Could not project source document for source {safe_source_id}"
                ) from None
            try:
                download_url = resolver.resolve(
                    source.document_id,
                    workspace=source.workspace,
                )
            except Exception:
                download_url = None
            if not download_url:
                logger.info(
                    "source_download_projection_outcome",
                    extra={"outcome": "invalid", "source_id": safe_source_id},
                )
                raise SourceDownloadInvariantError(
                    f"Could not project download locator for source {safe_source_id}"
                ) from None
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "resolved", "source_id": safe_source_id},
            )
        elif resolver is not None:
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "unauthorized", "source_id": safe_source_id},
            )
        projected.append(
            SourceReferencePayload(
                id=source.id,
                title=source.title,
                type=source.type,
                source_uri=source.source_uri,
                download_url=download_url,
                cited_chunk_ids=source.cited_chunk_ids,
                chunks=source.chunks,
            )
        )
    return projected


def retrieval_payload(
    result: RetrievalResult,
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
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
    )
    contexts = project_contexts_for_client(result.contexts, image_url_prefix=image_url_prefix)
    return {
        "answer": result.answer,
        "contexts": contexts,
        "sources": [source.model_dump() for source in source_payloads],
        "trace": result.trace,
        "image_descriptions": result.image_descriptions,
        "current_image_ids": result.current_image_ids,
    }


def answer_payload(
    result: RetrievalResult,
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    image_url_prefix: str | None = "/images",
) -> dict[str, Any]:
    """Project generated answer results into a client-safe response dictionary."""
    source_payloads = project_source_payloads(
        result.sources,
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
    )
    return {
        "answer": result.answer,
        "contexts": project_contexts_for_client(
            result.contexts,
            image_url_prefix=image_url_prefix,
        ),
        "references": [{"id": ref.id, "title": ref.title} for ref in result.references],
        "sources": [source.model_dump() for source in source_payloads],
        "answer_images": result.answer_images,
        "answer_blocks": result.answer_blocks,
        "trace": result.trace,
        "image_descriptions": result.image_descriptions,
        "current_image_ids": result.current_image_ids,
    }


__all__ = [
    "SourceDownloadInvariantError",
    "answer_payload",
    "metadata_filter_from_payload",
    "project_contexts_for_client",
    "project_source_payloads",
    "retrieval_payload",
]
