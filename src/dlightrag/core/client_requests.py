# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request projection for client-facing adapters."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from dlightrag.core.client_contracts import IngestDocument, IngestSpec, dump_optional_list
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
        s3_region = _get(payload, "s3_region")
        s3_key = _get(payload, "s3_key")
        prefix = _get(payload, "prefix")
        if s3_region:
            kwargs["s3_region"] = s3_region
        if s3_key:
            kwargs["s3_key"] = s3_key
        if prefix is not None:
            kwargs["prefix"] = prefix
    elif source_type == "url":
        url = _get(payload, "url")
        urls = _get(payload, "urls")
        filename = _get(payload, "filename")
        if url:
            kwargs["url"] = url
        if urls:
            kwargs["urls"] = urls
        if filename:
            kwargs["filename"] = filename
        source_uri = _get(payload, "source_uri")
        source_uris = _get(payload, "source_uris")
        if source_uri:
            kwargs["source_uri"] = source_uri
        if source_uris is not None:
            kwargs["source_uris"] = source_uris

    for name in (
        "replace",
        "retain_source_file",
        "title",
        "author",
        "metadata",
        "metadata_policy",
        "documents",
    ):
        value = _get(payload, name)
        if value is not None:
            kwargs[name] = dump_optional_list(value) if name == "documents" else value

    return kwargs


def ingest_spec_from_payload(payload: Any) -> IngestSpec:
    """Return an IngestSpec from REST, MCP, CLI, or SDK-shaped payload objects."""
    source_type = _get(payload, "source_type")
    return IngestSpec(source_type=source_type, **ingest_kwargs_from_payload(payload))


def managed_local_ingest_path(
    *,
    source_type: str,
    path: str | None,
    input_dir: Path,
    workspace: str,
) -> str | None:
    """Return a local ingest path constrained to this workspace's input root."""
    if source_type != "local" or not path:
        return path

    from dlightrag.utils import normalize_workspace

    root = (input_dir / normalize_workspace(workspace)).resolve()
    if "\0" in path or path.startswith(("~", "/", "\\")):
        raise ValueError(
            "local ingest paths from REST/MCP must be relative to input_dir/<workspace>"
        )
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(
            "local ingest paths from REST/MCP must be relative to input_dir/<workspace>"
        )

    parts = tuple(part for part in re.split(r"[\\/]+", path) if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(
            "local ingest paths from REST/MCP must be relative to input_dir/<workspace>"
        )
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            "local ingest paths from REST/MCP must be under input_dir/<workspace>"
        ) from None
    return str(resolved)


def managed_local_ingest_documents(
    *,
    source_type: str,
    documents: list[IngestDocument] | None,
    input_dir: Path,
    workspace: str,
) -> list[IngestDocument] | None:
    """Constrain local manifest paths to this workspace's input root."""
    if source_type != "local" or documents is None:
        return documents
    return [
        document.model_copy(
            update={
                "path": managed_local_ingest_path(
                    source_type=source_type,
                    path=document.path,
                    input_dir=input_dir,
                    workspace=workspace,
                )
            }
        )
        for document in documents
    ]


def query_image_blocks_from_urls(values: Sequence[str]) -> list[dict[str, Any]]:
    """Wrap CLI-supplied image URLs/data URIs as modern image_url content blocks."""
    return [{"type": "image_url", "image_url": {"url": value}} for value in values]


__all__ = [
    "ingest_kwargs_from_payload",
    "ingest_spec_from_payload",
    "managed_local_ingest_documents",
    "managed_local_ingest_path",
    "query_image_blocks_from_urls",
    "query_kwargs_from_payload",
]
