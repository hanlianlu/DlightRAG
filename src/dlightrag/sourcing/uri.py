# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""URI parsing helpers for ingestion sources.

Centralises the ``scheme://...`` parsing that was previously duplicated
between ``RAGService.aretry_failed_docs`` and the file-serving routes,
so that adding a new source scheme is a single-edit change.
"""

from __future__ import annotations

from typing import Any, Literal

SourceType = Literal["local", "azure_blob", "s3"]


def parse_remote_uri(file_path: str) -> tuple[SourceType, dict[str, Any]]:
    """Split a stored ``file_path`` into ``(source_type, ingest_kwargs)``.

    Used by retry / re-ingest paths that need to dispatch a stored doc
    URI back through ``RAGService.aingest(source_type, **kwargs)``.

    >>> parse_remote_uri("/var/data/foo.pdf")
    ('local', {'path': '/var/data/foo.pdf'})
    >>> parse_remote_uri("azure://container/path/to/blob.pdf")
    ('azure_blob', {'container_name': 'container', 'blob_path': 'path/to/blob.pdf'})
    >>> parse_remote_uri("s3://bucket/key/with/slashes")
    ('s3', {'bucket': 'bucket', 'key': 'key/with/slashes'})

    Raises ``ValueError`` for malformed URIs (missing container/bucket or
    payload) or unsupported schemes.
    """
    if file_path.startswith("azure://"):
        stripped = file_path[len("azure://") :]
        container, _, blob_path = stripped.partition("/")
        if not container or not blob_path:
            raise ValueError(f"malformed azure URI (need container/path): {file_path!r}")
        return "azure_blob", {"container_name": container, "blob_path": blob_path}

    if file_path.startswith("s3://"):
        stripped = file_path[len("s3://") :]
        bucket, _, key = stripped.partition("/")
        if not bucket or not key:
            raise ValueError(f"malformed s3 URI (need bucket/key): {file_path!r}")
        return "s3", {"bucket": bucket, "key": key}

    if "://" in file_path:
        scheme = file_path.split("://", 1)[0]
        raise ValueError(f"unsupported URI scheme {scheme!r} in {file_path!r}")

    return "local", {"path": file_path}


__all__ = ["SourceType", "parse_remote_uri"]
