# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base classes for data sources."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlightrag.core.retrieval.metadata_fields import MetadataIngestPolicy


@dataclass(frozen=True)
class SourceDocument:
    """One source document discovered by a streaming SDK connector."""

    key: str
    source_uri: str | None = None
    download_uri: str | None = None
    display_filename: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: Mapping[str, Any] | None = None
    metadata_policy: MetadataIngestPolicy | None = None


class AsyncDataSource(ABC):
    """Abstract base class for async data sources.

    All async data sources must implement this interface.
    """

    async def alist_documents(self, prefix: str | None = None) -> list[SourceDocument]:
        """Collect available documents into a list."""
        return [document async for document in self.aiter_documents(prefix=prefix)]

    @abstractmethod
    def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[SourceDocument]:
        """Stream available documents.

        Adapters should implement this as the primary discovery path. Batch
        metadata comes from the ingest call; per-document metadata belongs on
        each ``SourceDocument`` and overlays those batch defaults.
        """
        raise NotImplementedError

    @abstractmethod
    async def amaterialize_document(self, document: SourceDocument, destination: Path) -> None:
        """Write document content to *destination* without returning bytes."""
        raise NotImplementedError


__all__ = ["AsyncDataSource", "SourceDocument"]
