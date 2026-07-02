# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base classes for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path


class AsyncDataSource(ABC):
    """Abstract base class for async data sources.

    All async data sources must implement this interface.
    """

    async def alist_documents(self, prefix: str | None = None) -> list[str]:
        """Collect available document identifiers into a list."""
        return [doc_id async for doc_id in self.aiter_documents(prefix=prefix)]

    @abstractmethod
    def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[str]:
        """Stream available document identifiers.

        Adapters should implement this as the primary discovery path. Small
        sources can yield from an in-memory list; paginated backends should
        yield page by page.
        """
        ...

    @abstractmethod
    async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
        """Write document content to *destination* without returning bytes."""
        ...


__all__ = ["AsyncDataSource"]
