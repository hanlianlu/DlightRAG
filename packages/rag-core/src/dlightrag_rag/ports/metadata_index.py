# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral metadata index interface."""

from typing import Any, Protocol

from dlightrag_rag.retrieval.models import MetadataFilter


class MetadataIndexProtocol(Protocol):
    """Common interface for PGMetadataIndex and test doubles."""

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    async def merge_custom_metadata(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        raise NotImplementedError

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def get_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    async def query(self, filters: MetadataFilter) -> list[str]:
        raise NotImplementedError

    async def delete(self, doc_id: str) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def find_by_filename(self, name: str) -> list[str]:
        raise NotImplementedError

    async def find_by_download_locator(self, download_locator: str) -> list[str]:
        raise NotImplementedError


__all__ = [
    "MetadataIndexProtocol",
]
