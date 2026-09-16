# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Load an earlier Run's Resource when the consuming Run's lineage admits it.

The authorization is the durable row's own Session stamp: a Resource another Run
registered for this same Agent Session and whose bytes are still retained. A row
belonging to another Session or owner is not visible to this loader at all, and
nothing here reads message text or accepts a handle on trust.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    SNAPSHOT_KIND,
    LineageResourceBytes,
)
from dlightrag.engine.runtime.records import RunFetchedResource

_ADOPTABLE_KINDS = frozenset({"web", "tool_attachment"})


class LineageResourceStore(Protocol):
    """The one durable read lineage adoption needs, already Session-scoped."""

    async def lineage_resource(
        self, *, owner_id: str, session_id: str, resource_id: str
    ) -> tuple[RunFetchedResource, ...]: ...


class BlobReader(Protocol):
    """Stream one owner-scoped Blob by digest."""

    def stream(self, *, owner_id: str, digest: str) -> AsyncIterator[bytes]: ...


class RetainedResourceLoader:
    """Implements the tool seam's loader for one consuming Run."""

    def __init__(
        self,
        *,
        store: LineageResourceStore,
        blobs: BlobReader,
        owner_id: str,
        session_id: str,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._owner_id = owner_id
        self._session_id = session_id

    async def load(self, resource_id: str) -> LineageResourceBytes | None:
        rows = await self._store.lineage_resource(
            owner_id=self._owner_id,
            session_id=self._session_id,
            resource_id=resource_id,
        )
        source = next(
            (
                row
                for row in rows
                if row.resource_id == resource_id
                and _resource_kind(row) in _ADOPTABLE_KINDS
                and row.digest
            ),
            None,
        )
        if source is None:
            return None
        snapshot_row = _first(rows, SNAPSHOT_KIND)
        return LineageResourceBytes(
            resource_id=resource_id,
            origin_run_id=_origin_run_id(source),
            filename=source.filename or resource_id,
            media_type=source.mime_type or "application/octet-stream",
            content=await self._read(source.digest),
            conversion_snapshot=(
                await self._read(snapshot_row.digest) if snapshot_row is not None else None
            ),
            assets={
                row.resource_id: await self._read(row.digest)
                for row in rows
                if _resource_kind(row) == ASSET_KIND and row.digest
            },
            source_url=_source_url(source),
        )

    async def _read(self, digest: str) -> bytes:
        pieces = [
            piece async for piece in self._blobs.stream(owner_id=self._owner_id, digest=digest)
        ]
        return b"".join(pieces)


def _resource_kind(row: RunFetchedResource) -> str:
    return str(row.capabilities.get("resource_kind") or "")


def _first(rows: Sequence[RunFetchedResource], kind: str) -> RunFetchedResource | None:
    return next((row for row in rows if _resource_kind(row) == kind and row.digest), None)


def _origin_run_id(row: RunFetchedResource) -> str:
    return str(row.capabilities.get("origin_run_id") or "")


def _source_url(row: RunFetchedResource) -> str | None:
    locator = row.source_locator
    if not locator:
        return None
    try:
        decoded = locator.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return decoded if decoded.startswith(("http://", "https://")) else None


__all__ = ["BlobReader", "LineageResourceStore", "RetainedResourceLoader"]
