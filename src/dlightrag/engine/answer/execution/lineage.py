# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Load an earlier Run's Resource when the consuming Run's lineage admits it.

The authorization is the durable row's own Session stamp: a Resource another Run
registered for this same Agent Session and whose bytes are still retained. A row
belonging to another Session or owner is not visible to this loader at all, and
nothing here reads message text or accepts a handle on trust.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    SNAPSHOT_KIND,
    LineageResourceBytes,
)
from dlightrag.engine.runtime.records import RunFetchedResource

if TYPE_CHECKING:
    from dlightrag.engine.answer.execution.executor import RunBlobReader

logger = logging.getLogger(__name__)

_ADOPTABLE_KINDS = frozenset({"web", "tool_attachment"})


class LineageResourceStore(Protocol):
    """The one durable read lineage adoption needs, already Session-scoped."""

    async def lineage_resource_rows(
        self, *, owner_id: str, session_id: str, resource_id: str
    ) -> tuple[RunFetchedResource, ...]: ...


class RetainedResourceLoader:
    """Implements the tool seam's loader for one consuming Run."""

    def __init__(
        self,
        *,
        store: LineageResourceStore,
        blobs: RunBlobReader,
        owner_id: str,
        session_id: str,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._owner_id = owner_id
        self._session_id = session_id

    async def load(self, resource_id: str) -> LineageResourceBytes | None:
        rows = await self._store.lineage_resource_rows(
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
        content = await self._read(source.digest)
        if content is None:
            return None
        snapshot_row = _first(rows, SNAPSHOT_KIND)
        return LineageResourceBytes(
            resource_id=resource_id,
            origin_run_id=_origin_run_id(source),
            filename=source.filename or resource_id,
            media_type=source.mime_type or "application/octet-stream",
            content=content,
            conversion_snapshot=(
                await self._read(snapshot_row.digest) if snapshot_row is not None else None
            ),
            assets={
                row.resource_id: asset
                for row in rows
                if _resource_kind(row) == ASSET_KIND and row.digest
                for asset in (await self._read(row.digest),)
                if asset is not None
            },
            source_url=_source_url(source),
        )

    async def _read(self, digest: str) -> bytes | None:
        """Read one retained Blob, or nothing when those bytes are no longer there.

        Retention is what makes an earlier Run's Resource adoptable, so absent or
        altered bytes refuse the adoption instead of failing the tool internally.
        """
        pieces = [
            piece async for piece in self._blobs.stream(owner_id=self._owner_id, digest=digest)
        ]
        content = b"".join(pieces)
        if not content:
            return None
        if hashlib.sha256(content).hexdigest() != digest:
            logger.warning("Adoptable Resource bytes no longer match their recorded digest")
            return None
        return content


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


__all__ = ["LineageResourceStore", "RetainedResourceLoader"]
