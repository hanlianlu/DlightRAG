# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Load an earlier Run's Resource when the consuming Run's lineage admits it.

The authorization is the durable row's own Session stamp: a Resource another Run
registered for this same Agent Session and whose bytes are still retained. A row
belonging to another Session or owner is not visible to this loader at all, and
nothing here reads message text or accepts a handle on trust.

An adoption is recorded through the same loader, fenced by the consuming Run's
lease, as that Run's own Resources.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from dlightrag.engine.agent.tools import ResourceAttachmentBytes
from dlightrag.engine.answer.research.resource_settlement import attached_resource_update
from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    SNAPSHOT_KIND,
    LineageResourceBytes,
)
from dlightrag.engine.runtime.records import RunFetchedResource
from dlightrag.engine.runtime.settlements import FetchedResourceSettlementUpdate

if TYPE_CHECKING:
    from dlightrag.engine.answer.execution.executor import RunBlobReader
    from dlightrag.engine.answer.resources.registry import ResourceEffectOwner

logger = logging.getLogger(__name__)

#: What a later Run of the same Agent Session may adopt by naming a handle: a
#: (capability, resource kind) pair each. One declaration drives both this loader's
#: gate and the adapter's read, so "adoptable" is stated where the kind is known
#: rather than remembered in two layers.
#:
#: A published Artifact is here for the same reason a fetched Web body is: it is
#: digest-addressed bytes that outlive the Run that produced them, and a
#: conversation that keeps working on one deliverable must be able to read the
#: version it published earlier.
ADOPTABLE_LINEAGE_KINDS: tuple[tuple[str, str], ...] = (
    ("web", "fetched_blob"),
    ("tool_attachment", "fetched_blob"),
    ("published_artifact", "published_artifact"),
)

_ADOPTABLE_KINDS = frozenset(capability for capability, _kind in ADOPTABLE_LINEAGE_KINDS)


class LineageResourceStore(Protocol):
    """The durable read lineage adoption needs, already Session-scoped, and its write."""

    async def lineage_resource_rows(
        self, *, owner_id: str, session_id: str, resource_id: str
    ) -> tuple[RunFetchedResource, ...]: ...

    async def record_lineage_adoption(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        resources: tuple[FetchedResourceSettlementUpdate, ...],
    ) -> None: ...


class RetainedResourceLoader:
    """Implements the tool seam's loader for one consuming Run and its claim."""

    def __init__(
        self,
        *,
        store: LineageResourceStore,
        blobs: RunBlobReader,
        owner_id: str,
        session_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._owner_id = owner_id
        self._session_id = session_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._fencing_epoch = fencing_epoch

    async def record(
        self, resources: tuple[ResourceAttachmentBytes, ...], owner: ResourceEffectOwner
    ) -> None:
        """Write one adoption as this Run's own Resources, in one write under its lease.

        The rows are the ones a settlement of the adopting call would write, so
        recovery restores them like any other Resource of this Run and never reads
        the lineage again. A lost lease raises ``LeaseLostError`` and writes nothing.
        """
        await self._store.record_lineage_adoption(
            owner_id=self._owner_id,
            run_id=self._run_id,
            worker_id=self._worker_id,
            fencing_epoch=self._fencing_epoch,
            resources=tuple(
                attached_resource_update(
                    resource,
                    session_id=owner.execution_scope,
                    intent_id=owner.intent_id.value,
                )
                for resource in resources
            ),
        )

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


__all__ = ["ADOPTABLE_LINEAGE_KINDS", "LineageResourceStore", "RetainedResourceLoader"]
