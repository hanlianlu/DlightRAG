# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Adopt one earlier Run's Resource into this Run, keeping its earlier handle.

A later Run on the same Agent Session may re-materialize what it can still see.
The model asks with a handle it read in its own context, the host loads the
earlier Run's retained bytes, and this module turns them into a Resource of the
consuming Run: a fresh handle, the earlier handle as an alias, and the stored
conversion view adopted as that Resource's view. The loader records all of it
as the consuming Run's own Resources, under its fence, before any of it can be
reached, so an adoption is durable or absent whatever the call that asked for it
does next.

Nothing here reads message text or grants authority. The loader decides what the
lineage rule admits, and a loader that returns nothing leaves the tool's ordinary
refusal in place.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from dlightrag.engine.agent.tools import ResourceAttachmentBytes
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.resources.registry import ResourceEffectOwner, ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot

LINEAGE_ADOPTION_KIND = "lineage_adoption"
SNAPSHOT_KIND = "conversion_snapshot"
ASSET_KIND = "conversion_asset"


@dataclass(frozen=True, slots=True)
class LineageResourceBytes:
    """One earlier Run's Resource, authorized for this Run by the loader.

    ``conversion_snapshot`` and ``assets`` are the stored conversion view. When the
    snapshot is present it is adopted verbatim: re-running parser selection would
    build a different history than the earlier Run recorded.
    """

    resource_id: str
    origin_run_id: str
    filename: str
    media_type: str
    content: bytes
    conversion_snapshot: bytes | None = None
    assets: Mapping[str, bytes] = field(default_factory=dict)
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("lineage resource id cannot be empty")
        if not self.filename.strip():
            raise ValueError("lineage resource filename cannot be empty")
        if not self.content:
            raise ValueError("lineage resource bytes cannot be empty")


class LineageResourceLoader(Protocol):
    """This Run's lineage authorization: one earlier Run's Resource, read and adopted.

    ``record`` makes an adoption durable as this Run's own Resources, fenced by
    this Run's lease, and raises when it could not.
    """

    async def load(self, resource_id: str) -> LineageResourceBytes | None: ...

    async def record(
        self, resources: tuple[ResourceAttachmentBytes, ...], owner: ResourceEffectOwner
    ) -> None: ...


class LineageSnapshotError(RuntimeError):
    """An authorized adoption whose stored conversion view cannot be trusted.

    Re-running parser selection instead would build a history the earlier Run never
    recorded, so the caller refuses with this reason rather than repairing quietly.
    """


class LineageAdoptionConflict(RuntimeError):
    """An adoption the store refused: this Run records other bytes or another view.

    The store keeps one view per Resource, so the whole adoption is refused and
    nothing of it is recorded or bound.
    """


async def adopt_lineage_resource(
    registry: ResourceRegistry,
    loaded: LineageResourceBytes,
    *,
    record: Callable[[tuple[ResourceAttachmentBytes, ...]], Awaitable[None]],
    needs_text: bool = False,
) -> str:
    """Register the earlier bytes as this Run's Resource under the earlier handle.

    The returned id is this Run's canonical handle, and the handle the model used
    stays readable as an alias, so a later call in the same Run needs no second read.

    The stored view is checked before anything is registered. Binding the alias
    first would let a refused adoption succeed on the next call by converting the
    bytes afresh; checking first means asking again refuses again. The bytes are
    registered as stored-view-only for the same reason: a ``view`` may adopt a
    document the earlier Run never converted, and a later ``read`` of it must
    refuse rather than build the view that Run never had.

    The adoption row and the view it brings are recorded in one write before the
    registry binds the earlier handle. The view is the earlier Run's, verbatim,
    recorded as the view of this Run's Resource under this Run's handle, so a
    recovery restores it like any view this Run made, and a later turn can adopt
    this Run's Resource by the handle this Run printed.
    """
    snapshot = _restore_snapshot(loaded)

    async def durable(canonical: str, view: ConversionSnapshot | None) -> None:
        await record(
            (
                ResourceAttachmentBytes(
                    resource_id=canonical,
                    filename=loaded.filename,
                    mime_type=loaded.media_type,
                    source_locator=canonical,
                    content=loaded.content,
                    resource_kind=LINEAGE_ADOPTION_KIND,
                    aliases=(loaded.resource_id,),
                ),
                *(view.effects() if view is not None else ()),
            )
        )

    return await registry.adopt(
        ResourceInput(
            filename=loaded.filename,
            declared_mime=loaded.media_type,
            content=loaded.content,
        ),
        alias=loaded.resource_id,
        view=snapshot,
        record=durable,
        needs_text=needs_text,
    )


def _restore_snapshot(loaded: LineageResourceBytes) -> ConversionSnapshot | None:
    """Decode the stored view, refusing rather than repairing a broken one."""
    if loaded.conversion_snapshot is None:
        return None
    try:
        snapshot = ConversionSnapshot.restore(loaded.conversion_snapshot, dict(loaded.assets))
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageSnapshotError("the earlier Run's stored conversion view is unusable") from exc
    if snapshot.resource_id != loaded.resource_id:
        raise LineageSnapshotError("conversion snapshot does not belong to this resource")
    if snapshot.input_digest != hashlib.sha256(loaded.content).hexdigest():
        raise LineageSnapshotError(
            "the earlier Run's stored conversion view does not belong to these bytes"
        )
    return snapshot


__all__ = [
    "ASSET_KIND",
    "LINEAGE_ADOPTION_KIND",
    "SNAPSHOT_KIND",
    "LineageAdoptionConflict",
    "LineageResourceBytes",
    "LineageResourceLoader",
    "LineageSnapshotError",
    "adopt_lineage_resource",
]
