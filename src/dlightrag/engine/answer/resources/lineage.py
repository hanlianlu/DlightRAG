# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Adopt one earlier Run's Resource into this Run, keeping its earlier handle.

A later Run on the same Agent Session may re-materialize what it can still see.
The model asks with a handle it read in its own context, the host loads the
earlier Run's retained bytes, and this module turns them into a Resource of the
consuming Run: a fresh handle, the earlier handle as an alias, the stored
conversion snapshot adopted as-is, and settlement effects that pin the adopted
bytes under the consuming Run's fence.

Nothing here reads message text or grants authority. The loader decides what the
lineage rule admits, and a loader that returns nothing leaves the tool's ordinary
refusal in place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from dlightrag.engine.agent.tools import ResourceAttachmentBytes
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.resources.registry import ResourceRegistry
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
    """Read one earlier Run's Resource through this Run's lineage authorization."""

    async def load(self, resource_id: str) -> LineageResourceBytes | None: ...


class LineageSnapshotError(RuntimeError):
    """An authorized adoption whose stored conversion view cannot be trusted.

    Re-running parser selection instead would build a history the earlier Run never
    recorded, so the caller refuses with this reason rather than repairing quietly.
    """


def adopt_lineage_resource(registry: ResourceRegistry, loaded: LineageResourceBytes) -> str:
    """Register the earlier bytes as this Run's Resource under the earlier handle.

    The returned id is this Run's canonical handle, and the handle the model used
    stays readable as an alias, so a later call in the same Run needs no second read.
    """
    adopted = registry.register(
        ResourceInput(
            filename=loaded.filename,
            declared_mime=loaded.media_type,
            content=loaded.content,
        ),
        aliases=(loaded.resource_id,),
    )
    if loaded.conversion_snapshot is not None:
        try:
            snapshot = ConversionSnapshot.restore(loaded.conversion_snapshot, dict(loaded.assets))
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageSnapshotError(
                "the earlier Run's stored conversion view is unusable"
            ) from exc
        if snapshot.resource_id != loaded.resource_id:
            raise LineageSnapshotError("conversion snapshot does not belong to this resource")
        registry.adopt_conversion_snapshot(snapshot)
    return adopted


def lineage_adoption_effects(
    loaded: LineageResourceBytes, adopted_resource_id: str
) -> tuple[ResourceAttachmentBytes, ...]:
    """Pin the adopted bytes, snapshot, and assets under the consuming Run.

    Settlement writes these as this Run's own Resources, so origin-Run cleanup can
    never invalidate what this Run adopted, and recovery re-materializes them
    through the same restore path a same-Run fetch already uses.
    """
    effects = [
        ResourceAttachmentBytes(
            resource_id=adopted_resource_id,
            filename=loaded.filename,
            mime_type=loaded.media_type,
            source_locator=loaded.resource_id,
            content=loaded.content,
            resource_kind=LINEAGE_ADOPTION_KIND,
        )
    ]
    if loaded.conversion_snapshot is None:
        return tuple(effects)
    effects.append(
        ResourceAttachmentBytes(
            resource_id=f"{adopted_resource_id}-conversion",
            filename="conversion.json",
            mime_type="application/json",
            source_locator=adopted_resource_id,
            content=loaded.conversion_snapshot,
            resource_kind=SNAPSHOT_KIND,
        )
    )
    media_types = _snapshot_asset_media_types(loaded.conversion_snapshot)
    effects.extend(
        ResourceAttachmentBytes(
            resource_id=asset_id,
            filename=asset_id,
            mime_type=media_types.get(asset_id, "application/octet-stream"),
            source_locator=adopted_resource_id,
            content=content,
            resource_kind=ASSET_KIND,
        )
        for asset_id, content in sorted(loaded.assets.items())
    )
    return tuple(effects)


def _snapshot_asset_media_types(encoded: bytes) -> dict[str, str]:
    """Asset media types as the adopted snapshot recorded them."""
    payload = json.loads(encoded)
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("conversion snapshot assets are missing")
    return {
        str(asset["resource_id"]): str(asset.get("media_type") or "application/octet-stream")
        for asset in assets
        if isinstance(asset, dict) and isinstance(asset.get("resource_id"), str)
    }


__all__ = [
    "ASSET_KIND",
    "LINEAGE_ADOPTION_KIND",
    "SNAPSHOT_KIND",
    "LineageResourceBytes",
    "LineageResourceLoader",
    "LineageSnapshotError",
    "adopt_lineage_resource",
    "lineage_adoption_effects",
]
