# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""How a Resource a Tool attached becomes one durable row of its Run.

A Tool result settles its attached Resources this way, a lineage adoption records
the Resource it adopts the same way before any Tool result settles, and
publication records the conversion view of a product it publishes the same way,
so none of them can describe one Resource differently.
"""

from __future__ import annotations

from dataclasses import asdict

from dlightrag.engine.agent.tools import ResourceAttachmentBytes
from dlightrag.engine.runtime.blob_chunks import blob_digest, plan_blob
from dlightrag.engine.runtime.settlements import (
    CompleteBlobDescriptor,
    FetchedResourceSettlementUpdate,
    OpaqueFetchedResourceWrite,
)

#: What a row's capabilities say about its bytes; an attribute may add to them, not set them.
_ROW_CAPABILITIES = frozenset({"resource_kind", "visual_source", "resource_aliases"})


def attached_resource_update(
    attached: ResourceAttachmentBytes, *, session_id: str, intent_id: str | None
) -> FetchedResourceSettlementUpdate:
    """Describe one attached Resource as its row and complete Blob.

    ``intent_id`` names the Tool effect that attached it; publication, which is no
    Tool effect, records its views with none. Its ``attributes`` join the row's
    capabilities, and may not rewrite what the row itself says about the bytes.
    """
    collisions = sorted({name for name, _value in attached.attributes} & _ROW_CAPABILITIES)
    if collisions:
        raise ValueError(f"attachment attributes cannot set {', '.join(collisions)}")
    plan = plan_blob(attached.content)
    return FetchedResourceSettlementUpdate(
        resource=OpaqueFetchedResourceWrite(
            resource_id=attached.resource_id,
            ordinal=0,
            safe_name=attached.filename,
            media_type=attached.mime_type,
            capabilities={
                **dict(attached.attributes),
                "resource_kind": attached.resource_kind,
                "visual_source": asdict(attached.source) if attached.source is not None else None,
                **({"resource_aliases": list(attached.aliases)} if attached.aliases else {}),
            },
            blob_digest=plan.digest,
            source_locator_digest=blob_digest(attached.source_locator.encode("utf-8")),
            source_locator=attached.source_locator.encode("utf-8"),
            session_id=session_id,
            intent_id=intent_id,
        ),
        complete_blob=CompleteBlobDescriptor(
            digest=plan.digest,
            total_bytes=plan.total_bytes,
            chunks=tuple(plan.chunk(attached.content, index) for index in range(plan.chunk_count)),
        ),
    )


__all__ = ["attached_resource_update"]
