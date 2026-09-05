# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One atomic host-effect aggregate stored with each tool settlement."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class OpaqueEvidenceWrite:
    """One evidence fact: stable identity, canonical payload, and digests."""

    session_id: str
    intent_id: str
    result_ordinal: int
    content_digest: str
    locator_digest: str
    content: bytes
    locator: bytes

    def __post_init__(self) -> None:
        if self.result_ordinal < 0:
            raise ValueError("evidence result ordinal cannot be negative")
        for name in ("content_digest", "locator_digest"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"evidence {name} must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class OpaqueEvidenceResourceWrite:
    """One evidence-backed source handle committed with an evidence settlement."""

    resource_id: str
    safe_name: str
    media_type: str
    capabilities: Any
    session_id: str
    intent_id: str
    result_ordinal: int
    locator_digest: str

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("resource id cannot be empty")
        if len(self.locator_digest) != 64:
            raise ValueError("resource locator digest must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class OpaqueFetchedResourceWrite:
    """One fetched body registered as a complete blob-backed resource."""

    resource_id: str
    ordinal: int
    safe_name: str
    media_type: str
    capabilities: Any
    blob_digest: str
    source_locator_digest: str
    source_locator: bytes
    session_id: str
    intent_id: str

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("resource id cannot be empty")
        if self.ordinal < 0:
            raise ValueError("fetched resource ordinal cannot be negative")
        for name in ("blob_digest", "source_locator_digest"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"fetched resource {name} must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class CompleteBlobDescriptor:
    """A fully verified blob: every chunk and the complete-content digest.

    Metadata existence means complete: one transaction writes all chunks and
    inserts metadata last, so a partial blob is never visible.
    """

    digest: str
    total_bytes: int
    chunks: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if len(self.digest) != 64:
            raise ValueError("blob digest must be a SHA-256 hex digest")
        if self.total_bytes != sum(len(chunk) for chunk in self.chunks):
            raise ValueError("blob total bytes must equal its chunk sum")


@dataclass(frozen=True, slots=True)
class FetchedResourceSettlementUpdate:
    """One fetched blob resource, its evidence, and the complete blob."""

    resource: OpaqueFetchedResourceWrite
    complete_blob: CompleteBlobDescriptor
    evidence: Sequence[OpaqueEvidenceWrite] = ()


@dataclass(frozen=True, slots=True)
class InventoryPathRecord:
    """One current-epoch path observation."""

    relative_path: str
    entry_type: str
    size_bytes: int
    mode: int | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.relative_path.strip():
            raise ValueError("inventory path cannot be empty")
        if self.size_bytes < 0:
            raise ValueError("inventory size cannot be negative")
        if self.content_digest is not None and len(self.content_digest) != 64:
            raise ValueError("inventory content digest must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class WorkspaceInventoryUpdate:
    """Inventory upserts/deletes, or a full replace after bash/handoff scan."""

    upserts: Sequence[InventoryPathRecord] = ()
    deletes: Sequence[str] = ()
    replace_all: bool = False


@dataclass(frozen=True, slots=True)
class CommittedSpillUpdate:
    """Private spill bytes on the volume, addressed by a Resource Handle."""

    resource_id: str
    content_digest: str
    size_bytes: int
    session_id: str
    intent_id: str

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("spill resource id cannot be empty")
        if len(self.content_digest) != 64:
            raise ValueError("spill content digest must be a SHA-256 hex digest")
        if self.size_bytes < 0:
            raise ValueError("spill size cannot be negative")


@dataclass(frozen=True, slots=True)
class ArtifactAttachmentUpdate:
    """One validated Artifact root attached by a settled parent tool call."""

    relative_path: str
    label: str
    content_digest: str
    size_bytes: int
    presentation: str
    session_id: str
    intent_id: str

    def __post_init__(self) -> None:
        if not self.relative_path.strip():
            raise ValueError("Artifact attachment path cannot be empty")
        if len(self.content_digest) != 64:
            raise ValueError("Artifact attachment digest must be a SHA-256 hex digest")
        if self.size_bytes < 0:
            raise ValueError("Artifact attachment size cannot be negative")
        if self.presentation not in {"image", "markdown", "html", "pdf", "text", "download"}:
            raise ValueError("Artifact attachment presentation is invalid")
        if not self.session_id.strip() or not self.intent_id.strip():
            raise ValueError("Artifact attachment provenance cannot be empty")


@dataclass(frozen=True, slots=True)
class MemoryOperationSettlement:
    """Owner-safe product receipt projected from one settled Memory tool call."""

    operation: Literal["remember", "forget", "undo"]
    outcome: Literal["changed", "unchanged", "rejected", "conflict"]
    change_id: str | None = None
    memory_ids: Sequence[str] = ()
    kind: str | None = None
    body: str = ""
    supersedes_id: str | None = None
    target_change_id: str | None = None


@dataclass(frozen=True, slots=True)
class EffectHostUpdate:
    """Complete host-side effect batch committed with one model-visible result."""

    evidence: Sequence[OpaqueEvidenceWrite] = ()
    resources: Sequence[OpaqueEvidenceResourceWrite] = ()
    fetched: Sequence[FetchedResourceSettlementUpdate] = ()
    committed_outputs: Sequence[CommittedSpillUpdate] = ()
    workspace_inventory: WorkspaceInventoryUpdate | None = None
    artifact_attachment: ArtifactAttachmentUpdate | None = None
    memory_operation: MemoryOperationSettlement | None = None


__all__ = [
    "ArtifactAttachmentUpdate",
    "CommittedSpillUpdate",
    "CompleteBlobDescriptor",
    "EffectHostUpdate",
    "FetchedResourceSettlementUpdate",
    "InventoryPathRecord",
    "MemoryOperationSettlement",
    "OpaqueEvidenceResourceWrite",
    "OpaqueEvidenceWrite",
    "OpaqueFetchedResourceWrite",
    "WorkspaceInventoryUpdate",
]
