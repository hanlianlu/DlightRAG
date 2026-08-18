# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The M3 HostUpdate union: opaque settlement facts Runtime stores undecoded.

Only these two discriminated M3 variants exist; spill, workspace inventory, and
child-session variants arrive in M4/M6 with their first writers. Runtime stores
identity, digests, and opaque payload bytes and never parses citations,
locators, chunks, or resource policy (M3 HostUpdate contract).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


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
        for name in ("blob_digest", "source_locator_digest"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"fetched resource {name} must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class CompleteBlobDescriptor:
    """A fully verified blob: every chunk and the complete-content digest.

    Metadata existence means complete: one transaction writes all chunks and
    inserts metadata last, so a partial blob is never visible (M3-D21).
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
class EvidenceSettlementUpdate:
    """Evidence plus evidence-backed source handles for one settlement."""

    evidence: Sequence[OpaqueEvidenceWrite] = ()
    resources: Sequence[OpaqueEvidenceResourceWrite] = ()


@dataclass(frozen=True, slots=True)
class FetchedResourceSettlementUpdate:
    """One fetched blob resource, its evidence, and the complete blob."""

    resource: OpaqueFetchedResourceWrite
    complete_blob: CompleteBlobDescriptor
    evidence: Sequence[OpaqueEvidenceWrite] = ()


type M3HostUpdate = EvidenceSettlementUpdate | FetchedResourceSettlementUpdate


__all__ = [
    "CompleteBlobDescriptor",
    "EvidenceSettlementUpdate",
    "FetchedResourceSettlementUpdate",
    "M3HostUpdate",
    "OpaqueEvidenceResourceWrite",
    "OpaqueEvidenceWrite",
    "OpaqueFetchedResourceWrite",
]
