# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical digest helpers for durable Evidence and resource identity checks.

Answer computes ``content_digest`` over canonical evidence content and
``locator_digest`` over the canonical opaque locator payload; Runtime stores
identity and digests without decoding product payloads. An existing identity
with both digests equal is idempotent; any mismatch is a deterministic,
terminal ``evidence_settlement_conflict`` — there is no update-in-place and no
new-identity retry (M3 Evidence contract).
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dlightrag_agent.session.effects import canonical_json


def canonical_content_digest(content: Any) -> str:
    """Digest one canonical evidence content payload."""
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def canonical_locator_digest(locator: Any) -> str:
    """Digest one canonical opaque locator payload."""
    return hashlib.sha256(canonical_json(locator).encode("utf-8")).hexdigest()


def canonical_evidence_identity(
    *,
    owner_id: str,
    run_id: str,
    session_id: str,
    intent_id: str,
    result_ordinal: int,
) -> str:
    """Return the canonical string form of one evidence identity tuple."""
    if result_ordinal < 0:
        raise ValueError("evidence result ordinal cannot be negative")
    return canonical_json(
        {
            "owner_id": owner_id,
            "run_id": run_id,
            "session_id": session_id,
            "intent_id": intent_id,
            "result_ordinal": result_ordinal,
        }
    )


@dataclass(frozen=True, slots=True)
class EvidenceDigests:
    """The two digests that make one evidence identity idempotent."""

    content_digest: str
    locator_digest: str

    def __post_init__(self) -> None:
        for name in ("content_digest", "locator_digest"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"evidence {name} must be a SHA-256 hex digest")


def evidence_digests(*, content: Any, locator: Any) -> EvidenceDigests:
    """Compute the canonical digests for one evidence settlement payload."""
    return EvidenceDigests(
        content_digest=canonical_content_digest(content),
        locator_digest=canonical_locator_digest(locator),
    )


def digests_match(existing: EvidenceDigests, candidate: EvidenceDigests) -> bool:
    """Return whether one candidate settlement equals the existing identity.

    Equality requires both digests; any differing field is a conflict value,
    never a partial match.
    """
    return (
        existing.content_digest == candidate.content_digest
        and existing.locator_digest == candidate.locator_digest
    )


def digests_conflict_reason(existing: EvidenceDigests, candidate: EvidenceDigests) -> str | None:
    """Return the deterministic conflict reason, or None when digests match."""
    if digests_match(existing, candidate):
        return None
    return (
        "evidence_settlement_conflict: "
        f"content={existing.content_digest[:12]}… vs {candidate.content_digest[:12]}…, "
        f"locator={existing.locator_digest[:12]}… vs {candidate.locator_digest[:12]}…"
    )


def resource_record_digest(record: Mapping[str, Any]) -> str:
    """Digest one canonical resource record for stable-identity comparison."""
    return canonical_content_digest(record)


__all__ = [
    "EvidenceDigests",
    "canonical_content_digest",
    "canonical_evidence_identity",
    "canonical_locator_digest",
    "digests_conflict_reason",
    "digests_match",
    "evidence_digests",
    "resource_record_digest",
]
