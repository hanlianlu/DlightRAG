# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral records for one durable run lifecycle."""

import datetime
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dlightrag.runtime.contracts import AnswerRunPhase, AnswerRunStatus

type AnswerRunEventType = Literal["progress", "token", "reset", "done", "error"]
type ArtifactReferenceKind = Literal["current_attachment", "history_attachment", "fetched_resource"]
#: How a graceful shutdown left one owned run.
type ShutdownOutcome = Literal["requeued", "cancelled", "lease_lost"]
#: Whether a fenced worker's artifact write landed, or why it was refused.
type ArtifactAttachOutcome = Literal["attached", "lease_lost", "turn_mismatch"]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def canonical_run_request_json(request: Mapping[str, Any]) -> str:
    """Serialize a run request to its one durable JSON representation."""
    return json.dumps(
        dict(request),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def answer_run_request_fingerprint(request: Mapping[str, Any]) -> str:
    """Digest one canonical public request for idempotency comparison."""
    encoded = canonical_run_request_json(request).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_digest(content: bytes) -> str:
    """Content address for one immutable run artifact."""
    return hashlib.sha256(content).hexdigest()


class IdempotencyKeyConflict(RuntimeError):
    """One owner reused an idempotency key with different normalized input."""


@dataclass(frozen=True, slots=True)
class AnswerRunRecord:
    """Authoritative lifecycle state of one durable run."""

    owner_id: str
    run_id: str
    idempotency_key: str | None
    request: Mapping[str, Any]
    status: AnswerRunStatus
    phase: AnswerRunPhase | None
    stop_reason: str | None
    completed_turns: int
    cancel_requested_at: datetime.datetime | None
    lease_owner: str | None
    lease_expires_at: datetime.datetime | None
    fencing_epoch: int
    recovery_count: int
    next_event_sequence: int
    events_trimmed_at: datetime.datetime | None
    result: Mapping[str, Any] | None
    error_kind: str | None
    error_message: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_requested_at is not None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """Restorable agent state plus the turn number copied from the run row."""

    version: int
    completed_turns: int
    state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AnswerRunEvent:
    """One durable event in a run's gap-free sequence."""

    sequence: int
    event_type: AnswerRunEventType
    payload: Mapping[str, Any]
    created_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class RunCreation:
    """Result of an owner-scoped create, including idempotent replays."""

    run: AnswerRunRecord
    replayed: bool


@dataclass(frozen=True, slots=True)
class ClaimedRun:
    """A run this worker now owns, with the checkpoint it must restore."""

    run: AnswerRunRecord
    checkpoint: RunCheckpoint | None


@dataclass(frozen=True, slots=True)
class LeaseRenewal:
    """Whether a fenced worker still owns its run and its cancellation state."""

    renewed: bool
    cancel_requested: bool


@dataclass(frozen=True, slots=True)
class CheckpointCommit:
    """Definite outcome of one control-turn compare-and-set."""

    outcome: Literal["committed", "lease_lost", "corrupt"]
    completed_turns: int


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Result of a fenced terminal transition and its single terminal event."""

    committed: bool
    status: AnswerRunStatus | None
    event_sequence: int | None


@dataclass(frozen=True, slots=True)
class CancellationOutcome:
    """Result of an owner-scoped cancellation request."""

    outcome: Literal["unknown", "cancelled", "pending", "already_terminal"]
    run: AnswerRunRecord | None


@dataclass(frozen=True, slots=True)
class SweepOutcome:
    """Rows the slot-free sweeper finalized in one pass."""

    cancelled: int
    abandoned: int


@dataclass(frozen=True, slots=True)
class RunDeletion:
    """Rows and now-unreferenced blobs removed by deletion or retention."""

    runs: int
    artifacts: int


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    """Immutable bytes in one owner's content-addressed namespace."""

    content: bytes

    @property
    def digest(self) -> str:
        return artifact_digest(self.content)


@dataclass(frozen=True, slots=True)
class PendingArtifactReference:
    """One ordered run input or discovered resource pointing at stored bytes."""

    resource_id: str
    reference_kind: ArtifactReferenceKind
    ordinal: int
    digest: str
    filename: str
    mime_type: str
    transform_locator: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunArtifactReference:
    """A stored run-artifact reference read back with its creation time."""

    resource_id: str
    reference_kind: ArtifactReferenceKind
    ordinal: int
    digest: str
    filename: str
    mime_type: str
    transform_locator: Mapping[str, Any]
    created_at: datetime.datetime


__all__ = [
    "AnswerRunEvent",
    "AnswerRunEventType",
    "AnswerRunRecord",
    "ArtifactAttachOutcome",
    "ArtifactReferenceKind",
    "CancellationOutcome",
    "CheckpointCommit",
    "ClaimedRun",
    "IdempotencyKeyConflict",
    "LeaseRenewal",
    "PendingArtifact",
    "PendingArtifactReference",
    "RunArtifactReference",
    "RunCheckpoint",
    "RunCreation",
    "RunDeletion",
    "ShutdownOutcome",
    "SweepOutcome",
    "TerminalOutcome",
    "answer_run_request_fingerprint",
    "artifact_digest",
    "canonical_run_request_json",
]
