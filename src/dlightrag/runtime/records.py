# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral records for one durable run lifecycle."""

import datetime
import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dlightrag.agent.session.ids import SessionId
from dlightrag.agent.session.store import AgentSessionStore
from dlightrag.runtime.contracts import AnswerRunPhase, AnswerRunStatus
from dlightrag.runtime.policy import MAX_RECLAIMS_WITHOUT_PROGRESS
from dlightrag.runtime.progress import RunProgressStore
from dlightrag.runtime.settlements import EffectHostUpdate
from dlightrag.runtime.workspace import WorkspaceStore

type AnswerRunEventType = Literal[
    "progress",
    "token",
    "reset",
    "tool_start",
    "tool_progress",
    "tool_end",
    "memory_operation_settled",
    "done",
    "error",
]
type ArtifactReferenceKind = Literal[
    "current_attachment",
    "history_attachment",
    "fetched_resource",
    "primary_report",
    "published_artifact",
]
#: How a graceful shutdown left one owned run.
type ShutdownOutcome = Literal["requeued", "cancelled", "lease_lost"]

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


def parse_run_id(run_id: str) -> uuid.UUID | None:
    """Parse an opaque run id; malformed caller input reads as unknown."""
    try:
        return uuid.UUID(str(run_id))
    except ValueError:
        return None


class IdempotencyKeyConflict(RuntimeError):
    """One owner reused an idempotency key with different normalized input."""


@dataclass(frozen=True, slots=True)
class AnswerRunRecord:
    """Authoritative lifecycle state of one durable run.

    Durable progress replaces the checkpoint-era turn and recovery counters:
    ``durable_progress_version`` advances only on live fenced work — model
    turn appends, compaction appends, live effect settlements, and Fast stage
    settlements. Recovery prelude (interrupted ``never`` intents, contract
    changes, workspace-epoch handoff) does not advance it.
    """

    owner_id: str
    run_id: str
    idempotency_key: str | None
    prepared_input: Mapping[str, Any] | None
    status: AnswerRunStatus
    phase: AnswerRunPhase | None
    stop_reason: str | None
    cancel_requested_at: datetime.datetime | None
    lease_owner: str | None
    lease_expires_at: datetime.datetime | None
    fencing_epoch: int
    durable_progress_version: int
    last_reclaim_progress_version: int
    reclaims_without_progress: int
    next_event_sequence: int
    events_trimmed_at: datetime.datetime | None
    result: Mapping[str, Any] | None
    error_kind: str | None
    error_message: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    workspace_epoch: int | None = None
    #: The bounded public envelope (query, workspaces, mode, attachment
    #: identities) that survives the terminal transition: prepared_input_json
    #: is cleared at finish, so post-terminal readers project from this.
    accepted_input: Mapping[str, Any] | None = None

    def request_input(self) -> Mapping[str, Any]:
        """The run's public request for projection: envelope first, then input.

        Terminal transitions clear ``prepared_input_json``; the accepted
        envelope is the durable public face readers use for history, workspace
        download authorization, and turn projection.
        """
        return self.accepted_input or self.prepared_input or {}

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_requested_at is not None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


def accepted_input_envelope(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the terminal-surviving public envelope from a prepared input.

    Continuations are ordinary newly authorized runs, but they must be able to
    reconstruct the selected run's accepted context after execution-private
    prepared input is cleared. Pinned model facts and resource manifests remain
    execution-only; normalized history, retrieval controls, and input-resource
    identities remain in this bounded public envelope.
    """
    envelope = {
        "query": str(prepared.get("query") or ""),
        "workspaces": [str(value) for value in prepared.get("workspaces") or ()],
        "history": [dict(item) for item in prepared.get("history") or ()],
        "episodic_summary": str(prepared.get("episodic_summary") or ""),
        "top_k": prepared.get("top_k"),
        "chunk_top_k": prepared.get("chunk_top_k"),
        "filters": (
            dict(prepared["filters"]) if isinstance(prepared.get("filters"), Mapping) else None
        ),
        "semantic_highlights": bool(prepared.get("semantic_highlights")),
        "mode": str(prepared["mode"]) if prepared.get("mode") else None,
        "links": [dict(item) for item in prepared.get("links") or ()],
        "attachments": [dict(item) for item in prepared.get("attachments") or ()],
        "history_attachments": [dict(item) for item in prepared.get("history_attachments") or ()],
        "agent_session_id": str(prepared.get("agent_session_id") or ""),
        "agent_lane_id": str(prepared.get("agent_lane_id") or "main"),
        "source_lane_id": (
            str(prepared["source_lane_id"]) if prepared.get("source_lane_id") else None
        ),
    }
    if prepared.get("parent_run_id"):
        envelope["parent_run_id"] = str(prepared["parent_run_id"])
        envelope["continuation_kind"] = str(prepared.get("continuation_kind") or "")
    return envelope


@dataclass(frozen=True, slots=True)
class ReclaimState:
    """The three durable-progress counters one reclaim consults and updates."""

    durable_progress_version: int
    last_reclaim_progress_version: int
    reclaims_without_progress: int


@dataclass(frozen=True, slots=True)
class ReclaimDecision:
    """What one expired-lease reclaim decided: claimable or abandoned."""

    abandoned: bool
    reclaims_without_progress: int
    last_reclaim_progress_version: int


def advance_reclaim(
    state: ReclaimState,
    *,
    max_reclaims: int = MAX_RECLAIMS_WITHOUT_PROGRESS,
) -> ReclaimDecision:
    """Advance the reclaim counters for one expired-lease reclaim.

    Progress since the last reclaim resets the no-progress counter to one (this
    reclaim itself). Consecutive reclaims without durable progress abandon the
    run once the declared bound is reached: the fourth such reclaim abandons
    under the default bound (M3 durable progress contract).
    """
    if max_reclaims < 1:
        raise ValueError("max_reclaims must be positive")
    if state.durable_progress_version > state.last_reclaim_progress_version:
        return ReclaimDecision(
            abandoned=False,
            reclaims_without_progress=1,
            last_reclaim_progress_version=state.durable_progress_version,
        )
    reclaims = state.reclaims_without_progress + 1
    return ReclaimDecision(
        abandoned=reclaims >= max_reclaims,
        reclaims_without_progress=reclaims,
        last_reclaim_progress_version=state.last_reclaim_progress_version,
    )


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
class RunExecutionContext:
    """The immutable claim-bound execution surface one worker receives.

    The PostgreSQL adapter creates this binding at claim/reclaim: owner id,
    run id, worker id, lease owner, and fencing epoch are embedded, and the
    bound session and progress stores carry no fencing parameters, so callers
    can neither pass nor mutate them (M3 claim-bound execution stores).
    """

    owner_id: str
    run_id: str
    worker_id: str
    lease_owner: str
    fencing_epoch: int
    session_store: AgentSessionStore[EffectHostUpdate]
    progress_store: RunProgressStore
    workspace_store: WorkspaceStore | None = None


@dataclass(frozen=True, slots=True)
class ClaimedRun:
    """A run this worker now owns, with its claim-bound execution surface."""

    run: AnswerRunRecord
    execution: RunExecutionContext
    pinned_session_id: SessionId | None = None


@dataclass(frozen=True, slots=True)
class LeaseRenewal:
    """Whether a fenced worker still owns its run and its cancellation state."""

    renewed: bool
    cancel_requested: bool


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
class PendingPublication:
    """Staged workspace bytes to attach at successful terminal commit."""

    resource_id: str
    reference_kind: ArtifactReferenceKind
    filename: str
    mime_type: str
    content: bytes


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
    "ArtifactReferenceKind",
    "CancellationOutcome",
    "ClaimedRun",
    "IdempotencyKeyConflict",
    "LeaseRenewal",
    "PendingArtifact",
    "PendingPublication",
    "PendingArtifactReference",
    "ReclaimDecision",
    "ReclaimState",
    "RunArtifactReference",
    "RunCreation",
    "RunDeletion",
    "RunExecutionContext",
    "ShutdownOutcome",
    "SweepOutcome",
    "TerminalOutcome",
    "advance_reclaim",
    "answer_run_request_fingerprint",
    "artifact_digest",
    "canonical_run_request_json",
    "parse_run_id",
]
