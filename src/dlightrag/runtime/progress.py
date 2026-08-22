# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Fast-stage progress: the run-progress store and its closed outcomes.

Fast Answers settle three typed stages (planner, retrieval, final_generation)
against a durable progress version. Every settlement includes the live
lease/epoch predicate in the same transaction (M3-D22, M3-D28). Stage outcomes
are closed values, never database exceptions (M3-D3).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from dlightrag.agent.session.ids import StageIntentId

type StageCommitOutcome = Literal[
    "committed", "version_conflict", "lease_lost", "stage_conflict", "evidence_conflict"
]


@dataclass(frozen=True, slots=True)
class StageCommit:
    """One committed stage settlement: new progress version and stage facts."""

    progress_version: int
    stage_intent_id: StageIntentId
    evidence_count: int


@dataclass(frozen=True, slots=True)
class StageProgressConflict:
    """The expected progress version no longer matches the stored version."""

    expected_progress_version: int
    current_progress_version: int


@dataclass(frozen=True, slots=True)
class StageConflict:
    """A stage with this stable intent id was already settled differently."""

    stage_intent_id: StageIntentId


@dataclass(frozen=True, slots=True)
class StageLeaseLost:
    """The caller's lease no longer owns this run."""


@dataclass(frozen=True, slots=True)
class StageEvidenceConflict:
    """A host evidence update collided with an existing identity."""


type StageCommitResult = (
    StageCommit | StageProgressConflict | StageLeaseLost | StageConflict | StageEvidenceConflict
)


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One committed Fast stage: identity, name, versioned state, and evidence."""

    stage_intent_id: StageIntentId
    stage_name: str
    progress_version: int
    state: Any
    state_digest: str
    evidence_count: int
    settled_at: str | None = None


class RunProgressStore(Protocol):
    """Claim-bound durable progress storage for one Fast Answer run.

    The PostgreSQL adapter binds owner, run, lease owner, and fencing epoch at
    claim time; public methods carry no fencing fields, so callers can neither
    pass nor mutate them (M3 claim-bound execution stores).
    """

    async def load_stage(self, stage_intent_id: StageIntentId) -> StageRecord | None: ...

    async def settle_stage(
        self,
        *,
        expected_progress_version: int,
        stage_intent_id: StageIntentId,
        stage_name: str,
        state: Any,
        evidence: Sequence[Any],
    ) -> StageCommitResult: ...


__all__ = [
    "RunProgressStore",
    "StageCommit",
    "StageCommitOutcome",
    "StageCommitResult",
    "StageConflict",
    "StageEvidenceConflict",
    "StageLeaseLost",
    "StageProgressConflict",
    "StageRecord",
]
