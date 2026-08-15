# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable operations required by the storage-neutral run coordinator."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from dlightrag.runtime.contracts import AnswerRunPhase
from dlightrag.runtime.records import (
    AnswerRunEvent,
    AnswerRunRecord,
    ArtifactAttachOutcome,
    CheckpointCommit,
    ClaimedRun,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    RunDeletion,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
)


class AnswerRunStore(Protocol):
    """The durable operations a run coordinator may perform."""

    async def claim_next(self, *, worker_id: str) -> ClaimedRun | None: ...

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal: ...

    async def record_phase(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: AnswerRunPhase,
    ) -> int | None: ...

    async def append_token_batch(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int, text: str
    ) -> int | None: ...

    async def append_reset(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> int | None: ...

    async def commit_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        expected_completed_turns: int,
        version: int,
        state: Mapping[str, object],
    ) -> CheckpointCommit: ...

    async def attach_artifacts(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        expected_completed_turns: int,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> ArtifactAttachOutcome: ...

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, object],
        stop_reason: str | None = None,
    ) -> TerminalOutcome: ...

    async def finish_failure(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        error_kind: str,
        error_message: str,
    ) -> TerminalOutcome: ...

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome: ...

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> ShutdownOutcome: ...

    async def sweep_once(self) -> SweepOutcome: ...

    async def trim_expired_event_logs(self) -> int: ...

    async def prune_expired_runs(self) -> RunDeletion: ...

    async def get_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None: ...

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[AnswerRunEvent, ...]: ...


__all__ = ["AnswerRunStore"]
