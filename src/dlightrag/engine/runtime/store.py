# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable operations required by the storage-neutral run coordinator."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from dlightrag.engine.runtime.contracts import AnswerRunPhase
from dlightrag.engine.runtime.records import (
    AnswerRunEvent,
    AnswerRunRecord,
    ClaimedRun,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    PendingPublication,
    RunArtifactReference,
    RunCreation,
    RunDeletion,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
)


class AnswerRunStore(Protocol):
    """The durable operations a run coordinator may perform.

    Claim returns a run with its claim-bound execution surface; checkpoint-era
    commit/attach methods are gone. Accepted attachments register through the
    acceptance transaction, evidence and fetched resources through effect or
    stage settlements.
    """

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

    async def append_tool_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        event_type: str,
        payload: Mapping[str, object],
    ) -> int | None: ...

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, object],
        stop_reason: str | None = None,
        publications: Sequence[PendingPublication] = (),
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

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[AnswerRunRecord, ...]: ...

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]: ...

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[AnswerRunEvent, ...]: ...


class AnswerAcceptanceStore(Protocol):
    """Acceptance-side durable operations for one new run.

    Prepared input, accepted blob resources, and the run row commit in one
    atomic acceptance transaction; queued and running rows store exactly one
    bounded ``prepared_input_json``.
    """

    async def accept_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        idempotency_key: str | None,
        prepared_input: Mapping[str, object],
        resources: Sequence[Mapping[str, object]],
        blobs: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
    ) -> RunCreation: ...


__all__ = ["AnswerAcceptanceStore", "AnswerRunStore"]
