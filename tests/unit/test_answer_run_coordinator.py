# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for the local Answer run coordinator and its subscribers.

The coordinator is the only local owner of accepted-run execution: it reserves
an execution slot before it claims a row, heartbeats a fenced lease, checkpoints
completed control turns, coalesces tokens, and writes exactly one terminal
event. Subscribers replay durable events and detach without touching the run.
"""

import asyncio
import datetime
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest

from dlightrag.core.answer_runs.coordinator import (
    AnswerRunCoordinator,
    LeaseLostError,
    RunCancelledError,
    RunSession,
)
from dlightrag.core.answer_runs.models import CheckpointError
from dlightrag.storage.answer_runs import (
    AnswerRunEvent,
    AnswerRunRecord,
    CheckpointCommit,
    ClaimedRun,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    RunCheckpoint,
    TerminalOutcome,
    artifact_digest,
)

_OWNER = "owner-alpha"


class _MemoryStore:
    """An in-memory stand-in for the fenced PostgreSQL run store."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[AnswerRunEvent]] = {}
        self.sweeps = 0
        self.claims = 0
        self.artifact_writes: list[dict[str, Any]] = []
        self.claim_gate: asyncio.Event | None = None
        self.heartbeats = 0
        self.heartbeat_failures = 0
        self.heartbeat_result: Any = None

    # -- test helpers -------------------------------------------------
    def add_run(self, run_id: str, **overrides: Any) -> None:
        self.runs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "completed_turns": 0,
            "recovery_count": 0,
            "fencing_epoch": 0,
            "lease_owner": None,
            "lease_live": False,
            "cancel_requested": False,
            "next_event_sequence": 1,
            "checkpoint": None,
            "result": None,
            "error_kind": None,
            "request": {"query": "why"},
            **overrides,
        }
        self.events.setdefault(run_id, [])

    def _record(self, row: Mapping[str, Any]) -> AnswerRunRecord:
        now = datetime.datetime.now(datetime.UTC)
        return AnswerRunRecord(
            owner_id=_OWNER,
            run_id=str(row["run_id"]),
            idempotency_key=None,
            request=row["request"],
            status=row["status"],
            phase=None,
            stop_reason=None,
            completed_turns=int(row["completed_turns"]),
            cancel_requested_at=now if row["cancel_requested"] else None,
            lease_owner=row["lease_owner"],
            lease_expires_at=now if row["lease_owner"] else None,
            fencing_epoch=int(row["fencing_epoch"]),
            recovery_count=int(row["recovery_count"]),
            next_event_sequence=int(row["next_event_sequence"]),
            events_trimmed_at=None,
            result=row["result"],
            error_kind=row["error_kind"],
            error_message=None,
            created_at=now,
            updated_at=now,
            started_at=now,
            finished_at=now if row["status"] in ("succeeded", "failed", "cancelled") else None,
        )

    def _owns(self, row: Mapping[str, Any], worker_id: str, epoch: int) -> bool:
        return (
            row["status"] == "running"
            and row["lease_owner"] == worker_id
            and row["lease_live"]
            and int(row["fencing_epoch"]) == epoch
        )

    def _append(self, row: dict[str, Any], event_type: str, payload: Mapping[str, Any]) -> int:
        sequence = int(row["next_event_sequence"])
        row["next_event_sequence"] = sequence + 1
        self.events[str(row["run_id"])].append(
            AnswerRunEvent(
                sequence=sequence,
                event_type=event_type,  # type: ignore[arg-type]
                payload=dict(payload),
                created_at=datetime.datetime.now(datetime.UTC),
            )
        )
        return sequence

    # -- store surface -------------------------------------------------
    async def claim_next(self, *, worker_id: str) -> ClaimedRun | None:
        if self.claim_gate is not None:
            self.claim_gate.set()
        self.claims += 1
        for row in self.runs.values():
            eligible = row["status"] == "queued" or (
                row["status"] == "running" and not row["lease_live"]
            )
            if not eligible or row["cancel_requested"]:
                continue
            if row["status"] == "running":
                row["recovery_count"] = int(row["recovery_count"]) + 1
            row["status"] = "running"
            row["lease_owner"] = worker_id
            row["lease_live"] = True
            row["fencing_epoch"] = int(row["fencing_epoch"]) + 1
            checkpoint = row["checkpoint"]
            return ClaimedRun(
                run=self._record(row),
                checkpoint=(
                    RunCheckpoint(
                        version=int(checkpoint["version"]),
                        completed_turns=int(checkpoint["completed_turns"]),
                        state=checkpoint["state"],
                    )
                    if checkpoint is not None
                    else None
                ),
            )
        return None

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal:
        self.heartbeats += 1
        if self.heartbeat_failures > 0:
            self.heartbeat_failures -= 1
            raise RuntimeError("transient heartbeat failure")
        if self.heartbeat_result is not None:
            return cast(LeaseRenewal, self.heartbeat_result)
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return LeaseRenewal(renewed=False, cancel_requested=False)
        return LeaseRenewal(renewed=True, cancel_requested=bool(row["cancel_requested"]))

    async def record_phase(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int, phase: str
    ) -> int | None:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return None
        return self._append(row, "progress", {"phase": phase})

    async def append_token_batch(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int, text: str
    ) -> int | None:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return None
        return self._append(row, "token", {"text": text})

    async def append_reset(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> int | None:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return None
        return self._append(row, "reset", {})

    async def commit_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        expected_completed_turns: int,
        version: int,
        state: Mapping[str, Any],
    ) -> CheckpointCommit:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return CheckpointCommit(
                outcome="lease_lost", completed_turns=int(row["completed_turns"])
            )
        if int(row["completed_turns"]) != expected_completed_turns:
            return CheckpointCommit(outcome="corrupt", completed_turns=int(row["completed_turns"]))
        row["completed_turns"] = expected_completed_turns + 1
        row["recovery_count"] = 0
        row["checkpoint"] = {
            "version": version,
            "completed_turns": row["completed_turns"],
            "state": dict(state),
        }
        return CheckpointCommit(outcome="committed", completed_turns=int(row["completed_turns"]))

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, Any],
        stop_reason: str | None = None,
    ) -> TerminalOutcome:
        row = self.runs[run_id]
        if row["cancel_requested"]:
            return await self.finish_cancelled(
                owner_id=owner_id, run_id=run_id, worker_id=worker_id, fencing_epoch=fencing_epoch
            )
        return self._finish(
            row,
            worker_id,
            fencing_epoch,
            status="succeeded",
            result=dict(result),
            event_type="done",
            payload={"status": "succeeded", "result": dict(result)},
        )

    async def finish_failure(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        error_kind: str,
        error_message: str,
    ) -> TerminalOutcome:
        row = self.runs[run_id]
        row["error_kind"] = error_kind
        return self._finish(
            row,
            worker_id,
            fencing_epoch,
            status="failed",
            result=None,
            event_type="error",
            payload={"kind": error_kind, "message": error_message},
        )

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome:
        return self._finish(
            self.runs[run_id],
            worker_id,
            fencing_epoch,
            status="cancelled",
            result=None,
            event_type="done",
            payload={"status": "cancelled"},
        )

    def _finish(
        self,
        row: dict[str, Any],
        worker_id: str,
        fencing_epoch: int,
        *,
        status: str,
        result: dict[str, Any] | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> TerminalOutcome:
        if not self._owns(row, worker_id, fencing_epoch):
            return TerminalOutcome(committed=False, status=None, event_sequence=None)
        row["status"] = status
        row["result"] = result
        row["lease_owner"] = None
        row["lease_live"] = False
        row["checkpoint"] = None
        sequence = self._append(row, event_type, payload)
        return TerminalOutcome(committed=True, status=status, event_sequence=sequence)  # type: ignore[arg-type]

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> str:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return "lease_lost"
        if row["cancel_requested"]:
            self._finish(
                row,
                worker_id,
                fencing_epoch,
                status="cancelled",
                result=None,
                event_type="done",
                payload={"status": "cancelled"},
            )
            return "cancelled"
        row["status"] = "queued"
        row["lease_owner"] = None
        row["lease_live"] = False
        return "requeued"

    async def sweep_once(self) -> Any:
        self.sweeps += 1
        return None

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[AnswerRunEvent, ...]:
        return tuple(
            event for event in self.events.get(run_id, []) if event.sequence > after_sequence
        )

    async def get_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None:
        row = self.runs.get(run_id)
        return self._record(row) if row is not None else None

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> Any:
        self.runs[run_id]["cancel_requested"] = True
        return None

    async def attach_artifacts(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        expected_completed_turns: int,
        artifacts: Sequence[Any] = (),
        references: Sequence[Any] = (),
    ) -> str:
        row = self.runs[run_id]
        if not self._owns(row, worker_id, fencing_epoch):
            return "lease_lost"
        if int(row["completed_turns"]) != expected_completed_turns:
            return "turn_mismatch"
        self.artifact_writes.append(
            {
                "run_id": run_id,
                "epoch": fencing_epoch,
                "expected_completed_turns": expected_completed_turns,
                "artifacts": list(artifacts),
                "references": list(references),
            }
        )
        return "attached"

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        return ()

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None:
        return None


class _Executor:
    """A deterministic stand-in for the manager's answer execution."""

    def __init__(self, body: Any) -> None:
        self._body = body
        self.sessions: list[RunSession] = []

    async def execute(self, session: RunSession) -> Mapping[str, Any]:
        self.sessions.append(session)
        return await self._body(session)


async def _settle(predicate: Any, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


def _coordinator(store: _MemoryStore, executor: _Executor, *, max_async: int = 2):
    return AnswerRunCoordinator(store=store, executor=executor, max_async=max_async)


class TestSchedulingAndLease:
    async def test_reserves_a_local_slot_before_it_claims_a_row(self) -> None:
        store = _MemoryStore()
        store.claim_gate = asyncio.Event()
        release = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await release.wait()
            return {"answer": "done"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        store.add_run("run-b")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "running")
            claims_while_busy = store.claims
            await asyncio.sleep(0.05)

            assert store.claims == claims_while_busy
            assert store.runs["run-b"]["status"] == "queued"
            release.set()
            await _settle(lambda: store.runs["run-b"]["status"] == "succeeded")
        finally:
            release.set()
            await coordinator.aclose()

    async def test_lease_loss_stops_execution_and_forbids_later_writes(self) -> None:
        store = _MemoryStore()
        stopped = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            store.runs["run-a"]["lease_owner"] = "another-worker"
            try:
                await asyncio.sleep(5)
            finally:
                stopped.set()
            return {"answer": "unreachable"}

        coordinator = AnswerRunCoordinator(
            store=store, executor=_Executor(body), max_async=1, heartbeat_seconds=0.01
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(stopped.is_set)
            await _settle(lambda: not coordinator.active_runs)

            assert store.runs["run-a"]["status"] == "running"
            assert [event.event_type for event in store.events["run-a"]] == []
        finally:
            await coordinator.aclose()

    async def test_a_worker_that_lost_its_lease_cannot_append_or_finish(self) -> None:
        store = _MemoryStore()
        outcome: dict[str, Any] = {}

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.enter_phase("generating")
            store.runs["run-a"]["fencing_epoch"] = 99
            try:
                await session.emit_token("late")
                await session.flush_tokens()
            except LeaseLostError as exc:
                outcome["raised"] = exc
                raise
            return {"answer": "no"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: "raised" in outcome)
            await _settle(lambda: not coordinator.active_runs)

            assert [event.event_type for event in store.events["run-a"]] == ["progress"]
        finally:
            await coordinator.aclose()

    async def test_sweeper_runs_without_holding_an_execution_slot(self) -> None:
        store = _MemoryStore()
        hold = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await hold.wait()
            return {"answer": "done"}

        coordinator = AnswerRunCoordinator(
            store=store, executor=_Executor(body), max_async=1, sweep_seconds=0.01
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "running")
            await _settle(lambda: store.sweeps >= 2)
        finally:
            hold.set()
            await coordinator.aclose()


class TestDurableProgress:
    async def test_tokens_are_coalesced_in_order_with_one_terminal_event(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.enter_phase("generating")
            for token in ("alpha ", "beta ", "gamma"):
                await session.emit_token(token)
            return {"answer": "alpha beta gamma"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        kinds = [event.event_type for event in store.events["run-a"]]
        assert kinds == ["progress", "token", "done"]
        assert store.events["run-a"][1].payload["text"] == "alpha beta gamma"
        assert kinds.count("done") + kinds.count("error") == 1

    async def test_checkpoint_advances_one_turn_and_survives_a_restart(self) -> None:
        store = _MemoryStore()
        attempts: list[int] = []

        async def body(session: RunSession) -> Mapping[str, Any]:
            attempts.append(session.completed_turns)
            if session.completed_turns == 0:
                await session.commit_checkpoint(
                    {"version": 1, "completed_turns": 1, "state": {"episode": "one"}}
                )
                raise RuntimeError("process died")
            return {"answer": "resumed", "turns": session.completed_turns}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
            store.runs["run-a"].update(
                status="running",
                lease_live=False,
                error_kind=None,
                checkpoint={"version": 1, "completed_turns": 1, "state": {"episode": "one"}},
                completed_turns=1,
            )
            coordinator.wake()
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert attempts == [0, 1]
        assert store.runs["run-a"]["result"] == {"answer": "resumed", "turns": 1}

    async def test_resumed_run_emits_reset_before_regenerated_output(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a", next_event_sequence=4)
        store.events["run-a"] = []

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.emit_token("fresh draft")
            return {"answer": "fresh draft"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert [event.event_type for event in store.events["run-a"]] == [
            "reset",
            "token",
            "done",
        ]

    async def test_first_attempt_never_emits_reset(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.emit_token("draft")
            return {"answer": "draft"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert [event.event_type for event in store.events["run-a"]] == ["token", "done"]

    async def test_unreadable_checkpoint_fails_the_run_with_its_public_kind(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            raise CheckpointError("checkpoint_too_large", "too big")

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["error_kind"] == "checkpoint_too_large"
        assert store.events["run-a"][-1].payload["kind"] == "checkpoint_too_large"

    async def test_missing_checkpoint_for_a_resumed_turn_is_incompatible(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a", completed_turns=2, checkpoint=None)
        executed = False

        async def body(session: RunSession) -> Mapping[str, Any]:
            nonlocal executed
            executed = True
            return {"answer": "never"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        assert executed is False
        assert store.runs["run-a"]["error_kind"] == "checkpoint_incompatible"

    async def test_fetched_artifacts_are_attached_under_the_live_fence(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.attach_artifacts(
                artifacts=[PendingArtifact(content=b"page-bytes")],
                references=[
                    PendingArtifactReference(
                        resource_id="res-1",
                        reference_kind="fetched_resource",
                        ordinal=0,
                        digest=artifact_digest(b"page-bytes"),
                        filename="page.html",
                        mime_type="text/html",
                    )
                ],
            )
            return {"answer": "ok"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert len(store.artifact_writes) == 1
        write = store.artifact_writes[0]
        assert write["run_id"] == "run-a"
        assert write["epoch"] == 1
        assert write["expected_completed_turns"] == 0
        assert write["references"][0].reference_kind == "fetched_resource"

    async def test_attaching_against_a_stale_turn_fails_the_run_as_corrupt(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")

        async def body(session: RunSession) -> Mapping[str, Any]:
            store.runs["run-a"]["completed_turns"] = 5
            await session.attach_artifacts(
                artifacts=[PendingArtifact(content=b"page-bytes")],
                references=[
                    PendingArtifactReference(
                        resource_id="res-1",
                        reference_kind="fetched_resource",
                        ordinal=0,
                        digest=artifact_digest(b"page-bytes"),
                        filename="page.html",
                        mime_type="text/html",
                    )
                ],
            )
            return {"answer": "never"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "failed")
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["error_kind"] == "checkpoint_corrupt"


class TestHeartbeatResilience:
    """A store hiccup is not lease loss, and a dead heartbeat is not an outcome."""

    async def test_transient_heartbeat_failures_never_end_the_run(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")
        store.heartbeat_failures = 3

        async def body(session: RunSession) -> Mapping[str, Any]:
            await _settle(lambda: store.heartbeats >= 5)
            await session.check_cancelled()
            return {"answer": "survived"}

        coordinator = AnswerRunCoordinator(
            store=store, executor=_Executor(body), max_async=1, heartbeat_seconds=0.01
        )
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["result"] == {"answer": "survived"}

    async def test_an_authoritative_non_renewal_still_stops_the_run(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")
        store.heartbeat_failures = 2

        async def body(session: RunSession) -> Mapping[str, Any]:
            await _settle(lambda: store.heartbeats >= 3)
            row = store.runs["run-a"]
            row["lease_owner"] = "another-worker"
            row["fencing_epoch"] = int(row["fencing_epoch"]) + 1
            await asyncio.sleep(5)
            return {"answer": "never"}

        executor = _Executor(body)
        coordinator = AnswerRunCoordinator(
            store=store, executor=executor, max_async=1, heartbeat_seconds=0.01
        )
        await coordinator.start()
        try:
            await _settle(lambda: bool(executor.sessions) and executor.sessions[0].lease_lost)
            await _settle(lambda: coordinator.active_runs == ())
        finally:
            await coordinator.aclose()

        assert store.runs["run-a"]["status"] == "running"
        assert store.events["run-a"] == []

    async def test_a_failing_heartbeat_task_never_replaces_the_run_outcome(self) -> None:
        store = _MemoryStore()
        store.add_run("run-a")
        store.heartbeat_result = _UnreadableRenewal()
        executions: list[asyncio.Task[Any]] = []

        async def body(session: RunSession) -> Mapping[str, Any]:
            task = asyncio.current_task()
            assert task is not None
            executions.append(task)
            await _settle(lambda: store.heartbeats >= 1)
            return {"answer": "kept"}

        coordinator = AnswerRunCoordinator(
            store=store, executor=_Executor(body), max_async=1, heartbeat_seconds=0.01
        )
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
            await _settle(lambda: bool(executions) and executions[0].done())
        finally:
            await coordinator.aclose()

        assert executions[0].exception() is None
        assert store.runs["run-a"]["result"] == {"answer": "kept"}


class _UnreadableRenewal:
    """A renewal this worker cannot interpret; reading it must not kill the run."""

    @property
    def renewed(self) -> bool:
        raise RuntimeError("lease renewal could not be interpreted")


class TestCancellationAndShutdown:
    async def test_cancellation_is_observed_at_a_boundary_and_commits_cancelled(self) -> None:
        store = _MemoryStore()
        entered = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            entered.set()
            for _ in range(200):
                await session.check_cancelled()
                await asyncio.sleep(0.005)
            return {"answer": "unreachable"}

        coordinator = AnswerRunCoordinator(
            store=store, executor=_Executor(body), max_async=1, heartbeat_seconds=0.01
        )
        store.add_run("run-a")
        await coordinator.start()
        try:
            await entered.wait()
            await store.request_cancellation(owner_id=_OWNER, run_id="run-a")
            await _settle(lambda: store.runs["run-a"]["status"] == "cancelled")
        finally:
            await coordinator.aclose()

        assert store.events["run-a"][-1].payload == {"status": "cancelled"}

    async def test_pending_tokens_are_flushed_before_a_terminal_transition(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.emit_token("partial")
            raise RunCancelledError

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "cancelled")
        finally:
            await coordinator.aclose()

        assert [event.event_type for event in store.events["run-a"]] == ["token", "done"]

    async def test_graceful_shutdown_requeues_without_crash_recovery(self) -> None:
        store = _MemoryStore()
        running = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            running.set()
            await asyncio.sleep(30)
            return {"answer": "unreachable"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        await running.wait()

        await coordinator.aclose()

        assert store.runs["run-a"]["status"] == "queued"
        assert store.runs["run-a"]["recovery_count"] == 0
        assert store.runs["run-a"]["lease_owner"] is None
        assert not coordinator.active_runs

    async def test_shutdown_leaves_no_background_task_running(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            return {"answer": "done"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        before = {task for task in asyncio.all_tasks()}
        await coordinator.start()
        await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        await coordinator.aclose()

        leaked = {task for task in asyncio.all_tasks() if task not in before and not task.done()}
        assert leaked == set()


class TestSubscriptions:
    async def test_replays_durable_events_then_follows_to_the_terminal_event(self) -> None:
        store = _MemoryStore()
        gate = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.enter_phase("generating")
            await session.emit_token("one")
            await session.flush_tokens()
            await gate.wait()
            return {"answer": "one"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: len(store.events["run-a"]) == 2)
            seen: list[AnswerRunEvent] = []

            async def consume() -> None:
                async for event in coordinator.subscribe(owner_id=_OWNER, run_id="run-a"):
                    seen.append(event)

            reader = asyncio.create_task(consume())
            await _settle(lambda: len(seen) == 2)
            gate.set()
            await asyncio.wait_for(reader, timeout=2)
        finally:
            gate.set()
            await coordinator.aclose()

        assert [event.sequence for event in seen] == [1, 2, 3]
        assert seen[-1].event_type == "done"

    async def test_reconnect_after_a_cursor_has_no_gap_or_duplicate(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.enter_phase("generating")
            await session.emit_token("one")
            await session.flush_tokens()
            return {"answer": "one"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
            first: list[int] = []
            async for event in coordinator.subscribe(owner_id=_OWNER, run_id="run-a"):
                first.append(event.sequence)
                if len(first) == 2:
                    break
            second = [
                event.sequence
                async for event in coordinator.subscribe(
                    owner_id=_OWNER, run_id="run-a", after_sequence=first[-1]
                )
            ]
        finally:
            await coordinator.aclose()

        assert first == [1, 2]
        assert second == [3]

    async def test_detaching_a_subscriber_never_cancels_the_run(self) -> None:
        store = _MemoryStore()
        release = asyncio.Event()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.enter_phase("generating")
            await release.wait()
            return {"answer": "still finished"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:
            stream = coordinator.subscribe(owner_id=_OWNER, run_id="run-a")
            assert (await anext(stream)).event_type == "progress"
            await stream.aclose()

            assert store.runs["run-a"]["cancel_requested"] is False
            release.set()
            await _settle(lambda: store.runs["run-a"]["status"] == "succeeded")
        finally:
            release.set()
            await coordinator.aclose()

    async def test_two_subscribers_each_see_the_whole_stream(self) -> None:
        store = _MemoryStore()

        async def body(session: RunSession) -> Mapping[str, Any]:
            await session.emit_token("shared")
            return {"answer": "shared"}

        coordinator = _coordinator(store, _Executor(body), max_async=1)
        store.add_run("run-a")
        await coordinator.start()
        try:

            async def consume() -> list[int]:
                return [
                    event.sequence
                    async for event in coordinator.subscribe(owner_id=_OWNER, run_id="run-a")
                ]

            first, second = await asyncio.gather(consume(), consume())
        finally:
            await coordinator.aclose()

        assert first == second == [1, 2]

    async def test_unknown_run_yields_nothing_and_closes(self) -> None:
        store = _MemoryStore()
        coordinator = _coordinator(store, _Executor(_noop), max_async=1)

        events = [event async for event in coordinator.subscribe(owner_id=_OWNER, run_id="missing")]

        assert events == []


async def _noop(session: RunSession) -> Mapping[str, Any]:
    return {"answer": ""}


@pytest.mark.parametrize("max_async", [1, 4])
def test_execution_slots_are_bounded_by_max_async(max_async: int) -> None:
    coordinator = AnswerRunCoordinator(
        store=_MemoryStore(), executor=_Executor(_noop), max_async=max_async
    )
    assert coordinator.max_async == max_async
