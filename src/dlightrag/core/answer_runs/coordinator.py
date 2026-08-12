# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The local owner of accepted Answer run execution.

One coordinator per process schedules, executes, and finalizes durable runs. It
reserves a local execution slot *before* it claims a row, so a worker never
holds a lease while waiting for capacity, and every durable write it makes is
predicated on its own lease owner and fencing epoch. Lease duration, heartbeat
cadence, sweep cadence, and token coalescing are fixed internal constants; the
only public bound is ``max_async``, which already bounds concurrent LLM calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from dlightrag.core.answer.errors import classify_answer_error
from dlightrag.core.answer_runs.models import CheckpointError
from dlightrag.core.answer_runs.subscription import RunEventBroker, follow_run_events
from dlightrag.storage.answer_runs import (
    ANSWER_RUN_HEARTBEAT_SECONDS,
    AnswerRunEvent,
    AnswerRunPhase,
    AnswerRunRecord,
    ClaimedRun,
    PendingArtifact,
    PendingArtifactReference,
    RunCheckpoint,
)

logger = logging.getLogger(__name__)

#: How often an idle worker rechecks the queue for another host's work.
SWEEP_SECONDS = 1.0
#: Coalescing bounds for durable token batches.
TOKEN_BATCH_CHARS = 512
TOKEN_BATCH_SECONDS = 0.25

_FAILED_MESSAGE = "Answer run failed."


class RunCancelledError(Exception):
    """The run's owner requested cancellation and the worker observed it."""


class LeaseLostError(Exception):
    """This worker no longer owns the run and must persist nothing further."""


class AnswerRunStore(Protocol):
    """The durable operations a coordinator is allowed to perform."""

    async def claim_next(self, *, worker_id: str) -> ClaimedRun | None: ...

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> Any: ...

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
        state: Mapping[str, Any],
    ) -> Any: ...

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
    ) -> str: ...

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, Any],
        stop_reason: str | None = None,
    ) -> Any: ...

    async def finish_failure(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        error_kind: str,
        error_message: str,
    ) -> Any: ...

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> Any: ...

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> str: ...

    async def sweep_once(self) -> Any: ...

    async def get_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None: ...

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[AnswerRunEvent, ...]: ...


class AnswerRunExecutor(Protocol):
    """Executes one claimed run and returns its canonical result payload."""

    async def execute(self, session: RunSession) -> Mapping[str, Any]: ...


class RunSession:
    """One claimed run's fenced view of its durable state.

    Every write is predicated on this worker's lease owner and fencing epoch and
    is shielded, so a shutdown cancellation never tears a small fenced write in
    half. The first zero-row write means the lease is gone: the session latches
    closed so no later event, checkpoint, or terminal transition can be written
    by a worker the run no longer belongs to.
    """

    def __init__(
        self,
        store: AnswerRunStore,
        claimed: ClaimedRun,
        *,
        broker: RunEventBroker,
        notify: Callable[[], None] | None = None,
    ) -> None:
        run = claimed.run
        self.owner_id = run.owner_id
        self.run_id = run.run_id
        self.worker_id = str(run.lease_owner or "")
        self.fencing_epoch = run.fencing_epoch
        self.request: Mapping[str, Any] = run.request
        self.completed_turns = run.completed_turns
        self.checkpoint: RunCheckpoint | None = claimed.checkpoint
        self._store = store
        self._broker = broker
        self._notify = notify
        self._cancel_requested = run.cancel_requested
        self._lease_lost = False
        self._pending_tokens: list[str] = []
        self._pending_chars = 0
        self._flush_deadline: float | None = None
        # A run that already committed an event has a partial draft somewhere;
        # regenerated output must clear it before the first new token.
        self._reset_pending = run.next_event_sequence > 1

    # -- state ---------------------------------------------------------
    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost

    def observe_cancellation(self) -> None:
        self._cancel_requested = True

    def observe_lease_loss(self) -> None:
        self._lease_lost = True

    async def check_cancelled(self) -> None:
        """Raise at a control boundary once the owner asked to cancel."""
        if self._lease_lost:
            raise LeaseLostError
        if self._cancel_requested:
            raise RunCancelledError

    # -- durable writes -------------------------------------------------
    async def enter_phase(self, phase: AnswerRunPhase) -> None:
        await self.flush_tokens()
        await self._fenced(
            self._store.record_phase(
                owner_id=self.owner_id,
                run_id=self.run_id,
                worker_id=self.worker_id,
                fencing_epoch=self.fencing_epoch,
                phase=phase,
            )
        )

    async def emit_token(self, text: str) -> None:
        """Buffer generated text into bounded durable batches."""
        if not text:
            return
        self._pending_tokens.append(text)
        self._pending_chars += len(text)
        now = asyncio.get_running_loop().time()
        if self._flush_deadline is None:
            self._flush_deadline = now + TOKEN_BATCH_SECONDS
        if self._pending_chars >= TOKEN_BATCH_CHARS or now >= self._flush_deadline:
            await self.flush_tokens()
            await self.check_cancelled()

    async def flush_tokens(self) -> None:
        """Commit any buffered text; a reset clears the previous draft first."""
        if not self._pending_tokens:
            return
        text = "".join(self._pending_tokens)
        self._pending_tokens.clear()
        self._pending_chars = 0
        self._flush_deadline = None
        if self._reset_pending:
            self._reset_pending = False
            await self._fenced(
                self._store.append_reset(
                    owner_id=self.owner_id,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    fencing_epoch=self.fencing_epoch,
                )
            )
        await self._fenced(
            self._store.append_token_batch(
                owner_id=self.owner_id,
                run_id=self.run_id,
                worker_id=self.worker_id,
                fencing_epoch=self.fencing_epoch,
                text=text,
            )
        )

    async def commit_checkpoint(self, envelope: Mapping[str, Any]) -> None:
        """Advance one completed control turn and its checkpoint atomically."""
        if self._lease_lost:
            raise LeaseLostError
        expected = self.completed_turns
        commit = await asyncio.shield(
            self._store.commit_checkpoint(
                owner_id=self.owner_id,
                run_id=self.run_id,
                worker_id=self.worker_id,
                fencing_epoch=self.fencing_epoch,
                expected_completed_turns=expected,
                version=int(envelope["version"]),
                state=envelope["state"],
            )
        )
        if commit.outcome == "lease_lost":
            self._lease_lost = True
            raise LeaseLostError
        if commit.outcome == "corrupt":
            raise CheckpointError(
                "checkpoint_corrupt",
                "Answer run state no longer matches its authoritative turn count.",
            )
        self.completed_turns = commit.completed_turns

    async def attach_artifacts(
        self,
        *,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> None:
        """Persist validated fetched bytes before they may enter a checkpoint."""
        if self._lease_lost:
            raise LeaseLostError
        outcome = await asyncio.shield(
            self._store.attach_artifacts(
                owner_id=self.owner_id,
                run_id=self.run_id,
                worker_id=self.worker_id,
                fencing_epoch=self.fencing_epoch,
                expected_completed_turns=self.completed_turns,
                artifacts=artifacts,
                references=references,
            )
        )
        if outcome == "turn_mismatch":
            # The run advanced a turn without this worker; its replay slots no
            # longer describe the durable state, so the run terminalizes here
            # instead of being handed back to crash recovery.
            raise CheckpointError(
                "checkpoint_corrupt",
                "Answer run artifacts no longer match its authoritative turn count.",
            )
        if outcome != "attached":
            self._lease_lost = True
            raise LeaseLostError

    async def _fenced(self, operation: Awaitable[int | None]) -> None:
        if self._lease_lost:
            raise LeaseLostError
        sequence = await asyncio.shield(operation)
        if sequence is None:
            self._lease_lost = True
            raise LeaseLostError
        self._broker.notify(self.owner_id, self.run_id)
        if self._notify is not None:
            self._notify()


class AnswerRunCoordinator:
    """Schedule, execute, and finalize this process's durable Answer runs."""

    def __init__(
        self,
        *,
        store: AnswerRunStore,
        executor: AnswerRunExecutor,
        max_async: int,
        worker_id: str | None = None,
        heartbeat_seconds: float = ANSWER_RUN_HEARTBEAT_SECONDS,
        sweep_seconds: float = SWEEP_SECONDS,
    ) -> None:
        self._store = store
        self._executor = executor
        self._max_async = max(1, int(max_async))
        self._worker_id = worker_id or f"answer-worker-{uuid.uuid4().hex}"
        self._heartbeat_seconds = heartbeat_seconds
        self._sweep_seconds = sweep_seconds
        self._slots = asyncio.Semaphore(self._max_async)
        self._broker = RunEventBroker()
        self._wake = asyncio.Event()
        self._closing = False
        self._scheduler: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._runs: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, RunSession] = {}

    @property
    def max_async(self) -> int:
        return self._max_async

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def active_runs(self) -> tuple[str, ...]:
        return tuple(self._runs)

    async def start(self) -> None:
        """Begin claiming accepted runs and sweeping abandoned ones."""
        if self._scheduler is not None:
            return
        self._closing = False
        self._scheduler = asyncio.create_task(self._schedule_forever())
        self._sweeper = asyncio.create_task(self._sweep_forever())

    def wake(self) -> None:
        """Nudge this process after it accepted a run; polling remains the truth."""
        self._wake.set()

    async def aclose(self) -> None:
        """Stop claiming, let fenced writes settle, then requeue owned work."""
        self._closing = True
        self._wake.set()
        for task in (self._scheduler, self._sweeper):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._scheduler = None
        self._sweeper = None
        running = list(self._runs.values())
        for task in running:
            task.cancel()
        for task in running:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._runs.clear()
        self._sessions.clear()

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[AnswerRunEvent]:
        """Follow one run's durable events; detaching never mutates the run."""

        async def _is_finished() -> bool:
            run = await self._store.get_run(owner_id=owner_id, run_id=run_id)
            return run is None or run.terminal

        return follow_run_events(
            self._store,
            self._broker,
            owner_id=owner_id,
            run_id=run_id,
            after_sequence=after_sequence,
            is_finished=_is_finished,
        )

    # -- scheduling -----------------------------------------------------
    async def _schedule_forever(self) -> None:
        while not self._closing:
            await self._slots.acquire()
            if self._closing:
                self._slots.release()
                return
            claimed: ClaimedRun | None = None
            try:
                claimed = await self._store.claim_next(worker_id=self._worker_id)
            except Exception:
                logger.warning("Answer run claim failed", exc_info=True)
            if claimed is None:
                self._slots.release()
                await self._idle()
                continue
            run_id = claimed.run.run_id
            task = asyncio.create_task(self._execute(claimed))
            self._runs[run_id] = task
            task.add_done_callback(lambda _task, key=run_id: self._forget(key))

    async def _idle(self) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=self._sweep_seconds)

    def _forget(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
        self._sessions.pop(run_id, None)
        self._slots.release()
        self._wake.set()

    async def _sweep_forever(self) -> None:
        """Finalize abandoned and cancel-pending rows without holding a slot."""
        while True:
            try:
                await self._store.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Answer run sweep failed", exc_info=True)
            await asyncio.sleep(self._sweep_seconds)

    # -- execution ------------------------------------------------------
    async def _execute(self, claimed: ClaimedRun) -> None:
        session = RunSession(self._store, claimed, broker=self._broker, notify=self._wake.set)
        self._sessions[session.run_id] = session
        heartbeat = asyncio.create_task(self._heartbeat_forever(session))
        try:
            if claimed.run.completed_turns > 0 and claimed.checkpoint is None:
                raise CheckpointError(
                    "checkpoint_incompatible",
                    "Answer run has completed turns but no restorable checkpoint.",
                )
            result = await self._executor.execute(session)
            await session.flush_tokens()
            await self._finish_success(session, result)
        except asyncio.CancelledError:
            await self._release(session)
            raise
        except RunCancelledError:
            await self._finish_cancelled(session)
        except LeaseLostError:
            logger.info(
                "Answer run %s lost its lease; leaving recovery to the next owner", session.run_id
            )
        except CheckpointError as exc:
            await self._finish_failure(session, exc.kind, exc.public_message)
        except Exception as exc:
            logger.warning("Answer run %s failed", session.run_id, exc_info=True)
            await self._finish_failure(session, classify_answer_error(exc), _FAILED_MESSAGE)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            except Exception:
                # A dead heartbeat is never this run's outcome.
                logger.warning(
                    "Answer run %s heartbeat ended in failure", session.run_id, exc_info=True
                )

    async def _heartbeat_forever(self, session: RunSession) -> None:
        """Renew an unexpired fenced lease and surface pending cancellation.

        A store that fails to answer is a transient fault, not lease loss: the
        renewal is retried on the next cadence and the run stays owned until the
        store authoritatively refuses to renew.
        """
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                renewal = await self._store.heartbeat(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Answer run %s heartbeat failed; retrying next cadence",
                    session.run_id,
                    exc_info=True,
                )
                continue
            if not renewal.renewed:
                session.observe_lease_loss()
                task = self._runs.get(session.run_id)
                if task is not None:
                    task.cancel()
                return
            if renewal.cancel_requested:
                session.observe_cancellation()

    async def _finish_success(self, session: RunSession, result: Mapping[str, Any]) -> None:
        if session.lease_lost:
            return
        outcome = await asyncio.shield(
            self._store.finish_success(
                owner_id=session.owner_id,
                run_id=session.run_id,
                worker_id=session.worker_id,
                fencing_epoch=session.fencing_epoch,
                result=result,
            )
        )
        self._broker.notify(session.owner_id, session.run_id)
        if not outcome.committed:
            session.observe_lease_loss()

    async def _finish_failure(self, session: RunSession, kind: str, message: str) -> None:
        if session.lease_lost:
            return
        with contextlib.suppress(Exception):
            await session.flush_tokens()
        with contextlib.suppress(Exception):
            await asyncio.shield(
                self._store.finish_failure(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                    error_kind=kind,
                    error_message=message,
                )
            )
        self._broker.notify(session.owner_id, session.run_id)

    async def _finish_cancelled(self, session: RunSession) -> None:
        if session.lease_lost:
            return
        with contextlib.suppress(Exception):
            await session.flush_tokens()
        await asyncio.shield(
            self._store.finish_cancelled(
                owner_id=session.owner_id,
                run_id=session.run_id,
                worker_id=session.worker_id,
                fencing_epoch=session.fencing_epoch,
            )
        )
        self._broker.notify(session.owner_id, session.run_id)

    async def _release(self, session: RunSession) -> None:
        """Requeue owned work on shutdown; this is not crash recovery."""
        if session.lease_lost:
            return
        with contextlib.suppress(Exception):
            await asyncio.shield(
                self._store.release_for_shutdown(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                )
            )
        self._broker.notify(session.owner_id, session.run_id)


__all__ = [
    "SWEEP_SECONDS",
    "TOKEN_BATCH_CHARS",
    "TOKEN_BATCH_SECONDS",
    "AnswerRunCoordinator",
    "AnswerRunExecutor",
    "AnswerRunStore",
    "LeaseLostError",
    "RunCancelledError",
    "RunSession",
]
