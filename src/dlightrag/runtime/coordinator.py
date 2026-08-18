# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The local owner of accepted durable run execution.

One coordinator per process schedules, executes, and finalizes durable runs. It
reserves a local execution slot *before* it claims a row, so a worker never
holds a lease while waiting for capacity, and every durable write it makes is
predicated on its own lease owner and fencing epoch. Lease duration, heartbeat
cadence, sweep cadence, and token coalescing are fixed internal constants; the
public worker bound is ``runtime.answer_worker_concurrency``. AI provider calls
and RAG pipeline work have independent admission owners.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol

from dlightrag.runtime.contracts import AnswerRunPhase
from dlightrag.runtime.errors import RunExecutionError
from dlightrag.runtime.records import AnswerRunEvent, ClaimedRun
from dlightrag.runtime.store import AnswerRunStore
from dlightrag.runtime.subscription import RunEventBroker, follow_run_events

logger = logging.getLogger(__name__)

#: How often an idle worker rechecks the queue for another host's work.
SWEEP_SECONDS = 1.0
#: Cadence for renewing an owned run's storage lease.
RUN_HEARTBEAT_SECONDS = 20.0
#: How often each process runs the 30-day event trim and run/artifact prune.
#: Every pass is bounded, ``SKIP LOCKED``, and idempotent, so running it on every
#: run-owning process needs no leader election and exposes no operator knob.
MAINTENANCE_SECONDS = 3600.0
#: Share of one cadence a process may defer its first retention pass by, so a
#: fleet that restarts together does not trim on the same instant.
_MAINTENANCE_JITTER_FRACTION = 0.1
#: Pause between drained retention batches, so a large backlog stays background
#: work rather than a burst against the pool.
_MAINTENANCE_BATCH_PAUSE_SECONDS = 0.05
#: How long a graceful shutdown waits for writes that were already in flight.
SHUTDOWN_WRITE_GRACE_SECONDS = 5.0
#: Coalescing bounds for durable token batches.
TOKEN_BATCH_CHARS = 512
TOKEN_BATCH_SECONDS = 0.25

_RUN_EXECUTION_FAILED = "run_execution_failed"
_RUN_EXECUTION_FAILED_MESSAGE = "Run execution failed."


def _startup_jitter(cadence: float) -> float:
    return random.uniform(0.0, max(0.0, cadence) * _MAINTENANCE_JITTER_FRACTION)  # noqa: S311


class RunCancelledError(Exception):
    """The run's owner requested cancellation and the worker observed it."""


class LeaseLostError(Exception):
    """This worker no longer owns the run and must persist nothing further."""


class DurableWrites:
    """Keeps every shielded durable write joinable across a shutdown.

    ``asyncio.shield`` stops a cancelled worker from tearing a small fenced write
    in half, but on its own it leaves that write running as an unreferenced task:
    a graceful shutdown could return before the requeue or terminal transition it
    already started ever reached PostgreSQL. Registering each one lets ``aclose``
    join them within the shutdown grace.
    """

    def __init__(self) -> None:
        self._writes: set[asyncio.Task[Any]] = set()

    def shield[T](self, operation: Awaitable[T]) -> Awaitable[T]:
        task = asyncio.ensure_future(operation)
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)
        return asyncio.shield(task)

    async def drain(self, timeout: float) -> None:
        """Wait out the writes already in flight; never start new ones."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self._writes:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "Shutdown left %d durable Answer writes in flight", len(self._writes)
                )
                return
            await asyncio.wait(tuple(self._writes), timeout=remaining)


class RunExecutor(Protocol):
    """Executes one claimed run or raises an owner-classified failure."""

    async def execute(self, session: RunSession) -> Mapping[str, Any]: ...


class RunSession:
    """One claimed run's fenced view of its durable state.

    Every write is predicated on this worker's lease owner and fencing epoch and
    is shielded, so a shutdown cancellation never tears a small fenced write in
    half. The first zero-row write means the lease is gone: the session latches
    closed so no later event, settlement, or terminal transition can be written
    by a worker the run no longer belongs to. Checkpoint and artifact methods
    are gone: journal settlements and acceptance carry those facts.
    """

    def __init__(
        self,
        store: AnswerRunStore,
        claimed: ClaimedRun,
        *,
        broker: RunEventBroker,
        writes: DurableWrites,
        notify: Callable[[], None] | None = None,
    ) -> None:
        execution = claimed.execution
        run = claimed.run
        self.owner_id = execution.owner_id
        self.run_id = execution.run_id
        self.worker_id = execution.worker_id
        self.fencing_epoch = execution.fencing_epoch
        self.execution = execution
        self.prepared_input: Mapping[str, Any] | None = run.prepared_input
        self._store = store
        self._broker = broker
        self._writes = writes
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
        self._guard()
        if self._cancel_requested:
            raise RunCancelledError

    def _guard(self) -> None:
        """Refuse every further durable write once this session lost coherence."""
        if self._lease_lost:
            raise LeaseLostError

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

    async def _fenced(self, operation: Awaitable[int | None]) -> None:
        self._guard()
        sequence = await self._writes.shield(operation)
        if sequence is None:
            self._lease_lost = True
            raise LeaseLostError
        self._broker.notify(self.owner_id, self.run_id)
        if self._notify is not None:
            self._notify()


class RunCoordinator:
    """Schedule, execute, and finalize this process's durable runs."""

    def __init__(
        self,
        *,
        store: AnswerRunStore,
        executor: RunExecutor,
        answer_worker_concurrency: int,
        worker_id: str | None = None,
        heartbeat_seconds: float = RUN_HEARTBEAT_SECONDS,
        sweep_seconds: float = SWEEP_SECONDS,
        maintenance_seconds: float = MAINTENANCE_SECONDS,
    ) -> None:
        if answer_worker_concurrency < 1:
            raise ValueError("answer_worker_concurrency must be positive")
        self._store = store
        self._executor = executor
        self._answer_worker_concurrency = int(answer_worker_concurrency)
        self._worker_id = worker_id or f"answer-worker-{uuid.uuid4().hex}"
        self._heartbeat_seconds = heartbeat_seconds
        self._sweep_seconds = sweep_seconds
        self._maintenance_seconds = maintenance_seconds
        self._slots = asyncio.Semaphore(self._answer_worker_concurrency)
        self._broker = RunEventBroker()
        self._writes = DurableWrites()
        self._wake = asyncio.Event()
        self._acceptance_lock = asyncio.Lock()
        self._closing = False
        self._scheduler: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._maintainer: asyncio.Task[None] | None = None
        self._runs: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, RunSession] = {}

    @property
    def answer_worker_concurrency(self) -> int:
        return self._answer_worker_concurrency

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def active_runs(self) -> tuple[str, ...]:
        return tuple(self._runs)

    @property
    def is_started(self) -> bool:
        """Whether this process can currently execute newly accepted runs."""
        tasks = (self._scheduler, self._sweeper, self._maintainer)
        return not self._closing and all(task is not None and not task.done() for task in tasks)

    @contextlib.asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        """Keep shutdown from crossing one short durable acceptance write."""
        async with self._acceptance_lock:
            yield self.is_started

    async def start(self) -> None:
        """Begin claiming accepted runs and sweeping abandoned ones."""
        async with self._acceptance_lock:
            if self._scheduler is not None:
                return
            self._closing = False
            self._scheduler = asyncio.create_task(self._schedule_forever())
            self._sweeper = asyncio.create_task(self._sweep_forever())
            self._maintainer = asyncio.create_task(self._maintain_forever())

    def cancel_local(self, owner_id: str, run_id: str) -> None:
        """Signal a locally leased run's task; the listener re-read authority first.

        Cancelling the task interrupts the executor at its next control
        boundary; its shielded writes settle and the coordinator commits the
        single cancelled terminal transition.
        """
        task = self._runs.get(run_id)
        if task is not None and not task.done():
            task.cancel()

    def wake(self) -> None:
        """Nudge this process after it accepted a run; polling remains the truth."""
        self._wake.set()

    async def aclose(self) -> None:
        """Stop claiming, let fenced writes settle, then requeue owned work."""
        async with self._acceptance_lock:
            self._closing = True
            self._wake.set()
            for task in (self._scheduler, self._sweeper, self._maintainer):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self._scheduler = None
            self._sweeper = None
            self._maintainer = None
            running = list(self._runs.values())
            for task in running:
                task.cancel()
            for task in running:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self._writes.drain(SHUTDOWN_WRITE_GRACE_SECONDS)
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

    async def _maintain_forever(self) -> None:
        """Apply 30-day retention without reserving an execution slot.

        The first pass is deferred by a bounded share of the cadence so a fleet
        that restarts together spreads its trims out instead of aligning them.
        """
        await asyncio.sleep(_startup_jitter(self._maintenance_seconds))
        while True:
            try:
                await self._maintain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Answer run retention failed", exc_info=True)
            await asyncio.sleep(self._maintenance_seconds)

    async def _maintain_once(self) -> None:
        """Drain both retention passes.

        Each pass is a bounded batch that strictly shrinks its own candidate set,
        so draining is finite; pausing between batches keeps a large backlog from
        monopolizing the pool, and a transient fault defers only the remainder.
        """
        while await self._store.trim_expired_event_logs() > 0:
            await asyncio.sleep(_MAINTENANCE_BATCH_PAUSE_SECONDS)
        while (await self._store.prune_expired_runs()).runs > 0:
            await asyncio.sleep(_MAINTENANCE_BATCH_PAUSE_SECONDS)

    # -- execution ------------------------------------------------------
    async def _execute(self, claimed: ClaimedRun) -> None:
        session = RunSession(
            self._store,
            claimed,
            broker=self._broker,
            writes=self._writes,
            notify=self._wake.set,
        )
        self._sessions[session.run_id] = session
        heartbeat = asyncio.create_task(self._heartbeat_forever(session))
        try:
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
        except RunExecutionError as exc:
            await self._finish_failure(session, exc.kind, exc.public_message)
        except Exception:
            logger.warning(
                "Run %s failed with an unclassified executor error", session.run_id, exc_info=True
            )
            await self._finish_failure(
                session,
                _RUN_EXECUTION_FAILED,
                _RUN_EXECUTION_FAILED_MESSAGE,
            )
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
        outcome = await self._writes.shield(
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
            await self._writes.shield(
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
        await self._writes.shield(
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
            await self._writes.shield(
                self._store.release_for_shutdown(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                )
            )
        self._broker.notify(session.owner_id, session.run_id)


__all__ = [
    "MAINTENANCE_SECONDS",
    "RUN_HEARTBEAT_SECONDS",
    "SHUTDOWN_WRITE_GRACE_SECONDS",
    "SWEEP_SECONDS",
    "TOKEN_BATCH_CHARS",
    "TOKEN_BATCH_SECONDS",
    "DurableWrites",
    "RunCoordinator",
    "RunExecutor",
    "LeaseLostError",
    "RunCancelledError",
    "RunSession",
]
