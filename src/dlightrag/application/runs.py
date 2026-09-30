# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Admission, the owner-scoped service, and views for the common durable Run lifecycle."""

import datetime
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias
from uuid import uuid7

from dlightrag.application.errors import (
    ApplicationConflictError,
    ApplicationInputError,
    ApplicationUnavailableError,
)
from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunPhase, RunStatus
from dlightrag.engine.runtime.records import CancellationOutcome as RuntimeCancellationOutcome
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)
from dlightrag.engine.runtime.records import (
    PreparedInputTooLargeError,
    PreparedRunEnvelope,
    RunAccessScope,
    require_prepared_input_bounds,
)
from dlightrag.engine.runtime.records import (
    RunAdmissionLimitExceededError as RuntimeRunAdmissionLimitExceededError,
)
from dlightrag.engine.runtime.records import RunCreation as RuntimeRunCreation
from dlightrag.engine.runtime.records import RunEvent as RuntimeRunEvent
from dlightrag.engine.runtime.records import RunRecord as RuntimeRunRecord

RunCancellationResult: TypeAlias = Literal[  # noqa: UP040
    "unknown", "cancelled", "pending", "already_terminal", "rejected"
]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_MAX_REPAIR_TEXT_CHARS = 512


def _repair_text(record: RuntimeRunRecord, key: str) -> str | None:
    if record.phase != "waiting_for_repair" or not isinstance(record.checkpoint, Mapping):
        return None
    value = record.checkpoint.get(key)
    return str(value)[:_MAX_REPAIR_TEXT_CHARS] if isinstance(value, str) and value else None


class IdempotencyKeyConflict(ApplicationConflictError):
    """A caller reused a submission key with different normalized input.

    The store's own text names the owner and key, so it stays in the cause.
    """

    def __init__(self) -> None:
        super().__init__("Idempotency key was reused with a different request")


class RunAdmissionLimitExceededError(ApplicationUnavailableError):
    """The deployment-wide nonterminal admission limit was reached."""

    def __init__(self) -> None:
        super().__init__("Deployment-wide nonterminal admission limit reached")


class RunRuntimeUnavailableError(ApplicationUnavailableError):
    """No local common Run scheduler can safely accept new work."""


class RunCancelledError(RuntimeError):
    """The run this caller waited on was cancelled by its owner."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id} was cancelled")
        self.run_id = run_id


class RunFailedError(RuntimeError):
    """The run this caller waited on failed with one public error."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.error_kind = kind
        self.public_message = message


@dataclass(frozen=True, slots=True)
class RunView:
    """Caller-facing common lifecycle state without lease or persistence internals."""

    run_id: str
    run_kind: RunKind
    lane: RunLane
    submitted_by: str
    access_scope_kind: Literal["owner", "workspace"]
    access_scope_id: str
    status: RunStatus
    phase: RunPhase | None
    durable_progress_version: int
    next_event_sequence: int
    events_trimmed_at: datetime.datetime | None
    cancel_requested: bool
    result: Mapping[str, Any] | None
    error_kind: str | None
    error_message: str | None
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    request: Mapping[str, Any]
    repair_reason: str | None = None
    repair_remedy: str | None = None

    @classmethod
    def from_runtime(cls, record: RuntimeRunRecord) -> RunView:
        """Drop worker-only state at the Engine-to-Application boundary."""
        return cls(
            run_id=record.run_id,
            run_kind=record.run_kind,
            lane=record.lane,
            submitted_by=record.submitted_by,
            access_scope_kind=record.access_scope.kind,
            access_scope_id=record.access_scope.scope_id,
            status=record.status,
            phase=record.phase,
            durable_progress_version=record.durable_progress_version,
            next_event_sequence=record.next_event_sequence,
            events_trimmed_at=record.events_trimmed_at,
            cancel_requested=record.cancel_requested,
            result=dict(record.result) if record.result is not None else None,
            error_kind=record.error_kind,
            error_message=record.error_message,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            request=dict(record.request_input()),
            repair_reason=_repair_text(record, "repair_reason"),
            repair_remedy=_repair_text(record, "repair_remedy"),
        )

    def request_input(self) -> Mapping[str, Any]:
        """Return the bounded accepted request used by caller projections."""
        return self.request

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One caller-facing event in a Run's gap-free durable sequence."""

    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    created_at: datetime.datetime

    @classmethod
    def from_runtime(cls, event: RuntimeRunEvent) -> RunEvent:
        return cls(
            sequence=event.sequence,
            event_type=event.event_type,
            payload=dict(event.payload),
            created_at=event.created_at,
        )


@dataclass(frozen=True, slots=True)
class RunCancellation:
    """Caller-facing result of an owner-scoped cancellation request."""

    outcome: RunCancellationResult
    run: RunView | None

    @classmethod
    def from_runtime(cls, outcome: RuntimeCancellationOutcome) -> RunCancellation:
        return cls(
            outcome=outcome.outcome,
            run=RunView.from_runtime(outcome.run) if outcome.run is not None else None,
        )


@dataclass(frozen=True, slots=True)
class RunCreation:
    """Caller-facing accepted Run, including idempotent replay state."""

    run: RunView
    replayed: bool

    @classmethod
    def from_runtime(cls, creation: RuntimeRunCreation) -> RunCreation:
        return cls(run=RunView.from_runtime(creation.run), replayed=bool(creation.replayed))


class RunReplayer[T](Protocol):
    """Where one kind looks up the Run a submission key already accepted."""

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> T | None: ...


class RunAccept[T](Protocol):
    """One kind's durable acceptance of an admitted envelope."""

    def __call__(self, *, envelope: PreparedRunEnvelope, run_id: str) -> Awaitable[T]: ...


class RunAdmissionScheduler(Protocol):
    """The local scheduler a submission is admitted by and wakes."""

    @property
    def is_started(self) -> bool: ...

    def admission(self) -> AbstractAsyncContextManager[bool]: ...

    def wake(self) -> None: ...


@contextmanager
def _public_refusals() -> Iterator[None]:
    """Name the store's refusals in the Application's words; the store's own stay the cause."""
    try:
        yield
    except RuntimeIdempotencyKeyConflict as exc:
        raise IdempotencyKeyConflict() from exc
    except RuntimeRunAdmissionLimitExceededError as exc:
        raise RunAdmissionLimitExceededError() from exc


@dataclass(frozen=True, slots=True)
class RunAdmission:
    """The one sequence every durable Run kind is admitted through.

    A use case keeps what is its own: normalizing its request, preparing its
    input, and the acceptor that stores it. Admission does the rest, in order:
    it replays a retried key, bounds the prepared input, takes this process's
    admission slot, accepts under it, wakes the scheduler, and names every
    refusal in the Application's words.
    """

    run_kind: RunKind
    lane: RunLane
    #: Names the runtime in its refusal: "{runtime} runtime is unavailable".
    runtime: str
    retention_seconds: int
    #: The kind's own input error, which an oversized prepared input raises.
    input_error: type[ApplicationInputError]
    scheduler: RunAdmissionScheduler

    async def replay[T](
        self,
        acceptor: RunReplayer[T],
        *,
        submitted_by: str,
        idempotency_key: str | None,
        fingerprint: str,
    ) -> T | None:
        """Return the Run a retried key already accepted; changed input is a conflict."""
        if idempotency_key is None:
            return None
        with _public_refusals():
            return await acceptor.replay_run(
                owner_id=submitted_by,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
                run_kind=self.run_kind,
            )

    def require_runtime(self) -> None:
        """Refuse while this process cannot run what it would accept."""
        if not self.scheduler.is_started:
            raise self._unavailable()

    async def admit[T](
        self,
        accept: RunAccept[T],
        *,
        submitted_by: str,
        access_scope: RunAccessScope,
        idempotency_key: str | None,
        fingerprint: str,
        payload: Mapping[str, Any],
        accepted_input: Mapping[str, Any],
        supersedes_run_id: str | None = None,
        run_id: str | None = None,
    ) -> T:
        """Bound, admit, and accept one prepared submission, then wake the scheduler.

        An unkeyed submission is keyed by its own run id, which admission draws
        unless the kind already staged work under one.
        """
        try:
            require_prepared_input_bounds(payload)
        except PreparedInputTooLargeError as exc:
            raise self.input_error(str(exc)) from exc
        self.require_runtime()
        with _public_refusals():
            async with self.scheduler.admission() as available:
                if not available:
                    raise self._unavailable()
                run_id = run_id or str(uuid7())
                accepted = await accept(
                    envelope=PreparedRunEnvelope(
                        run_kind=self.run_kind,
                        lane=self.lane,
                        submitted_by=submitted_by,
                        access_scope=access_scope,
                        submission_key=idempotency_key or run_id,
                        request_fingerprint=fingerprint,
                        payload=payload,
                        accepted_input=accepted_input,
                        retention_seconds=self.retention_seconds,
                        supersedes_run_id=supersedes_run_id,
                    ),
                    run_id=run_id,
                )
                # A linking acceptor accepts nothing when it finds nothing to link.
                if accepted is not None:
                    self.scheduler.wake()
        return accepted

    def _unavailable(self) -> RunRuntimeUnavailableError:
        return RunRuntimeUnavailableError(f"{self.runtime} runtime is unavailable")


class RunRepository(Protocol):
    async def get_run(self, *, owner_id: str, run_id: str) -> RuntimeRunRecord | None: ...
    async def get_run_global(self, *, run_id: str) -> RuntimeRunRecord | None: ...
    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RuntimeRunRecord, ...]: ...
    async def request_cancellation(
        self, *, owner_id: str, run_id: str
    ) -> RuntimeCancellationOutcome: ...
    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool: ...


class RunScheduler(Protocol):
    def cancel_local(self, owner_id: str, run_id: str) -> None: ...
    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RuntimeRunEvent]: ...


class RunService:
    """The sole generic lifecycle authority exposed by Application.

    ``on_cancelled`` hears of each Run that cancellation ended while it was
    queued: no executor runs it again, so whatever it held is released there.
    """

    def __init__(
        self,
        *,
        store: RunRepository,
        scheduler: RunScheduler,
        on_cancelled: Callable[[RunView], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._scheduler = scheduler
        self._on_cancelled = on_cancelled

    async def get(self, *, owner_id: str, run_id: str) -> RunView | None:
        record = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        return RunView.from_runtime(record) if record is not None else None

    async def get_global(self, *, run_id: str) -> RunView | None:
        """Return a Run for a transport to authorize by its declared scope."""
        record = await self._store.get_run_global(run_id=run_id)
        return RunView.from_runtime(record) if record is not None else None

    async def list(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunView, ...]:
        records = await self._store.list_runs(
            owner_id=owner_id, after_run_id=after_run_id, limit=limit
        )
        return tuple(RunView.from_runtime(record) for record in records)

    async def cancel(self, *, owner_id: str, run_id: str) -> RunCancellation:
        outcome = await self._store.request_cancellation(owner_id=owner_id, run_id=run_id)
        if outcome.outcome == "pending":
            self._scheduler.cancel_local(owner_id, run_id)
        cancellation = RunCancellation.from_runtime(outcome)
        if (
            cancellation.outcome == "cancelled"
            and cancellation.run is not None
            and self._on_cancelled is not None
        ):
            await self._on_cancelled(cancellation.run)
        return cancellation

    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool:
        resumed = await self._store.resume_repair(owner_id=owner_id, run_id=run_id)
        if resumed:
            wake = getattr(self._scheduler, "wake", None)
            if callable(wake):
                wake()
        return resumed

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RunEvent]:
        events = self._scheduler.subscribe(
            owner_id=owner_id, run_id=run_id, after_sequence=after_sequence
        )

        async def _views() -> AsyncIterator[RunEvent]:
            async for event in events:
                yield RunEvent.from_runtime(event)

        return _views()


__all__ = [
    "IdempotencyKeyConflict",
    "RunAccept",
    "RunAdmission",
    "RunAdmissionScheduler",
    "RunCancellation",
    "RunCancelledError",
    "RunAdmissionLimitExceededError",
    "RunCreation",
    "RunEvent",
    "RunFailedError",
    "RunKind",
    "RunLane",
    "RunRuntimeUnavailableError",
    "RunPhase",
    "RunReplayer",
    "RunRepository",
    "RunScheduler",
    "RunService",
    "RunStatus",
    "RunView",
]
