# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The admission contract every durable Run kind shares, pinned for each kind.

Answer, Retrieval, and Corpus Mutation acceptance each replay a retried key,
bound the prepared input, take this process's admission slot, accept, wake the
scheduler, and name every refusal in the Application's words. Each test drives
all three kinds through their public acceptance entry points, so a change to
any step shows whichever kind it reaches.
"""

import datetime
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from dlightrag.application.answer_runs import AnswerRequestError
from dlightrag.application.corpus_admin import CorpusMutationInputError
from dlightrag.application.corpus_admin.mutations import CorpusMutationService, UploadLimits
from dlightrag.application.errors import ApplicationUnavailableError
from dlightrag.application.retrieval import (
    RetrievalInputError,
    RetrievalService,
    RetrievalSettings,
    RetrieveRequest,
)
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunCreation,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.runtime.policy import (
    CORPUS_MUTATION_RUN_RETENTION_SECONDS,
    DEFAULT_RUN_RETENTION_SECONDS,
    RETRIEVAL_RUN_RETENTION_SECONDS,
)
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)
from dlightrag.engine.runtime.records import (
    PreparedInputTooLargeError,
    PreparedRunEnvelope,
    RunAccessScope,
    RunRecord,
)
from dlightrag.engine.runtime.records import (
    RunAdmissionLimitExceededError as RuntimeRunAdmissionLimitExceededError,
)
from dlightrag.engine.runtime.records import RunCreation as RuntimeRunCreation
from tests.unit.test_answer_service import _request as answer_request
from tests.unit.test_answer_service import _service as answer_service

_OWNER = "owner-1"
_NOW = datetime.datetime(2026, 9, 30, tzinfo=datetime.UTC)
_STORED_RUN = "0199a0a0-0000-7000-8000-0000000000aa"
_STAGED_RUN = "0199a0a0-0000-7000-8000-0000000000bb"
# The store's own wording names owner and key; the caller must never see it.
_STORE_CONFLICT = "owner owner-1 reused idempotency key key-1 with different normalized input"
_STORE_LIMIT = "lane query already holds 200 nonterminal runs"
_PUBLIC_CONFLICT = "Idempotency key was reused with a different request"
_PUBLIC_LIMIT = "Deployment-wide nonterminal admission limit reached"


class _Store:
    """Replays and accepts in memory, recording what each kind submitted."""

    def __init__(self) -> None:
        self.replays: list[dict[str, Any]] = []
        self.accepted: list[tuple[PreparedRunEnvelope, str]] = []
        self.replay_result: RuntimeRunCreation | None = None
        self.replay_error: BaseException | None = None
        self.accept_error: BaseException | None = None
        # The run a concurrent acceptance of the same key already stored, if any.
        self.accept_replays: str | None = None

    async def replay_run(self, **call: Any) -> RuntimeRunCreation | None:
        self.replays.append(call)
        if self.replay_error is not None:
            raise self.replay_error
        return self.replay_result

    async def accept_run(
        self, *, envelope: PreparedRunEnvelope, run_id: str, **_projections: Any
    ) -> RuntimeRunCreation:
        if self.accept_error is not None:
            raise self.accept_error
        self.accepted.append((envelope, run_id))
        if self.accept_replays is not None:
            return RuntimeRunCreation(run=_record(envelope, self.accept_replays), replayed=True)
        return RuntimeRunCreation(run=_record(envelope, run_id), replayed=False)

    # Answer's acceptor protocol names the same acceptance ``create_run``.
    create_run = accept_run


class _Scheduler:
    """A started runtime whose admission slot a test can close under a submission."""

    def __init__(self) -> None:
        self.is_started = True
        self.admits = True
        self.wakes = 0

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        yield self.is_started and self.admits

    def wake(self) -> None:
        self.wakes += 1


def _record(envelope: PreparedRunEnvelope, run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        run_kind=envelope.run_kind,
        lane=envelope.lane,
        submitted_by=envelope.submitted_by,
        access_scope=envelope.access_scope,
        submission_key=envelope.submission_key,
        request_fingerprint=envelope.request_fingerprint,
        prepared_input=dict(envelope.payload),
        accepted_input=dict(envelope.accepted_input),
        status="queued",
        phase=None,
        stop_reason=None,
        cancel_requested_at=None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        durable_progress_version=0,
        last_reclaim_progress_version=0,
        reclaims_without_progress=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=None,
        error_kind=None,
        error_message=None,
        created_at=_NOW,
        updated_at=_NOW,
        started_at=None,
        finished_at=None,
    )


@dataclass(frozen=True, slots=True)
class _Kind:
    """One Run kind: how it is built and submitted, and what it is accepted as."""

    name: str
    runtime: str
    run_kind: str
    lane: str
    scope: tuple[str, str]
    retention_seconds: int
    input_error: type[Exception]
    # Which refusal wins when the input is too large and the runtime is down.
    unavailable_before_oversized: bool
    build: Callable[[_Store, _Scheduler, Path], Any]
    submit: Callable[[Any, str | None], Awaitable[RunCreation]]


def _answer(store: _Store, scheduler: _Scheduler, _root: Path) -> Any:
    return answer_service(store=store, coordinator=scheduler)


async def _submit_answer(service: Any, key: str | None) -> RunCreation:
    return await service.create(request=answer_request(), owner_id=_OWNER, idempotency_key=key)


def _retrieval(store: _Store, scheduler: _Scheduler, _root: Path) -> RetrievalService:
    return RetrievalService(
        pool=AsyncMock(),
        planners=Mock(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=40,
            default_chunk_top_k=20,
            timeout_seconds=300,
            query_image_limit=3,
        ),
        telemetry=NoopTelemetry(),
        store=cast(Any, store),
        coordinator=cast(Any, scheduler),
        model_profile_for_role=lambda _role: ModelProfile(context_window_tokens=128_000),
        model_invocation_fingerprint_for_role=lambda _role: ModelInvocationFingerprint(
            provider="openai",
            model="query-model",
            endpoint_fingerprint=None,
            api_family="chat_completion",
        ),
    )


async def _submit_retrieval(service: Any, key: str | None) -> RunCreation:
    return await service.create(
        request=RetrieveRequest(query="q", workspaces=("finance",)),
        owner_id=_OWNER,
        idempotency_key=key,
    )


async def _registered(_workspace: str) -> bool:
    return True


def _mutation(store: _Store, scheduler: _Scheduler, root: Path) -> CorpusMutationService:
    return CorpusMutationService(
        source_root=root / "inputs",
        corpus_root=root / "corpus",
        store=cast(Any, store),
        coordinator=scheduler,
        upload_limits=UploadLimits(file_bytes=10, request_bytes=15),
        workspace_exists=_registered,
    )


async def _submit_mutation(service: Any, key: str | None) -> RunCreation:
    return await service.create_delete(
        workspace="default",
        submitted_by=_OWNER,
        document_ids=["doc-1"],
        idempotency_key=key,
    )


_KINDS = (
    _Kind(
        name="answer",
        runtime="Answer",
        run_kind="answer",
        lane="query",
        scope=("owner", _OWNER),
        retention_seconds=DEFAULT_RUN_RETENTION_SECONDS,
        input_error=AnswerRequestError,
        unavailable_before_oversized=True,
        build=_answer,
        submit=_submit_answer,
    ),
    _Kind(
        name="retrieval",
        runtime="Retrieval",
        run_kind="retrieval",
        lane="query",
        scope=("owner", _OWNER),
        retention_seconds=RETRIEVAL_RUN_RETENTION_SECONDS,
        input_error=RetrievalInputError,
        unavailable_before_oversized=True,
        build=_retrieval,
        submit=_submit_retrieval,
    ),
    _Kind(
        name="corpus_mutation",
        runtime="Corpus Mutation",
        run_kind="corpus_mutation",
        lane="corpus_mutation",
        scope=("workspace", "default"),
        retention_seconds=CORPUS_MUTATION_RUN_RETENTION_SECONDS,
        input_error=CorpusMutationInputError,
        unavailable_before_oversized=False,
        build=_mutation,
        submit=_submit_mutation,
    ),
)


@pytest.fixture(params=_KINDS, ids=[kind.name for kind in _KINDS])
def kind(request: pytest.FixtureRequest) -> _Kind:
    return request.param


@pytest.fixture
def store() -> _Store:
    return _Store()


@pytest.fixture
def scheduler() -> _Scheduler:
    return _Scheduler()


@pytest.fixture
def service(kind: _Kind, store: _Store, scheduler: _Scheduler, tmp_path: Path) -> Any:
    return kind.build(store, scheduler, tmp_path)


def _stored(kind: _Kind) -> RuntimeRunCreation:
    envelope = PreparedRunEnvelope(
        run_kind=cast(Any, kind.run_kind),
        lane=cast(Any, kind.lane),
        submitted_by=_OWNER,
        access_scope=RunAccessScope(kind=cast(Any, kind.scope[0]), scope_id=kind.scope[1]),
        submission_key="key-1",
        request_fingerprint="stored-fingerprint",
        payload={},
        accepted_input={},
        retention_seconds=kind.retention_seconds,
    )
    return RuntimeRunCreation(run=_record(envelope, _STORED_RUN), replayed=True)


async def test_a_keyed_submission_is_replayed_then_accepted_once_and_wakes_the_scheduler(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    creation = await kind.submit(service, "key-1")

    [replay] = store.replays
    assert replay["owner_id"] == _OWNER
    assert replay["idempotency_key"] == "key-1"
    assert replay["run_kind"] == kind.run_kind
    [(envelope, run_id)] = store.accepted
    assert uuid.UUID(run_id).version == 7
    assert (envelope.run_kind, envelope.lane) == (kind.run_kind, kind.lane)
    assert envelope.submitted_by == _OWNER
    assert (envelope.access_scope.kind, envelope.access_scope.scope_id) == kind.scope
    assert envelope.submission_key == "key-1"
    assert envelope.request_fingerprint == replay["idempotency_fingerprint"]
    assert envelope.retention_seconds == kind.retention_seconds
    assert envelope.supersedes_run_id is None
    assert scheduler.wakes == 1
    assert isinstance(creation, RunCreation)
    assert (creation.run.run_id, creation.replayed) == (run_id, False)


async def test_an_unkeyed_submission_is_keyed_by_its_own_run_id(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    await kind.submit(service, None)

    assert store.replays == []
    [(envelope, run_id)] = store.accepted
    assert envelope.submission_key == run_id
    assert scheduler.wakes == 1


async def test_a_retried_key_returns_its_run_even_while_the_runtime_is_down(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    store.replay_result = _stored(kind)
    scheduler.is_started = False

    creation = await kind.submit(service, "key-1")

    assert (creation.run.run_id, creation.replayed) == (_STORED_RUN, True)
    assert store.accepted == []
    assert scheduler.wakes == 0


async def test_a_key_reused_with_changed_input_is_the_public_conflict(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    store.replay_error = RuntimeIdempotencyKeyConflict(_STORE_CONFLICT)

    with pytest.raises(IdempotencyKeyConflict) as raised:
        await kind.submit(service, "key-1")

    assert str(raised.value) == _PUBLIC_CONFLICT
    assert raised.value.__cause__ is store.replay_error
    assert store.accepted == []
    assert scheduler.wakes == 0


async def test_a_conflict_met_while_accepting_is_the_same_public_conflict(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    store.accept_error = RuntimeIdempotencyKeyConflict(_STORE_CONFLICT)

    with pytest.raises(IdempotencyKeyConflict) as raised:
        await kind.submit(service, "key-1")

    assert str(raised.value) == _PUBLIC_CONFLICT
    assert raised.value.__cause__ is store.accept_error
    assert scheduler.wakes == 0


async def test_the_admission_limit_is_a_public_retryable_refusal(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    store.accept_error = RuntimeRunAdmissionLimitExceededError(_STORE_LIMIT)

    with pytest.raises(RunAdmissionLimitExceededError) as raised:
        await kind.submit(service, "key-1")

    assert isinstance(raised.value, ApplicationUnavailableError)
    assert str(raised.value) == _PUBLIC_LIMIT
    assert raised.value.__cause__ is store.accept_error
    assert scheduler.wakes == 0


async def test_an_unstarted_runtime_refuses_before_anything_is_accepted(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    scheduler.is_started = False

    with pytest.raises(RunRuntimeUnavailableError) as raised:
        await kind.submit(service, "key-1")

    assert str(raised.value) == f"{kind.runtime} runtime is unavailable"
    assert raised.value.__cause__ is None
    assert len(store.replays) == 1
    assert store.accepted == []
    assert scheduler.wakes == 0


async def test_an_admission_slot_closed_under_the_submission_refuses_alike(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    scheduler.admits = False

    with pytest.raises(RunRuntimeUnavailableError) as raised:
        await kind.submit(service, "key-1")

    assert str(raised.value) == f"{kind.runtime} runtime is unavailable"
    assert raised.value.__cause__ is None
    assert store.accepted == []
    assert scheduler.wakes == 0


async def test_an_oversized_prepared_input_is_the_callers_to_fix(
    kind: _Kind,
    store: _Store,
    scheduler: _Scheduler,
    service: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.runtime.records.MAX_PREPARED_INPUT_BYTES", 64)

    with pytest.raises(kind.input_error) as raised:
        await kind.submit(service, "key-1")

    assert str(raised.value).startswith("prepared_input_too_large: ")
    assert str(raised.value).endswith(" bytes exceed the 64 byte bound")
    assert isinstance(raised.value.__cause__, PreparedInputTooLargeError)
    assert store.accepted == []
    assert scheduler.wakes == 0


async def test_an_oversized_input_meets_a_stopped_runtime_in_each_kinds_order(
    kind: _Kind,
    store: _Store,
    scheduler: _Scheduler,
    service: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.runtime.records.MAX_PREPARED_INPUT_BYTES", 64)
    scheduler.is_started = False
    expected = RunRuntimeUnavailableError if kind.unavailable_before_oversized else kind.input_error

    with pytest.raises(expected):
        await kind.submit(service, "key-1")

    assert store.accepted == []


async def test_other_store_failures_pass_through_untranslated(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    store.accept_error = ConnectionError("store unavailable")

    with pytest.raises(ConnectionError, match="store unavailable"):
        await kind.submit(service, "key-1")

    assert scheduler.wakes == 0


async def test_a_run_the_store_replays_while_accepting_still_wakes_the_scheduler(
    kind: _Kind, store: _Store, scheduler: _Scheduler, service: Any
) -> None:
    store.accept_replays = _STORED_RUN

    creation = await kind.submit(service, "key-1")

    assert (creation.run.run_id, creation.replayed) == (_STORED_RUN, True)
    assert scheduler.wakes == 1


async def test_a_linked_answer_acceptance_that_links_nothing_wakes_nothing(
    store: _Store, scheduler: _Scheduler
) -> None:
    class _NoConversation:
        async def replay_run(self, **_call: Any) -> None:
            return None

        async def create_run(self, **_call: Any) -> None:
            return None

    service = answer_service(store=store, coordinator=scheduler)

    linked = await service.accept(
        request=answer_request(),
        owner_id=_OWNER,
        idempotency_key="key-1",
        idempotency_fingerprint="submission-fingerprint",
        acceptor=_NoConversation(),
    )

    assert linked is None
    assert scheduler.wakes == 0


class _Upload:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self, _size: int) -> bytes:
        content, self._content = self._content, b""
        return content


async def test_a_staged_upload_is_accepted_under_its_stage_without_a_separate_replay(
    store: _Store, scheduler: _Scheduler, tmp_path: Path
) -> None:
    service = _mutation(store, scheduler, tmp_path)
    (staged,) = await service.stage_uploads(
        workspace="default", run_id=_STAGED_RUN, uploads=[("a.pdf", _Upload(b"pdf"))]
    )

    creation = await service.create_staged_ingest(
        workspace="default",
        run_id=_STAGED_RUN,
        staged=staged,
        submitted_by=_OWNER,
        idempotency_key="key-1",
    )

    assert store.replays == []
    [(envelope, run_id)] = store.accepted
    assert (run_id, envelope.submission_key) == (_STAGED_RUN, "key-1")
    assert (creation.run.run_id, creation.replayed) == (_STAGED_RUN, False)
    assert scheduler.wakes == 1
    assert staged.path.exists()


async def test_a_staged_upload_whose_key_another_run_holds_drops_its_stage(
    store: _Store, scheduler: _Scheduler, tmp_path: Path
) -> None:
    service = _mutation(store, scheduler, tmp_path)
    (staged,) = await service.stage_uploads(
        workspace="default", run_id=_STAGED_RUN, uploads=[("a.pdf", _Upload(b"pdf"))]
    )
    store.accept_replays = _STORED_RUN

    creation = await service.create_staged_ingest(
        workspace="default",
        run_id=_STAGED_RUN,
        staged=staged,
        submitted_by=_OWNER,
        idempotency_key="key-1",
    )

    assert (creation.run.run_id, creation.replayed) == (_STORED_RUN, True)
    assert not staged.path.exists()
