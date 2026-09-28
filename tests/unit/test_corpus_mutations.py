# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Corpus Mutation acceptance and public projection contracts."""

import asyncio
import datetime
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, create_autospec

import pytest

from dlightrag.application.corpus_admin import IngestSpec, UploadTooLargeError
from dlightrag.application.corpus_admin.mutations import (
    CorpusMutationExecutor,
    CorpusMutationService,
    UploadLimits,
    _join_public_operation,
    _result,
    validate_corpus_mutation_prepared_input,
)
from dlightrag.engine.dependencies import TransientDependencyError
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
from dlightrag.engine.runtime.records import (
    Deferred,
    Succeeded,
    WaitingForRepair,
)

_RUN_ID = "0199a0a0-0000-7000-8000-000000000001"
_TRACK_ID = f"dlightrag-corpus-{_RUN_ID}"
_LIMITS = UploadLimits(file_bytes=10, request_bytes=15, request_files=3)


class _Reader:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self, _size: int) -> bytes:
        content, self._content = self._content, b""
        return content


def _service(tmp_path: Path) -> CorpusMutationService:
    return CorpusMutationService(
        input_root=tmp_path,
        store=AsyncMock(),
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=_LIMITS,
    )


async def test_reset_exposes_supersession_as_a_typed_run_envelope_field(
    tmp_path: Path,
) -> None:
    store = AsyncMock()
    store.accept_run.side_effect = RuntimeError("captured envelope")

    @asynccontextmanager
    async def admission():
        yield True

    coordinator = SimpleNamespace(is_started=True, admission=admission, wake=lambda: None)
    service = CorpusMutationService(
        input_root=tmp_path,
        store=store,
        coordinator=cast(Any, coordinator),
        upload_limits=_LIMITS,
    )
    superseded = "0199a0a0-0000-7000-8000-000000000002"

    with pytest.raises(RuntimeError, match="captured envelope"):
        await service.create_reset(
            workspace="default",
            submitted_by="operator",
            supersedes_run_id=superseded,
        )

    envelope = store.accept_run.await_args.kwargs["envelope"]
    assert envelope.supersedes_run_id == superseded


async def test_stage_upload_digest_mismatch_removes_the_run_exclusive_stage(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        await service.stage_upload(
            workspace="default",
            run_id=_RUN_ID,
            filename="report.pdf",
            reader=_Reader(b"actual"),
            max_bytes=1024,
            content_sha256=hashlib.sha256(b"different").hexdigest(),
        )

    assert not (tmp_path / "default" / ".runs" / _RUN_ID).exists()


async def test_discard_staged_run_owns_the_private_stage_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "default" / ".runs" / _RUN_ID
    run_root.mkdir(parents=True)
    (run_root / "source").write_bytes(b"bytes")

    await _service(tmp_path).discard_staged_run(workspace="default", run_id=_RUN_ID)

    assert not run_root.exists()


async def test_stage_uploads_bounds_one_file_by_the_per_file_cap(tmp_path: Path) -> None:
    """A single file may not borrow the larger per-request budget."""
    with pytest.raises(UploadTooLargeError):
        await _service(tmp_path).stage_uploads(
            workspace="default",
            run_id=_RUN_ID,
            uploads=[("report.pdf", _Reader(b"x" * 11))],
        )

    assert not (tmp_path / "default" / ".runs" / _RUN_ID).exists()


async def test_stage_uploads_bounds_each_file_by_the_remaining_request_budget(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    with pytest.raises(UploadTooLargeError):
        await service.stage_uploads(
            workspace="default",
            run_id=_RUN_ID,
            uploads=[("first.pdf", _Reader(b"a" * 8)), ("second.pdf", _Reader(b"b" * 8))],
        )

    # The failed file removes the whole Run-exclusive stage, earlier files included.
    assert not (tmp_path / "default" / ".runs" / _RUN_ID).exists()

    staged = await service.stage_uploads(
        workspace="default",
        run_id=_RUN_ID,
        uploads=[("first.pdf", _Reader(b"a" * 8)), ("second.pdf", _Reader(b"b" * 7))],
    )
    assert [item.size_bytes for item in staged] == [8, 7]


async def test_stage_uploads_refuses_counts_and_digests_it_cannot_honour(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(UploadTooLargeError, match="more than 3 files"):
        await service.stage_uploads(
            workspace="default",
            run_id=_RUN_ID,
            uploads=[(f"{index}.pdf", _Reader(b"x")) for index in range(4)],
        )
    with pytest.raises(ValueError, match="only for a single upload"):
        await service.stage_uploads(
            workspace="default",
            run_id=_RUN_ID,
            uploads=[("a.pdf", _Reader(b"a")), ("b.pdf", _Reader(b"b"))],
            content_sha256=hashlib.sha256(b"a").hexdigest(),
        )
    assert not (tmp_path / "default").exists()


async def test_stage_upload_preserves_a_safe_relative_folder_path(tmp_path) -> None:
    content = b"durable upload"
    staged = await _service(tmp_path).stage_upload(
        workspace="default",
        run_id=_RUN_ID,
        filename="reports/annual.pdf",
        reader=_Reader(content),
        max_bytes=1024,
    )

    assert staged.filename == "reports/annual.pdf"
    assert staged.path == (
        tmp_path / "default" / ".runs" / _RUN_ID / "sources" / "reports" / "annual.pdf"
    )
    assert staged.path.read_bytes() == content
    assert staged.content_sha256 == hashlib.sha256(content).hexdigest()
    assert staged.size_bytes == len(content)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "action": "retry",
                "workspace": "default",
                "track_id": _TRACK_ID,
                "document_ids": "doc-1",
                "selector": None,
            },
            id="retry-document-ids-must-be-a-list",
        ),
        pytest.param(
            {
                "action": "delete",
                "workspace": "default",
                "track_id": _TRACK_ID,
                "file_paths": [],
                "filenames": [],
                "document_ids": ["doc-1"],
                "unexpected": True,
            },
            id="closed-field-set",
        ),
        pytest.param(
            {
                "action": "ingest",
                "workspace": "default",
                "track_id": _TRACK_ID,
                "source": {
                    "source_type": "s3",
                    "bucket": "documents",
                    "prefix": "reports/",
                    "replace": False,
                },
                "staged_sources": [{"path": "/private/source", "content_sha256": "a" * 64}],
            },
            id="closed-staged-source-record",
        ),
        pytest.param(
            {
                "action": "delete_workspace",
                "workspace": "research",
                "track_id": _TRACK_ID,
                "supersedes_run_id": None,
            },
            id="workspace-delete-carries-no-selector",
        ),
    ],
)
def test_prepared_input_validator_rejects_noncanonical_shapes(payload) -> None:
    with pytest.raises(ValueError):
        validate_corpus_mutation_prepared_input(payload)


def test_prepared_input_validator_accepts_exact_retry_selector() -> None:
    action, workspace = validate_corpus_mutation_prepared_input(
        {
            "action": "retry",
            "workspace": "default",
            "track_id": _TRACK_ID,
            "document_ids": [],
            "selector": "all_retryable",
        }
    )

    assert action == "retry"
    assert workspace == "default"


def test_prepared_input_validator_accepts_workspace_delete() -> None:
    action, workspace = validate_corpus_mutation_prepared_input(
        {"action": "delete_workspace", "workspace": "research", "track_id": _TRACK_ID}
    )

    assert action == "delete_workspace"
    assert workspace == "research"


def test_public_result_is_bounded_and_drops_paths_and_diagnostics() -> None:
    documents = [
        {
            "doc_id": f"doc-{index}",
            "status": "ready",
            "file_path": f"/private/{index}.pdf",
            "error": "secret diagnostic",
        }
        for index in range(101)
    ]

    result = _result("ingest", documents, {"track_id": _TRACK_ID})

    assert result["document_count"] == 100
    assert result["details_truncated"] is True
    assert len(result["documents"]) == 100
    assert all("file_path" not in item and "error" not in item for item in result["documents"])


class _Session:
    def __init__(
        self,
        prepared_input: dict[str, Any],
        *,
        handoff_started: bool = False,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.prepared_input = prepared_input
        self.owner_id = "default"
        self.run_id = _RUN_ID
        self.handoff_started = handoff_started
        self.checkpoint = checkpoint
        self.checkpoints: list[tuple[dict[str, Any], str | None]] = []
        self.phases: list[str] = []

    async def checkpoint_state(self, checkpoint, *, phase=None) -> None:
        value = dict(checkpoint)
        self.checkpoint = value
        self.checkpoints.append((value, phase))

    async def begin_handoff(self, checkpoint) -> None:
        self.handoff_started = True
        await self.checkpoint_state(checkpoint, phase="handoff_started")

    async def enter_phase(self, phase: str) -> None:
        self.phases.append(phase)


class _Maintenance:
    def __init__(self) -> None:
        self.unregistered: list[str] = []

    @asynccontextmanager
    async def workspace_write_gate(self, _workspace: str):
        yield

    async def unregister_workspace(self, workspace: str) -> bool:
        self.unregistered.append(workspace)
        return True


def _payload(action: str, **fields: Any) -> dict[str, Any]:
    return {
        "action": action,
        "workspace": "default",
        "track_id": _TRACK_ID,
        **fields,
    }


def _executor(
    runtime: Any,
    *,
    now=None,
    acquire_error: Exception | None = None,
    maintenance: _Maintenance | None = None,
    store: Any = None,
):
    pool = SimpleNamespace(
        acquire=AsyncMock(side_effect=acquire_error, return_value=runtime),
        evict=AsyncMock(),
    )
    store = store or SimpleNamespace(record_corpus_window=AsyncMock(return_value=True))
    executor = CorpusMutationExecutor(
        pool=cast(Any, pool),
        maintenance=cast(Any, maintenance or _Maintenance()),
        store=cast(Any, store),
        now=now,
    )
    return executor, pool, store


def _runtime(*, tracked: dict[str, Any] | None = None) -> SimpleNamespace:
    lightrag = SimpleNamespace(
        aget_docs_by_track_id=AsyncMock(return_value=tracked or {}),
        apipeline_process_enqueue_documents=AsyncMock(),
    )
    # Autospecced so a call the real WorkspaceRag signature rejects fails here too.
    real = create_autospec(WorkspaceRag, instance=True)
    real.aingest.return_value = {
        "processed": 1,
        "errors": [],
        "results": [{"doc_id": "doc-1", "chunks": ["chunk-1"]}],
    }
    real.adelete_files.return_value = [{"identifier": "doc-1", "status": "deleted"}]
    real.aretryable_document_ids.return_value = ("doc-1",)
    real.aretry_failed_docs.return_value = {
        "retried": 1,
        "succeeded": 1,
        "failed": 0,
        "succeeded_docs": [{"doc_id": "doc-1"}],
        "failed_docs": [],
    }
    real.areset.return_value = {"documents_deleted": 1, "errors": []}
    return SimpleNamespace(
        lightrag=lightrag,
        aingest=real.aingest,
        adelete_files=real.adelete_files,
        aretryable_document_ids=real.aretryable_document_ids,
        aretry_failed_docs=real.aretry_failed_docs,
        areset=real.areset,
    )


async def test_recovered_delete_handoff_waits_for_repair_without_repeating_mutation() -> None:
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)
    session = _Session(
        _payload(
            "delete",
            file_paths=[],
            filenames=[],
            document_ids=["doc-1"],
        ),
        handoff_started=True,
        checkpoint={"resolved_documents": [{"document_ids": ["doc-1"]}]},
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, WaitingForRepair)
    assert outcome.checkpoint["repair_reason"]
    assert outcome.checkpoint["repair_remedy"]
    runtime.adelete_files.assert_not_awaited()


async def test_explicit_repair_resume_is_consumed_before_delete_reexecution() -> None:
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)
    session = _Session(
        _payload(
            "delete",
            file_paths=[],
            filenames=[],
            document_ids=["doc-1"],
        ),
        handoff_started=True,
        checkpoint={
            "resolved_documents": [{"document_ids": ["doc-1"]}],
            "repair_resume_confirmed": True,
        },
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    assert any(
        checkpoint.get("repair_resume_confirmed") is False and phase == "repair_attempt_started"
        for checkpoint, phase in session.checkpoints
    )
    runtime.adelete_files.assert_awaited_once_with(
        file_paths=["doc-1"], filenames=[], dry_run=False
    )


@pytest.mark.parametrize(
    ("action", "fields", "forbidden_method"),
    [
        ("retry", {"document_ids": ["doc-1"], "selector": None}, "aretry_failed_docs"),
        ("reset", {"supersedes_run_id": None}, "areset"),
        ("delete_workspace", {}, "areset"),
    ],
)
async def test_recovered_destructive_handoff_does_not_blindly_repeat(
    action: str,
    fields: dict[str, Any],
    forbidden_method: str,
) -> None:
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)
    session = _Session(_payload(action, **fields), handoff_started=True)

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, WaitingForRepair)
    getattr(runtime, forbidden_method).assert_not_awaited()


async def test_fresh_ingest_handoffs_once_and_records_its_settled_window() -> None:
    runtime = _runtime()
    executor, _pool, store = _executor(runtime)
    session = _Session(
        _payload(
            "ingest",
            source={"source_type": "s3", "bucket": "documents", "replace": False},
            staged_sources=[],
        )
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    assert session.handoff_started is True
    runtime.aingest.assert_awaited_once()
    store.record_corpus_window.assert_awaited_once_with(
        run_id=_RUN_ID,
        workspace="default",
        window_number=1,
        docs=1,
        chunks=1,
    )


@pytest.mark.parametrize("active_status", ["parsing", "analyzing", "processing", "preprocessed"])
async def test_recovered_ingest_does_not_redrive_an_active_tracked_pipeline(
    active_status: str,
) -> None:
    runtime = _runtime(tracked={"doc-1": {"status": active_status}})
    executor, _pool, _store = _executor(runtime)
    session = _Session(
        _payload(
            "ingest",
            source={"source_type": "s3", "bucket": "documents", "replace": False},
            staged_sources=[],
        ),
        handoff_started=True,
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Deferred)
    runtime.lightrag.apipeline_process_enqueue_documents.assert_not_awaited()
    runtime.aretry_failed_docs.assert_not_awaited()


async def test_recovered_replace_reconciles_tracked_state_without_repeating_replace() -> None:
    runtime = _runtime(tracked={"doc-1": {"status": "processed"}})
    executor, _pool, _store = _executor(runtime)
    session = _Session(
        _payload(
            "replace",
            source={"source_type": "s3", "bucket": "documents", "replace": True},
            staged_sources=[],
        ),
        handoff_started=True,
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    runtime.aingest.assert_not_awaited()
    runtime.aretry_failed_docs.assert_awaited_once_with(
        cohort_doc_ids=("doc-1",), track_id=_TRACK_ID
    )


async def test_fresh_retry_seals_the_exact_cohort_before_handoff() -> None:
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)
    session = _Session(_payload("retry", document_ids=[], selector="all_retryable"))

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    sealed = [checkpoint for checkpoint, phase in session.checkpoints if phase == "cohort_sealed"]
    assert sealed and sealed[-1]["cohort_doc_ids"] == ["doc-1"]
    assert sealed[-1]["cohort_sealed"] is True
    runtime.aretry_failed_docs.assert_awaited_once_with(
        cohort_doc_ids=["doc-1"], track_id=_TRACK_ID
    )


async def test_fresh_reset_preserves_later_run_sources_and_evicts_only_runtime() -> None:
    runtime = _runtime()
    executor, pool, _store = _executor(runtime)
    session = _Session(_payload("reset", supersedes_run_id=None))

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    runtime.areset.assert_awaited_once_with(preserve_run_sources_after=_RUN_ID)
    pool.evict.assert_awaited_once_with("default")


async def test_settled_delete_checkpoint_finishes_without_repeating_upstream() -> None:
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)
    session = _Session(
        _payload(
            "delete",
            file_paths=[],
            filenames=[],
            document_ids=["doc-1"],
        ),
        handoff_started=True,
        checkpoint={
            "operation_settled": True,
            "document_outcomes": [{"identifier": "doc-1", "status": "deleted"}],
        },
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    runtime.adelete_files.assert_not_awaited()


async def test_transient_dependency_deferral_uses_bounded_exponential_backoff() -> None:
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    payload = _payload(
        "ingest",
        source={"source_type": "s3", "bucket": "documents", "replace": False},
        staged_sources=[],
    )
    error = TransientDependencyError("corpus_storage", "temporarily unavailable")
    first_executor, _pool, _store = _executor(_runtime(), now=lambda: now, acquire_error=error)
    first = await first_executor.execute(cast(Any, _Session(payload)))
    assert isinstance(first, Deferred)
    assert (first.next_attempt_at - now).total_seconds() == 2

    second_executor, _pool, _store = _executor(_runtime(), now=lambda: now, acquire_error=error)
    second = await second_executor.execute(
        cast(Any, _Session(payload, checkpoint=dict(first.checkpoint)))
    )
    assert isinstance(second, Deferred)
    assert (second.next_attempt_at - now).total_seconds() == 4
    assert second.checkpoint["corpus_unavailable_attempt"] == 2


async def test_uncertain_destructive_failure_after_handoff_requires_repair() -> None:
    runtime = _runtime()
    runtime.adelete_files.side_effect = [
        [{"identifier": "doc-1", "matched_doc_ids": ["doc-1"]}],
        ConnectionError("connection dropped"),
    ]
    executor, _pool, _store = _executor(runtime)
    session = _Session(
        _payload(
            "delete",
            file_paths=[],
            filenames=[],
            document_ids=["doc-1"],
        )
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, WaitingForRepair)
    assert session.handoff_started is True


async def test_public_operation_finishes_after_outer_task_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation() -> str:
        started.set()
        await release.wait()
        return "settled"

    joined = asyncio.create_task(_join_public_operation(operation()))
    await started.wait()
    joined.cancel()
    await asyncio.sleep(0)
    assert not joined.done()

    release.set()
    assert await joined == "settled"


@pytest.mark.asyncio
async def test_a_reader_refuses_every_corpus_write_before_it_happens(tmp_path: Path) -> None:
    """A read-only replica declines the write instead of accepting a Run it cannot run.

    A `reader` process registers no corpus-mutation executor, so accepting a write staged
    bytes for a Run that could never execute and failed it later. Every entry point now
    refuses first, with the remedy, and the workspace stays empty.
    """
    from dlightrag.application.corpus_admin import CorpusMutationUnavailableError

    service = CorpusMutationService(
        input_root=tmp_path,
        store=AsyncMock(),
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=_LIMITS,
        writable=False,
    )

    from dlightrag.application.corpus_admin import IngestSpec

    spec = IngestSpec(source_type="url", url="https://example.com/report.pdf")
    calls = (
        ("the ingest", service.create_ingest(workspace="default", spec=spec, submitted_by="o")),
        (
            "the delete",
            service.create_delete(workspace="default", document_ids=["doc-1"], submitted_by="o"),
        ),
        ("the retry", service.create_retry(workspace="default", submitted_by="o")),
        ("the Corpus Reset", service.create_reset(workspace="default", submitted_by="o")),
        (
            "the Workspace Delete",
            service.create_workspace_delete(workspace="research", submitted_by="o"),
        ),
        (
            "the upload",
            service.stage_upload(
                workspace="default",
                run_id="run-1",
                filename="report.pdf",
                reader=AsyncMock(),
                max_bytes=1024,
            ),
        ),
    )
    for request, call in calls:
        with pytest.raises(CorpusMutationUnavailableError) as refused:
            await call
        assert str(refused.value) == (
            "This deployment is a read-only replica of the knowledge base: it accepts no "
            f"corpus writes. Send {request} to a writer."
        )
    assert list(tmp_path.rglob("*")) == []


def _run_record(run_id: str, *, status: str = "queued", run_kind: str = "corpus_mutation"):
    return SimpleNamespace(
        run_id=run_id,
        run_kind=run_kind,
        terminal=status in {"succeeded", "failed", "cancelled"},
    )


def _successor_store(*pages: tuple[Any, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        list_runs=AsyncMock(side_effect=list(pages)),
        request_cancellation=AsyncMock(),
    )


def _workspace_delete_session(**kwargs: Any) -> _Session:
    session = _Session(
        {"action": "delete_workspace", "workspace": "research", "track_id": _TRACK_ID},
        **kwargs,
    )
    session.owner_id = "research"
    return session


async def test_workspace_delete_resets_everything_then_retires_identity_and_successors() -> None:
    runtime = _runtime()
    maintenance = _Maintenance()
    store = _successor_store(
        (
            _run_record("run-queued"),
            _run_record("run-done", status="succeeded"),
            _run_record("run-answer", run_kind="answer"),
        ),
    )
    executor, pool, _store = _executor(runtime, maintenance=maintenance, store=store)
    session = _workspace_delete_session()

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    assert outcome.result["action"] == "delete_workspace"
    # No source is preserved: every queued successor is cancelled, never replayed.
    runtime.areset.assert_awaited_once_with()
    assert session.handoff_started is True
    assert session.phases == ["resetting_corpus", "removing_workspace"]
    assert maintenance.unregistered == ["research"]
    store.list_runs.assert_awaited_once_with(owner_id="research", after_run_id=_RUN_ID, limit=100)
    store.request_cancellation.assert_awaited_once_with(owner_id="research", run_id="run-queued")
    pool.evict.assert_awaited_once_with("research")


async def test_workspace_delete_pages_through_every_queued_successor() -> None:
    first = tuple(_run_record(f"run-{index:03d}") for index in range(100))
    store = _successor_store(first, (_run_record("run-last"),))
    executor, _pool, _store = _executor(_runtime(), store=store)

    outcome = await executor.execute(cast(Any, _workspace_delete_session()))

    assert isinstance(outcome, Succeeded)
    assert store.list_runs.await_args_list[1].kwargs["after_run_id"] == "run-099"
    assert store.request_cancellation.await_count == 101


async def test_settled_workspace_delete_repeats_only_idempotent_retirement() -> None:
    runtime = _runtime()
    maintenance = _Maintenance()
    store = _successor_store(())
    executor, pool, _store = _executor(runtime, maintenance=maintenance, store=store)
    session = _workspace_delete_session(
        handoff_started=True,
        checkpoint={"operation_settled": True, "document_outcomes": [{"documents_deleted": 1}]},
    )

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    runtime.areset.assert_not_awaited()
    assert maintenance.unregistered == ["research"]
    pool.evict.assert_awaited_once_with("research")


async def test_workspace_delete_with_reset_errors_waits_for_repair_and_keeps_identity() -> None:
    runtime = _runtime()
    runtime.areset.return_value = {"errors": ["Phase 1 (chunks): dropped connection"]}
    maintenance = _Maintenance()
    store = _successor_store(())
    executor, pool, _store = _executor(runtime, maintenance=maintenance, store=store)

    outcome = await executor.execute(cast(Any, _workspace_delete_session()))

    assert isinstance(outcome, WaitingForRepair)
    assert maintenance.unregistered == []
    store.list_runs.assert_not_awaited()
    pool.evict.assert_not_awaited()


async def test_workspace_delete_refuses_the_deployment_default(tmp_path: Path) -> None:
    store = AsyncMock()
    service = CorpusMutationService(
        input_root=tmp_path,
        store=store,
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=_LIMITS,
        default_workspace="research",
    )

    with pytest.raises(ValueError, match="default workspace cannot be deleted"):
        await service.create_workspace_delete(workspace="research", submitted_by="owner")
    store.accept_run.assert_not_awaited()


async def test_workspace_delete_is_accepted_as_a_workspace_scoped_mutation(
    tmp_path: Path,
) -> None:
    store = AsyncMock()
    store.accept_run.side_effect = RuntimeError("captured envelope")

    @asynccontextmanager
    async def admission():
        yield True

    coordinator = SimpleNamespace(is_started=True, admission=admission, wake=lambda: None)
    service = CorpusMutationService(
        input_root=tmp_path,
        store=store,
        coordinator=cast(Any, coordinator),
        upload_limits=_LIMITS,
    )

    with pytest.raises(RuntimeError, match="captured envelope"):
        await service.create_workspace_delete(workspace="research", submitted_by="operator")

    envelope = store.accept_run.await_args.kwargs["envelope"]
    assert envelope.access_scope.scope_id == "research"
    assert envelope.payload["action"] == "delete_workspace"
    assert envelope.accepted_input == {"action": "delete_workspace", "workspace": "research"}


def _local_spec(path: Path) -> IngestSpec:
    return IngestSpec(source_type="local", path=str(path))


async def test_blank_selectors_are_the_callers_to_fix(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    with pytest.raises(CorpusMutationInputError, match="at least one exact document identifier"):
        await _service(tmp_path).create_delete(
            workspace="default", submitted_by="o", filenames=["  "]
        )


async def test_selector_bounds_and_blank_retries_are_the_callers_to_fix(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    service = _service(tmp_path)
    with pytest.raises(CorpusMutationInputError, match="at most 100 document identifiers"):
        await service.create_delete(
            workspace="default",
            submitted_by="o",
            document_ids=[f"doc-{index}" for index in range(101)],
        )
    with pytest.raises(CorpusMutationInputError, match="provide document_ids"):
        await service.create_retry(workspace="default", submitted_by="o", document_ids=["  "])


async def test_an_oversized_corpus_request_is_the_callers_to_fix(
    tmp_path: Path, monkeypatch
) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    monkeypatch.setattr("dlightrag.engine.runtime.records.MAX_PREPARED_INPUT_BYTES", 64)
    store = AsyncMock()
    service = CorpusMutationService(
        input_root=tmp_path,
        store=store,
        coordinator=cast(Any, SimpleNamespace(is_started=True)),
        upload_limits=_LIMITS,
    )

    with pytest.raises(CorpusMutationInputError, match="prepared_input_too_large"):
        await service.create_delete(
            workspace="default", submitted_by="o", document_ids=["doc-" + "x" * 80]
        )
    store.accept_run.assert_not_awaited()


def test_a_local_folder_cannot_link_outside_its_workspace(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    folder = tmp_path / "default" / "docs"
    folder.mkdir(parents=True)
    (folder / "ok.txt").write_text("ok", encoding="utf-8")
    (folder / "leak.txt").symlink_to(outside / "secret.txt")

    with pytest.raises(CorpusMutationInputError, match="cannot contain symlinks"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))

    # Nothing was staged, least of all the linked file's bytes.
    assert not (tmp_path / "default" / ".runs" / _RUN_ID).exists()


def test_a_local_folder_may_hold_only_files_and_folders(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    folder = tmp_path / "default" / "docs"
    folder.mkdir(parents=True)
    os.mkfifo(folder / "pipe")

    with pytest.raises(CorpusMutationInputError, match="only regular files and folders"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))


def test_a_missing_or_oversized_local_source_is_the_callers_to_fix(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    service = _service(tmp_path)
    with pytest.raises(CorpusMutationInputError, match="does not exist"):
        service._snapshot_local_spec(
            _RUN_ID, "default", _local_spec(tmp_path / "default" / "missing.pdf")
        )

    folder = tmp_path / "default" / "many"
    folder.mkdir(parents=True)
    for index in range(101):
        (folder / f"{index}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(CorpusMutationInputError, match="more than 100 files"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))


def test_an_oversized_local_folder_refuses_before_copying(tmp_path: Path, monkeypatch) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError, mutations

    copied: list[Path] = []
    real_copy = mutations._copy_regular_file

    def counting_copy(fd: int, target: Path) -> dict:
        copied.append(target)
        return real_copy(fd, target)

    monkeypatch.setattr(mutations, "_copy_regular_file", counting_copy)
    folder = tmp_path / "default" / "many"
    folder.mkdir(parents=True)
    for index in range(101):
        (folder / f"{index}.txt").write_text("x", encoding="utf-8")

    with pytest.raises(CorpusMutationInputError, match="more than 100 files"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))
    assert copied == []


def test_a_folder_swapped_for_a_link_while_copying_refuses(tmp_path: Path, monkeypatch) -> None:
    """The listing is not trusted: every file is reopened without following a link."""
    from dlightrag.application.corpus_admin import CorpusMutationInputError, mutations

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    folder = tmp_path / "default" / "docs"
    (folder / "zsub").mkdir(parents=True)
    (folder / "a.txt").write_text("a", encoding="utf-8")
    (folder / "zsub" / "secret.txt").write_text("inside", encoding="utf-8")
    real_list = mutations._list_local_tree

    def list_then_swap(source: int, *, max_files: int):
        listed = real_list(source, max_files=max_files)
        (folder / "zsub" / "secret.txt").unlink()
        (folder / "zsub").rmdir()
        (folder / "zsub").symlink_to(outside, target_is_directory=True)
        return listed

    monkeypatch.setattr(mutations, "_list_local_tree", list_then_swap)

    with pytest.raises(CorpusMutationInputError):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))
    staged = list((tmp_path / "default").rglob("secret.txt"))
    assert staged == [], "no byte behind the link was staged"


def test_a_file_that_became_a_link_refuses(tmp_path: Path) -> None:
    """A source resolved before the swap is still opened without following a link."""
    from dlightrag.application.corpus_admin import CorpusMutationInputError
    from dlightrag.application.corpus_admin.mutations import _snapshot_local_source

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    workspace = tmp_path / "default"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "report.txt").symlink_to(outside)

    with pytest.raises(CorpusMutationInputError, match="cannot contain symlinks"):
        _snapshot_local_source(
            workspace, ("docs", "report.txt"), tmp_path / "copy.txt", max_files=100
        )
    assert not (tmp_path / "copy.txt").exists()


def test_a_linked_folder_refuses(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    folder = tmp_path / "default" / "docs"
    folder.mkdir(parents=True)
    (folder / "ok.txt").write_text("ok", encoding="utf-8")
    (folder / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CorpusMutationInputError, match="cannot contain symlinks"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))


def test_a_workspace_root_source_copies_only_what_a_scan_ingests(tmp_path: Path) -> None:
    """Run stages, dot entries and parser sidecars stay out of the copy, and its count."""
    workspace = tmp_path / "default"
    (workspace / "reports").mkdir(parents=True)
    (workspace / "reports" / "q3.txt").write_text("quarter", encoding="utf-8")
    (workspace / "top.txt").write_text("top", encoding="utf-8")
    (workspace / ".staging").mkdir()
    (workspace / ".staging" / "partial.txt").write_text("x", encoding="utf-8")
    (workspace / ".hidden.txt").write_text("x", encoding="utf-8")
    (workspace / "__parsed__").mkdir()
    for index in range(150):
        (workspace / "__parsed__" / f"{index}.md").write_text("x", encoding="utf-8")

    spec, run_root, manifest = _service(tmp_path)._snapshot_local_spec(
        _RUN_ID, "default", _local_spec(workspace)
    )

    copied = Path(cast(str, spec.path))
    assert sorted(p.relative_to(copied).as_posix() for p in copied.rglob("*") if p.is_file()) == [
        "reports/q3.txt",
        "top.txt",
    ]
    assert run_root == workspace.resolve() / ".runs" / _RUN_ID
    assert {Path(item["path"]).name for item in manifest} == {"q3.txt", "top.txt"}
    top = next(item for item in manifest if item["path"].endswith("top.txt"))
    assert top["size_bytes"] == 3
    assert top["content_sha256"] == hashlib.sha256(b"top").hexdigest()


def test_a_local_source_with_nothing_to_ingest_or_no_path_is_the_callers_to_fix(
    tmp_path: Path,
) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    workspace = tmp_path / "default"
    (workspace / "empty" / ".git").mkdir(parents=True)
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    service = _service(tmp_path)

    with pytest.raises(CorpusMutationInputError, match="no files to ingest"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(workspace / "empty"))
    with pytest.raises(CorpusMutationInputError, match="does not exist"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(workspace / "a.txt" / "x"))
