# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Corpus Mutation acceptance and public projection contracts."""

import asyncio
import datetime
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from unittest.mock import AsyncMock, create_autospec

import pytest

from dlightrag.application.corpus_admin import (
    IngestSpec,
    UploadTooLargeError,
    WorkspaceNotFoundError,
)
from dlightrag.application.corpus_admin.mutations import (
    CorpusMutationExecutor,
    CorpusMutationService,
    UploadLimits,
    _join_public_operation,
    _result,
    validate_corpus_mutation_prepared_input,
)
from dlightrag.application.errors import ApplicationUnavailableError
from dlightrag.engine.dependencies import (
    MAX_DEPENDENCY_DEFERRALS,
    DependencyRetriesExhausted,
    TransientDependencyError,
)
from dlightrag.engine.rag.workspace.ports import WorkspaceWriteFencedError
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
from dlightrag.engine.runtime.records import (
    Deferred,
    Failed,
    Succeeded,
    WaitingForRepair,
    run_request_fingerprint,
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


async def _registered(_workspace: str) -> bool:
    return True


class _Roots(TypedDict):
    source_root: Path
    corpus_root: Path


def _roots(tmp_path: Path) -> _Roots:
    """Operators' source folder and this service's own corpus directory."""
    return {"source_root": tmp_path / "inputs", "corpus_root": tmp_path / "corpus"}


def _service(tmp_path: Path, *, upload_limits: UploadLimits = _LIMITS) -> CorpusMutationService:
    return CorpusMutationService(
        **_roots(tmp_path),
        store=AsyncMock(),
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=upload_limits,
        workspace_exists=_registered,
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
        **_roots(tmp_path),
        store=store,
        coordinator=cast(Any, coordinator),
        upload_limits=_LIMITS,
        workspace_exists=_registered,
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


def _accepting_service(
    tmp_path: Path, store: AsyncMock, *, workspace_exists: Any = _registered
) -> CorpusMutationService:
    @asynccontextmanager
    async def admission():
        yield True

    coordinator = SimpleNamespace(is_started=True, admission=admission, wake=lambda: None)
    return CorpusMutationService(
        **_roots(tmp_path),
        store=store,
        coordinator=cast(Any, coordinator),
        upload_limits=_LIMITS,
        workspace_exists=workspace_exists,
    )


async def test_workspace_delete_may_supersede_the_waiting_mutation(tmp_path: Path) -> None:
    """A mutation waiting for repair would otherwise hold the FIFO lane ahead of it."""
    store = AsyncMock()
    store.accept_run.side_effect = RuntimeError("captured envelope")
    service = _accepting_service(tmp_path, store)
    superseded = "0199a0a0-0000-7000-8000-000000000002"

    with pytest.raises(RuntimeError, match="captured envelope"):
        await service.create_workspace_delete(
            workspace="research", submitted_by="operator", supersedes_run_id=superseded
        )

    envelope = store.accept_run.await_args.kwargs["envelope"]
    assert envelope.supersedes_run_id == superseded
    assert envelope.payload["supersedes_run_id"] == superseded
    action, workspace = validate_corpus_mutation_prepared_input(envelope.payload)
    assert (action, workspace) == ("delete_workspace", "research")


async def test_plain_workspace_delete_keeps_its_prepared_input_and_fingerprint(
    tmp_path: Path,
) -> None:
    store = AsyncMock()
    store.accept_run.side_effect = RuntimeError("captured envelope")
    service = _accepting_service(tmp_path, store)

    with pytest.raises(RuntimeError, match="captured envelope"):
        await service.create_workspace_delete(workspace="research", submitted_by="operator")

    envelope = store.accept_run.await_args.kwargs["envelope"]
    assert envelope.supersedes_run_id is None
    assert "supersedes_run_id" not in envelope.payload
    assert envelope.request_fingerprint == run_request_fingerprint(
        {"action": "delete_workspace", "workspace": "research"}
    )


async def test_workspace_delete_refuses_a_workspace_that_is_gone(tmp_path: Path) -> None:
    store = AsyncMock()

    async def gone(_workspace: str) -> bool:
        return False

    service = _accepting_service(tmp_path, store, workspace_exists=gone)

    with pytest.raises(WorkspaceNotFoundError, match="no longer exists"):
        await service.create_workspace_delete(workspace="research", submitted_by="operator")
    store.accept_run.assert_not_awaited()


async def _gone(_workspace: str) -> bool:
    return False


@pytest.mark.parametrize(
    "write_to",
    [
        lambda service: service.create_ingest(
            workspace="reserch",
            spec=IngestSpec(source_type="url", url="https://example.com/a.pdf"),
            submitted_by="o",
        ),
        lambda service: service.create_delete(
            workspace="reserch", submitted_by="o", document_ids=["doc-a"]
        ),
        lambda service: service.create_retry(
            workspace="reserch", submitted_by="o", document_ids=["doc-a"]
        ),
        lambda service: service.create_reset(workspace="reserch", submitted_by="o"),
    ],
    ids=["ingest", "delete", "retry", "reset"],
)
async def test_every_corpus_write_needs_a_created_workspace(tmp_path: Path, write_to) -> None:
    """A misspelled Workspace is refused, never a write to an uncatalogued corpus."""
    store = AsyncMock()
    service = _accepting_service(tmp_path, store, workspace_exists=_gone)

    with pytest.raises(WorkspaceNotFoundError, match="does not exist; create it first"):
        await write_to(service)
    store.accept_run.assert_not_awaited()


async def test_a_local_source_or_upload_for_a_missing_workspace_stages_nothing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "inputs" / "reserch"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "a.txt").write_text("a", encoding="utf-8")
    service = _accepting_service(tmp_path, AsyncMock(), workspace_exists=_gone)

    with pytest.raises(WorkspaceNotFoundError):
        await service.create_ingest(
            workspace="reserch", spec=_local_spec(workspace / "docs"), submitted_by="o"
        )
    with pytest.raises(WorkspaceNotFoundError):
        await service.stage_uploads(
            workspace="reserch", run_id=_RUN_ID, uploads=[("a.pdf", _Reader(b"a"))]
        )
    assert not (tmp_path / "corpus").exists()


async def test_a_retried_ingest_key_replays_before_the_workspace_check(tmp_path: Path) -> None:
    lookups: list[str] = []

    async def gone(workspace: str) -> bool:
        lookups.append(workspace)
        return False

    service = _accepting_service(tmp_path, AsyncMock(), workspace_exists=gone)
    service.replay = AsyncMock(return_value="receipt")  # type: ignore[method-assign]

    receipt = await service.create_ingest(
        workspace="reserch",
        spec=IngestSpec(source_type="url", url="https://example.com/a.pdf"),
        submitted_by="o",
        idempotency_key="key",
    )

    assert receipt == "receipt"
    assert lookups == []


async def test_workspace_delete_replays_its_receipt_after_the_workspace_is_gone(
    tmp_path: Path,
) -> None:
    """The existence check follows replay, so a retried key still gets its Run."""
    store = AsyncMock()
    lookups: list[str] = []

    async def gone(workspace: str) -> bool:
        lookups.append(workspace)
        return False

    service = _accepting_service(tmp_path, store, workspace_exists=gone)
    service.replay = AsyncMock(return_value="receipt")  # type: ignore[method-assign]

    assert (
        await service.create_workspace_delete(
            workspace="research", submitted_by="operator", idempotency_key="key"
        )
        == "receipt"
    )
    assert lookups == []


async def test_workspace_delete_fails_closed_when_the_catalog_is_unreadable(
    tmp_path: Path,
) -> None:
    store = AsyncMock()

    async def unreadable(_workspace: str) -> bool:
        raise RuntimeError("registry down")

    service = _accepting_service(tmp_path, store, workspace_exists=unreadable)

    with pytest.raises(ApplicationUnavailableError, match="temporarily unavailable"):
        await service.create_workspace_delete(workspace="research", submitted_by="operator")
    store.accept_run.assert_not_awaited()


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

    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()


async def test_discard_staged_run_owns_the_private_stage_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "corpus" / "default" / ".runs" / _RUN_ID
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

    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()


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
    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()

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
    assert not (tmp_path / "corpus" / "default").exists()


async def test_stage_upload_keeps_the_relative_path_as_identity_and_the_name_on_disk(
    tmp_path: Path,
) -> None:
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
        tmp_path / "corpus" / "default" / ".runs" / _RUN_ID / "sources" / "0" / "annual.pdf"
    )
    assert staged.path.read_bytes() == content
    assert staged.content_sha256 == hashlib.sha256(content).hexdigest()
    assert staged.size_bytes == len(content)


async def test_staging_never_writes_the_operators_source_folder(tmp_path: Path) -> None:
    """Local snapshots and uploads stage only in the corpus directory, whatever their names."""
    inputs = tmp_path / "inputs" / "default"
    (inputs / "docs").mkdir(parents=True)
    (inputs / "docs" / "report.pdf").write_bytes(b"operator copy")
    (inputs / "report.pdf").write_bytes(b"operator original")
    before = {path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()}
    service = _service(tmp_path)

    _spec, manifest = service._snapshot_local_spec(_RUN_ID, "default", _local_spec(inputs / "docs"))
    upload = await service.stage_upload(
        workspace="default",
        run_id="0199a0a0-0000-7000-8000-000000000002",
        filename="report.pdf",
        reader=_Reader(b"uploaded"),
        max_bytes=1024,
    )

    assert {path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()} == before
    corpus = (tmp_path / "corpus").resolve()
    assert manifest and all(Path(item["path"]).is_relative_to(corpus) for item in manifest)
    assert upload.path.is_relative_to(corpus)


async def test_uploads_that_would_become_one_document_are_refused_before_staging(
    tmp_path: Path,
) -> None:
    """LightRAG names a document by its basename, whatever folder it was uploaded from."""
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    with pytest.raises(CorpusMutationInputError, match="same document 'report.pdf'"):
        await _service(tmp_path).stage_uploads(
            workspace="default",
            run_id=_RUN_ID,
            uploads=[("q1/report.pdf", _Reader(b"a")), ("q2/report.pdf", _Reader(b"b"))],
        )
    assert not (tmp_path / "corpus").exists()


async def test_a_folder_upload_leaves_out_what_a_folder_listing_skips(tmp_path: Path) -> None:
    """A dot file such as .DS_Store, and anything in a dot or corpus folder, is no document."""
    limits = UploadLimits(file_bytes=10, request_bytes=100, request_files=10)
    staged = await _service(tmp_path, upload_limits=limits).stage_uploads(
        workspace="default",
        run_id=_RUN_ID,
        uploads=[
            ("docs/a/report.pdf", _Reader(b"a")),
            ("docs/a/.DS_Store", _Reader(b"x")),
            ("docs/b/notes.md", _Reader(b"b")),
            ("docs/b/.DS_Store", _Reader(b"x")),
            ("docs/.git/HEAD", _Reader(b"x")),
            ("docs/__parsed__/old.pdf", _Reader(b"x")),
        ],
    )

    assert [item.filename for item in staged] == ["docs/a/report.pdf", "docs/b/notes.md"]
    assert [item.path.name for item in staged] == ["report.pdf", "notes.md"]


@pytest.mark.parametrize("filename", [".DS_Store", ".runs", "__parsed__", "docs/.git/config"])
async def test_an_upload_with_nothing_left_to_ingest_is_refused(
    tmp_path: Path, filename: str
) -> None:
    """A document's parser input takes its name in the corpus directory, beside its stages."""
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    with pytest.raises(CorpusMutationInputError, match="no files to ingest"):
        await _service(tmp_path).stage_uploads(
            workspace="default", run_id=_RUN_ID, uploads=[(filename, _Reader(b"x"))]
        )
    assert not (tmp_path / "corpus").exists()


async def test_a_nested_upload_batch_lists_every_file_under_the_callers_run(
    tmp_path: Path,
) -> None:
    """The Run ingests exactly what was staged, from any folder, under the run id given."""
    store = AsyncMock()
    store.accept_run.side_effect = RuntimeError("captured envelope")
    service = _accepting_service(tmp_path, store)
    staged = await service.stage_uploads(
        workspace="default",
        run_id=_RUN_ID,
        uploads=[
            ("reports/q1/a.pdf", _Reader(b"a")),
            ("reports/q2/b.pdf", _Reader(b"b")),
            ("c.pdf", _Reader(b"c")),
        ],
    )

    with pytest.raises(RuntimeError, match="captured envelope"):
        await service.create_staged_batch(
            workspace="default", run_id=_RUN_ID, staged=staged, submitted_by="o"
        )

    kwargs = store.accept_run.await_args.kwargs
    payload = kwargs["envelope"].payload
    assert kwargs["run_id"] == _RUN_ID
    assert payload["track_id"] == _TRACK_ID
    assert payload["source"]["documents"] == [{"path": str(item.path)} for item in staged]
    assert [item["path"] for item in payload["staged_sources"]] == [
        str(item.path) for item in staged
    ]
    assert validate_corpus_mutation_prepared_input(payload) == ("ingest", "default")
    # An unaccepted Run's stage goes with its refusal.
    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()


async def test_a_single_upload_is_one_listed_document_with_its_defaults(tmp_path: Path) -> None:
    store = AsyncMock()
    store.accept_run.side_effect = RuntimeError("captured envelope")
    service = _accepting_service(tmp_path, store)
    (staged,) = await service.stage_uploads(
        workspace="default", run_id=_RUN_ID, uploads=[("report.pdf", _Reader(b"pdf"))]
    )

    with pytest.raises(RuntimeError, match="captured envelope"):
        await service.create_staged_ingest(
            workspace="default",
            run_id=_RUN_ID,
            staged=staged,
            submitted_by="o",
            title="Annual",
            replace=True,
        )

    payload = store.accept_run.await_args.kwargs["envelope"].payload
    assert payload["source"] == {
        "source_type": "local",
        "documents": [{"path": str(staged.path)}],
        "replace": True,
        "title": "Annual",
    }
    assert validate_corpus_mutation_prepared_input(payload) == ("replace", "default")


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
                "supersedes_run_id": "",
            },
            id="workspace-delete-supersedes-a-blank-run",
        ),
        pytest.param(
            {"action": "reset", "workspace": "research", "track_id": _TRACK_ID},
            id="reset-omits-its-supersession-field",
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


def _staged(*paths: str) -> list[dict[str, Any]]:
    return [{"path": path, "content_sha256": "a" * 64, "size_bytes": 1} for path in paths]


def test_a_local_run_must_list_exactly_its_staged_files_in_order() -> None:
    """The check is on data alone: the Run ingests the documents it lists, no folder."""
    listed = {"source_type": "local", "documents": [{"path": "/s/0/a.pdf"}, {"path": "/s/1/b.pdf"}]}
    staged = _staged("/s/0/a.pdf", "/s/1/b.pdf")

    assert validate_corpus_mutation_prepared_input(
        _payload("ingest", source={**listed, "replace": False}, staged_sources=staged)
    ) == ("ingest", "default")
    for source, records in (
        ({**listed, "replace": False}, list(reversed(staged))),
        ({**listed, "replace": False}, staged[:1]),
        ({"source_type": "local", "path": "/s", "replace": False}, staged),
        ({"source_type": "local", "documents": [{"path": "/s/0/a.pdf"}], "replace": False}, []),
    ):
        with pytest.raises(ValueError):
            validate_corpus_mutation_prepared_input(
                _payload("ingest", source=source, staged_sources=records)
            )


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


#: A corpus directory that holds nothing, for executor tests that never stage.
_NO_CORPUS = Path("/nonexistent/dlightrag-test/corpus")


def _executor(
    runtime: Any,
    *,
    now=None,
    acquire_error: Exception | None = None,
    maintenance: _Maintenance | None = None,
    store: Any = None,
    corpus_root: Path = _NO_CORPUS,
    workspace_exists: Any = _registered,
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
        corpus_root=corpus_root,
        workspace_exists=workspace_exists,
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


def _local_payload(*paths: Path) -> dict[str, Any]:
    return _payload(
        "ingest",
        source={
            "source_type": "local",
            "documents": [{"path": str(path)} for path in paths],
            "replace": False,
        },
        staged_sources=_staged(*(str(path) for path in paths)),
    )


async def test_a_local_run_hands_the_engine_exactly_its_listed_files(tmp_path: Path) -> None:
    first, second = tmp_path / "0" / "a.pdf", tmp_path / "1" / "b.pdf"
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b"x")
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)

    outcome = await executor.execute(cast(Any, _Session(_local_payload(first, second))))

    assert isinstance(outcome, Succeeded)
    runtime.aingest.assert_awaited_once_with(
        "local",
        documents=[{"path": str(first)}, {"path": str(second)}],
        replace=False,
        _track_id=_TRACK_ID,
    )


async def test_a_local_run_missing_a_staged_file_fails_before_its_handoff(tmp_path: Path) -> None:
    present = tmp_path / "0" / "a.pdf"
    present.parent.mkdir()
    present.write_bytes(b"x")
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime)
    session = _Session(_local_payload(present, tmp_path / "1" / "gone.pdf"))

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Failed)
    assert outcome.error_kind == "corpus_source_unavailable"
    assert session.handoff_started is False
    runtime.aingest.assert_not_awaited()


def _staged_run(corpus_root: Path) -> tuple[Path, dict[str, Any]]:
    """One staged file of this Run, and the Run's prepared input."""
    source = corpus_root / "default" / ".runs" / _RUN_ID / "sources" / "0" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pdf")
    return source.parents[2], _local_payload(source)


@pytest.mark.parametrize(
    "result",
    [
        {"processed": 1, "errors": [], "results": [{"doc_id": "doc-1", "chunks": []}]},
        {"processed": 0, "errors": ["report.pdf: document processing failed"], "results": []},
    ],
    ids=["succeeded", "failed"],
)
async def test_a_settled_local_run_removes_its_stage(tmp_path: Path, result: dict) -> None:
    """Each ingested document has its own copy in the corpus directory by then."""
    stage, payload = _staged_run(tmp_path)
    runtime = _runtime()
    runtime.aingest.return_value = result
    executor, _pool, _store = _executor(runtime, corpus_root=tmp_path)

    outcome = await executor.execute(cast(Any, _Session(payload)))

    assert isinstance(outcome, Succeeded | Failed)
    assert not stage.exists()


async def test_a_replacement_whose_parser_input_cannot_be_placed_fails_before_any_effect(
    tmp_path: Path,
) -> None:
    """Placement precedes the replacement's cleanup, so there is nothing to repair."""
    from dlightrag.engine.rag.corpus.ingestion.errors import ParserInputPlacementError

    stage, payload = _staged_run(tmp_path)
    payload = {**payload, "action": "replace", "source": {**payload["source"], "replace": True}}
    runtime = _runtime()
    runtime.aingest.side_effect = ParserInputPlacementError("no space left on the corpus disk")
    executor, _pool, _store = _executor(runtime, corpus_root=tmp_path)
    session = _Session(payload)

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Failed)
    assert outcome.error_kind == "corpus_source_unavailable"
    assert session.handoff_started is True
    assert not stage.exists()


async def test_a_run_that_may_run_again_keeps_its_stage(tmp_path: Path) -> None:
    stage, payload = _staged_run(tmp_path)
    error = TransientDependencyError("corpus_storage", "temporarily unavailable")
    executor, _pool, _store = _executor(_runtime(), acquire_error=error, corpus_root=tmp_path)

    outcome = await executor.execute(cast(Any, _Session(payload)))

    assert isinstance(outcome, Deferred)
    assert (stage / "sources" / "0" / "report.pdf").read_bytes() == b"pdf"


async def test_a_run_cancelled_before_its_handoff_removes_its_stage(tmp_path: Path) -> None:
    from dlightrag.engine.runtime.coordinator import RunCancellationObserved

    stage, payload = _staged_run(tmp_path)
    executor, _pool, _store = _executor(_runtime(), corpus_root=tmp_path)
    session = _Session(payload)
    session.begin_handoff = AsyncMock(side_effect=RunCancellationObserved)  # type: ignore[method-assign]

    with pytest.raises(RunCancellationObserved):
        await executor.execute(cast(Any, session))

    assert not stage.exists()


async def test_a_corpus_run_cancelled_while_queued_loses_its_stage(tmp_path: Path) -> None:
    from dlightrag.application.runs import RunView

    stage = tmp_path / "corpus" / "default" / ".runs" / _RUN_ID
    (stage / "sources" / "0").mkdir(parents=True)
    service = _service(tmp_path)
    cancelled = cast(
        RunView,
        SimpleNamespace(
            run_id=_RUN_ID,
            run_kind="corpus_mutation",
            access_scope_kind="workspace",
            access_scope_id="default",
        ),
    )

    await service.discard_cancelled_run(cancelled)

    assert not stage.exists()


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


async def test_fresh_reset_evicts_only_runtime() -> None:
    runtime = _runtime()
    executor, pool, _store = _executor(runtime)
    session = _Session(_payload("reset", supersedes_run_id=None))

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    runtime.areset.assert_awaited_once_with()
    pool.evict.assert_awaited_once_with("default")


async def test_a_reset_drops_only_the_stage_of_the_run_it_supersedes(tmp_path: Path) -> None:
    """A Run it supersedes ended at acceptance; a Run queued behind it keeps its stage."""
    superseded, queued = (
        "0199a0a0-0000-7000-8000-000000000002",
        "0199a0a0-0000-7000-8000-000000000003",
    )
    stages = tmp_path / "default" / ".runs"
    for run_id in (superseded, queued):
        (stages / run_id / "sources" / "0").mkdir(parents=True)
        (stages / run_id / "sources" / "0" / "report.pdf").write_bytes(b"pdf")
    executor, _pool, _store = _executor(_runtime(), corpus_root=tmp_path)

    outcome = await executor.execute(
        cast(Any, _Session(_payload("reset", supersedes_run_id=superseded)))
    )

    assert isinstance(outcome, Succeeded)
    assert not (stages / superseded).exists()
    assert (stages / queued / "sources" / "0" / "report.pdf").read_bytes() == b"pdf"


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


async def test_a_local_run_whose_outages_spent_its_deferrals_removes_its_stage(
    tmp_path: Path,
) -> None:
    stage, payload = _staged_run(tmp_path)
    error = TransientDependencyError("corpus_storage", "temporarily unavailable")
    executor, _pool, _store = _executor(_runtime(), acquire_error=error, corpus_root=tmp_path)
    session = _Session(payload, checkpoint={"dependency_deferrals": MAX_DEPENDENCY_DEFERRALS})

    with pytest.raises(DependencyRetriesExhausted):
        await executor.execute(cast(Any, session))

    assert not stage.exists()


async def test_a_write_fence_defers_a_run_whose_outages_spent_its_deferrals() -> None:
    """Waiting behind a fence is no outage, so it never ends the Run."""
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    payload = _payload(
        "ingest",
        source={"source_type": "s3", "bucket": "documents", "replace": False},
        staged_sources=[],
    )
    fence = WorkspaceWriteFencedError(workspace="default", retry_after_seconds=30)
    executor, _pool, _store = _executor(_runtime(), now=lambda: now, acquire_error=fence)
    session = _Session(payload, checkpoint={"dependency_deferrals": MAX_DEPENDENCY_DEFERRALS})

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Deferred)
    assert outcome.checkpoint["dependency_deferrals"] == MAX_DEPENDENCY_DEFERRALS


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
        **_roots(tmp_path),
        store=AsyncMock(),
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=_LIMITS,
        workspace_exists=_registered,
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


async def test_a_reader_refuses_uploads_without_reading_the_catalog(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationUnavailableError

    lookups: list[str] = []

    async def exists(workspace: str) -> bool:
        lookups.append(workspace)
        return True

    service = CorpusMutationService(
        **_roots(tmp_path),
        store=AsyncMock(),
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=_LIMITS,
        workspace_exists=exists,
        writable=False,
    )

    with pytest.raises(CorpusMutationUnavailableError):
        await service.stage_uploads(
            workspace="default", run_id=_RUN_ID, uploads=[("a.pdf", _Reader(b"a"))]
        )
    assert lookups == []


_UNLISTED_RUNS = [
    pytest.param(_local_payload, id="local-ingest"),
    pytest.param(
        lambda: _payload("delete", file_paths=[], filenames=[], document_ids=["doc-1"]),
        id="delete",
    ),
    pytest.param(lambda: _payload("retry", document_ids=["doc-1"], selector=None), id="retry"),
    pytest.param(lambda: _payload("reset", supersedes_run_id=None), id="reset"),
]


@pytest.mark.parametrize("prepared", _UNLISTED_RUNS)
async def test_a_run_whose_workspace_is_gone_fails_before_any_effect(
    tmp_path: Path, prepared: Any
) -> None:
    """A Run accepted while its Workspace was being deleted runs after that delete."""
    stage, payload = _staged_run(tmp_path)
    if prepared is not _local_payload:
        payload = prepared()
    runtime = _runtime()
    executor, pool, _store = _executor(runtime, corpus_root=tmp_path, workspace_exists=_gone)
    session = _Session(payload)

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Failed)
    assert outcome.error_kind == "workspace_not_found"
    assert session.handoff_started is False
    pool.acquire.assert_not_awaited()
    runtime.aingest.assert_not_awaited()
    if prepared is _local_payload:
        assert not stage.exists()


async def test_workspace_delete_proceeds_once_its_workspace_is_unlisted() -> None:
    """It unlists the Workspace itself, so its recovery must not refuse on that."""
    runtime = _runtime()
    executor, _pool, _store = _executor(runtime, store=_successor_store(()), workspace_exists=_gone)

    outcome = await executor.execute(cast(Any, _workspace_delete_session()))

    assert isinstance(outcome, Succeeded)
    runtime.areset.assert_awaited_once_with()


async def test_an_unreadable_catalog_defers_the_run() -> None:
    async def unreadable(_workspace: str) -> bool:
        raise ConnectionError("registry down")

    executor, pool, _store = _executor(_runtime(), workspace_exists=unreadable)

    outcome = await executor.execute(cast(Any, _Session(_payload("reset", supersedes_run_id=None))))

    assert isinstance(outcome, Deferred)
    pool.acquire.assert_not_awaited()


def _run_record(run_id: str, *, status: str = "queued", run_kind: str = "corpus_mutation"):
    return SimpleNamespace(
        run_id=run_id,
        run_kind=run_kind,
        terminal=status in {"succeeded", "failed", "cancelled"},
    )


def _successor_store(*pages: tuple[Any, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        list_runs=AsyncMock(side_effect=list(pages)),
        request_cancellation=AsyncMock(return_value=SimpleNamespace(outcome="cancelled")),
    )


def _run_ids(count: int) -> list[str]:
    return [f"0199a0a0-0000-7000-8000-{index:012d}" for index in range(10, 10 + count)]


def _workspace_delete_session(**kwargs: Any) -> _Session:
    session = _Session(
        {"action": "delete_workspace", "workspace": "research", "track_id": _TRACK_ID},
        **kwargs,
    )
    session.owner_id = "research"
    return session


async def test_workspace_delete_resets_everything_then_retires_identity_and_successors(
    tmp_path: Path,
) -> None:
    queued, done, answer = _run_ids(3)
    stages = tmp_path / "research" / ".runs"
    for run_id in (queued, done):
        (stages / run_id / "sources").mkdir(parents=True)
    runtime = _runtime()
    maintenance = _Maintenance()
    store = _successor_store(
        (
            _run_record(queued),
            _run_record(done, status="succeeded"),
            _run_record(answer, run_kind="answer"),
        ),
    )
    executor, pool, _store = _executor(
        runtime, maintenance=maintenance, store=store, corpus_root=tmp_path
    )
    session = _workspace_delete_session()

    outcome = await executor.execute(cast(Any, session))

    assert isinstance(outcome, Succeeded)
    assert outcome.result["action"] == "delete_workspace"
    runtime.areset.assert_awaited_once_with()
    assert session.handoff_started is True
    assert session.phases == ["resetting_corpus", "removing_workspace"]
    assert maintenance.unregistered == ["research"]
    store.list_runs.assert_awaited_once_with(owner_id="research", after_run_id=_RUN_ID, limit=100)
    store.request_cancellation.assert_awaited_once_with(owner_id="research", run_id=queued)
    pool.evict.assert_awaited_once_with("research")
    # Every successor ended, so its stage went, and with it the Workspace's folders.
    assert not (tmp_path / "research").exists()


async def test_workspace_delete_keeps_a_stage_it_did_not_end(tmp_path: Path) -> None:
    """An upload still staging for the deleted Workspace keeps its folder; nothing races it."""
    running, uploading = _run_ids(2)
    stages = tmp_path / "research" / ".runs"
    for run_id in (running, uploading):
        (stages / run_id / "sources").mkdir(parents=True)
    store = _successor_store((_run_record(running),))
    store.request_cancellation.return_value = SimpleNamespace(outcome="pending")
    executor, _pool, _store = _executor(_runtime(), store=store, corpus_root=tmp_path)

    outcome = await executor.execute(cast(Any, _workspace_delete_session()))

    assert isinstance(outcome, Succeeded)
    assert (stages / running / "sources").is_dir()
    assert (stages / uploading / "sources").is_dir()


async def test_workspace_delete_pages_through_every_queued_successor() -> None:
    run_ids = _run_ids(101)
    first = tuple(_run_record(run_id) for run_id in run_ids[:100])
    store = _successor_store(first, (_run_record(run_ids[100]),))
    executor, _pool, _store = _executor(_runtime(), store=store)

    outcome = await executor.execute(cast(Any, _workspace_delete_session()))

    assert isinstance(outcome, Succeeded)
    assert store.list_runs.await_args_list[1].kwargs["after_run_id"] == run_ids[99]
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
        **_roots(tmp_path),
        store=store,
        coordinator=cast(Any, SimpleNamespace()),
        upload_limits=_LIMITS,
        workspace_exists=_registered,
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
        **_roots(tmp_path),
        store=store,
        coordinator=cast(Any, coordinator),
        upload_limits=_LIMITS,
        workspace_exists=_registered,
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
        **_roots(tmp_path),
        store=store,
        coordinator=cast(Any, SimpleNamespace(is_started=True)),
        upload_limits=_LIMITS,
        workspace_exists=_registered,
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
    folder = tmp_path / "inputs" / "default" / "docs"
    folder.mkdir(parents=True)
    (folder / "ok.txt").write_text("ok", encoding="utf-8")
    (folder / "leak.txt").symlink_to(outside / "secret.txt")

    with pytest.raises(CorpusMutationInputError, match="cannot contain symlinks"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))

    # Nothing was staged, least of all the linked file's bytes.
    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()


def test_a_local_folder_may_hold_only_files_and_folders(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    folder = tmp_path / "inputs" / "default" / "docs"
    folder.mkdir(parents=True)
    os.mkfifo(folder / "pipe")

    with pytest.raises(CorpusMutationInputError, match="only regular files and folders"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))


def test_a_missing_or_oversized_local_source_is_the_callers_to_fix(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    service = _service(tmp_path)
    with pytest.raises(CorpusMutationInputError, match="does not exist"):
        service._snapshot_local_spec(
            _RUN_ID, "default", _local_spec(tmp_path / "inputs" / "default" / "missing.pdf")
        )

    folder = tmp_path / "inputs" / "default" / "many"
    folder.mkdir(parents=True)
    for index in range(101):
        (folder / f"{index}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(CorpusMutationInputError, match="more than 100 files"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))


def test_an_oversized_local_folder_refuses_before_copying(tmp_path: Path, monkeypatch) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError, mutations

    copied: list[tuple[object, ...]] = []
    real_copy = mutations._copy_regular_file

    def counting_copy(*args, **kwargs) -> dict:
        copied.append(args)
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(mutations, "_copy_regular_file", counting_copy)
    folder = tmp_path / "inputs" / "default" / "many"
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
    folder = tmp_path / "inputs" / "default" / "docs"
    (folder / "zsub").mkdir(parents=True)
    (folder / "a.txt").write_text("a", encoding="utf-8")
    (folder / "zsub" / "secret.txt").write_text("inside", encoding="utf-8")
    real_members = mutations._local_source_members

    def list_then_swap(source: int, name: str, **kwargs: Any):
        listed = real_members(source, name, **kwargs)
        (folder / "zsub" / "secret.txt").unlink()
        (folder / "zsub").rmdir()
        (folder / "zsub").symlink_to(outside, target_is_directory=True)
        return listed

    monkeypatch.setattr(mutations, "_local_source_members", list_then_swap)

    with pytest.raises(CorpusMutationInputError):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))
    staged = list((tmp_path / "corpus").rglob("secret.txt"))
    assert staged == [], "no byte behind the link was staged"


def test_a_file_that_became_a_link_refuses(tmp_path: Path) -> None:
    """A source resolved before the swap is still opened without following a link."""
    from dlightrag.application.corpus_admin import CorpusMutationInputError
    from dlightrag.application.corpus_admin.mutations import _open_below

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    workspace = tmp_path / "inputs" / "default"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "report.txt").symlink_to(outside)

    root = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(CorpusMutationInputError, match="cannot contain symlinks"):
            _open_below(root, ("docs", "report.txt"))
    finally:
        os.close(root)


def test_a_local_source_too_deep_or_too_wide_refuses(tmp_path: Path, monkeypatch) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError, mutations

    workspace = tmp_path / "inputs" / "default"
    deep = workspace / "deep"
    nested = deep.joinpath(*[f"d{index}" for index in range(mutations._MAX_LOCAL_DEPTH + 1)])
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("a", encoding="utf-8")
    deepest = workspace / "deepest"
    allowed = deepest.joinpath(*[f"d{index}" for index in range(mutations._MAX_LOCAL_DEPTH)])
    allowed.mkdir(parents=True)
    (allowed / "a.txt").write_text("a", encoding="utf-8")
    wide = workspace / "wide"
    wide.mkdir()
    (wide / "a.txt").write_text("a", encoding="utf-8")
    for index in range(20):
        (wide / f".skipped-{index}").write_text("x", encoding="utf-8")
    service = _service(tmp_path)

    with pytest.raises(CorpusMutationInputError, match="more than 32 deep"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(deep))
    # Folders nest exactly as deep as the bound allows.
    _spec, manifest = service._snapshot_local_spec(_RUN_ID, "default", _local_spec(deepest))
    assert [Path(item["path"]).name for item in manifest] == ["a.txt"]
    # Entries the listing skips still count toward the entry bound: they are read.
    monkeypatch.setattr(mutations, "_MAX_LOCAL_ENTRIES", 10)
    with pytest.raises(CorpusMutationInputError, match="more than 10 entries"):
        service._snapshot_local_spec(
            "0199a0a0-0000-7000-8000-000000000002", "default", _local_spec(wide)
        )


def test_an_unreadable_local_source_is_the_callers_to_fix(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    locked = tmp_path / "inputs" / "default" / "locked"
    (locked / "inner").mkdir(parents=True)
    (locked / "inner" / "a.txt").write_text("a", encoding="utf-8")
    locked.chmod(0)
    try:
        with pytest.raises(CorpusMutationInputError, match="cannot be read"):
            _service(tmp_path)._snapshot_local_spec(
                _RUN_ID, "default", _local_spec(locked / "inner")
            )
    finally:
        locked.chmod(0o755)


def test_a_linked_folder_refuses(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    folder = tmp_path / "inputs" / "default" / "docs"
    folder.mkdir(parents=True)
    (folder / "ok.txt").write_text("ok", encoding="utf-8")
    (folder / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CorpusMutationInputError, match="cannot contain symlinks"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))


def test_a_workspace_root_source_copies_only_what_ingestion_reads(tmp_path: Path) -> None:
    """Dot entries and parser folders stay out of the copy, and out of its file count."""
    workspace = tmp_path / "inputs" / "default"
    (workspace / "reports").mkdir(parents=True)
    (workspace / "reports" / "q3.txt").write_text("quarter", encoding="utf-8")
    (workspace / "reports" / "__uploads__").write_text("x", encoding="utf-8")
    (workspace / "top.txt").write_text("top", encoding="utf-8")
    (workspace / ".staging").mkdir()
    (workspace / ".staging" / "partial.txt").write_text("x", encoding="utf-8")
    (workspace / ".hidden.txt").write_text("x", encoding="utf-8")
    (workspace / "__parsed__").mkdir()
    for index in range(150):
        (workspace / "__parsed__" / f"{index}.md").write_text("x", encoding="utf-8")

    spec, manifest = _service(tmp_path)._snapshot_local_spec(
        _RUN_ID, "default", _local_spec(workspace)
    )

    sources = (tmp_path / "corpus").resolve() / "default" / ".runs" / _RUN_ID / "sources"
    assert [item["path"] for item in manifest] == [
        str(sources / "0" / "q3.txt"),
        str(sources / "1" / "top.txt"),
    ]
    assert [document.path for document in spec.documents or ()] == [
        item["path"] for item in manifest
    ]
    assert spec.path is None
    top = manifest[1]
    assert top["size_bytes"] == 3
    assert top["content_sha256"] == hashlib.sha256(b"top").hexdigest()


def test_a_local_source_with_nothing_to_ingest_or_no_path_is_the_callers_to_fix(
    tmp_path: Path,
) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    workspace = tmp_path / "inputs" / "default"
    (workspace / "empty" / ".git").mkdir(parents=True)
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    service = _service(tmp_path)

    with pytest.raises(CorpusMutationInputError, match="no files to ingest"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(workspace / "empty"))
    with pytest.raises(CorpusMutationInputError, match="does not exist"):
        service._snapshot_local_spec(_RUN_ID, "default", _local_spec(workspace / "a.txt" / "x"))


def test_a_local_file_is_staged_under_its_own_name(tmp_path: Path) -> None:
    """LightRAG names the document by the parser input's basename, so the stage keeps it."""
    source = tmp_path / "inputs" / "default" / "docs" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")

    spec, manifest = _service(tmp_path)._snapshot_local_spec(
        _RUN_ID, "default", _local_spec(source)
    )

    (record,) = manifest
    assert Path(record["path"]).name == "report.pdf"
    assert Path(record["path"]).read_bytes() == b"%PDF"
    assert [document.path for document in spec.documents or ()] == [record["path"]]


def test_a_manifest_keeps_each_documents_fields_and_names_only_files(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError
    from dlightrag.engine.rag.corpus.contracts import IngestDocument

    workspace = tmp_path / "inputs" / "default"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "a.pdf").write_bytes(b"a")
    (workspace / "docs" / "b.pdf").write_bytes(b"b")
    service = _service(tmp_path)
    spec = IngestSpec(
        source_type="local",
        title="batch title",
        documents=[
            IngestDocument(path=str(workspace / "docs" / "a.pdf"), metadata={"n": 1}),
            IngestDocument(path=str(workspace / "docs" / "b.pdf"), filename="renamed.pdf"),
        ],
    )

    executed, manifest = service._snapshot_local_spec(_RUN_ID, "default", spec)

    documents = executed.documents or []
    assert [document.path for document in documents] == [item["path"] for item in manifest]
    assert documents[0].metadata == {"n": 1}
    assert documents[1].filename == "renamed.pdf"
    assert executed.title == "batch title"

    folder = IngestSpec(
        source_type="local", documents=[IngestDocument(path=str(workspace / "docs"))]
    )
    with pytest.raises(CorpusMutationInputError, match="must name a file"):
        service._snapshot_local_spec("0199a0a0-0000-7000-8000-000000000002", "default", folder)


@pytest.mark.parametrize("name", [".hidden.pdf", ".runs", "__parsed__", "__remote_sources__"])
def test_a_local_file_named_like_a_corpus_entry_is_refused(tmp_path: Path, name: str) -> None:
    """Its parser input would take a Run stage's, a temporary copy's or a corpus folder's place."""
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    source = tmp_path / "inputs" / "default" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")

    with pytest.raises(CorpusMutationInputError, match="cannot be named"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(source))
    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()


def test_two_local_files_that_would_become_one_document_are_refused(tmp_path: Path) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationInputError

    folder = tmp_path / "inputs" / "default" / "docs"
    (folder / "a").mkdir(parents=True)
    (folder / "b").mkdir()
    (folder / "a" / "report.pdf").write_bytes(b"a")
    (folder / "b" / "report.pdf").write_bytes(b"b")

    with pytest.raises(CorpusMutationInputError, match="same document 'report.pdf'"):
        _service(tmp_path)._snapshot_local_spec(_RUN_ID, "default", _local_spec(folder))
    assert not (tmp_path / "corpus" / "default" / ".runs" / _RUN_ID).exists()


def test_a_workspace_named_sources_stages_and_validates_like_any_other(tmp_path: Path) -> None:
    source = tmp_path / "inputs" / "sources" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")

    spec, manifest = _service(tmp_path)._snapshot_local_spec(
        _RUN_ID, "sources", _local_spec(source)
    )

    action, workspace = validate_corpus_mutation_prepared_input(
        _payload(
            "ingest",
            workspace="sources",
            source=spec.model_dump(mode="json", exclude_none=True),
            staged_sources=manifest,
        )
    )
    assert (action, workspace) == ("ingest", "sources")
