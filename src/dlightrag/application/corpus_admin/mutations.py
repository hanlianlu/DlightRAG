# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Corpus Mutation acceptance and execution on the common RunRuntime."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import errno
import hashlib
import logging
import os
import shutil
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, assert_never, cast
from uuid import UUID, uuid7

from dlightrag.application.errors import ApplicationError, ApplicationUnavailableError
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunCreation,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.dependencies import (
    DependencyComponent,
    classify_transient_dependency,
    next_dependency_retry,
)
from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError
from dlightrag.engine.rag.corpus.ingestion.paths import excluded_from_directory_scan
from dlightrag.engine.rag.corpus.ingestion.uploads import safe_upload_relative_path
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.rag.workspace.ports import CorpusMaintenanceStore, WorkspaceWriteFencedError
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id
from dlightrag.engine.runtime.coordinator import RunExecutor, RunSession
from dlightrag.engine.runtime.policy import CORPUS_MUTATION_RUN_RETENTION_SECONDS
from dlightrag.engine.runtime.records import (
    Deferred,
    Failed,
    PreparedInputTooLargeError,
    PreparedRunEnvelope,
    RunAccessScope,
    RunExecutionOutcome,
    Succeeded,
    WaitingForRepair,
    require_prepared_input_bounds,
    run_request_fingerprint,
)
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)
from dlightrag.engine.runtime.records import (
    RunAdmissionLimitExceededError as RuntimeRunAdmissionLimitExceededError,
)

from .errors import (
    CorpusMutationInputError,
    CorpusMutationUnavailableError,
    CorpusStageUnavailableError,
    UnsafeUploadNameError,
    UploadTooLargeError,
    WorkspaceNotFoundError,
)
from .service import IngestSpec, safe_upload_basename

type CorpusMutationAction = Literal[
    "ingest", "replace", "delete", "retry", "reset", "delete_workspace"
]
type RetrySelector = Literal["all_retryable"]


@dataclass(frozen=True, slots=True)
class _ActionSpec:
    """What one Corpus Mutation action accepts and how its recovery behaves."""

    # Prepared-input fields beside action, workspace, and track_id.
    fields: frozenset[str]
    # Reads a source (ingest/replace) rather than selecting existing documents.
    source_based: bool
    # A recovered handoff is never repeated without repair evidence.
    destructive: bool
    # Enqueues LightRAG pipeline work under the Run's track id, so recovery first
    # reconciles what upstream already did.
    tracks_upstream: bool


# The one list of actions; validation, recovery, and projections derive from it.
_ACTIONS: Mapping[CorpusMutationAction, _ActionSpec] = MappingProxyType(
    {
        "ingest": _ActionSpec(frozenset({"source", "staged_sources"}), True, False, True),
        "replace": _ActionSpec(frozenset({"source", "staged_sources"}), True, True, True),
        "delete": _ActionSpec(
            frozenset({"file_paths", "filenames", "document_ids"}), False, True, False
        ),
        "retry": _ActionSpec(frozenset({"document_ids", "selector"}), False, True, True),
        "reset": _ActionSpec(frozenset({"supersedes_run_id"}), False, True, False),
        "delete_workspace": _ActionSpec(frozenset({"supersedes_run_id"}), False, True, False),
    }
)

_REPAIR_REASON = "The upstream corpus outcome is not safe to repeat automatically."
_REPAIR_REMEDY = "Inspect the public LightRAG state, repair it, then resume this Run."
logger = logging.getLogger(__name__)

_MAX_RESULT_DOCUMENTS = 100
_DESTRUCTIVE_ACTIONS = frozenset(action for action, spec in _ACTIONS.items() if spec.destructive)
_SUCCESSOR_PAGE_LIMIT = 100
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_DEFER_BASE_SECONDS = 2
_DEFER_MAX_SECONDS = 60


class CorpusMutationStore(Protocol):
    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: Literal["corpus_mutation"],
    ) -> Any: ...

    async def accept_run(self, *, envelope: PreparedRunEnvelope, run_id: str) -> Any: ...

    async def record_corpus_window(
        self,
        *,
        run_id: str,
        workspace: str,
        window_number: int,
        docs: int,
        chunks: int,
    ) -> bool: ...

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> Sequence[Any]: ...

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> Any: ...


class CorpusMutationScheduler(Protocol):
    @property
    def is_started(self) -> bool: ...

    def admission(self) -> Any: ...
    def wake(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StagedCorpusSource:
    """One bounded upload atomically committed to a Run-exclusive source path."""

    path: Path
    filename: str
    content_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class UploadLimits:
    """Bounds every upload surface shares: one file, one request, and file count."""

    file_bytes: int
    request_bytes: int
    request_files: int = 100


class CorpusMutationService:
    """Accept all product corpus writes as generic ``corpus_mutation`` Runs."""

    def __init__(
        self,
        *,
        input_root: Path,
        store: CorpusMutationStore,
        coordinator: CorpusMutationScheduler,
        upload_limits: UploadLimits,
        workspace_exists: Callable[[str], Awaitable[bool]],
        writable: bool = True,
        default_workspace: str = "default",
    ) -> None:
        self._input_root = Path(input_root)
        self._workspace_exists = workspace_exists
        self._store = store
        self._coordinator = coordinator
        self._upload_limits = upload_limits
        self._writable = writable
        self._default_workspace = require_canonical_workspace_id(default_workspace)

    def _require_writable(self, request: str) -> None:
        """Refuse a corpus write before any of it happens, and say who can take it."""
        if not self._writable:
            raise CorpusMutationUnavailableError(request=request)

    async def replay(
        self,
        *,
        submitted_by: str,
        idempotency_key: str | None,
        normalized_request: Mapping[str, Any],
    ) -> RunCreation | None:
        """Resolve a supplied digest/key before receiving an upload body."""
        if idempotency_key is None:
            return None
        fingerprint = run_request_fingerprint(normalized_request)
        try:
            replay = await self._store.replay_run(
                owner_id=submitted_by,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
                run_kind="corpus_mutation",
            )
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict() from exc
        return RunCreation.from_runtime(replay) if replay is not None else None

    async def create_ingest(
        self,
        *,
        workspace: str,
        spec: IngestSpec,
        submitted_by: str,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        self._require_writable("the ingest")
        action: CorpusMutationAction = "replace" if bool(spec.replace) else "ingest"
        request = {
            "action": action,
            "workspace": require_canonical_workspace_id(workspace),
            "source": spec.model_dump(mode="json", exclude_none=True),
        }
        if spec.source_type != "local":
            replay = await self.replay(
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=request,
            )
            if replay is not None:
                return replay
        await self._require_workspace(request["workspace"], action)

        run_id = str(uuid7())
        execution_spec = spec
        staged_root: Path | None = None
        staged_sources: list[dict[str, Any]] = []
        if spec.source_type == "local":
            execution_spec, staged_root, staged_sources = await asyncio.to_thread(
                self._snapshot_local_spec, run_id, workspace, spec
            )
        normalized_request = {
            **request,
            **(
                {
                    "staged_sources": [
                        {
                            "content_sha256": item["content_sha256"],
                            "size_bytes": item["size_bytes"],
                        }
                        for item in staged_sources
                    ]
                }
                if staged_sources
                else {}
            ),
        }
        if staged_root is not None:
            replay = await self.replay(
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=normalized_request,
            )
            if replay is not None:
                await asyncio.to_thread(shutil.rmtree, staged_root, True)
                return replay
        payload = {
            **request,
            "source": execution_spec.model_dump(mode="json", exclude_none=True),
            "staged_sources": staged_sources,
            "track_id": _track_id(run_id),
        }
        try:
            creation = await self._accept(
                run_id=run_id,
                workspace=workspace,
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=normalized_request,
                payload=payload,
            )
        except BaseException:
            if staged_root is not None:
                await asyncio.to_thread(shutil.rmtree, staged_root, True)
            raise
        if staged_root is not None and creation.replayed and creation.run.run_id != run_id:
            await asyncio.to_thread(shutil.rmtree, staged_root, True)
        return creation

    async def create_staged_ingest(
        self,
        *,
        workspace: str,
        staged: StagedCorpusSource,
        submitted_by: str,
        idempotency_key: str | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> RunCreation:
        self._require_writable("the upload")
        run_id = staged.path.parents[1].name
        action: CorpusMutationAction = "replace" if replace else "ingest"
        source_identity = {
            "source_type": "local",
            "filename": staged.filename,
            "content_sha256": staged.content_sha256,
            "size_bytes": staged.size_bytes,
            **({"title": title} if title is not None else {}),
            **({"author": author} if author is not None else {}),
            **({"metadata": dict(metadata)} if metadata is not None else {}),
        }
        request = {
            "action": action,
            "workspace": require_canonical_workspace_id(workspace),
            "source": source_identity,
        }
        payload = {
            **request,
            "source": {
                "source_type": "local",
                "path": str(staged.path),
                "replace": replace,
                **({"title": title} if title is not None else {}),
                **({"author": author} if author is not None else {}),
                **({"metadata": dict(metadata)} if metadata is not None else {}),
            },
            "staged_sources": [_staged_source_record(staged)],
            "track_id": _track_id(run_id),
        }
        try:
            creation = await self._accept(
                run_id=run_id,
                workspace=workspace,
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=request,
                payload=payload,
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, staged.path.parents[1], True)
            raise
        if creation.replayed and creation.run.run_id != run_id:
            await asyncio.to_thread(shutil.rmtree, staged.path.parents[1], True)
        return creation

    async def create_staged_batch(
        self,
        *,
        workspace: str,
        staged: Sequence[StagedCorpusSource],
        submitted_by: str,
        idempotency_key: str | None = None,
        replace: bool = False,
    ) -> RunCreation:
        self._require_writable("the upload")
        """Accept one already-staged multipart cohort as one ingest or replace Run."""
        if not staged:
            raise ValueError("at least one staged source is required")
        run_id = staged[0].path.parents[1].name
        if any(item.path.parents[1].name != run_id for item in staged):
            raise ValueError("staged sources do not belong to one Run")
        action: CorpusMutationAction = "replace" if replace else "ingest"
        request = {
            "action": action,
            "workspace": require_canonical_workspace_id(workspace),
            "sources": [
                {
                    "filename": item.filename,
                    "content_sha256": item.content_sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in staged
            ],
        }
        payload = {
            **request,
            "source": {
                "source_type": "local",
                "path": str(staged[0].path.parent),
                "replace": replace,
            },
            "staged_sources": [_staged_source_record(item) for item in staged],
            "track_id": _track_id(run_id),
        }
        run_root = staged[0].path.parents[1]
        try:
            creation = await self._accept(
                run_id=run_id,
                workspace=workspace,
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=request,
                payload=payload,
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, run_root, True)
            raise
        if creation.replayed and creation.run.run_id != run_id:
            await asyncio.to_thread(shutil.rmtree, run_root, True)
        return creation

    async def create_delete(
        self,
        *,
        workspace: str,
        submitted_by: str,
        file_paths: Sequence[str] = (),
        filenames: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> RunCreation:
        self._require_writable("the delete")
        selectors = {
            "file_paths": _bounded_unique(file_paths),
            "filenames": _bounded_unique(filenames),
            "document_ids": _bounded_unique(document_ids),
        }
        if not any(selectors.values()):
            raise CorpusMutationInputError("at least one exact document identifier is required")
        return await self._create_action(
            action="delete",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            fields=selectors,
        )

    async def create_retry(
        self,
        *,
        workspace: str,
        submitted_by: str,
        document_ids: Sequence[str] = (),
        selector: RetrySelector | None = None,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        self._require_writable("the retry")
        ids = _bounded_unique(document_ids)
        if bool(ids) == bool(selector):
            raise CorpusMutationInputError(
                "provide document_ids or selector='all_retryable', but not both"
            )
        if selector not in {None, "all_retryable"}:
            raise CorpusMutationInputError("unknown retry selector")
        return await self._create_action(
            action="retry",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            fields={"document_ids": ids, "selector": selector},
        )

    async def create_reset(
        self,
        *,
        workspace: str,
        submitted_by: str,
        supersedes_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        self._require_writable("the Corpus Reset")
        return await self._create_action(
            action="reset",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            fields={"supersedes_run_id": supersedes_run_id},
        )

    async def create_workspace_delete(
        self,
        *,
        workspace: str,
        submitted_by: str,
        supersedes_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        """Accept the Workspace's final mutation: a full reset, then identity removal.

        Like a Corpus Reset it may supersede the Workspace's mutation waiting for
        repair, which would otherwise hold the FIFO lane ahead of it for good.
        """
        self._require_writable("the Workspace Delete")
        if require_canonical_workspace_id(workspace) == self._default_workspace:
            raise CorpusMutationInputError(
                "The default workspace cannot be deleted; reset its corpus instead."
            )
        return await self._create_action(
            action="delete_workspace",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            # Omitted when absent, so a plain delete keeps its request fingerprint.
            fields={"supersedes_run_id": supersedes_run_id} if supersedes_run_id else {},
        )

    async def _create_action(
        self,
        *,
        action: CorpusMutationAction,
        workspace: str,
        submitted_by: str,
        idempotency_key: str | None,
        fields: Mapping[str, Any],
    ) -> RunCreation:
        canonical = require_canonical_workspace_id(workspace)
        request = {"action": action, "workspace": canonical, **dict(fields)}
        replay = await self.replay(
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            normalized_request=request,
        )
        if replay is not None:
            return replay
        await self._require_workspace(canonical, action)
        run_id = str(uuid7())
        return await self._accept(
            run_id=run_id,
            workspace=canonical,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            normalized_request=request,
            payload={**request, "track_id": _track_id(run_id)},
        )

    async def _require_workspace(self, workspace: str, action: CorpusMutationAction) -> None:
        """Refuse a corpus write to a Workspace the catalog does not list.

        Every write needs a created Workspace: data written under an unlisted name
        is invisible to the catalog and its access rules, and a misspelled reset
        would otherwise "succeed" on an empty corpus. An unreadable catalog refuses
        rather than letting the write through.
        """
        try:
            registered = await self._workspace_exists(workspace)
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationUnavailableError(
                "Workspace catalog is temporarily unavailable"
            ) from exc
        if not registered:
            raise WorkspaceNotFoundError(
                "Workspace no longer exists"
                if action == "delete_workspace"
                else "Workspace does not exist; create it first"
            )

    async def _accept(
        self,
        *,
        run_id: str,
        workspace: str,
        submitted_by: str,
        idempotency_key: str | None,
        normalized_request: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> RunCreation:
        try:
            require_prepared_input_bounds(payload)
        except PreparedInputTooLargeError as exc:
            raise CorpusMutationInputError(str(exc)) from exc
        coordinator = self._coordinator
        if not coordinator.is_started:
            raise RunRuntimeUnavailableError("Corpus Mutation runtime is unavailable")
        envelope = PreparedRunEnvelope(
            run_kind="corpus_mutation",
            lane="corpus_mutation",
            submitted_by=submitted_by,
            access_scope=RunAccessScope(kind="workspace", scope_id=workspace),
            submission_key=idempotency_key or run_id,
            request_fingerprint=run_request_fingerprint(normalized_request),
            payload=payload,
            accepted_input={
                "action": str(payload["action"]),
                "workspace": workspace,
                **_accepted_selector(payload),
            },
            retention_seconds=CORPUS_MUTATION_RUN_RETENTION_SECONDS,
            supersedes_run_id=(
                str(payload["supersedes_run_id"])
                if payload.get("supersedes_run_id") is not None
                else None
            ),
        )
        try:
            async with coordinator.admission() as available:
                if not available:
                    raise RunRuntimeUnavailableError("Corpus Mutation runtime is unavailable")
                creation = await self._store.accept_run(envelope=envelope, run_id=run_id)
                coordinator.wake()
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict() from exc
        except RuntimeRunAdmissionLimitExceededError as exc:
            raise RunAdmissionLimitExceededError() from exc
        return RunCreation.from_runtime(creation)

    @property
    def upload_limits(self) -> UploadLimits:
        return self._upload_limits

    async def stage_uploads(
        self,
        *,
        workspace: str,
        run_id: str,
        uploads: Sequence[tuple[str, Any]],
        content_sha256: str | None = None,
    ) -> list[StagedCorpusSource]:
        """Stage one request's files under the shared per-file and per-request caps.

        Every file is bounded by the per-file cap and by what the request has left,
        so no surface can let one file use the whole request budget. A failed file
        removes the whole Run-exclusive stage.
        """
        limits = self._upload_limits
        if not uploads:
            raise CorpusMutationInputError("at least one upload is required")
        await self._require_workspace(require_canonical_workspace_id(workspace), "ingest")
        if len(uploads) > limits.request_files:
            raise UploadTooLargeError(f"upload contains more than {limits.request_files} files")
        if content_sha256 is not None and len(uploads) != 1:
            raise CorpusMutationInputError("content_sha256 is supported only for a single upload")
        staged: list[StagedCorpusSource] = []
        remaining = limits.request_bytes
        for filename, reader in uploads:
            if remaining <= 0:
                await self.discard_staged_run(workspace=workspace, run_id=run_id)
                raise UploadTooLargeError(f"upload exceeds {limits.request_bytes} bytes")
            item = await self.stage_upload(
                workspace=workspace,
                run_id=run_id,
                filename=filename,
                reader=reader,
                max_bytes=min(limits.file_bytes, remaining),
                content_sha256=content_sha256,
            )
            staged.append(item)
            remaining -= item.size_bytes
        return staged

    async def stage_upload(
        self,
        *,
        workspace: str,
        run_id: str,
        filename: str,
        reader: Any,
        max_bytes: int,
        content_sha256: str | None = None,
    ) -> StagedCorpusSource:
        """Stream, hash, bound, and atomically commit one source outside Run blobs."""
        self._require_writable("the upload")
        canonical = require_canonical_workspace_id(workspace)
        try:
            safe_path = safe_upload_relative_path(filename)
        except ValueError:
            raise UnsafeUploadNameError(f"Unsafe filename: {filename!r}") from None
        expected = content_sha256.lower() if content_sha256 else None
        if expected is not None and (
            len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected)
        ):
            raise CorpusMutationInputError(
                "content_sha256 must be a lowercase or uppercase SHA-256 hex digest"
            )

        stage, source_root = await asyncio.to_thread(
            _open_run_stage, self._input_root, canonical, run_id, exclusive=False
        )
        run_root = source_root.parent
        temporary = f"{run_id}.part"
        staging = parent = None
        try:
            staging = await asyncio.to_thread(_open_upload_staging, self._input_root, canonical)
            parent = await asyncio.to_thread(_stage_parents, stage, safe_path.parts[:-1])
            try:
                os.stat(safe_path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise CorpusMutationInputError("upload contains duplicate source filenames")
            digest = hashlib.sha256()
            size = 0
            written = os.open(temporary, _CREATE_NO_FOLLOW, 0o600, dir_fd=staging)
            with os.fdopen(written, "wb") as stream:
                while True:
                    chunk = await reader.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLargeError(f"upload exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            actual = digest.hexdigest()
            if expected is not None and actual != expected:
                raise CorpusMutationInputError("content_sha256 does not match the uploaded bytes")
            os.replace(temporary, safe_path.name, src_dir_fd=staging, dst_dir_fd=parent)
            return StagedCorpusSource(
                path=source_root / safe_path,
                filename=safe_path.as_posix(),
                content_sha256=actual,
                size_bytes=size,
            )
        except BaseException:
            if staging is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=staging)
            await asyncio.to_thread(shutil.rmtree, run_root, True)
            raise
        finally:
            for fd in (parent, staging, stage):
                if fd is not None:
                    os.close(fd)

    async def discard_staged_run(self, *, workspace: str, run_id: str) -> None:
        """Delete one unaccepted Run-exclusive upload stage without leaking layout."""
        canonical = require_canonical_workspace_id(workspace)
        safe_run_id = str(UUID(run_id))
        run_root = self._input_root / canonical / ".runs" / safe_run_id
        await asyncio.to_thread(shutil.rmtree, run_root, True)

    def _snapshot_local_spec(
        self, run_id: str, workspace: str, spec: IngestSpec
    ) -> tuple[IngestSpec, Path, list[dict[str, Any]]]:
        canonical = require_canonical_workspace_id(workspace)
        workspace_root = (self._input_root / canonical).resolve()
        stage, source_root = _open_run_stage(self._input_root, canonical, run_id, exclusive=True)
        run_root = source_root.parent
        manifest: list[dict[str, Any]] = []

        def copy_source(raw: str, ordinal: int) -> str:
            try:
                source = Path(raw).resolve(strict=True)
            except FileNotFoundError, NotADirectoryError:
                raise CorpusMutationInputError("local corpus source does not exist") from None
            except OSError:
                raise CorpusMutationInputError("local corpus source cannot be read") from None
            if not source.is_relative_to(workspace_root):
                raise CorpusMutationInputError(
                    "local corpus sources must stay under input_dir/<workspace>"
                )
            try:
                name = f"{ordinal:04d}-{safe_upload_basename(source.name)}"
            except ValueError:
                raise UnsafeUploadNameError(f"Unsafe filename: {source.name!r}") from None
            target = source_root / name
            manifest.extend(
                _snapshot_local_source(
                    workspace_root,
                    source.relative_to(workspace_root).parts,
                    stage,
                    name,
                    target,
                    max_files=_MAX_RESULT_DOCUMENTS - len(manifest),
                )
            )
            return str(target)

        try:
            if spec.documents is not None:
                documents = [
                    document.model_copy(update={"path": copy_source(cast(str, document.path), i)})
                    for i, document in enumerate(spec.documents)
                ]
                return spec.model_copy(update={"documents": documents}), run_root, manifest
            copied = copy_source(cast(str, spec.path), 0)
            return spec.model_copy(update={"path": copied}), run_root, manifest
        except BaseException:
            shutil.rmtree(run_root, ignore_errors=True)
            raise
        finally:
            os.close(stage)


class _TrackedPipelineNotSettled(RuntimeError):
    """A recoverable tracked LightRAG cohort has not reached a terminal status."""


_TRACKED_PIPELINE_ACTIVE_STATUSES = {
    "parsing",
    "analyzing",
    "processing",
    "preprocessed",
}


class CorpusMutationExecutor(RunExecutor):
    """Recoverable executor for every Corpus Mutation action, via public LightRAG operations."""

    def __init__(
        self,
        *,
        pool: WorkspacePool,
        maintenance: CorpusMaintenanceStore,
        store: CorpusMutationStore,
        now: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._maintenance = maintenance
        self._store = store
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))

    async def execute(self, session: RunSession) -> RunExecutionOutcome:
        raw = session.prepared_input
        if not isinstance(raw, Mapping):
            return Failed("invalid_corpus_mutation", "Corpus Mutation input is unavailable.")
        try:
            action, workspace = validate_corpus_mutation_prepared_input(raw)
        except ValueError:
            return Failed("invalid_corpus_mutation", "Corpus Mutation input is invalid.")
        if workspace != session.owner_id:
            return Failed("invalid_corpus_mutation", "Corpus Mutation scope does not match input.")

        recovered_after_handoff = session.handoff_started
        checkpoint = dict(session.checkpoint or {})
        checkpoint.update(
            action=action,
            workspace=workspace,
            track_id=str(raw.get("track_id") or _track_id(session.run_id)),
        )
        try:
            runtime = await self._pool.acquire(workspace)
            if _ACTIONS[action].tracks_upstream:
                await session.enter_phase("reconciling_upstream")
                upstream = await runtime.lightrag.aget_docs_by_track_id(checkpoint["track_id"])
                checkpoint["upstream_documents"] = _public_upstream_state(upstream)
                await session.checkpoint_state(checkpoint, phase="reconciled")

            if recovered_after_handoff and _requires_repair_resume(action, checkpoint):
                if checkpoint.get("repair_resume_confirmed") is not True:
                    return WaitingForRepair(_repair_checkpoint(checkpoint, ()))
                checkpoint["repair_resume_confirmed"] = False
                checkpoint["phase"] = "repair_attempt_started"
                await session.checkpoint_state(checkpoint, phase="repair_attempt_started")

            match action:
                case "ingest" | "replace":
                    return await self._ingest(session, runtime, raw, checkpoint)
                case "delete":
                    return await self._delete(session, runtime, raw, checkpoint)
                case "retry":
                    return await self._retry(session, runtime, raw, checkpoint)
                case "delete_workspace":
                    return await self._delete_workspace(session, runtime, checkpoint)
                case "reset":
                    return await self._reset(session, runtime, checkpoint)
                case _:
                    assert_never(action)
        except WorkspaceWriteFencedError, _TrackedPipelineNotSettled:
            return _deferred(checkpoint, "corpus_storage", now=self._now)
        except RetryOutcomeUncertainError:
            if action in _DESTRUCTIVE_ACTIONS and session.handoff_started:
                return WaitingForRepair(_repair_checkpoint(checkpoint, ()))
            return _deferred(checkpoint, "corpus_storage", now=self._now)
        except FileNotFoundError:
            return Failed(
                "corpus_source_unavailable",
                "A complete accepted corpus source is no longer available.",
                result=_result(action, (), checkpoint),
            )
        except ValueError:
            return Failed(
                "invalid_corpus_mutation",
                "Corpus Mutation input failed validation.",
                result=_result(action, (), checkpoint),
            )
        except Exception as exc:
            if action in _DESTRUCTIVE_ACTIONS and session.handoff_started:
                return WaitingForRepair(_repair_checkpoint(checkpoint, ()))
            component = classify_transient_dependency(exc)
            if component is not None:
                return _deferred(checkpoint, component, now=self._now)
            raise

    async def _ingest(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        source = raw.get("source")
        if not isinstance(source, Mapping):
            return Failed("invalid_corpus_mutation", "Corpus source is unavailable.")
        if checkpoint.get("operation_settled") is True:
            return await self._settle_ingest(session, str(raw["action"]), checkpoint)

        kwargs = dict(source)
        source_type = str(kwargs.pop("source_type", ""))
        upstream = list(checkpoint.get("upstream_documents") or ())
        if session.handoff_started and upstream:
            result = await self._reconcile_tracked_ingest(session, runtime, checkpoint)
            documents = _retry_outcomes(
                result,
                [str(item.get("document_id") or "") for item in upstream],
            )
        else:
            if source_type == "local" and not await asyncio.to_thread(
                _local_source_complete,
                kwargs,
                raw.get("staged_sources"),
            ):
                raise FileNotFoundError
            kwargs["replace"] = raw.get("action") == "replace"
            kwargs["_track_id"] = checkpoint["track_id"]
            await session.checkpoint_state(
                {**checkpoint, "phase": "source_staged"}, phase="source_staged"
            )
            if not session.handoff_started:
                await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
            await session.enter_phase("upstream_pipeline")
            async with self._maintenance.workspace_write_gate(session.owner_id):
                result = await _join_public_operation(runtime.aingest(source_type, **kwargs))
            documents = _document_outcomes(result)
        errors = result.get("errors") if isinstance(result, Mapping) else None
        processed = (
            int(result.get("processed") or len(documents)) if isinstance(result, Mapping) else 0
        )
        checkpoint.update(
            document_outcomes=documents,
            operation_settled=True,
            result_had_errors=bool(errors),
            processed_count=max(0, processed),
        )
        await session.checkpoint_state(checkpoint, phase="documents_reconciled")
        return await self._settle_ingest(session, str(raw["action"]), checkpoint)

    async def _settle_ingest(
        self,
        session: RunSession,
        action: str,
        checkpoint: Mapping[str, Any],
    ) -> RunExecutionOutcome:
        documents = _checkpoint_documents(checkpoint)
        public = _result(action, documents, checkpoint)
        if checkpoint.get("result_had_errors") is True or any(
            item.get("status") == "failed" for item in documents
        ):
            return Failed(
                "corpus_mutation_document_failed",
                "One or more documents did not become ready.",
                result=public,
            )
        await self._store.record_corpus_window(
            run_id=session.run_id,
            workspace=session.owner_id,
            window_number=1,
            docs=_nonnegative_checkpoint_int(checkpoint, "processed_count"),
            chunks=sum(_chunk_count(item.get("chunks")) for item in documents),
        )
        return Succeeded(public)

    async def _reconcile_tracked_ingest(
        self,
        session: RunSession,
        runtime: Any,
        checkpoint: dict[str, Any],
    ) -> Mapping[str, Any]:
        """Settle one already-handed-off cohort without replaying replace deletion."""
        states = {
            str(item.get("document_id") or ""): str(item.get("status") or "").lower()
            for item in checkpoint.get("upstream_documents") or ()
            if isinstance(item, Mapping) and item.get("document_id")
        }
        if any(status in _TRACKED_PIPELINE_ACTIVE_STATUSES for status in states.values()):
            # Another LightRAG queue owner is still advancing this cohort. Calling
            # the public sweep here can register a pending follow-up sweep that
            # reprocesses the same files after the authoritative run finalizes and
            # archives its staging source.
            raise _TrackedPipelineNotSettled
        if any(status not in {"processed", "failed"} for status in states.values()):
            await session.enter_phase("recovering_upstream_pipeline")
            async with self._maintenance.workspace_write_gate(session.owner_id):
                await _join_public_operation(runtime.lightrag.apipeline_process_enqueue_documents())
            refreshed = await runtime.lightrag.aget_docs_by_track_id(checkpoint["track_id"])
            checkpoint["upstream_documents"] = _public_upstream_state(refreshed)
            await session.checkpoint_state(checkpoint, phase="reconciled")
            states = {
                str(item.get("document_id") or ""): str(item.get("status") or "").lower()
                for item in checkpoint["upstream_documents"]
                if isinstance(item, Mapping) and item.get("document_id")
            }
        if not states or any(status not in {"processed", "failed"} for status in states.values()):
            raise _TrackedPipelineNotSettled
        await session.enter_phase("finalizing_tracked_documents")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            return await _join_public_operation(
                runtime.aretry_failed_docs(
                    cohort_doc_ids=tuple(states),
                    track_id=checkpoint["track_id"],
                )
            )

    async def _delete(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        file_paths = [str(value) for value in raw.get("file_paths") or ()]
        filenames = [str(value) for value in raw.get("filenames") or ()]
        document_ids = [str(value) for value in raw.get("document_ids") or ()]
        identifiers = [*file_paths, *filenames, *document_ids]
        if checkpoint.get("operation_settled") is True:
            return _settled_delete_outcome(checkpoint)
        if "resolved_documents" not in checkpoint:
            preview = await runtime.adelete_files(
                file_paths=[*file_paths, *document_ids], filenames=filenames, dry_run=True
            )
            checkpoint["resolved_documents"] = [
                {
                    "identifier": str(item.get("identifier") or ""),
                    "document_ids": [str(value) for value in item.get("matched_doc_ids") or ()][
                        :_MAX_RESULT_DOCUMENTS
                    ],
                    "file_paths": [str(value) for value in item.get("matched_file_paths") or ()][
                        :_MAX_RESULT_DOCUMENTS
                    ],
                }
                for item in preview
                if isinstance(item, Mapping)
            ][:_MAX_RESULT_DOCUMENTS]
            await session.checkpoint_state(checkpoint, phase="identities_resolved")
        if not session.handoff_started:
            await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
        await session.enter_phase("deleting_upstream")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            results = await _join_public_operation(
                runtime.adelete_files(
                    file_paths=[*file_paths, *document_ids], filenames=filenames, dry_run=False
                )
            )
        documents = [dict(item) for item in results if isinstance(item, Mapping)]
        if any(str(item.get("status")) == "waiting_for_repair" for item in documents):
            return WaitingForRepair(_repair_checkpoint(checkpoint, documents))
        if not identifiers:
            return Failed("invalid_corpus_mutation", "No delete identifiers were accepted.")
        checkpoint.update(document_outcomes=documents, operation_settled=True)
        await session.checkpoint_state(checkpoint, phase="delete_settled")
        return _settled_delete_outcome(checkpoint)

    async def _retry(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        cohort = [str(value) for value in checkpoint.get("cohort_doc_ids") or ()]
        if checkpoint.get("operation_settled") is True:
            return await self._settle_retry(session, checkpoint)
        if not checkpoint.get("cohort_sealed"):
            requested = [str(value) for value in raw.get("document_ids") or ()]
            if requested:
                cohort = list(dict.fromkeys(requested))
            else:
                cohort = list(await runtime.aretryable_document_ids())
            checkpoint.update(cohort_doc_ids=cohort, cohort_sealed=True, phase="cohort_sealed")
            await session.checkpoint_state(checkpoint, phase="cohort_sealed")
        if not cohort:
            return Succeeded(_result("retry", (), checkpoint))
        if not session.handoff_started:
            await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
        await session.enter_phase("retrying_documents")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            result = await _join_public_operation(
                runtime.aretry_failed_docs(
                    cohort_doc_ids=cohort,
                    track_id=checkpoint["track_id"],
                )
            )
        documents = _retry_outcomes(result, cohort)
        checkpoint.update(
            document_outcomes=documents,
            operation_settled=True,
            retry_failed_count=(
                max(0, int(result.get("failed") or 0))
                if isinstance(result, Mapping)
                else len(documents)
            ),
            retry_succeeded_count=(
                max(0, int(result.get("succeeded") or 0)) if isinstance(result, Mapping) else 0
            ),
        )
        await session.checkpoint_state(checkpoint, phase="retry_settled")
        return await self._settle_retry(session, checkpoint)

    async def _settle_retry(
        self,
        session: RunSession,
        checkpoint: Mapping[str, Any],
    ) -> RunExecutionOutcome:
        documents = _checkpoint_documents(checkpoint)
        public = _result("retry", documents, checkpoint)
        if _nonnegative_checkpoint_int(checkpoint, "retry_failed_count") > 0 or any(
            item.get("status") == "failed" for item in documents
        ):
            return Failed(
                "corpus_retry_document_failed",
                "One or more retry documents did not become ready.",
                result=public,
            )
        await self._store.record_corpus_window(
            run_id=session.run_id,
            workspace=session.owner_id,
            window_number=1,
            docs=_nonnegative_checkpoint_int(checkpoint, "retry_succeeded_count"),
            chunks=0,
        )
        return Succeeded(public)

    async def _reset(
        self,
        session: RunSession,
        runtime: Any,
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        waiting = await self._settle_full_reset(
            session, runtime, checkpoint, preserve_run_sources_after=session.run_id
        )
        if waiting is not None:
            return waiting
        await self._pool.evict(session.owner_id)
        return Succeeded(_result("reset", _checkpoint_documents(checkpoint), checkpoint))

    async def _delete_workspace(
        self,
        session: RunSession,
        runtime: Any,
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        """Reset the whole corpus, then retire the Workspace and every queued successor.

        The per-Workspace FIFO lane guarantees that every earlier mutation is
        terminal and no successor has started. Identity removal precedes the
        successor sweep so a browser submission can no longer join the queue.
        Both are idempotent, so recovery after the settled reset repeats them.
        """
        workspace = session.owner_id
        # No later source survives: every queued successor is cancelled below.
        waiting = await self._settle_full_reset(
            session, runtime, checkpoint, preserve_run_sources_after=None
        )
        if waiting is not None:
            return waiting
        await session.enter_phase("removing_workspace")
        await self._maintenance.unregister_workspace(workspace)
        await self._cancel_successors(workspace, session.run_id)
        await self._pool.evict(workspace)
        return Succeeded(_result("delete_workspace", _checkpoint_documents(checkpoint), checkpoint))

    async def _settle_full_reset(
        self,
        session: RunSession,
        runtime: Any,
        checkpoint: dict[str, Any],
        *,
        preserve_run_sources_after: str | None,
    ) -> WaitingForRepair | None:
        """Reset the whole corpus once, or say why it needs repair.

        Corpus Reset keeps the sources of Runs queued after it; Workspace Delete
        keeps none. A settled reset is checkpointed, so recovery never repeats it.
        """
        if checkpoint.get("operation_settled") is True:
            return None
        if not session.handoff_started:
            await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
        await session.enter_phase("resetting_corpus")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            result = await _join_public_operation(
                runtime.areset(preserve_run_sources_after=preserve_run_sources_after)
            )
        documents = [dict(result)] if isinstance(result, Mapping) else []
        if not isinstance(result, Mapping) or result.get("errors"):
            return WaitingForRepair(_repair_checkpoint(checkpoint, documents))
        checkpoint.update(document_outcomes=documents, operation_settled=True)
        await session.checkpoint_state(checkpoint, phase="reset_settled")
        return None

    async def _cancel_successors(self, workspace: str, run_id: str) -> None:
        """Cancel every Corpus Mutation queued behind this one in its Workspace."""
        after = run_id
        while True:
            page = await self._store.list_runs(
                owner_id=workspace, after_run_id=after, limit=_SUCCESSOR_PAGE_LIMIT
            )
            for record in page:
                if record.run_kind == "corpus_mutation" and not record.terminal:
                    await self._store.request_cancellation(owner_id=workspace, run_id=record.run_id)
            if len(page) < _SUCCESSOR_PAGE_LIMIT:
                return
            after = page[-1].run_id


async def _join_public_operation[T](operation: Awaitable[T]) -> T:
    """Never let task cancellation abandon an in-process upstream operation."""
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Handoff already occurred before every public operation joined here.
        # Finish and classify the authoritative outcome instead of labelling a
        # completed destructive effect as cancelled.
        return await asyncio.shield(task)


# Every local source and stage path component is opened relative to its parent's
# descriptor and never through a link, so no swap made while a copy runs can reach
# outside the workspace input root or move the stage elsewhere. O_NONBLOCK keeps a
# FIFO that replaced a file from stalling the open; its type check then refuses it.
_READ_NO_FOLLOW = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIRECTORY_NO_FOLLOW = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_CREATE_NO_FOLLOW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
#: A local source is a folder tree an operator placed under input_dir; these bound
#: the listing so a pathological tree cannot occupy a worker thread for minutes.
_MAX_LOCAL_DEPTH = 32
_MAX_LOCAL_ENTRIES = 10_000


def _open_no_follow(parent: int, name: str, *, directory: bool = False) -> int:
    try:
        return os.open(name, _READ_NO_FOLLOW | (os.O_DIRECTORY if directory else 0), dir_fd=parent)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise CorpusMutationInputError("local corpus sources cannot contain symlinks") from None
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            raise CorpusMutationInputError(
                "local corpus source changed while it was being copied"
            ) from None
        if exc.errno == errno.EACCES:
            raise CorpusMutationInputError("local corpus source cannot be read") from None
        if exc.errno == errno.ENXIO:
            raise CorpusMutationInputError(
                "local corpus sources may contain only regular files and folders"
            ) from None
        if exc.errno == errno.ENAMETOOLONG:
            raise CorpusMutationInputError("local corpus source has a name too long") from None
        raise


def _open_below(root: int, parts: Sequence[str], *, directory: bool = False) -> int:
    """Open ``parts`` under the ``root`` descriptor, one unfollowed component at a time."""
    fd = os.dup(root)
    try:
        for index, name in enumerate(parts):
            last = index == len(parts) - 1
            opened = _open_no_follow(fd, name, directory=directory or not last)
            os.close(fd)
            fd = opened
        return fd
    except BaseException:
        os.close(fd)
        raise


def _private_directory(parent: int, name: str, *, create: bool, exclusive: bool = False) -> int:
    """Open one stage directory this service alone owns and writes.

    It is created 0700 when absent, and one this service created earlier with a
    wider mode is narrowed to 0700. A link, a missing directory that was not to be
    created, or a directory another account owns refuses: whoever could redirect or
    write the stage could read what it stages or add files the ingest would read.
    """
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            if exclusive:
                raise
    try:
        fd = os.open(name, _DIRECTORY_NO_FOLLOW, dir_fd=parent)
    except OSError as exc:
        logger.warning("Corpus stage directory %r cannot be opened as a directory: %s", name, exc)
        raise CorpusStageUnavailableError() from None
    status = os.fstat(fd)
    if status.st_uid != os.geteuid():
        os.close(fd)
        logger.warning("Corpus stage directory %r is owned by uid %d", name, status.st_uid)
        raise CorpusStageUnavailableError()
    if status.st_mode & 0o077:
        os.fchmod(fd, 0o700)
    return fd


def _open_workspace(input_root: Path, workspace: str) -> int:
    """Open ``input_root/<workspace>``, creating it, never through a link at the workspace.

    The workspace folder may be shared with the operator who places sources there;
    only the stages below it are this service's own.
    """
    input_root.mkdir(parents=True, exist_ok=True)
    root = os.open(input_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(workspace, dir_fd=root)
        try:
            return os.open(workspace, _DIRECTORY_NO_FOLLOW, dir_fd=root)
        except OSError:
            logger.warning("Workspace input folder %r is not a plain directory", workspace)
            raise CorpusStageUnavailableError() from None
    finally:
        os.close(root)


def _open_run_stage(
    input_root: Path, workspace: str, run_id: str, *, exclusive: bool
) -> tuple[int, Path]:
    """Open ``input_root/<workspace>/.runs/<run_id>/sources`` without following links.

    Returns the descriptor every staged file is written through and the path the
    Run records for its ingest.
    """
    opened = [_open_workspace(input_root, workspace)]
    try:
        opened.append(_private_directory(opened[-1], ".runs", create=True))
        opened.append(_private_directory(opened[-1], run_id, create=True, exclusive=exclusive))
        sources = _private_directory(opened[-1], "sources", create=True)
    finally:
        for fd in opened:
            os.close(fd)
    return sources, input_root.resolve() / workspace / ".runs" / run_id / "sources"


def _open_upload_staging(input_root: Path, workspace: str) -> int:
    """Open the workspace's ``.staging`` folder, where an upload is written first.

    It sits beside ``.runs``, so a half-written upload is never inside a Run's
    sources, and it is as private as the stages themselves.
    """
    workspace_fd = _open_workspace(input_root, workspace)
    try:
        return _private_directory(workspace_fd, ".staging", create=True)
    finally:
        os.close(workspace_fd)


def _stage_parents(stage: int, parts: Sequence[str]) -> int:
    """Open, creating as needed, the private folders that hold one staged file."""
    fd = os.dup(stage)
    try:
        for name in parts:
            opened = _private_directory(fd, name, create=True)
            os.close(fd)
            fd = opened
        return fd
    except BaseException:
        os.close(fd)
        raise


@dataclass(slots=True)
class _LocalListing:
    files: list[tuple[str, ...]]
    entries: int = 0


def _list_directory(
    fd: int, prefix: tuple[str, ...], listing: _LocalListing, *, max_files: int
) -> None:
    with os.scandir(fd) as entries:
        listed = sorted(entries, key=lambda entry: entry.name)
    for entry in listed:
        listing.entries += 1
        if listing.entries > _MAX_LOCAL_ENTRIES:
            raise CorpusMutationInputError(
                f"local corpus source holds more than {_MAX_LOCAL_ENTRIES} entries"
            )
        is_dir = entry.is_dir(follow_symlinks=False)
        if excluded_from_directory_scan(entry.name, is_dir=is_dir):
            continue
        path = (*prefix, entry.name)
        if entry.is_symlink():
            raise CorpusMutationInputError("local corpus sources cannot contain symlinks")
        if is_dir:
            if len(path) >= _MAX_LOCAL_DEPTH:
                raise CorpusMutationInputError(
                    f"local corpus source nests folders more than {_MAX_LOCAL_DEPTH} deep"
                )
            child = _open_no_follow(fd, entry.name, directory=True)
            try:
                _list_directory(child, path, listing, max_files=max_files)
            finally:
                os.close(child)
        elif entry.is_file(follow_symlinks=False):
            listing.files.append(path)
            if len(listing.files) > max_files:
                raise CorpusMutationInputError(
                    f"local corpus source contains more than {_MAX_RESULT_DOCUMENTS} files"
                )
        else:
            raise CorpusMutationInputError(
                "local corpus sources may contain only regular files and folders"
            )


def _list_local_tree(source: int, *, max_files: int) -> list[tuple[str, ...]]:
    """List the files a scan of the copy would ingest, before anything is copied.

    Links and special files refuse, and so do a tree holding more files than the
    request allows and one deeper or wider than a local source may be. Entries the
    scan skips (dot entries, parser sidecars, staging) are left out, which also
    keeps a workspace root from copying its own Run stages into themselves. Each
    folder is opened once, holding one descriptor per level.
    """
    listing = _LocalListing(files=[])
    _list_directory(source, (), listing, max_files=max_files)
    return sorted(listing.files)


def _copy_regular_file(source: int, parent: int, name: str, target: Path) -> dict[str, Any]:
    """Copy one opened regular file into a new 0600 file, hashing the bytes it writes."""
    status = os.fstat(source)
    if not stat.S_ISREG(status.st_mode):
        raise CorpusMutationInputError(
            "local corpus sources may contain only regular files and folders"
        )
    digest = hashlib.sha256()
    size = 0
    written = os.open(name, _CREATE_NO_FOLLOW, 0o600, dir_fd=parent)
    try:
        while chunk := os.read(source, _UPLOAD_CHUNK_BYTES):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                view = view[os.write(written, view) :]
            size += len(chunk)
        os.utime(written, ns=(status.st_atime_ns, status.st_mtime_ns))
    finally:
        os.close(written)
    return {"path": str(target), "content_sha256": digest.hexdigest(), "size_bytes": size}


def _snapshot_local_source(
    workspace_root: Path,
    parts: Sequence[str],
    stage: int,
    name: str,
    target: Path,
    *,
    max_files: int,
) -> list[dict[str, Any]]:
    """Copy one local source below ``workspace_root`` into the stage as ``name``.

    Returns the manifest of what was copied. A directory is listed and counted in
    full before anything is copied, then every file is reopened through its own
    unfollowed path, so a folder swapped for a link after the listing refuses
    instead of staging what the link points at. Only the folders that hold a
    copied file are created, each 0700, and every file is written 0600.
    """
    root = os.open(workspace_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        source = _open_below(root, parts)
    finally:
        os.close(root)
    try:
        mode = os.fstat(source).st_mode
        if stat.S_ISREG(mode):
            if max_files < 1:
                raise CorpusMutationInputError(
                    f"local corpus source contains more than {_MAX_RESULT_DOCUMENTS} files"
                )
            return [_copy_regular_file(source, stage, name, target)]
        if not stat.S_ISDIR(mode):
            raise CorpusMutationInputError("local corpus source is not a regular file or directory")
        files = _list_local_tree(source, max_files=max_files)
        if not files:
            raise CorpusMutationInputError("local corpus source contains no files to ingest")
        copied = _private_directory(stage, name, create=True, exclusive=True)
        try:
            manifest = []
            for path in files:
                parent = _stage_parents(copied, path[:-1])
                try:
                    fd = _open_below(source, path)
                    try:
                        manifest.append(
                            _copy_regular_file(fd, parent, path[-1], target.joinpath(*path))
                        )
                    finally:
                        os.close(fd)
                finally:
                    os.close(parent)
            return manifest
        finally:
            os.close(copied)
    finally:
        os.close(source)


def _stage_matches_manifest(manifest: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a Run's stage holds exactly the files its manifest recorded.

    The ingest scans the stage folder, so a file added beside the recorded ones, a
    link anywhere, or a recorded file that changed would be read with them. The
    stage is walked from its private folders without following links, and every
    file the scan would ingest must be a recorded one with its recorded size and
    digest.
    """
    paths = [Path(str(item.get("path") or "")) for item in manifest]
    stages = set()
    for path in paths:
        parts = path.parts
        if "sources" not in parts:
            return False
        index = parts.index("sources")
        if index < 3 or parts[index - 2] != ".runs":
            return False
        stages.add(Path(*parts[: index + 1]))
    if len(stages) != 1:
        return False
    stage = stages.pop()
    expected = {
        path.relative_to(stage).parts: item for path, item in zip(paths, manifest, strict=True)
    }
    workspace = os.open(stage.parents[2], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    opened = [workspace]
    try:
        for name in (".runs", stage.parent.name, "sources"):
            opened.append(_private_directory(opened[-1], name, create=False))
        sources = opened[-1]
        found = _list_local_tree(sources, max_files=len(expected))
        if set(found) != set(expected):
            return False
        for parts, item in expected.items():
            fd = _open_below(sources, parts)
            try:
                status = os.fstat(fd)
                if not stat.S_ISREG(status.st_mode) or status.st_size != item.get("size_bytes"):
                    return False
                digest = hashlib.sha256()
                while chunk := os.read(fd, _UPLOAD_CHUNK_BYTES):
                    digest.update(chunk)
                if digest.hexdigest() != item.get("content_sha256"):
                    return False
            finally:
                os.close(fd)
        return True
    finally:
        for fd in opened:
            os.close(fd)


def validate_corpus_mutation_prepared_input(
    raw: Mapping[str, Any],
) -> tuple[CorpusMutationAction, str]:
    """Validate the closed durable action schema used for recovery compatibility."""
    action = cast(CorpusMutationAction, str(raw.get("action") or ""))
    action_spec = _ACTIONS.get(action)
    if action_spec is None:
        raise ValueError("unknown Corpus Mutation action")
    workspace = require_canonical_workspace_id(str(raw.get("workspace") or ""))
    track_id = str(raw.get("track_id") or "")
    prefix = "dlightrag-corpus-"
    if not track_id.startswith(prefix):
        raise ValueError("Corpus Mutation track_id is invalid")
    try:
        if UUID(track_id.removeprefix(prefix)).version != 7:
            raise ValueError
    except ValueError:
        raise ValueError("Corpus Mutation track_id is invalid") from None
    expected_fields = frozenset({"action", "workspace", "track_id"}) | action_spec.fields
    allowed_field_sets = {expected_fields}
    if action == "delete_workspace":
        # Its supersession is omitted when absent (see create_workspace_delete).
        allowed_field_sets.add(expected_fields - {"supersedes_run_id"})
    if action_spec.source_based:
        allowed_field_sets.add(expected_fields | {"sources"})
    if frozenset(raw) not in allowed_field_sets:
        raise ValueError("Corpus Mutation input fields do not match its action")
    if action_spec.source_based:
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Corpus Mutation source is unavailable")
        spec = IngestSpec.model_validate(source)
        if bool(spec.replace) != (action == "replace"):
            raise ValueError("Corpus Mutation source action does not match replace mode")
        staged_sources = raw.get("staged_sources")
        if not isinstance(staged_sources, list) or len(staged_sources) > _MAX_RESULT_DOCUMENTS:
            raise ValueError("Corpus Mutation staged source manifest is invalid")
        for item in staged_sources:
            if not isinstance(item, Mapping) or set(item) != {
                "path",
                "content_sha256",
                "size_bytes",
            }:
                raise ValueError("Corpus Mutation staged source manifest is invalid")
            path = item.get("path")
            digest = item.get("content_sha256")
            size = item.get("size_bytes")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                raise ValueError("Corpus Mutation staged source manifest is invalid")
        if spec.source_type == "local" and not staged_sources:
            raise ValueError("local Corpus Mutation source manifest is unavailable")
        if spec.source_type != "local" and staged_sources:
            raise ValueError("remote Corpus Mutation cannot contain staged sources")
        sources = raw.get("sources")
        if sources is not None and (
            not isinstance(sources, list)
            or len(sources) != len(staged_sources)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"filename", "content_sha256", "size_bytes"}
                for item in sources
            )
        ):
            raise ValueError("Corpus Mutation source identity list is invalid")
    elif action == "delete":
        selectors = tuple(
            _prepared_string_list(raw.get(key), field=key)
            for key in ("file_paths", "filenames", "document_ids")
        )
        if not any(selectors):
            raise ValueError("delete requires an exact identifier")
    elif action == "retry":
        document_ids = _prepared_string_list(raw.get("document_ids"), field="document_ids")
        selector = raw.get("selector")
        if bool(document_ids) == bool(selector) or selector not in {None, "all_retryable"}:
            raise ValueError("retry requires one cohort selector")
    else:
        supersedes = raw.get("supersedes_run_id")
        if supersedes is not None and (not isinstance(supersedes, str) or not supersedes.strip()):
            raise ValueError("supersedes_run_id is invalid")
    return action, workspace


def _prepared_string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > _MAX_RESULT_DOCUMENTS:
        raise ValueError(f"{field} must be a bounded list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} contains an invalid identifier")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicate identifiers")
    return normalized


def _track_id(run_id: str) -> str:
    return f"dlightrag-corpus-{run_id}"


def _staged_source_record(source: StagedCorpusSource) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "content_sha256": source.content_sha256,
        "size_bytes": source.size_bytes,
    }


def _requires_repair_resume(
    action: CorpusMutationAction,
    checkpoint: Mapping[str, Any],
) -> bool:
    """Fence recovered destructive work unless durable evidence makes replay unnecessary."""
    if checkpoint.get("operation_settled") is True:
        return False
    if not _ACTIONS[action].destructive:
        return False
    if action == "replace" and bool(checkpoint.get("upstream_documents")):
        return False
    return True


def _checkpoint_documents(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = checkpoint.get("document_outcomes")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)][:_MAX_RESULT_DOCUMENTS]


def _nonnegative_checkpoint_int(checkpoint: Mapping[str, Any], key: str) -> int:
    value = checkpoint.get(key)
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _settled_delete_outcome(checkpoint: Mapping[str, Any]) -> RunExecutionOutcome:
    documents = _checkpoint_documents(checkpoint)
    public = _result("delete", documents, checkpoint)
    if any(str(item.get("status")) in {"failed", "rejected"} for item in documents):
        return Failed(
            "corpus_delete_rejected",
            "One or more documents could not be deleted.",
            result=public,
        )
    return Succeeded(public)


def _deferred(
    checkpoint: Mapping[str, Any],
    component: DependencyComponent,
    *,
    now: Callable[[], datetime.datetime],
) -> Deferred:
    retry_checkpoint, delay = next_dependency_retry(
        checkpoint,
        component,
        base_seconds=_DEFER_BASE_SECONDS,
        max_seconds=_DEFER_MAX_SECONDS,
    )
    return Deferred(
        checkpoint={
            **dict(checkpoint),
            **retry_checkpoint,
            "phase": "deferred_dependency",
        },
        next_attempt_at=now() + datetime.timedelta(seconds=delay),
    )


def _bounded_unique(values: Sequence[str]) -> list[str]:
    normalized = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if len(normalized) > _MAX_RESULT_DOCUMENTS:
        raise CorpusMutationInputError(
            f"at most {_MAX_RESULT_DOCUMENTS} document identifiers are allowed"
        )
    return normalized


def _accepted_selector(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = _ACTIONS.get(cast(CorpusMutationAction, str(payload.get("action") or "")))
    if spec is not None and not spec.source_based:
        return {
            key: value for key, value in payload.items() if key in spec.fields and value is not None
        }
    source = payload.get("source")
    return {
        "source_type": str(source.get("source_type") or "") if isinstance(source, Mapping) else ""
    }


def _local_source_complete(source: Mapping[str, Any], manifest: Any) -> bool:
    paths = [str(source.get("path") or "")]
    documents = source.get("documents")
    if isinstance(documents, list):
        paths = [str(item.get("path") or "") for item in documents if isinstance(item, Mapping)]
    if not paths or not all(path and Path(path).exists() for path in paths):
        return False
    if not isinstance(manifest, list) or not manifest:
        return False
    if not all(isinstance(item, Mapping) for item in manifest):
        return False
    try:
        return _stage_matches_manifest(manifest)
    except OSError, ValueError, ApplicationError:
        return False


def _public_upstream_state(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, str]] = []
    for doc_id, status in list(value.items())[:_MAX_RESULT_DOCUMENTS]:
        raw = (
            status.get("status") if isinstance(status, Mapping) else getattr(status, "status", None)
        )
        rows.append(
            {
                "document_id": str(doc_id),
                "status": str(getattr(raw, "value", raw) or "unknown"),
            }
        )
    return rows


def _document_outcomes(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Mapping):
        return [{"status": "failed", "phase": "pipeline"}]
    raw_results = result.get("results")
    rows = (
        [dict(item) for item in raw_results if isinstance(item, Mapping)]
        if isinstance(raw_results, list)
        else ([dict(result)] if result.get("doc_id") else [])
    )
    for row in rows:
        row.setdefault("status", "ready")
        row.setdefault("phase", "finalized")
    errors = [str(error)[:256] for error in result.get("errors") or ()]
    for error in errors[: max(0, _MAX_RESULT_DOCUMENTS - len(rows))]:
        rows.append({"status": "failed", "phase": "pipeline", "error": error})
    return rows[:_MAX_RESULT_DOCUMENTS]


def _retry_outcomes(result: Any, cohort: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(result, Mapping):
        return [{"document_id": value, "status": "failed"} for value in cohort]
    succeeded = {
        str(item.get("doc_id"))
        for item in result.get("succeeded_docs") or ()
        if isinstance(item, Mapping)
    }
    failed = {
        str(item.get("doc_id"))
        for item in result.get("failed_docs") or ()
        if isinstance(item, Mapping)
    }
    return [
        {
            "document_id": doc_id,
            "status": "ready" if doc_id in succeeded and doc_id not in failed else "failed",
            "phase": "finalized" if doc_id in succeeded and doc_id not in failed else "retry",
        }
        for doc_id in cohort[:_MAX_RESULT_DOCUMENTS]
    ]


def _chunk_count(value: Any) -> int:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return 0


def _public_document_outcome(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded result evidence without source locators or diagnostics."""
    projected: dict[str, Any] = {}
    document_id = item.get("document_id") or item.get("doc_id")
    if document_id:
        projected["document_id"] = str(document_id)[:256]
    if item.get("identifier"):
        projected["identifier"] = str(item["identifier"])[:256]
    for key in ("status", "phase", "source_kind", "reason"):
        if item.get(key) is not None:
            projected[key] = str(item[key])[:256]
    chunks = item.get("chunks")
    if chunks is not None:
        projected["chunk_count"] = _chunk_count(chunks)
    for key in (
        "replacement_count",
        "documents_deleted",
        "chunks_deleted",
        "entities_deleted",
        "relationships_deleted",
        "local_files_removed",
        "orphan_tables_cleaned",
    ):
        value = item.get(key)
        if isinstance(value, int):
            projected[key] = max(0, value)
    if not projected:
        projected = {"status": "completed"}
    return projected


def _result(
    action: str, documents: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "action": action,
        "documents": [_public_document_outcome(item) for item in documents[:_MAX_RESULT_DOCUMENTS]],
        "document_count": min(len(documents), _MAX_RESULT_DOCUMENTS),
        "details_truncated": len(documents) > _MAX_RESULT_DOCUMENTS,
        "track_id": str(checkpoint.get("track_id") or ""),
    }


def _repair_checkpoint(
    checkpoint: Mapping[str, Any], documents: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        **dict(checkpoint),
        "phase": "waiting_for_repair",
        "repair_reason": _REPAIR_REASON,
        "repair_remedy": _REPAIR_REMEDY,
        "documents": [dict(item) for item in documents[:_MAX_RESULT_DOCUMENTS]],
    }


__all__ = [
    "CorpusMutationAction",
    "CorpusMutationExecutor",
    "CorpusMutationService",
    "RetrySelector",
    "StagedCorpusSource",
    "UploadLimits",
    "validate_corpus_mutation_prepared_input",
]
