# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus workspace lifecycle and administration over canonical ids."""

import logging
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.application.access import WorkspaceRecord
from dlightrag.application.errors import CorpusUnavailableError, StorageSchemaError
from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.rag.corpus.contracts import IngestDocument, SourceType, VisualAssetSize
from dlightrag.engine.rag.corpus.downloads import (
    LocalDownloadTarget as _EngineLocalDownloadTarget,
)
from dlightrag.engine.rag.corpus.downloads import (
    RedirectDownloadTarget as _EngineRedirectDownloadTarget,
)
from dlightrag.engine.rag.corpus.downloads import (
    SourceDownloadInvalidError as _EngineSourceDownloadInvalidError,
)
from dlightrag.engine.rag.corpus.downloads import (
    SourceDownloadNotFoundError as _EngineSourceDownloadNotFoundError,
)
from dlightrag.engine.rag.corpus.downloads import (
    SourceDownloadUnavailableError as _EngineSourceDownloadUnavailableError,
)
from dlightrag.engine.rag.corpus.ingest_jobs import JOB_STATES_WITH_RESULT, IngestJobSchemaError
from dlightrag.engine.rag.corpus.ingestion.paths import is_explicit_upload_batch_dir
from dlightrag.engine.rag.corpus.ingestion.uploads import (
    UploadTooLargeError as _EngineUploadTooLargeError,
)
from dlightrag.engine.rag.corpus.ingestion.uploads import (
    ignored_upload,
    safe_upload_basename,
    safe_upload_destination,
    upload_batch_dir,
    write_upload_stream,
)
from dlightrag.engine.rag.corpus.reset import areset_orphaned_workspace
from dlightrag.engine.rag.retrieval import MetadataFilter
from dlightrag.engine.rag.retrieval.metadata_fields import (
    MetadataValidationError as _EngineMetadataValidationError,
)
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.rag.workspace.ports import (
    CorpusMaintenanceStore,
    CorpusSchemaError,
)
from dlightrag.engine.rag.workspace.ports import (
    CorpusUnavailableError as _EngineCorpusUnavailableError,
)
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
from dlightrag.engine.rag.workspace.workspaces import (
    normalize_workspace,
    require_canonical_workspace_id,
)

from .errors import (
    LocalDownloadTarget,
    MetadataValidationError,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadTarget,
    SourceDownloadUnavailableError,
    UnsafeUploadNameError,
    UploadTooLargeError,
)
from .file_panel import (
    FilePanelCursor,
    FilePanelCursorCodec,
    FilePanelPageRequest,
    FilePanelRowPage,
)
from .metadata_search import (
    MetadataMatchRowPage,
    MetadataSearchCursor,
    MetadataSearchCursorCodec,
    MetadataSearchPage,
    MetadataSearchPageRequest,
)

logger = logging.getLogger(__name__)

_WORKSPACE_FORBIDDEN_RE = re.compile(r'[/\\<>"\']')


async def _acquire_workspace(pool: WorkspacePool, workspace: str) -> WorkspaceRag:
    """Acquire one runtime with rag-core errors translated to product types."""
    try:
        return await pool.acquire(workspace)
    except CorpusSchemaError as exc:
        raise StorageSchemaError(str(exc)) from exc
    except _EngineCorpusUnavailableError as exc:
        raise CorpusUnavailableError(str(exc)) from exc


def validate_workspace_name(name: str, *, max_length: int = 64) -> str:
    """Validate and trim a user-facing workspace name.

    The returned value is still a display label. RAG owns conversion to the
    canonical internal workspace identifier.
    """
    label = name.strip()
    if not label:
        raise ValueError("Workspace name cannot be empty")
    if len(label) > max_length:
        raise ValueError(f"Workspace name too long (max {max_length} characters)")
    if _WORKSPACE_FORBIDDEN_RE.search(label):
        raise ValueError("Workspace name contains forbidden characters")
    return label


class _CorpusContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class IngestSpec(_CorpusContractModel):
    """Transport-neutral corpus ingest source specification."""

    source_type: SourceType
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    prefix: str | None = None
    bucket: str | None = None
    s3_region: str | None = None
    s3_key: str | None = None
    url: Annotated[str, Field(min_length=1)] | None = None
    urls: list[Annotated[str, Field(min_length=1)]] | None = None
    filename: str | None = None
    source_uri: str | None = None
    source_uris: list[str] | None = None
    download_uri: str | None = None
    download_uris: list[str] | None = None
    documents: list[IngestDocument] | None = None
    retain_source_file: bool | None = None
    replace: bool | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_source_fields(self) -> IngestSpec:
        self._validate_download_fields()
        if self.source_type == "local":
            if self.documents is not None:
                if self.path:
                    raise ValueError("'path' and 'documents' are mutually exclusive")
                _require_document_field(self.documents, "path", source_type=self.source_type)
                return self
            if not self.path:
                raise ValueError("'path' is required for local ingestion")
        elif self.source_type == "azure_blob":
            if not self.container_name:
                raise ValueError("'container_name' is required for azure_blob")
            if self.documents is not None:
                if self.blob_path or self.prefix is not None:
                    raise ValueError("'blob_path'/'prefix' and 'documents' are mutually exclusive")
                _require_document_field(self.documents, "key", source_type=self.source_type)
                return self
            if self.blob_path and self.prefix is not None:
                raise ValueError("'blob_path' and 'prefix' are mutually exclusive")
        elif self.source_type == "s3":
            if not self.bucket:
                raise ValueError("'bucket' is required for s3")
            if self.documents is not None:
                if self.s3_key or self.prefix is not None:
                    raise ValueError("'s3_key'/'prefix' and 'documents' are mutually exclusive")
                _require_document_field(self.documents, "key", source_type=self.source_type)
                return self
            if self.s3_key and self.prefix is not None:
                raise ValueError("'s3_key' and 'prefix' are mutually exclusive")
        elif self.source_type == "url":
            if self.documents is not None:
                if any(
                    value is not None
                    for value in (
                        self.url,
                        self.urls,
                        self.filename,
                        self.source_uri,
                        self.source_uris,
                    )
                ):
                    raise ValueError(
                        "'url'/'urls'/'filename'/'source_uri' and 'documents' are mutually exclusive"
                    )
                _require_document_field(self.documents, "url", source_type=self.source_type)
                return self
            url_count = int(self.url is not None) + len(self.urls or [])
            if url_count == 0:
                raise ValueError("'url' or 'urls' is required for url ingestion")
            if self.url and self.urls is not None:
                raise ValueError("'url' and 'urls' are mutually exclusive")
            if self.filename and url_count != 1:
                raise ValueError("'filename' can only be used with a single url")
            if self.source_uri and self.source_uris is not None:
                raise ValueError("'source_uri' and 'source_uris' are mutually exclusive")
            if self.source_uri and url_count != 1:
                raise ValueError("'source_uri' can only be used with a single url")
            if self.source_uris is not None and len(self.source_uris) != url_count:
                raise ValueError("'source_uris' must match the number of urls")
        return self

    def _validate_download_fields(self) -> None:
        top_level_present = self.download_uri is not None or self.download_uris is not None
        document_values = [
            document.download_uri
            for document in self.documents or []
            if document.download_uri is not None
        ]
        if self.source_type != "url":
            if top_level_present or document_values:
                raise ValueError("download_uri fields are only valid for URL ingestion")
            return
        if self.documents is not None:
            if top_level_present:
                raise ValueError(
                    "'download_uri'/'download_uris' and 'documents' are mutually exclusive"
                )
            return
        url_count = int(self.url is not None) + len(self.urls or [])
        if self.download_uri is not None and self.download_uris is not None:
            raise ValueError("'download_uri' and 'download_uris' are mutually exclusive")
        if self.download_uri is not None and url_count != 1:
            raise ValueError("'download_uri' can only be used with a single url")
        if self.download_uris is not None and len(self.download_uris) != url_count:
            raise ValueError("'download_uris' must match the number of urls")


def ingest_spec_from_payload(payload: Any) -> IngestSpec:
    """Build the strict corpus ingest contract from one wire or CLI object."""
    values: dict[str, Any] = {}
    for name in IngestSpec.model_fields:
        value = _payload_value(payload, name)
        if value is not None:
            values[name] = _json_safe(value)
    return IngestSpec.model_validate(values)


def ingest_kwargs_from_spec(spec: IngestSpec) -> dict[str, Any]:
    """Project one validated ingest contract into WorkspaceRag keyword values."""
    data = spec.model_dump(mode="json", exclude={"source_type"}, exclude_none=True)
    source_fields: dict[SourceType, tuple[str, ...]] = {
        "local": ("path",),
        "azure_blob": ("container_name", "blob_path", "prefix"),
        "s3": ("bucket", "s3_region", "s3_key", "prefix"),
        "url": (
            "url",
            "urls",
            "filename",
            "source_uri",
            "source_uris",
            "download_uri",
            "download_uris",
        ),
    }
    common = ("replace", "retain_source_file", "title", "author", "metadata", "documents")
    allowed = set((*source_fields[spec.source_type], *common))
    return {name: value for name, value in data.items() if name in allowed}


def managed_local_ingest_path(
    *,
    source_type: str,
    path: str | None,
    input_dir: Path,
    workspace: str,
) -> str | None:
    """Constrain a transport-supplied local path to one workspace input root."""
    if source_type != "local" or not path:
        return path

    root = (input_dir / normalize_workspace(workspace)).resolve()
    if "\0" in path or path.startswith(("~", "/", "\\")):
        raise ValueError(
            "local ingest paths from REST/MCP must be relative to input_dir/<workspace>"
        )
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(
            "local ingest paths from REST/MCP must be relative to input_dir/<workspace>"
        )

    parts = tuple(part for part in re.split(r"[\\/]+", path) if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(
            "local ingest paths from REST/MCP must be relative to input_dir/<workspace>"
        )
    resolved = root.joinpath(*parts).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            "local ingest paths from REST/MCP must be under input_dir/<workspace>"
        ) from None
    return str(resolved)


def managed_local_ingest_documents(
    *,
    source_type: str,
    documents: list[IngestDocument] | None,
    input_dir: Path,
    workspace: str,
) -> list[IngestDocument] | None:
    """Constrain every local manifest document to one workspace input root."""
    if source_type != "local" or documents is None:
        return documents
    return [
        document.model_copy(
            update={
                "path": managed_local_ingest_path(
                    source_type=source_type,
                    path=document.path,
                    input_dir=input_dir,
                    workspace=workspace,
                )
            }
        )
        for document in documents
    ]


class CorpusIngestError(RuntimeError):
    """A caller-awaited ingest job could not produce a successful result."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class FilePanelSnapshot(TypedDict):
    files: list[dict[str, Any]]
    pipeline_status: dict[str, Any]
    next_cursor: FilePanelCursor | None
    fetched_rows: int


class CorpusResetResult(TypedDict):
    workspaces: dict[str, dict[str, Any]]
    total_errors: int


@dataclass(frozen=True, slots=True)
class CorpusAdminSettings:
    default_workspace_id: str
    default_display_name: str
    default_embedding_model: str
    input_root: Path | str
    ingest_timeout_seconds: float | None
    read_only: bool

    def __post_init__(self) -> None:
        if self.ingest_timeout_seconds is not None and self.ingest_timeout_seconds < 0:
            raise ValueError("ingest timeout must be non-negative")


class IngestJobs(Protocol):
    async def start_recovery(self) -> None: ...

    async def start_job(
        self,
        workspace: str,
        source_type: SourceType,
        *,
        cleanup_paths: str | Path | Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    async def await_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None: ...

    async def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    async def cancel_job(self, job_id: str, *, workspace: str) -> bool: ...

    def has_active_workspace_job(self, workspace: str) -> bool: ...

    async def cancel_for_workspace(self, workspace: str) -> int: ...

    async def attach_reset_result(
        self,
        *,
        workspace: str,
        result: dict[str, Any],
        dry_run: bool,
    ) -> None: ...

    async def close(self) -> None: ...


class FilePanelStore(Protocol):
    async def list_processed_files(
        self,
        workspace: str,
        *,
        page: FilePanelPageRequest,
    ) -> FilePanelRowPage: ...


class MetadataSearchStore(Protocol):
    async def search_metadata_page(
        self,
        workspace: str,
        filters: MetadataFilter,
        *,
        page: MetadataSearchPageRequest,
    ) -> MetadataMatchRowPage: ...


class UploadReader(Protocol):
    """Minimal async reader contract for one streamed upload."""

    async def read(self, size: int = -1) -> bytes: ...


class SourceDownloadPreparer(Protocol):
    async def prepare(self, document_id: str) -> object: ...


type SourceDownloadFactory = Callable[[str], SourceDownloadPreparer]


class CorpusAdmin:
    """Own corpus administration without transport or storage implementations."""

    def __init__(
        self,
        *,
        settings: CorpusAdminSettings,
        pool: WorkspacePool,
        maintenance: CorpusMaintenanceStore,
        ingest_jobs: IngestJobs,
        file_panel: FilePanelStore,
        metadata_search: MetadataSearchStore,
        source_download_for: SourceDownloadFactory,
        file_panel_cursor_secret: bytes | None = None,
        metadata_search_cursor_secret: bytes | None = None,
    ) -> None:
        self._settings = settings
        self._pool = pool
        self._maintenance = maintenance
        self._ingest_jobs = ingest_jobs
        self._file_panel = file_panel
        self._metadata_search = metadata_search
        self._source_download_for = source_download_for
        self._file_panel_cursor_codec = FilePanelCursorCodec(file_panel_cursor_secret)
        self._metadata_search_cursor_codec = MetadataSearchCursorCodec(
            metadata_search_cursor_secret
        )

    async def initialize(self) -> None:
        default_workspace = require_canonical_workspace_id(self._settings.default_workspace_id)
        try:
            await self._maintenance.initialize(validate_only=self._settings.read_only)
            if not self._settings.read_only:
                await self._maintenance.register_workspace(
                    workspace=default_workspace,
                    display_name=self._settings.default_display_name,
                    embedding_model=self._settings.default_embedding_model,
                )
        except CorpusSchemaError as exc:
            raise StorageSchemaError(str(exc)) from exc

    async def start_recovery(self) -> None:
        if self._settings.read_only:
            return
        try:
            await self._ingest_jobs.start_recovery()
        except IngestJobSchemaError as exc:
            raise StorageSchemaError(str(exc)) from exc

    async def aclose(self) -> None:
        await self._ingest_jobs.close()

    async def alist_workspace_records(self) -> list[WorkspaceRecord]:
        """Return the canonical workspace catalog with a default fallback."""
        default_workspace = require_canonical_workspace_id(self._settings.default_workspace_id)
        default_record: WorkspaceRecord = {
            "workspace": default_workspace,
            "display_name": self._settings.default_display_name,
            "embedding_model": self._settings.default_embedding_model,
            "created_at": None,
            "updated_at": None,
        }
        try:
            rows = await self._maintenance.list_workspace_records()
            records = [_workspace_record(row) for row in rows]
        except Exception as exc:
            logger.warning("Failed to list workspaces from registry: %s", exc)
            return [default_record]
        if not records:
            return [default_record]
        if all(record["workspace"] != default_record["workspace"] for record in records):
            records.append(default_record)
        return records

    async def list_workspaces(self) -> list[str]:
        return [record["workspace"] for record in await self.alist_workspace_records()]

    @property
    def file_panel_cursor_codec(self) -> FilePanelCursorCodec:
        """Return the cursor codec shared with the browser Files adapter."""
        return self._file_panel_cursor_codec

    @property
    def metadata_search_cursor_codec(self) -> MetadataSearchCursorCodec:
        """Return the cursor codec shared with the REST metadata adapter."""
        return self._metadata_search_cursor_codec

    async def workspace_exists(self, workspace_id: str) -> bool:
        """Perform one bounded catalog lookup without warming a workspace runtime."""
        workspace = require_canonical_workspace_id(workspace_id)
        if workspace == require_canonical_workspace_id(self._settings.default_workspace_id):
            return True
        return await self._maintenance.workspace_exists(workspace)

    async def create_workspace(
        self,
        workspace_id: str,
        *,
        display_name: str | None = None,
    ) -> None:
        self._require_writer("workspace creation")
        workspace = require_canonical_workspace_id(workspace_id)
        runtime = await _acquire_workspace(self._pool, workspace)
        await runtime.aregister_workspace(display_name=display_name)

    async def ingest(self, workspace_id: str, spec: IngestSpec) -> dict[str, Any]:
        job = await self.start_ingest_job(workspace_id, spec)
        row = await self._ingest_jobs.await_job(
            str(job["job_id"]),
            timeout=self._settings.ingest_timeout_seconds,
        )
        if row is None:
            raise CorpusIngestError(f"Ingest job disappeared: {job['job_id']}")
        status = str(row.get("status") or "")
        if status in JOB_STATES_WITH_RESULT:
            result = row.get("result")
            return result if isinstance(result, dict) else {}
        if status == "failed":
            raw_errors = row.get("errors")
            errors = raw_errors if isinstance(raw_errors, list) else []
            raise CorpusIngestError(
                "; ".join(str(error) for error in errors) or "Ingest job failed"
            )
        return row

    async def start_ingest_job(
        self,
        workspace_id: str,
        spec: IngestSpec,
    ) -> dict[str, Any]:
        self._require_writer("ingestion")
        workspace = require_canonical_workspace_id(workspace_id)
        try:
            return await self._ingest_jobs.start_job(
                workspace,
                spec.source_type,
                cleanup_paths=_cleanup_paths_for_local_ingest(spec),
                **ingest_kwargs_from_spec(spec),
            )
        except IngestJobSchemaError as exc:
            raise StorageSchemaError(str(exc)) from exc

    async def get_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        self._require_writer("ingest job access")
        return await self._ingest_jobs.get_job(job_id)

    async def cancel_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        self._require_writer("ingest job cancellation")
        job = await self._ingest_jobs.get_job(job_id)
        if job is None:
            return None
        workspace = require_canonical_workspace_id(str(job.get("workspace") or ""))
        await self._ingest_jobs.cancel_job(job_id, workspace=workspace)
        return await self._ingest_jobs.get_job(job_id)

    async def file_panel_snapshot(
        self,
        workspace_id: str,
        *,
        page: FilePanelPageRequest | None = None,
    ) -> FilePanelSnapshot:
        workspace = require_canonical_workspace_id(workspace_id)
        requested = page or FilePanelPageRequest()
        if requested.cursor is not None and requested.cursor.workspace != workspace:
            raise ValueError("file-panel cursor belongs to another workspace")
        result = await self._file_panel.list_processed_files(workspace, page=requested)
        next_cursor = None
        if result.has_more:
            if not result.items:
                raise RuntimeError("file-panel store reported more rows after an empty page")
            last = result.items[-1]
            next_cursor = FilePanelCursor(
                workspace=workspace,
                updated_at=last.updated_at,
                doc_id=last.doc_id,
            )
        loaded_status = await self._pool.get_pipeline_status(workspace)
        if loaded_status is not None:
            pipeline_status = loaded_status
        elif self._ingest_jobs.has_active_workspace_job(workspace):
            pipeline_status = {
                "busy": True,
                "pending_enqueues": 0,
                "latest_message": "Starting ingest...",
            }
        else:
            pipeline_status = {
                "busy": False,
                "pending_enqueues": 0,
                "latest_message": "",
            }
        return {
            "files": [item.presentation() for item in result.items],
            "pipeline_status": pipeline_status,
            "next_cursor": next_cursor,
            "fetched_rows": result.fetched_rows,
        }

    async def prepare_source_download(
        self,
        workspace_id: str,
        document_id: str,
    ) -> SourceDownloadTarget:
        """Resolve one source download into an Application-owned target."""
        workspace = require_canonical_workspace_id(workspace_id)
        try:
            target = await self._source_download_for(workspace).prepare(document_id)
        except _EngineSourceDownloadInvalidError as exc:
            raise SourceDownloadInvalidError(str(exc)) from exc
        except _EngineSourceDownloadNotFoundError as exc:
            raise SourceDownloadNotFoundError(str(exc)) from exc
        except _EngineSourceDownloadUnavailableError as exc:
            raise SourceDownloadUnavailableError(str(exc)) from exc
        if isinstance(target, _EngineLocalDownloadTarget):
            return LocalDownloadTarget(
                path=target.path,
                media_type=target.media_type,
                filename=target.filename,
            )
        if isinstance(target, _EngineRedirectDownloadTarget):
            return RedirectDownloadTarget(url=target.url)
        raise SourceDownloadInvalidError("Source download target is invalid")

    async def stage_upload_stream(
        self,
        workspace_id: str,
        *,
        filename: str,
        reader: UploadReader,
        max_bytes: int,
    ) -> tuple[Path, str]:
        """Stage one streamed upload under the workspace input root.

        Returns the saved path and the safe basename; raises product errors for
        unsafe names and oversized payloads.
        """
        workspace = require_canonical_workspace_id(workspace_id)
        try:
            safe_name = safe_upload_basename(filename)
        except ValueError:
            raise UnsafeUploadNameError(f"Unsafe filename: {filename!r}") from None
        target_dir = Path(self._settings.input_root) / workspace
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        try:
            await write_upload_stream(reader, target_path, max_bytes=max_bytes)
        except _EngineUploadTooLargeError as exc:
            raise UploadTooLargeError(str(exc)) from exc
        return target_path, safe_name

    async def stage_upload_batch(
        self,
        workspace_id: str,
        files: Sequence[tuple[str, UploadReader]],
        *,
        per_file_max_bytes: int,
        batch_max_bytes: int,
    ) -> tuple[Path, list[Path]]:
        """Stage a multi-file upload batch and return its directory and saved paths.

        Ignored OS-junk names are skipped. Oversized payloads raise
        ``UploadTooLargeError`` after removing the batch directory.
        """
        workspace = require_canonical_workspace_id(workspace_id)
        upload_dir = upload_batch_dir(Path(self._settings.input_root) / workspace)
        saved_paths: list[Path] = []
        bytes_written = 0
        try:
            for filename, reader in files:
                if not filename or ignored_upload(filename):
                    continue
                try:
                    dest = safe_upload_destination(upload_dir, filename)
                except ValueError as exc:
                    raise UnsafeUploadNameError(f"Unsafe filename: {filename!r}") from exc
                bytes_written = await write_upload_stream(
                    reader,
                    dest,
                    max_bytes=min(batch_max_bytes, bytes_written + per_file_max_bytes),
                    bytes_written=bytes_written,
                )
                saved_paths.append(dest)
        except _EngineUploadTooLargeError as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise UploadTooLargeError(str(exc)) from exc
        except BaseException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        return upload_dir, saved_paths

    async def list_ingested_files(self, workspace_id: str) -> list[dict[str, Any]]:
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.alist_ingested_files()

    async def get_pipeline_status(self, workspace_id: str) -> dict[str, Any]:
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.aget_pipeline_status()

    async def delete_files(
        self,
        workspace_id: str,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        self._require_writer("file deletion")
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.adelete_files(
            file_paths=file_paths,
            filenames=filenames,
            dry_run=dry_run,
        )

    async def list_failed_docs(self, workspace_id: str) -> list[dict[str, Any]]:
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.alist_failed_docs()

    async def get_visual_asset(
        self,
        workspace_id: str,
        chunk_id: str,
        *,
        size: VisualAssetSize = "full",
    ) -> Any:
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.aget_visual_asset(chunk_id, size=size)

    async def retry_failed_docs(self, workspace_id: str) -> dict[str, Any]:
        self._require_writer("failed document retry")
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.aretry_failed_docs()

    async def get_metadata(self, workspace_id: str, document_id: str) -> dict[str, Any]:
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        return await runtime.aget_metadata(document_id)

    async def update_metadata(
        self,
        workspace_id: str,
        document_id: str,
        data: dict[str, Any],
    ) -> None:
        self._require_writer("metadata update")
        runtime = await _acquire_workspace(self._pool, require_canonical_workspace_id(workspace_id))
        try:
            await runtime.aupdate_metadata(document_id, data)
        except _EngineMetadataValidationError as exc:
            raise MetadataValidationError(str(exc)) from exc

    async def search_metadata(
        self,
        workspace_id: str,
        filters: MetadataFilter,
        *,
        page: MetadataSearchPageRequest | None = None,
    ) -> MetadataSearchPage:
        """Return one bounded matching document-id page without warming a runtime."""
        workspace = require_canonical_workspace_id(workspace_id)
        requested = page or MetadataSearchPageRequest()
        if requested.cursor is not None and requested.cursor.workspace != workspace:
            raise ValueError("metadata-search cursor belongs to another workspace")
        result = await self._metadata_search.search_metadata_page(
            workspace,
            filters,
            page=requested,
        )
        next_cursor = None
        if result.has_more:
            if not result.document_ids:
                raise RuntimeError("metadata-search store reported more rows after an empty page")
            next_cursor = MetadataSearchCursor(
                workspace=workspace,
                after_doc_id=result.document_ids[-1],
                mode=result.mode,
            )
        return MetadataSearchPage(
            document_ids=result.document_ids,
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    async def reset(
        self,
        *,
        workspace_ids: Sequence[str],
        keep_files: bool = False,
        dry_run: bool = False,
    ) -> CorpusResetResult:
        """Reset an explicit non-empty set of authorized canonical workspaces."""
        workspaces = _require_workspace_scope(workspace_ids)
        self._require_writer("workspace reset")
        known = set(await self.list_workspaces())
        results: dict[str, Any] = {}
        total_errors = 0

        for workspace in workspaces:
            if workspace not in known and not await self._pool.is_loaded(workspace):
                result = await self._reset_orphan(
                    workspace,
                    keep_files=keep_files,
                    dry_run=dry_run,
                )
            else:
                result = await self._reset_loaded(
                    workspace,
                    keep_files=keep_files,
                    dry_run=dry_run,
                )
            results[workspace] = result
            total_errors += len(result.get("errors", ()))
            if "error" in result:
                total_errors += 1

        return {"workspaces": results, "total_errors": total_errors}

    async def _reset_orphan(
        self,
        workspace: str,
        *,
        keep_files: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        cancelled = 0 if dry_run else await self._ingest_jobs.cancel_for_workspace(workspace)
        result = await areset_orphaned_workspace(
            workspace,
            maintenance=self._maintenance,
            keep_files=keep_files,
            dry_run=dry_run,
            input_dir=str(Path(self._settings.input_root)),
        )
        await self._ingest_jobs.attach_reset_result(
            workspace=workspace,
            result=result,
            dry_run=dry_run,
        )
        result["ingest_jobs_cancelled"] = cancelled
        return result

    async def _reset_loaded(
        self,
        workspace: str,
        *,
        keep_files: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        cancelled = 0 if dry_run else await self._ingest_jobs.cancel_for_workspace(workspace)
        try:
            runtime = await _acquire_workspace(self._pool, workspace)
            result = await runtime.areset(keep_files=keep_files, dry_run=dry_run)
            result["ingest_jobs_cancelled"] = cancelled
            await self._ingest_jobs.attach_reset_result(
                workspace=workspace,
                result=result,
                dry_run=dry_run,
            )
        except Exception as exc:
            logger.warning(
                "Failed to reset workspace '%s': %s",
                safe_log_text(workspace),
                safe_log_text(exc),
            )
            result = {
                "error": "workspace reset failed",
                "ingest_jobs_cancelled": cancelled,
            }
        if not dry_run:
            try:
                await self._pool.evict(workspace)
            except Exception:
                logger.warning(
                    "Failed to close workspace '%s'", safe_log_text(workspace), exc_info=True
                )
        return result

    def _require_writer(self, operation: str) -> None:
        if self._settings.read_only:
            raise PermissionError(f"{operation} requires a writer service role")


def _require_workspace_scope(workspace_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(workspace_ids, (str, bytes)) or not workspace_ids:
        raise ValueError("at least one canonical workspace id is required")
    workspaces: list[str] = []
    for workspace in workspace_ids:
        try:
            canonical = require_canonical_workspace_id(workspace)
        except ValueError as exc:
            raise ValueError("reset requires canonical workspace ids") from exc
        if canonical not in workspaces:
            workspaces.append(canonical)
    return tuple(workspaces)


def _payload_value(payload: Any, name: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _require_document_field(
    documents: list[IngestDocument],
    field_name: Literal["path", "key", "url"],
    *,
    source_type: SourceType,
) -> None:
    if not documents:
        raise ValueError("'documents' must contain at least one document")
    for index, document in enumerate(documents):
        if not getattr(document, field_name):
            raise ValueError(
                f"'documents[{index}].{field_name}' is required for {source_type} ingestion"
            )


def _cleanup_paths_for_local_ingest(spec: IngestSpec) -> list[str]:
    if spec.source_type != "local" or not spec.path:
        return []
    path = Path(spec.path).expanduser()
    return [str(path)] if is_explicit_upload_batch_dir(path) else []


def _workspace_record(row: dict[str, Any]) -> WorkspaceRecord:
    workspace = require_canonical_workspace_id(str(row.get("workspace") or ""))
    return {
        "workspace": workspace,
        "display_name": str(row.get("display_name") or workspace),
        "embedding_model": str(row.get("embedding_model") or ""),
        "created_at": _iso_or_none(row.get("created_at")),
        "updated_at": _iso_or_none(row.get("updated_at")),
    }


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat() if callable(isoformat) else value)


__all__ = [
    "CorpusAdmin",
    "CorpusAdminSettings",
    "CorpusIngestError",
    "CorpusResetResult",
    "FilePanelStore",
    "FilePanelSnapshot",
    "IngestSpec",
    "IngestJobs",
    "SourceDownloadFactory",
    "SourceDownloadPreparer",
    "UploadReader",
    "ingest_kwargs_from_spec",
    "ingest_spec_from_payload",
    "managed_local_ingest_documents",
    "managed_local_ingest_path",
    "safe_upload_basename",
    "validate_workspace_name",
]
