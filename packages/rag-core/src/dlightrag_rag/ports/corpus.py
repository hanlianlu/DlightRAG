# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral corpus backend composition interfaces."""

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag_rag.ports.ingest_jobs import IngestJobStore
from dlightrag_rag.ports.metadata_index import MetadataIndexProtocol
from dlightrag_rag.ports.retrieval import BM25Search, CorpusChunkStore, FilteredVectorSearch


class CorpusSchemaError(RuntimeError):
    """The deployed corpus schema is incompatible with this software revision."""


class CorpusUnavailableError(RuntimeError):
    """The configured corpus backend cannot currently be reached."""


class CorpusCoordination(Protocol):
    """Serialize workspace initialization and startup pipeline recovery."""

    def workspace_initialization(self) -> AbstractAsyncContextManager[None]: ...

    def pipeline_recovery(self) -> AbstractAsyncContextManager[None]: ...


class CorpusMaintenanceStore(Protocol):
    """Own storage-neutral workspace catalog maintenance operations."""

    async def initialize(self, *, validate_only: bool = False) -> None: ...

    async def clean_orphan_rows(self, workspace: str, *, dry_run: bool) -> int: ...

    async def delete_workspace_record(self, workspace: str) -> bool: ...

    async def list_workspace_records(self) -> tuple[dict[str, Any], ...]: ...

    async def register_workspace(
        self,
        *,
        workspace: str,
        display_name: str,
        embedding_model: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkspaceCorpusStores:
    """Retrieval and ingestion stores attached to one LightRAG runtime."""

    metadata_index: MetadataIndexProtocol
    chunks: CorpusChunkStore
    filtered_vectors: FilteredVectorSearch | None
    bm25: BM25Search | None
    bm25_languages: tuple[str, ...] = ()


class CorpusRuntimeBinder(Protocol):
    """Attach backend stores after the host has initialized LightRAG storage."""

    async def bind(self, lightrag: Any) -> WorkspaceCorpusStores: ...


@dataclass(frozen=True, slots=True)
class WorkspaceCorpusBackend:
    """Coherent backend capabilities bound to one workspace."""

    coordination: CorpusCoordination
    maintenance: CorpusMaintenanceStore
    runtime: CorpusRuntimeBinder
    ingest_jobs: IngestJobStore


class CorpusBackendFactory(Protocol):
    """Translate host settings once and create one workspace backend bundle."""

    def create(self) -> WorkspaceCorpusBackend: ...


__all__ = [
    "CorpusBackendFactory",
    "CorpusCoordination",
    "CorpusMaintenanceStore",
    "CorpusRuntimeBinder",
    "CorpusSchemaError",
    "CorpusUnavailableError",
    "WorkspaceCorpusBackend",
    "WorkspaceCorpusStores",
]
