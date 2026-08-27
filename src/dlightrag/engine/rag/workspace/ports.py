# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral corpus backend composition interfaces."""

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.engine.rag.corpus.ingest_jobs import IngestJobStore
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol
from dlightrag.engine.rag.retrieval.ports import BM25Search, CorpusChunkStore, FilteredVectorSearch
from dlightrag.engine.rag.workspace.settings import RagSettings


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


@dataclass(frozen=True, slots=True)
class CorpusRuntimeModels:
    """Model callbacks required to construct one LightRAG runtime."""

    default_llm_func: Any
    embedding_func: Any
    role_llm_configs: Any


class CorpusRuntimeBinder(Protocol):
    """Construct and attach one backend-specific LightRAG runtime."""

    def create(self, *, models: CorpusRuntimeModels, settings: RagSettings) -> Any: ...

    async def attach(self, lightrag: Any) -> WorkspaceCorpusStores: ...


@dataclass(frozen=True, slots=True)
class WorkspaceCorpusBackend:
    """Coherent backend capabilities bound to one workspace."""

    workspace_id: str
    read_only: bool
    coordination: CorpusCoordination
    maintenance: CorpusMaintenanceStore
    runtime: CorpusRuntimeBinder
    ingest_jobs: IngestJobStore


__all__ = [
    "CorpusCoordination",
    "CorpusMaintenanceStore",
    "CorpusRuntimeBinder",
    "CorpusRuntimeModels",
    "CorpusSchemaError",
    "CorpusUnavailableError",
    "WorkspaceCorpusBackend",
    "WorkspaceCorpusStores",
]
