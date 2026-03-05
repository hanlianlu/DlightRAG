# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared RAG Service Pool - Per-process singleton with async lazy initialization.

Provides a shared RAGService instance per process, avoiding the overhead of
creating new instances per task. The service is stateless (connects to PG/storage),
so it can safely be shared across concurrent async tasks within one event loop.

Thread-safety note:
    This module uses ``asyncio.Lock`` and module-level globals. It is safe for
    **multi-process** deployments (e.g. ``gunicorn --workers N``) but is NOT
    thread-safe for **multi-threaded** deployments (``gunicorn --threads M``).
    Deploy with async workers (uvicorn / UvicornWorker) and scale via processes
    or Kubernetes horizontal pod autoscaling, not via threads.

Usage:
    from dlightrag.pool import get_shared_rag_service

    rag_service = await get_shared_rag_service()
    result = await rag_service.aretrieve(query, ...)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

from dlightrag.service import RAGService

logger = logging.getLogger(__name__)

# Per-worker singleton
_shared_rag_service: RAGService | None = None
_rag_init_lock: asyncio.Lock | None = None

# RAG readiness state for health checks and auto-recovery
_rag_ready: bool = False
_rag_last_error: str | None = None
_rag_last_error_ts: float | None = None
_rag_retry_after: float = 30.0
_RAG_MAX_RETRY_INTERVAL: float = 300.0


class RAGServiceUnavailableError(Exception):
    """Raised when the RAG service is not ready."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


def _get_init_lock() -> asyncio.Lock:
    """Get or create the initialization lock (must be created within event loop).

    # hllyu init lock
    """
    global _rag_init_lock
    if _rag_init_lock is None:
        _rag_init_lock = asyncio.Lock()
    return _rag_init_lock


async def get_shared_rag_service(
    config: DlightragConfig | None = None,
    enable_vlm: bool = True,
    cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    url_transformer: Callable[[str], str] | None = None,
) -> RAGService:
    """Get or initialize the shared RAG service for this worker.

    Async-task-safe via asyncio.Lock (NOT thread-safe). First caller
    initializes; subsequent calls return the cached instance immediately.

    If initialization previously failed, attempts reinitialize with
    exponential backoff (30s -> 60s -> 120s -> 300s max).

    Args:
        config: Optional DlightragConfig. If None, uses get_config() singleton.
        enable_vlm: Whether to enable vision model support.
        cancel_checker: Optional cancellation callback.
        url_transformer: Optional URL generation callback.

    Returns:
        Initialized RAGService instance shared across all tasks.
    """
    global _shared_rag_service, _rag_ready, _rag_last_error, _rag_last_error_ts, _rag_retry_after

    # Fast path: already initialized and ready
    if _shared_rag_service is not None and _rag_ready:
        return _shared_rag_service

    # Check if we should attempt reinitialization (exponential backoff)
    if _rag_last_error_ts is not None:
        time_since_error = time.time() - _rag_last_error_ts
        if time_since_error < _rag_retry_after:
            raise RAGServiceUnavailableError(detail=_rag_last_error)

    # Slow path: need to initialize (with lock to prevent concurrent init)
    lock = _get_init_lock()
    async with lock:
        # Double-check after acquiring lock
        if _shared_rag_service is not None and _rag_ready:
            return _shared_rag_service

        # Check backoff again after acquiring lock
        if _rag_last_error_ts is not None:
            time_since_error = time.time() - _rag_last_error_ts
            if time_since_error < _rag_retry_after:
                raise RAGServiceUnavailableError(detail=_rag_last_error)

        logger.info("Initializing shared RAG service...")

        try:
            _shared_rag_service = await RAGService.create(
                config=config,
                enable_vlm=enable_vlm,
                cancel_checker=cancel_checker,
                url_transformer=url_transformer,
            )

            # Success - mark as ready and reset retry state
            _rag_ready = True
            _rag_last_error = None
            _rag_last_error_ts = None
            _rag_retry_after = 30.0

            logger.info("Shared RAG service initialized successfully")
            return _shared_rag_service

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            _rag_ready = False
            _rag_last_error = error_msg
            _rag_last_error_ts = time.time()
            _rag_retry_after = min(_rag_retry_after * 2, _RAG_MAX_RETRY_INTERVAL)

            logger.error(
                f"RAG service initialization failed: {error_msg}. Next retry in {_rag_retry_after}s"
            )
            raise RAGServiceUnavailableError(detail=error_msg) from e


async def close_shared_rag_service() -> None:
    """Close the shared RAG service during shutdown."""
    global _shared_rag_service, _rag_ready

    if _shared_rag_service is not None:
        logger.info("Closing shared RAG service...")
        try:
            await _shared_rag_service.close()
        except Exception:
            logger.warning("Failed to close shared RAG service", exc_info=True)
        finally:
            _shared_rag_service = None
            _rag_ready = False


def is_rag_service_initialized() -> bool:
    """Check if RAG service has been initialized and is ready."""
    return _shared_rag_service is not None and _rag_ready


def get_rag_error_info() -> dict[str, str | float | None]:
    """Get RAG initialization error info for health checks."""
    return {
        "last_error": _rag_last_error,
        "timestamp": _rag_last_error_ts,
        "retry_after": _rag_retry_after,
    }


def reset_shared_rag_service() -> None:
    """Reset the shared service state. Useful for testing."""
    global _shared_rag_service, _rag_init_lock, _rag_ready
    global _rag_last_error, _rag_last_error_ts, _rag_retry_after
    _shared_rag_service = None
    _rag_init_lock = None
    _rag_ready = False
    _rag_last_error = None
    _rag_last_error_ts = None
    _rag_retry_after = 30.0


# ---------------------------------------------------------------------------
# WorkspacePool — manages multiple RAGService instances keyed by workspace
# ---------------------------------------------------------------------------

_workspace_services: dict[str, RAGService] = {}
_workspace_lock: asyncio.Lock | None = None


def _get_workspace_lock() -> asyncio.Lock:
    """Get or create the workspace pool lock (must be created within event loop)."""
    global _workspace_lock
    if _workspace_lock is None:
        _workspace_lock = asyncio.Lock()
    return _workspace_lock


async def get_workspace_service(
    workspace: str,
    base_config: DlightragConfig | None = None,
    enable_vlm: bool = True,
    cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    url_transformer: Callable[[str], str] | None = None,
) -> RAGService:
    """Get or create a RAGService for a specific workspace.

    Services are cached per workspace name. All services share the same
    PG connection pool (LightRAG's ClientManager is a process-level singleton).
    """
    if workspace in _workspace_services:
        return _workspace_services[workspace]

    lock = _get_workspace_lock()
    async with lock:
        # Double-check after acquiring lock
        if workspace in _workspace_services:
            return _workspace_services[workspace]

        from dlightrag.config import get_config

        config = base_config or get_config()
        ws_config = config.model_copy(update={"workspace": workspace})
        svc = await RAGService.create(
            config=ws_config,
            enable_vlm=enable_vlm,
            cancel_checker=cancel_checker,
            url_transformer=url_transformer,
        )
        _workspace_services[workspace] = svc
        logger.info("Created RAGService for workspace '%s'", workspace)
        return svc


async def list_available_workspaces(
    base_config: DlightragConfig | None = None,
) -> list[str]:
    """Discover available workspaces based on storage backend.

    - PG backends: query dlightrag_file_hashes table
    - Filesystem backends (Json/Nano/NetworkX): scan working_dir subdirectories
    - Other backends: return cached workspace names from pool + default
    """
    from dlightrag.config import get_config

    config = base_config or get_config()

    # PG backends: query hash table
    if config.kv_storage.startswith("PG"):
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_user,
                password=config.postgres_password,
                database=config.postgres_database,
            )
            try:
                rows = await conn.fetch(
                    "SELECT DISTINCT workspace FROM dlightrag_file_hashes ORDER BY workspace"
                )
                workspaces = [row["workspace"] for row in rows]
                return workspaces if workspaces else [config.workspace]
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("Failed to list workspaces from PG: %s", exc)
            return [config.workspace]

    # Filesystem backends: scan working_dir for workspace subdirectories
    _FS_BACKENDS = {"JsonKVStorage", "JsonDocStatusStorage", "NanoVectorDBStorage",
                    "NetworkXStorage", "FaissVectorDBStorage"}
    if config.kv_storage in _FS_BACKENDS or config.vector_storage in _FS_BACKENDS:
        return _discover_filesystem_workspaces(config)

    # Fallback: return cached workspace names + default
    cached = list(_workspace_services.keys())
    if config.workspace not in cached:
        cached.append(config.workspace)
    return sorted(cached)


def _discover_filesystem_workspaces(config: DlightragConfig) -> list[str]:
    """Scan working_dir for subdirectories containing LightRAG data files."""
    working_dir = config.working_dir_path
    if not working_dir.exists():
        return [config.workspace]

    workspaces = []
    for entry in working_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Check for LightRAG data files as evidence of workspace usage
        if (any(entry.glob("kv_store_*.json"))
                or any(entry.glob("vdb_*.json"))
                or any(entry.glob("graph_*.graphml"))
                or any(entry.glob("file_content_hashes.json"))):
            workspaces.append(entry.name)

    return sorted(workspaces) if workspaces else [config.workspace]


async def close_workspace_services() -> None:
    """Close all workspace-specific RAGService instances."""
    global _workspace_services
    for ws, svc in _workspace_services.items():
        try:
            await svc.close()
        except Exception:
            logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
    _workspace_services = {}


def reset_workspace_pool() -> None:
    """Reset workspace pool state. Useful for testing."""
    global _workspace_services, _workspace_lock
    _workspace_services = {}
    _workspace_lock = None


__all__ = [
    "RAGServiceUnavailableError",
    "close_shared_rag_service",
    "close_workspace_services",
    "get_rag_error_info",
    "get_shared_rag_service",
    "get_workspace_service",
    "is_rag_service_initialized",
    "list_available_workspaces",
    "reset_shared_rag_service",
    "reset_workspace_pool",
]
