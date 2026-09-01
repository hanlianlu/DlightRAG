# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable ingest-job persistence interface and lifecycle policy."""

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

# How often the ingest coordinator re-checks a workspace promotion fence while
# a job waits for the cutover to finish. Jobs stay durably queued the whole
# time and resume automatically once the fence is gone.
FENCE_POLL_SECONDS = 2.0

JOB_RETENTION_SECONDS = 7 * 24 * 3600
JOB_LEASE_SECONDS = 300
JOB_HEARTBEAT_SECONDS = 60
JOB_ORPHAN_AFTER_SECONDS = 12 * JOB_LEASE_SECONDS
JOB_ABANDONED_ERROR = "ingest job abandoned after process exit"
JOB_STATES_WITH_RESULT = ("succeeded", "partial")


class IngestJobSchemaError(RuntimeError):
    """The durable ingest-job schema is incompatible with this revision."""


class IngestJobStore(Protocol):
    async def initialize(self) -> None: ...

    async def create(
        self,
        *,
        job_id: str,
        workspace: str,
        source_type: str,
        request: dict[str, Any],
    ) -> None: ...

    async def claim_running(self, job_id: str, *, lease_owner: str, lease_seconds: int) -> bool: ...

    async def release_running(self, job_id: str, *, lease_owner: str) -> bool:
        """Durably return one running job to queued so another claim can retry."""
        ...

    async def touch_queued(self, job_id: str) -> bool:
        """Refresh one still-queued job's liveness; false if gone or claimed."""
        ...

    async def cancel_queued(self, job_id: str, *, error: str) -> bool:
        """Terminally fail one still-queued job (explicit user cancellation)."""
        ...

    async def is_workspace_fenced(self, workspace: str) -> bool:
        """Report whether a promotion write fence currently blocks writes."""
        ...

    def workspace_write_gate(self, workspace: str) -> AbstractAsyncContextManager[None]:
        """Gate one ingest run behind the promotion fence and drain protocol."""
        ...

    async def heartbeat(self, job_id: str, *, lease_owner: str, lease_seconds: int) -> bool: ...

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        chunk_delta: int,
        current_window: int,
        errors: list[str],
        lease_owner: str,
        lease_seconds: int,
    ) -> bool: ...

    async def finish(self, job_id: str, *, result: dict[str, Any], lease_owner: str) -> bool: ...

    async def finish_failed_retry(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        lease_owner: str,
    ) -> bool: ...

    async def fail(self, job_id: str, *, error: str, lease_owner: str) -> bool: ...

    async def get(self, job_id: str) -> dict[str, Any] | None: ...

    async def get_active_for_workspace(
        self,
        workspace: str,
        *,
        source_type: str,
    ) -> dict[str, Any] | None: ...

    async def list_recoverable(self) -> list[dict[str, Any]]: ...

    async def prune(self) -> dict[str, int]: ...

    async def delete_for_workspace(self, workspace: str) -> int: ...


__all__ = [
    "IngestJobSchemaError",
    "IngestJobStore",
    "JOB_ABANDONED_ERROR",
    "JOB_HEARTBEAT_SECONDS",
    "JOB_LEASE_SECONDS",
    "JOB_ORPHAN_AFTER_SECONDS",
    "JOB_RETENTION_SECONDS",
    "JOB_STATES_WITH_RESULT",
    "FENCE_POLL_SECONDS",
]
