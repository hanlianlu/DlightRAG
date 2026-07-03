# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable ingest job lifecycle coordination."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from dlightrag.core.client_contracts import SourceType
from dlightrag.utils import normalize_workspace

if TYPE_CHECKING:
    from dlightrag.core.service import RAGService

logger = logging.getLogger(__name__)

_RECOVERABLE_SOURCE_TYPES = {"local", "azure_blob", "s3", "url"}


class IngestJobStore(Protocol):
    async def create(
        self,
        *,
        job_id: str,
        workspace: str,
        source_type: str,
        request: dict[str, Any],
    ) -> None: ...

    async def mark_running(self, job_id: str) -> None: ...

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        current_window: int,
        errors: list[str],
    ) -> None: ...

    async def finish(self, job_id: str, *, result: dict[str, Any]) -> None: ...
    async def fail(self, job_id: str, *, error: str) -> None: ...
    async def get(self, job_id: str) -> dict[str, Any] | None: ...
    async def list_recoverable(self) -> list[dict[str, Any]]: ...
    async def prune(self) -> dict[str, int]: ...
    async def delete_for_workspace(self, workspace: str) -> int: ...


class IngestJobCoordinator:
    """Manage durable ingest jobs, recovery, and in-process task lifecycle."""

    def __init__(self, get_service: Callable[[str], Awaitable[RAGService]]) -> None:
        self._get_service = get_service
        self._store: IngestJobStore | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._workspaces: dict[str, str] = {}
        self._recovery_started = False
        self._closing = False

    async def get_store(self) -> IngestJobStore:
        if self._store is None:
            from dlightrag.storage.ingest_jobs import PGIngestJobStore

            store = PGIngestJobStore()
            await store.initialize()
            self._store = store
            await self._prune_store(store)
            await self._recover_jobs(store)
        return self._store

    async def start_recovery(self) -> None:
        await self.get_store()

    async def _prune_store(self, store: IngestJobStore) -> None:
        try:
            summary = await store.prune()
        except Exception:
            logger.warning("Ingest job cleanup failed", exc_info=True)
            return
        if summary.get("failed_abandoned") or summary.get("deleted_completed"):
            logger.info("Ingest job cleanup: %s", summary)

    async def _recover_jobs(self, store: IngestJobStore) -> None:
        if self._recovery_started:
            return
        self._recovery_started = True

        try:
            rows = await store.list_recoverable()
        except Exception:
            logger.warning("Failed to list recoverable ingest jobs", exc_info=True)
            return

        recovered = 0
        for row in rows:
            if self.schedule_recovered_job(row, store):
                recovered += 1
        if recovered:
            logger.info("Recovered %d ingest job(s)", recovered)

    def schedule_recovered_job(self, row: dict[str, Any], store: IngestJobStore) -> bool:
        job_id = str(row.get("job_id") or "")
        if not job_id or job_id in self._tasks:
            return False

        raw_request = row.get("request")
        request: dict[str, Any] = raw_request if isinstance(raw_request, dict) else {}
        workspace = normalize_workspace(str(row.get("workspace") or request.get("workspace") or ""))
        source_type = str(row.get("source_type") or request.get("source_type") or "")
        raw_kwargs = request.get("kwargs")
        kwargs: dict[str, Any] = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        if not workspace or source_type not in _RECOVERABLE_SOURCE_TYPES:
            logger.warning("Skipping invalid recoverable ingest job row: %s", row)
            return False

        if "source" in kwargs:
            task = asyncio.create_task(
                store.fail(job_id, error="ingest job cannot recover an in-memory source object")
            )
            self._track(job_id, workspace, task)
            return True

        recovered_kwargs = dict(kwargs)
        resume_from_window = max(0, int(row.get("current_window") or 0))
        if resume_from_window:
            recovered_kwargs["_resume_from_window"] = resume_from_window
        cleanup_paths = _normalize_cleanup_paths(request.get("cleanup_paths"))

        task = asyncio.create_task(
            self._run_job(
                job_id=job_id,
                workspace=workspace,
                source_type=cast(SourceType, source_type),
                kwargs=recovered_kwargs,
                cleanup_paths=cleanup_paths,
            )
        )
        self._track(job_id, workspace, task)
        return True

    async def attach_reset_result(
        self,
        *,
        workspace: str,
        result: dict[str, Any],
        dry_run: bool,
    ) -> None:
        if dry_run:
            result["ingest_jobs_deleted"] = 0
            return

        try:
            store = await self.get_store()
            result["ingest_jobs_deleted"] = await store.delete_for_workspace(workspace)
        except Exception as exc:
            result["ingest_jobs_deleted"] = 0
            errors = result.setdefault("errors", [])
            if isinstance(errors, list):
                errors.append(f"ingest_jobs: {exc}")
            logger.warning("Failed to delete ingest jobs for workspace '%s': %s", workspace, exc)

    async def cancel_for_workspace(self, workspace: str) -> int:
        job_ids = [
            job_id
            for job_id, job_workspace in self._workspaces.items()
            if job_workspace == workspace
        ]
        if not job_ids:
            return 0

        tasks: list[asyncio.Task[None]] = []
        for job_id in job_ids:
            task = self._tasks.get(job_id)
            if task is None:
                self._forget(job_id)
            elif not task.done():
                task.cancel()
                tasks.append(task)
            else:
                self._forget(job_id)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for job_id in job_ids:
            self._forget(job_id)
        return len(tasks)

    async def start_job(
        self,
        workspace: str,
        source_type: SourceType,
        *,
        cleanup_paths: str | Path | Sequence[str | Path] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        workspace_id = normalize_workspace(workspace)
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        job_id = uuid.uuid4().hex
        store = await self.get_store()
        cleanup_path_tuple = _normalize_cleanup_paths(cleanup_paths)
        request = {
            "workspace": workspace_id,
            "source_type": source_type,
            "kwargs": _json_safe(kwargs),
        }
        if cleanup_path_tuple:
            request["cleanup_paths"] = [str(path) for path in cleanup_path_tuple]
        await store.create(
            job_id=job_id,
            workspace=workspace_id,
            source_type=source_type,
            request=request,
        )

        task = asyncio.create_task(
            self._run_job(
                job_id=job_id,
                workspace=workspace_id,
                source_type=source_type,
                kwargs=dict(kwargs),
                cleanup_paths=cleanup_path_tuple,
            )
        )
        self._track(job_id, workspace_id, task)

        row = await store.get(job_id)
        return row or {
            "job_id": job_id,
            "workspace": workspace_id,
            "source_type": source_type,
            "status": "queued",
        }

    async def await_job(
        self, job_id: str, *, timeout: float | None = None
    ) -> dict[str, Any] | None:
        task = self._tasks.get(job_id)
        if task is not None:
            try:
                if timeout is None:
                    await asyncio.shield(task)
                else:
                    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except TimeoutError:
                pass
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        store = await self.get_store()
        return await store.get(job_id)

    async def _run_job(
        self,
        *,
        job_id: str,
        workspace: str,
        source_type: SourceType,
        kwargs: dict[str, Any],
        cleanup_paths: tuple[Path, ...],
    ) -> None:
        store = await self.get_store()
        await store.mark_running(job_id)
        progress_seen = False
        cleanup_after_run = True

        async def _record_progress(progress: Any) -> None:
            nonlocal progress_seen
            progress_seen = True
            await store.record_window(
                job_id,
                total_delta=progress.total_delta,
                processed_delta=progress.processed_delta,
                failed_delta=progress.failed_delta,
                current_window=progress.batch_index + 1,
                errors=list(progress.errors),
            )

        try:
            svc = await self._get_service(workspace)
            await svc.aregister_workspace()
            result = await svc.aingest(
                source_type=source_type,
                **kwargs,
                _progress_callback=_record_progress,
            )
            if not progress_seen:
                total_items, processed_items, failed_items, errors = _infer_ingest_counts(result)
                await store.record_window(
                    job_id,
                    total_delta=total_items,
                    processed_delta=processed_items,
                    failed_delta=failed_items,
                    current_window=1,
                    errors=errors,
                )
            await store.finish(
                job_id, result=await self._job_result_with_totals(store, job_id, result)
            )
        except asyncio.CancelledError:
            if not self._closing:
                await store.fail(job_id, error="ingest job cancelled")
            else:
                cleanup_after_run = False
            raise
        except Exception as exc:
            await store.fail(job_id, error=f"{type(exc).__name__}: {exc}")
        finally:
            if cleanup_after_run and cleanup_paths:
                await asyncio.to_thread(_cleanup_ingest_paths, cleanup_paths)

    async def _job_result_with_totals(
        self,
        store: IngestJobStore,
        job_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        row = await store.get(job_id)
        if row is None:
            return result

        final_result = dict(result)
        if isinstance(final_result.get("processed"), int):
            final_result["processed"] = int(row.get("processed_items") or 0)
        if isinstance(final_result.get("errors"), list):
            final_result["errors"] = list(row.get("errors") or [])
        return final_result

    async def close(self) -> None:
        if self._tasks:
            self._closing = True
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()
            self._closing = False
        self._workspaces.clear()

    def _track(self, job_id: str, workspace: str, task: asyncio.Task[None]) -> None:
        self._tasks[job_id] = task
        self._workspaces[job_id] = workspace
        task.add_done_callback(lambda _task, _job_id=job_id: self._forget(_job_id))

    def _forget(self, job_id: str) -> None:
        self._tasks.pop(job_id, None)
        self._workspaces.pop(job_id, None)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _normalize_cleanup_paths(
    raw: str | Path | Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if raw is None:
        return ()
    values = [raw] if isinstance(raw, (str, Path)) else raw
    return tuple(Path(str(value)) for value in values if str(value))


def _cleanup_ingest_paths(paths: tuple[Path, ...]) -> None:
    from dlightrag.config import get_config

    input_root = get_config().input_dir_path.resolve(strict=False)
    for path in paths:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(input_root)
        except ValueError:
            logger.warning("Skipping ingest cleanup outside input_dir: %s", path)
            continue
        try:
            if resolved.is_dir() and not resolved.is_symlink():
                shutil.rmtree(resolved, ignore_errors=True)
            else:
                resolved.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to clean ingest path: %s", resolved, exc_info=True)


def _infer_ingest_counts(result: dict[str, Any]) -> tuple[int, int, int, list[str]]:
    errors = [str(error) for error in result.get("errors") or []]
    if isinstance(result.get("processed"), int):
        processed = max(0, int(result["processed"]))
        failed = len(errors)
        return processed + failed, processed, failed, errors

    if not result:
        return 0, 0, 0, errors
    if str(result.get("status") or "").lower() == "skipped":
        return 1, 0, 0, errors
    if result.get("doc_id") or result.get("chunks") or result.get("source_kind"):
        return 1, 1, 0, errors
    return 0, 0, len(errors), errors


__all__ = ["IngestJobCoordinator"]
