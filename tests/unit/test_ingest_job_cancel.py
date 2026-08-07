# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cancellation contract for in-flight ingest jobs."""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from dlightrag.access_control import ACTION_PRESETS, AccessAction
from dlightrag.core.ingest_job_coordinator import IngestJobCoordinator


def _coordinator() -> IngestJobCoordinator:
    async def _unused(workspace: str):  # pragma: no cover - never awaited here
        raise AssertionError("service lookup is not part of cancellation")

    return IngestJobCoordinator(_unused, input_root=Path("/tmp/dlightrag-test"))


async def _register(coordinator: IngestJobCoordinator, job_id: str, workspace: str) -> None:
    started = asyncio.Event()

    async def _forever() -> None:
        started.set()
        await asyncio.sleep(3600)

    coordinator._tasks[job_id] = asyncio.create_task(_forever())
    coordinator._workspaces[job_id] = workspace
    await started.wait()


async def test_cancel_job_stops_the_task_and_forgets_it() -> None:
    coordinator = _coordinator()
    await _register(coordinator, "job-1", "alpha")

    assert await coordinator.cancel_job("job-1", workspace="alpha") is True
    assert "job-1" not in coordinator._tasks


async def test_cancel_job_refuses_a_job_owned_by_another_workspace() -> None:
    coordinator = _coordinator()
    await _register(coordinator, "job-1", "alpha")

    assert await coordinator.cancel_job("job-1", workspace="beta") is False
    assert not coordinator._tasks["job-1"].done()

    await coordinator.cancel_job("job-1", workspace="alpha")


async def test_cancelling_twice_is_not_reported_as_a_second_cancellation() -> None:
    coordinator = _coordinator()
    await _register(coordinator, "job-1", "alpha")

    assert await coordinator.cancel_job("job-1", workspace="alpha") is True
    assert await coordinator.cancel_job("job-1", workspace="alpha") is False


async def test_cancel_for_workspace_leaves_other_workspaces_running() -> None:
    coordinator = _coordinator()
    await _register(coordinator, "job-1", "alpha")
    await _register(coordinator, "job-2", "alpha")
    await _register(coordinator, "job-3", "beta")

    assert await coordinator.cancel_for_workspace("alpha") == 2
    assert not coordinator._tasks["job-3"].done()

    await coordinator.cancel_for_workspace("beta")


def test_anyone_who_may_ingest_may_also_stop_their_own_job() -> None:
    editor = ACTION_PRESETS["editor"]

    assert AccessAction.WORKSPACE_INGEST in editor
    assert AccessAction.JOB_CANCEL in editor
    assert AccessAction.JOB_CANCEL not in ACTION_PRESETS["reader"]


async def test_cancelling_parks_docs_that_a_startup_sweep_would_otherwise_resume() -> None:
    """LightRAG resets PARSING/ANALYZING/PROCESSING to PENDING and re-runs them."""
    from dataclasses import dataclass, field

    from lightrag.base import DocStatus

    from dlightrag.core.service import RAGService

    @dataclass
    class _Doc:
        status: Any = DocStatus.PARSING
        error_msg: str = ""
        file_path: str = "book.pdf"
        chunks_list: list[str] = field(default_factory=list)

    written: dict[str, Any] = {}

    class _DocStatus:
        async def upsert(self, data: dict[str, Any]) -> None:
            written.update(data)

    class _Stores:
        doc_status = _DocStatus()

        async def docs_by_status(self, status: Any) -> dict[str, Any]:
            return {"doc-1": _Doc()} if status is DocStatus.PARSING else {}

    service = cast(Any, RAGService.__new__(RAGService))
    service._lightrag_stores = _Stores()
    service._ensure_initialized = lambda: None

    assert await service.afail_unfinished_docs(reason="ingest job cancelled") == 1
    assert written["doc-1"]["status"] is DocStatus.FAILED
    assert written["doc-1"]["error_msg"] == "ingest job cancelled"
    # The rest of the row must survive the round trip.
    assert written["doc-1"]["file_path"] == "book.pdf"


async def test_terminal_docs_are_left_alone() -> None:
    from lightrag.base import DocStatus

    from dlightrag.core.service import RAGService

    class _Stores:
        doc_status = None

        async def docs_by_status(self, status: Any) -> dict[str, Any]:
            assert status not in (DocStatus.PROCESSED, DocStatus.FAILED)
            return {}

    service = cast(Any, RAGService.__new__(RAGService))
    service._lightrag_stores = _Stores()
    service._ensure_initialized = lambda: None

    assert await service.afail_unfinished_docs(reason="x") == 0


async def test_the_sweeper_runs_at_startup_and_then_on_a_schedule(monkeypatch) -> None:
    """It replaces the one-shot startup prune, so the first pass must not wait."""
    import dlightrag.core.ingest_job_coordinator as coordinator_module

    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(coordinator_module.asyncio, "sleep", _sleep)

    class _Store:
        def __init__(self) -> None:
            self.passes = 0

        async def prune(self) -> dict[str, int]:
            self.passes += 1
            return {"failed_abandoned": 0, "deleted_completed": 0}

    store = _Store()
    with pytest.raises(asyncio.CancelledError):
        await _coordinator()._sweep_jobs(cast(Any, store))

    assert store.passes == 2
    assert slept == [coordinator_module._JOB_SWEEP_SECONDS] * 2


async def test_a_failing_sweep_does_not_stop_the_schedule(monkeypatch) -> None:
    import dlightrag.core.ingest_job_coordinator as coordinator_module

    calls: list[int] = []

    async def _sleep(_seconds: float) -> None:
        if len(calls) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(coordinator_module.asyncio, "sleep", _sleep)

    class _Store:
        async def prune(self) -> dict[str, int]:
            calls.append(1)
            raise RuntimeError("database down")

    with pytest.raises(asyncio.CancelledError):
        await _coordinator()._sweep_jobs(cast(Any, _Store()))

    assert len(calls) == 2
