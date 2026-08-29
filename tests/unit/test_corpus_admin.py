# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for corpus administration."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from dlightrag.application.corpus_admin import (
    CorpusAdmin,
    CorpusAdminSettings,
    CorpusIngestError,
    FilePanelCursor,
    FilePanelPageRequest,
    FilePanelRowPage,
    IngestSpec,
    ProcessedFileRow,
    RedirectDownloadTarget,
)


def _settings(
    *,
    read_only: bool = False,
    input_root: str | Path = "/tmp/inputs",
    default_workspace_id: str = "default",
) -> CorpusAdminSettings:
    return CorpusAdminSettings(
        default_workspace_id=default_workspace_id,
        default_display_name="Default",
        default_embedding_model="embedding-model",
        input_root=input_root,
        ingest_timeout_seconds=30.0,
        read_only=read_only,
    )


def _admin(
    *,
    read_only: bool = False,
    input_root: str | Path = "/tmp/inputs",
    default_workspace_id: str = "default",
) -> tuple[CorpusAdmin, Any, Any, Any, Any, Any]:
    runtime = AsyncMock()
    pool = SimpleNamespace(
        acquire=AsyncMock(return_value=runtime),
        get_pipeline_status=AsyncMock(return_value=None),
        is_loaded=AsyncMock(return_value=False),
        evict=AsyncMock(),
    )
    maintenance = SimpleNamespace(
        initialize=AsyncMock(),
        register_workspace=AsyncMock(),
        list_workspace_records=AsyncMock(return_value=[]),
        workspace_exists=AsyncMock(return_value=True),
    )
    jobs = SimpleNamespace(
        start_recovery=AsyncMock(),
        start_job=AsyncMock(return_value={"job_id": "job-1", "status": "queued"}),
        await_job=AsyncMock(),
        get_job=AsyncMock(),
        cancel_job=AsyncMock(),
        has_active_workspace_job=MagicMock(return_value=False),
        cancel_for_workspace=AsyncMock(return_value=0),
        attach_reset_result=AsyncMock(),
        close=AsyncMock(),
    )
    file_panel = SimpleNamespace(
        list_processed_files=AsyncMock(
            return_value=FilePanelRowPage(items=(), has_more=False, fetched_rows=0)
        )
    )
    download = SimpleNamespace(prepare=AsyncMock())
    admin = CorpusAdmin(
        settings=_settings(
            read_only=read_only,
            input_root=input_root,
            default_workspace_id=default_workspace_id,
        ),
        pool=cast(Any, pool),
        maintenance=cast(Any, maintenance),
        ingest_jobs=cast(Any, jobs),
        file_panel=cast(Any, file_panel),
        source_download_for=MagicMock(return_value=download),
    )
    return admin, pool, maintenance, jobs, file_panel, download


@pytest.mark.parametrize("workspace_ids", [(), ("*",), ("Finance Reports",)])
async def test_reset_rejects_empty_or_policy_shaped_scope_before_side_effects(
    workspace_ids: tuple[str, ...],
) -> None:
    admin, pool, maintenance, jobs, _, _ = _admin()

    with pytest.raises(ValueError, match="canonical workspace"):
        await admin.reset(workspace_ids=workspace_ids)

    pool.acquire.assert_not_awaited()
    pool.is_loaded.assert_not_awaited()
    maintenance.list_workspace_records.assert_not_awaited()
    jobs.cancel_for_workspace.assert_not_awaited()


async def test_reset_rejects_a_bare_string_scope_before_side_effects() -> None:
    admin, pool, maintenance, jobs, _, _ = _admin()

    with pytest.raises(ValueError, match="canonical workspace"):
        await admin.reset(workspace_ids=cast(Any, "finance"))

    pool.acquire.assert_not_awaited()
    maintenance.list_workspace_records.assert_not_awaited()
    jobs.cancel_for_workspace.assert_not_awaited()


async def test_reset_cancels_then_resets_and_evicts_loaded_runtime() -> None:
    admin, pool, maintenance, jobs, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [{"workspace": "finance"}]
    jobs.cancel_for_workspace.return_value = 2
    runtime = pool.acquire.return_value
    runtime.areset.return_value = {"workspace": "finance", "errors": []}

    result = await admin.reset(workspace_ids=("finance",), keep_files=True)

    jobs.cancel_for_workspace.assert_awaited_once_with("finance")
    runtime.areset.assert_awaited_once_with(keep_files=True, dry_run=False)
    jobs.attach_reset_result.assert_awaited_once_with(
        workspace="finance",
        result=result["workspaces"]["finance"],
        dry_run=False,
    )
    pool.evict.assert_awaited_once_with("finance")
    assert result["workspaces"]["finance"]["ingest_jobs_cancelled"] == 2
    assert result["total_errors"] == 0


async def test_reset_dry_run_neither_cancels_jobs_nor_evicts_runtime() -> None:
    admin, pool, maintenance, jobs, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [{"workspace": "finance"}]
    runtime = pool.acquire.return_value
    runtime.areset.return_value = {"workspace": "finance", "errors": []}

    await admin.reset(workspace_ids=("finance",), dry_run=True)

    jobs.cancel_for_workspace.assert_not_awaited()
    runtime.areset.assert_awaited_once_with(keep_files=False, dry_run=True)
    pool.evict.assert_not_awaited()


async def test_reset_failure_is_counted_and_runtime_is_still_evicted() -> None:
    admin, pool, maintenance, jobs, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [{"workspace": "finance"}]
    pool.acquire.return_value.areset.side_effect = RuntimeError("reset failed")

    result = await admin.reset(workspace_ids=("finance",))

    jobs.attach_reset_result.assert_not_awaited()
    pool.evict.assert_awaited_once_with("finance")
    assert result == {
        "workspaces": {
            "finance": {
                "error": "workspace reset failed",
                "ingest_jobs_cancelled": 0,
            }
        },
        "total_errors": 1,
    }


async def test_reset_completes_before_evict_and_evict_failure_preserves_result() -> None:
    admin, pool, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [{"workspace": "finance"}]
    order: list[str] = []

    async def reset_runtime(**_kwargs: Any) -> dict[str, Any]:
        order.append("reset")
        return {"workspace": "finance", "errors": []}

    async def fail_evict(_workspace: str) -> None:
        order.append("evict")
        raise RuntimeError("close failed")

    pool.acquire.return_value.areset.side_effect = reset_runtime
    pool.evict.side_effect = fail_evict

    result = await admin.reset(workspace_ids=("finance",))

    assert order == ["reset", "evict"]
    assert result["workspaces"]["finance"]["workspace"] == "finance"
    assert result["total_errors"] == 0


async def test_explicit_authorized_id_absent_from_catalog_uses_orphan_cleanup(
    tmp_path: Path,
) -> None:
    admin, pool, _, jobs, _, _ = _admin(input_root=tmp_path)
    orphan_result = {"workspace": "archived", "errors": []}

    with patch(
        "dlightrag.application.corpus_admin.service.areset_orphaned_workspace",
        new_callable=AsyncMock,
        return_value=orphan_result,
    ) as reset_orphan:
        result = await admin.reset(workspace_ids=("archived",))

    pool.is_loaded.assert_awaited_once_with("archived")
    pool.acquire.assert_not_awaited()
    reset_orphan.assert_awaited_once()
    call = reset_orphan.await_args
    assert call is not None
    assert call.args == ("archived",)
    assert call.kwargs["input_dir"] == str(tmp_path)
    jobs.attach_reset_result.assert_awaited_once_with(
        workspace="archived",
        result=orphan_result,
        dry_run=False,
    )
    assert result["workspaces"]["archived"] is orphan_result


async def test_initialize_registers_default_only_for_writer() -> None:
    writer, _, writer_maintenance, _, _, _ = _admin()
    reader, _, reader_maintenance, _, _, _ = _admin(read_only=True)

    await writer.initialize()
    await reader.initialize()

    writer_maintenance.initialize.assert_awaited_once_with(validate_only=False)
    writer_maintenance.register_workspace.assert_awaited_once_with(
        workspace="default",
        display_name="Default",
        embedding_model="embedding-model",
    )
    reader_maintenance.initialize.assert_awaited_once_with(validate_only=True)
    reader_maintenance.register_workspace.assert_not_awaited()


async def test_invalid_default_workspace_fails_on_initialize_not_construction() -> None:
    admin, _, maintenance, _, _, _ = _admin(default_workspace_id="")

    with pytest.raises(ValueError, match="canonical workspace"):
        await admin.initialize()

    maintenance.initialize.assert_not_awaited()


def test_zero_ingest_timeout_is_a_valid_immediate_wait_policy() -> None:
    settings = CorpusAdminSettings(
        default_workspace_id="default",
        default_display_name="Default",
        default_embedding_model="embedding-model",
        input_root="/tmp/inputs",
        ingest_timeout_seconds=0,
        read_only=False,
    )

    assert settings.ingest_timeout_seconds == 0


@pytest.mark.parametrize("payload", [{"url": ""}, {"urls": [""]}])
def test_url_ingest_rejects_empty_urls(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        IngestSpec.model_validate({"source_type": "url", **payload})


@pytest.mark.parametrize(
    "row",
    [
        None,
        {"job_id": "job-1", "status": "failed", "errors": ["parse failed"]},
    ],
)
async def test_caller_awaited_ingest_raises_typed_failure(row: dict[str, Any] | None) -> None:
    admin, _, _, jobs, _, _ = _admin()
    jobs.await_job.return_value = row

    with pytest.raises(CorpusIngestError):
        await admin.ingest("finance", IngestSpec(source_type="local", path="report.pdf"))


async def test_caller_awaited_ingest_returns_result_with_configured_timeout() -> None:
    admin, _, _, jobs, _, _ = _admin()
    jobs.await_job.return_value = {
        "job_id": "job-1",
        "status": "succeeded",
        "result": {"processed": 3},
    }

    result = await admin.ingest(
        "finance",
        IngestSpec(source_type="s3", bucket="documents", prefix="reports/"),
    )

    jobs.start_job.assert_awaited_once()
    jobs.await_job.assert_awaited_once_with("job-1", timeout=30.0)
    assert result == {"processed": 3}


async def test_caller_awaited_ingest_returns_running_row_after_timeout() -> None:
    admin, _, _, jobs, _, _ = _admin()
    running = {"job_id": "job-1", "status": "running"}
    jobs.await_job.return_value = running

    result = await admin.ingest(
        "finance",
        IngestSpec(source_type="local", path="report.pdf"),
    )

    assert result is running
    jobs.await_job.assert_awaited_once_with("job-1", timeout=30.0)


async def test_upload_batch_is_the_only_local_ingest_cleanup_path(tmp_path: Path) -> None:
    admin, _, _, jobs, _, _ = _admin()
    upload_batch = tmp_path / "default" / "__uploads__" / "batch-1"
    regular_path = tmp_path / "default" / "documents"

    await admin.start_ingest_job(
        "default",
        IngestSpec(source_type="local", path=str(upload_batch)),
    )
    await admin.start_ingest_job(
        "default",
        IngestSpec(source_type="local", path=str(regular_path)),
    )

    upload_call, regular_call = jobs.start_job.await_args_list
    assert upload_call.kwargs["cleanup_paths"] == [str(upload_batch)]
    assert regular_call.kwargs["cleanup_paths"] == []


async def test_workspace_catalog_serializes_rows_and_includes_default() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [
        {
            "workspace": "finance",
            "display_name": "Finance",
            "embedding_model": "embed-v2",
            "created_at": datetime(2026, 8, 17, tzinfo=UTC),
            "updated_at": None,
        }
    ]

    records = await admin.alist_workspace_records()

    assert records[0] == {
        "workspace": "finance",
        "display_name": "Finance",
        "embedding_model": "embed-v2",
        "created_at": "2026-08-17T00:00:00+00:00",
        "updated_at": None,
    }
    assert records[1]["workspace"] == "default"
    assert await admin.list_workspaces() == ["finance", "default"]


async def test_catalog_failure_falls_back_to_default() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.side_effect = RuntimeError("registry unavailable")

    assert await admin.list_workspaces() == ["default"]


@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    [
        ("create_workspace", ("finance",), {}),
        ("ingest", ("finance", IngestSpec(source_type="local", path="report.pdf")), {}),
        (
            "start_ingest_job",
            ("finance", IngestSpec(source_type="local", path="report.pdf")),
            {},
        ),
        ("cancel_ingest_job", ("job-1",), {}),
        ("delete_files", ("finance",), {}),
        ("retry_failed_docs", ("finance",), {}),
        ("update_metadata", ("finance", "doc-1", {"title": "Report"}), {}),
        ("reset", (), {"workspace_ids": ("finance",)}),
    ],
)
async def test_reader_rejects_corpus_mutations_before_touching_collaborators(
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    admin, pool, maintenance, jobs, _, _ = _admin(read_only=True)

    with pytest.raises(PermissionError, match="writer service role"):
        await getattr(admin, method)(*args, **kwargs)

    pool.acquire.assert_not_awaited()
    maintenance.list_workspace_records.assert_not_awaited()
    jobs.start_job.assert_not_awaited()
    jobs.cancel_job.assert_not_awaited()


async def test_recovery_skips_reader_and_close_always_joins_jobs() -> None:
    writer, _, _, writer_jobs, _, _ = _admin()
    reader, _, _, reader_jobs, _, _ = _admin(read_only=True)

    await writer.start_recovery()
    await reader.start_recovery()
    await writer.aclose()
    await reader.aclose()

    writer_jobs.start_recovery.assert_awaited_once_with()
    reader_jobs.start_recovery.assert_not_awaited()
    writer_jobs.close.assert_awaited_once_with()
    reader_jobs.close.assert_awaited_once_with()


async def test_cancel_ingest_job_uses_stored_canonical_workspace_and_returns_latest_row() -> None:
    admin, _, _, jobs, _, _ = _admin()
    jobs.get_job.side_effect = [
        {"job_id": "job-1", "workspace": "finance", "status": "running"},
        {"job_id": "job-1", "workspace": "finance", "status": "cancelled"},
    ]

    result = await admin.cancel_ingest_job("job-1")

    jobs.cancel_job.assert_awaited_once_with("job-1", workspace="finance")
    assert result is not None
    assert result["status"] == "cancelled"


async def test_file_panel_and_source_download_do_not_warm_cold_runtime() -> None:
    from dlightrag.engine.rag.corpus.downloads import (
        RedirectDownloadTarget as RagRedirectDownloadTarget,
    )

    admin, pool, _, jobs, file_panel, download = _admin()
    file_panel.list_processed_files.return_value = FilePanelRowPage(
        items=(
            ProcessedFileRow(
                doc_id="doc-1",
                file_path="/files/doc-1.pdf",
                updated_at=datetime(2026, 3, 4, 5, 6, 7),
            ),
        ),
        has_more=False,
        fetched_rows=1,
    )
    jobs.has_active_workspace_job.return_value = True
    target = RagRedirectDownloadTarget(url="https://cdn.example.com/report.pdf")
    download.prepare.return_value = target

    snapshot = await admin.file_panel_snapshot("finance")
    prepared = await admin.prepare_source_download("finance", "doc-1")

    assert snapshot == {
        "files": [
            {
                "doc_id": "doc-1",
                "file_path": "/files/doc-1.pdf",
                "status": "processed",
                "updated_at": "2026-03-04T05:06:07.000000",
            }
        ],
        "pipeline_status": {
            "busy": True,
            "pending_enqueues": 0,
            "latest_message": "Starting ingest...",
        },
        "next_cursor": None,
        "fetched_rows": 1,
    }
    assert isinstance(prepared, RedirectDownloadTarget)
    assert prepared.url == "https://cdn.example.com/report.pdf"
    pool.acquire.assert_not_awaited()


async def test_file_panel_snapshot_derives_bounded_cursor_and_rejects_foreign_cursor() -> None:
    admin, pool, _, _, file_panel, _ = _admin()
    timestamp = datetime(2026, 3, 4, 5, 6, 7, 123456)
    file_panel.list_processed_files.return_value = FilePanelRowPage(
        items=(
            ProcessedFileRow(doc_id="doc-a", file_path="/a", updated_at=timestamp),
            ProcessedFileRow(doc_id="doc-b", file_path="/b", updated_at=timestamp),
        ),
        has_more=True,
        fetched_rows=3,
    )

    snapshot = await admin.file_panel_snapshot(
        "finance",
        page=FilePanelPageRequest(limit=2),
    )

    assert snapshot["next_cursor"] == FilePanelCursor(
        workspace="finance",
        updated_at=timestamp,
        doc_id="doc-b",
    )
    assert snapshot["fetched_rows"] == 3
    file_panel.list_processed_files.assert_awaited_once_with(
        "finance",
        page=FilePanelPageRequest(limit=2),
    )
    pool.acquire.assert_not_awaited()

    file_panel.list_processed_files.reset_mock()
    with pytest.raises(ValueError, match="another workspace"):
        await admin.file_panel_snapshot(
            "finance",
            page=FilePanelPageRequest(
                cursor=FilePanelCursor(
                    workspace="legal",
                    updated_at=None,
                    doc_id="doc-z",
                )
            ),
        )
    file_panel.list_processed_files.assert_not_awaited()


async def test_workspace_exists_uses_default_fast_path_and_bounded_maintenance_lookup() -> None:
    admin, pool, maintenance, _, _, _ = _admin()
    maintenance.workspace_exists.side_effect = [True, False]

    assert await admin.workspace_exists("default") is True
    assert await admin.workspace_exists("finance") is True
    assert await admin.workspace_exists("legal") is False

    assert [item.args for item in maintenance.workspace_exists.await_args_list] == [
        ("finance",),
        ("legal",),
    ]
    maintenance.workspace_exists.side_effect = RuntimeError("registry unavailable")
    with pytest.raises(RuntimeError, match="registry unavailable"):
        await admin.workspace_exists("research")
    pool.acquire.assert_not_awaited()
