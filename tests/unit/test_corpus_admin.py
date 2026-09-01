# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for corpus administration."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
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
    FailedFileRow,
    FailedFileRowPage,
    FilePanelCursor,
    FilePanelPageRequest,
    FilePanelRowPage,
    IngestSpec,
    MetadataMatchRowPage,
    MetadataSearchCursor,
    MetadataSearchPageRequest,
    ProcessedFileRow,
    RedirectDownloadTarget,
    WorkspaceCatalogCursor,
    WorkspaceCatalogPageRequest,
    public_failure_diagnostic,
)
from dlightrag.engine.rag.retrieval import MetadataFilter


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


@asynccontextmanager
async def _noop_write_gate(workspace: str) -> AsyncIterator[None]:
    yield None


def _admin(
    *,
    read_only: bool = False,
    input_root: str | Path = "/tmp/inputs",
    default_workspace_id: str = "default",
    metadata_search: Any | None = None,
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
        list_workspace_records_page=AsyncMock(return_value=([], False)),
        workspace_exists=AsyncMock(return_value=True),
        get_workspace_record=AsyncMock(return_value=None),
        workspace_write_gate=_noop_write_gate,
    )
    jobs = SimpleNamespace(
        start_recovery=AsyncMock(),
        start_job=AsyncMock(return_value={"job_id": "job-1", "status": "queued"}),
        start_retry_failed_job=AsyncMock(return_value={"job_id": "retry-1", "status": "queued"}),
        await_job=AsyncMock(),
        get_job=AsyncMock(),
        get_active_retry_failed_job=AsyncMock(return_value=None),
        cancel_job=AsyncMock(),
        has_active_workspace_job=MagicMock(return_value=False),
        cancel_for_workspace=AsyncMock(return_value=0),
        attach_reset_result=AsyncMock(),
        close=AsyncMock(),
    )
    file_panel = SimpleNamespace(
        list_processed_files=AsyncMock(
            return_value=FilePanelRowPage(items=(), has_more=False, fetched_rows=0)
        ),
        list_failed_files=AsyncMock(
            return_value=FailedFileRowPage(items=(), has_more=False, fetched_rows=0)
        ),
    )
    metadata_store = metadata_search or SimpleNamespace(
        search_metadata_page=AsyncMock(
            return_value=MetadataMatchRowPage(
                document_ids=(),
                has_more=False,
                fetched_rows=0,
                mode="exact",
            )
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
        metadata_search=cast(Any, metadata_store),
        source_download_for=MagicMock(return_value=download),
        file_panel_cursor_secret=b"corpus-file-panel-test",
        metadata_search_cursor_secret=b"corpus-metadata-search-test",
        workspace_catalog_cursor_secret=b"corpus-workspace-catalog-test",
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
        ("start_retry_failed_docs", ("finance",), {}),
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
    jobs.start_retry_failed_job.assert_not_awaited()
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


async def test_cancel_ingest_job_rejects_unchanged_active_row() -> None:
    admin, _, _, jobs, _, _ = _admin()
    active = {"job_id": "job-1", "workspace": "finance", "status": "running"}
    jobs.get_job.side_effect = [active, active]
    jobs.cancel_job.return_value = False

    with pytest.raises(CorpusIngestError, match="cancellation was not committed"):
        await admin.cancel_ingest_job("job-1")


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


async def test_failed_document_retry_jobs_delegate_to_durable_coordinator() -> None:
    admin, pool, _, jobs, _, _ = _admin()
    jobs.get_active_retry_failed_job.return_value = {
        "job_id": "retry-1",
        "workspace": "finance",
        "status": "running",
    }
    jobs.await_job.return_value = {
        "job_id": "retry-1",
        "status": "partial",
        "result": {"retried": 2, "succeeded": 1, "failed": 1},
    }

    started = await admin.start_retry_failed_docs("finance")
    active = await admin.get_active_retry_failed_docs("finance")
    result = await admin.retry_failed_docs("finance")

    assert started == {"job_id": "retry-1", "status": "queued"}
    assert active == {
        "job_id": "retry-1",
        "workspace": "finance",
        "status": "running",
    }
    assert result == {"retried": 2, "succeeded": 1, "failed": 1}
    assert jobs.start_retry_failed_job.await_count == 2
    jobs.start_retry_failed_job.assert_awaited_with("finance")
    jobs.get_active_retry_failed_job.assert_awaited_once_with("finance")
    jobs.await_job.assert_awaited_once_with("retry-1")
    pool.acquire.assert_not_awaited()


async def test_reader_does_not_initialize_retry_coordinator_for_active_status() -> None:
    admin, _, _, jobs, _, _ = _admin(read_only=True)

    assert await admin.get_active_retry_failed_docs("finance") is None

    jobs.get_active_retry_failed_job.assert_not_awaited()
    jobs.start_recovery.assert_not_awaited()


async def test_failed_file_snapshot_is_bounded_and_projects_public_diagnostics() -> None:
    admin, pool, _, _, file_panel, _ = _admin()
    timestamp = datetime(2026, 3, 4, 5, 6, 7, 123456)
    file_panel.list_failed_files.return_value = FailedFileRowPage(
        items=(
            FailedFileRow(
                doc_id="doc-failed",
                file_path="/failed.pdf",
                error=(
                    "parser failed at /srv/private/report.pdf "
                    "https://user:password@example.test/file?token=secret " + "x" * 700
                ),
                updated_at=timestamp,
            ),
        ),
        has_more=True,
        fetched_rows=2,
    )

    snapshot = await admin.failed_file_snapshot(
        "finance",
        page=FilePanelPageRequest(limit=1),
    )

    assert snapshot == {
        "failed": [
            {
                "doc_id": "doc-failed",
                "file_path": "/failed.pdf",
                "error": snapshot["failed"][0]["error"],
                "updated_at": "2026-03-04T05:06:07.123456",
            }
        ],
        "next_cursor": FilePanelCursor(
            workspace="finance",
            updated_at=timestamp,
            doc_id="doc-failed",
            view="failed",
        ),
        "fetched_rows": 2,
    }
    diagnostic = snapshot["failed"][0]["error"]
    assert diagnostic == "Document processing failed."
    assert len(diagnostic) <= 512
    pool.acquire.assert_not_awaited()


@pytest.mark.parametrize(
    "private_value",
    [
        "Authorization: Bearer sk-live-123",
        "Authorization:Bearer private-token",
        r'headers={"Authorization":"Bearer sk-ant-api03-private\\\",continued"}',
        "Proxy-Authorization: Basic dXNlcjpwYXNz",
        "Authorization: Session private-value",
        "Cookie: session=private-value",
        "Cookie: PHPSESSID=private-session; sid=other-private",
        "Cookie: theme=light; session-id=private-value; locale=en",
        "headers={'Authorization': 'Basic dXNlcjpwYXNz'}",
        'headers={"Proxy-Authorization": "Basic cHJveHk6c2VjcmV0"}',
        "headers={'Cookie': 'session=private-value'}",
        'headers={"Set-Cookie": "session=private-value; HttpOnly"}',
        "client_secret=top-secret",
        "access_token=abc",
        "refresh-token='refresh private value'",
        'OPENAI_API_KEY="sk-live-provider"',
        "AWS_SECRET_ACCESS_KEY=provider-private",
        'password="correct horse"',
        "postgresql://user:password@db.internal/app",
        "urn:customer:private-record",
        "archive/private/report.pdf",
        "file:///srv/private/report.pdf",
        "/home dir/alice/report.pdf",
        r"C:\\Users\\Alice Smith\\report.pdf",
        r"客户\报告.pdf",
        r"\\server\share\private report.pdf",
        "../private/report.pdf",
        "pass\u200bword=secret\x00\u202e",
    ],
)
def test_public_failure_diagnostic_redacts_common_private_values(private_value: str) -> None:
    diagnostic = public_failure_diagnostic(f"parser failed: {private_value}")

    assert diagnostic == "Document processing failed."
    assert len(diagnostic) <= 512
    assert public_failure_diagnostic(diagnostic) == diagnostic


def test_public_failure_diagnostic_nfkc_normalizes_before_redaction() -> None:
    diagnostic = public_failure_diagnostic(
        'headers={"Ａｕｔｈｏｒｉｚａｔｉｏｎ": "Ｂａｓｉｃ dXNlcjpwYXNz"}'
    )

    assert diagnostic == "Document processing failed."
    assert public_failure_diagnostic(diagnostic) == diagnostic


def test_public_failure_diagnostic_allows_only_known_application_messages() -> None:
    assert public_failure_diagnostic("source metadata unavailable") == (
        "Source metadata unavailable."
    )
    assert public_failure_diagnostic("retry ingestion failed") == "Retry ingestion failed."
    assert public_failure_diagnostic("") == ""


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


# ---------------------------------------------------------------------------
# Metadata search — bounded cold path
# ---------------------------------------------------------------------------


async def test_search_metadata_never_warms_a_runtime_and_derives_next_cursor() -> None:
    admin, pool, _, _, _, _ = _admin(
        metadata_search=SimpleNamespace(
            search_metadata_page=AsyncMock(
                return_value=MetadataMatchRowPage(
                    document_ids=("doc-b", "doc-c"),
                    has_more=True,
                    fetched_rows=3,
                    mode="contains",
                )
            )
        )
    )

    page = await admin.search_metadata("finance", MetadataFilter(filename="Quarterly"))

    assert page.document_ids == ("doc-b", "doc-c")
    assert page.fetched_rows == 3
    assert page.next_cursor == MetadataSearchCursor(
        workspace="finance",
        after_doc_id="doc-c",
        mode="contains",
    )
    pool.acquire.assert_not_awaited()
    pool.get_pipeline_status.assert_not_awaited()


async def test_search_metadata_has_no_cursor_when_the_page_is_exhausted() -> None:
    store = SimpleNamespace(
        search_metadata_page=AsyncMock(
            return_value=MetadataMatchRowPage(
                document_ids=("doc-z",),
                has_more=False,
                fetched_rows=1,
                mode="exact",
            )
        )
    )
    admin, pool, _, _, _, _ = _admin(metadata_search=store)

    page = await admin.search_metadata(
        "finance",
        MetadataFilter(filename="Report"),
        page=MetadataSearchPageRequest(limit=25),
    )

    assert page.next_cursor is None
    called = store.search_metadata_page.await_args
    assert called.kwargs["page"].limit == 25
    assert called.args[0] == "finance"
    pool.acquire.assert_not_awaited()


async def test_search_metadata_rejects_cross_workspace_cursor_before_storage() -> None:
    store = SimpleNamespace(search_metadata_page=AsyncMock())
    admin, _, _, _, _, _ = _admin(metadata_search=store)

    with pytest.raises(ValueError, match="another workspace"):
        await admin.search_metadata(
            "finance",
            MetadataFilter(filename="Report"),
            page=MetadataSearchPageRequest(
                cursor=MetadataSearchCursor(
                    workspace="legal",
                    after_doc_id="doc-1",
                    mode="exact",
                )
            ),
        )

    store.search_metadata_page.assert_not_awaited()


async def test_workspace_catalog_page_delegates_and_derives_next_cursor() -> None:
    admin, pool, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records_page = AsyncMock(
        return_value=(
            [
                {
                    "workspace": "finance",
                    "display_name": "Finance",
                    "embedding_model": "voyage-multimodal-3.5",
                    "created_at": None,
                    "updated_at": None,
                },
            ],
            True,
        )
    )

    page = await admin.list_workspace_records_page(
        page=WorkspaceCatalogPageRequest(
            limit=50,
            cursor=WorkspaceCatalogCursor(after_workspace="default"),
        )
    )

    maintenance.list_workspace_records_page.assert_awaited_once_with(
        after_workspace="default",
        limit=50,
    )
    assert [item["workspace"] for item in page.items] == ["finance"]
    assert page.next_cursor == WorkspaceCatalogCursor(after_workspace="finance")
    assert page.fetched_rows == 2
    pool.acquire.assert_not_awaited()


async def test_workspace_catalog_page_rejects_empty_page_with_continuation() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records_page = AsyncMock(return_value=([], True))

    with pytest.raises(RuntimeError, match="empty page"):
        await admin.list_workspace_records_page()


async def test_workspace_catalog_full_reads_remain_full() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.list_workspace_records.return_value = [
        {
            "workspace": "default",
            "display_name": "Default",
            "embedding_model": "voyage-multimodal-3.5",
            "created_at": None,
            "updated_at": None,
        }
    ]

    records = await admin.alist_workspace_records()
    workspaces = await admin.list_workspaces()

    assert [record["workspace"] for record in records] == ["default"]
    assert workspaces == ["default"]
    maintenance.list_workspace_records_page.assert_not_awaited()


async def test_workspace_catalog_cursor_codec_is_exposed() -> None:
    from dlightrag.application.corpus_admin import WorkspaceCatalogCursorCodec

    admin, _, _, _, _, _ = _admin()

    assert isinstance(admin.workspace_catalog_cursor_codec, WorkspaceCatalogCursorCodec)


# ---------------------------------------------------------------------------
# Commit 3: promotion fence gates and storage status
# ---------------------------------------------------------------------------


async def test_delete_files_under_a_promotion_fence_raises_retryable_error() -> None:
    from dlightrag.application.errors import WorkspaceWriteFencedError
    from dlightrag.engine.rag.workspace.ports import (
        WorkspaceWriteFencedError as EngineWorkspaceWriteFencedError,
    )

    admin, _, maintenance, _, _, _ = _admin()

    @asynccontextmanager
    async def fenced_gate(workspace: str) -> AsyncIterator[None]:
        raise EngineWorkspaceWriteFencedError(workspace=workspace, retry_after_seconds=17.0)
        yield None  # pragma: no cover

    maintenance.workspace_write_gate = fenced_gate

    with pytest.raises(WorkspaceWriteFencedError) as excinfo:
        await admin.delete_files("finance", filenames=["report.pdf"])

    assert excinfo.value.retry_after_seconds == 17.0
    assert "finance" in str(excinfo.value)


async def test_dry_run_delete_skips_the_fence_gate() -> None:
    admin, pool, maintenance, _, _, _ = _admin()
    maintenance.workspace_write_gate = MagicMock(side_effect=AssertionError("gated"))
    pool.acquire.return_value.adelete_files.return_value = [{"dry_run": True}]

    result = await admin.delete_files("finance", filenames=["report.pdf"], dry_run=True)

    assert result == [{"dry_run": True}]
    maintenance.workspace_write_gate.assert_not_called()


async def test_storage_status_projects_registry_facts_bounded() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    now = datetime.now(UTC)
    maintenance.get_workspace_record.return_value = {
        "workspace": "finance",
        "storage_tier": "hot",
        "promotion_state": "failed",
        "ingested_docs_total": 42,
        "ingested_chunks_total": 900,
        "promotion_retry_count": 3,
        "promotion_last_error": "promotion failed: copy verification failed",
        "promotion_next_retry_at": now,
        "write_fence_owner": None,
        "write_fence_until": None,
    }

    status = await admin.get_workspace_storage_status("finance")
    assert status is not None

    assert status == {
        "workspace": "finance",
        "storage_tier": "hot",
        "promotion_state": "failed",
        "ingested_docs_total": 42,
        "ingested_chunks_total": 900,
        "promotion_retry_count": 3,
        "promotion_last_error": "promotion failed: copy verification failed",
        "promotion_next_retry_at": now.isoformat(),
        "write_fenced": False,
        "retry_after_seconds": None,
    }


async def test_storage_status_reports_active_fence_retry_window() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.get_workspace_record.return_value = {
        "workspace": "finance",
        "storage_tier": "shared",
        "promotion_state": "promoting",
        "ingested_docs_total": 0,
        "ingested_chunks_total": 0,
        "promotion_retry_count": 0,
        "promotion_last_error": None,
        "promotion_next_retry_at": None,
        "write_fence_owner": "worker#1",
        "write_fence_until": datetime.now(UTC) + timedelta(seconds=30),
    }

    status = await admin.get_workspace_storage_status("finance")
    assert status is not None

    assert status["write_fenced"] is True
    assert 25.0 <= status["retry_after_seconds"] <= 30.1
    assert status["storage_tier"] == "shared"
    assert status["promotion_state"] == "promoting"


async def test_storage_status_treats_stale_promoting_as_conservatively_fenced() -> None:
    admin, _, maintenance, _, _, _ = _admin()
    maintenance.get_workspace_record.return_value = {
        "workspace": "finance",
        "storage_tier": "shared",
        "promotion_state": "promoting",
        "ingested_docs_total": 0,
        "ingested_chunks_total": 0,
        "promotion_retry_count": 0,
        "promotion_last_error": None,
        "promotion_next_retry_at": None,
        "write_fence_owner": "dead-worker#1",
        "write_fence_until": datetime.now(UTC) - timedelta(seconds=60),  # expired
    }

    status = await admin.get_workspace_storage_status("finance")

    assert status is not None
    # A crashed worker's committed exclusion proofs keep the workspace
    # conservatively write-fenced with a small bounded retry window.
    assert status["write_fenced"] is True
    assert status["retry_after_seconds"] == 5.0
    assert status["promotion_state"] == "promoting"


async def test_start_promotion_worker_starts_only_on_writers() -> None:
    writer, _, _, _, _, _ = _admin()
    reader, _, _, _, _, _ = _admin(read_only=True)

    writer._promotion_worker = cast(Any, SimpleNamespace(start=MagicMock()))
    writer.start_promotion_worker()
    writer._promotion_worker.start.assert_called_once_with()  # type: ignore[union-attr]

    reader._promotion_worker = cast(Any, SimpleNamespace(start=MagicMock()))
    reader.start_promotion_worker()
    reader._promotion_worker.start.assert_not_called()  # type: ignore[union-attr]
