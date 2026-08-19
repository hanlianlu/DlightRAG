# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

import datetime
import json
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from mcp import Client, MCPError
from mcp.types import INVALID_PARAMS, CallToolResult, InputRequiredResult, TextContent

from dlightrag.access import RequestScope, owner_id_from_principal, request_scope_context
from dlightrag.config import AccessControlConfig, AccessControlRuleConfig, DlightragConfig
from dlightrag.mcp import server as mcp_server
from dlightrag.runtime import AnswerRunRecord
from dlightrag.services.corpora import IngestSpec
from dlightrag.services.retrieval import RetrieveResponse as ServiceResponse
from tests.unit.conftest import answer_capability_view

_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def _completed_tool_result(result: CallToolResult | InputRequiredResult) -> CallToolResult:
    assert isinstance(result, CallToolResult)
    return result


def _tool_text(result: CallToolResult | InputRequiredResult) -> str:
    result = _completed_tool_result(result)
    assert result.content
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _tool_json(result: CallToolResult | InputRequiredResult) -> Any:
    return json.loads(_tool_text(result))


@pytest.fixture
def mock_mcp_application(monkeypatch, test_config: DlightragConfig):
    application = AsyncMock()
    application.config = test_config
    application.corpora = SimpleNamespace(
        list_workspaces=AsyncMock(return_value=["default"]),
        alist_workspace_records=AsyncMock(return_value=[{"workspace": "default"}]),
        create_workspace=AsyncMock(),
        reset=AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0}),
        ingest=AsyncMock(),
        start_ingest_job=AsyncMock(),
        get_ingest_job=AsyncMock(),
        cancel_ingest_job=AsyncMock(),
        list_ingested_files=AsyncMock(return_value=[]),
        delete_files=AsyncMock(return_value=[]),
    )
    application.retrieval = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=ServiceResponse(
                contexts={"chunks": [], "entities": [], "relationships": []},
                sources=(),
                trace={},
                image_descriptions=(),
            )
        )
    )
    capability_view = answer_capability_view()
    application.answers = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(run=_run_record(), replayed=False)),
        get=AsyncMock(return_value=_run_record()),
        cancel=AsyncMock(
            return_value=SimpleNamespace(outcome="cancelled", run=_run_record(status="cancelled"))
        ),
        capabilities=capability_view.read,
    )
    application.config.answer.max_attachments = 6
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    return application


_RUN_ID = "019893f4-0000-7000-8000-000000000001"
_CREATED_AT = datetime.datetime(2026, 8, 12, tzinfo=datetime.UTC)
#: MCP with ``auth_mode="none"`` collapses callers into the deployment owner.
_EXPECTED_OWNER = owner_id_from_principal(auth_mode="none", user_id="anonymous")


def _run_record(
    *,
    status: str = "queued",
    result: dict[str, Any] | None = None,
    cancel_requested: bool = False,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> AnswerRunRecord:
    terminal = status in ("succeeded", "failed", "cancelled")
    return AnswerRunRecord(
        owner_id=_EXPECTED_OWNER,
        run_id=_RUN_ID,
        idempotency_key=None,
        prepared_input={"query": "Follow up", "workspaces": ["default"]},
        status=status,  # type: ignore[arg-type]
        phase=None,
        stop_reason=None,
        cancel_requested_at=_CREATED_AT if cancel_requested else None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        durable_progress_version=0,
        last_reclaim_progress_version=0,
        reclaims_without_progress=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=result,
        error_kind=error_kind,
        error_message=error_message,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
        started_at=None,
        finished_at=_CREATED_AT if terminal else None,
    )


def _stored_result() -> dict[str, Any]:
    return {
        "answer": "Answer [1-1].",
        "contexts": {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "Evidence",
                    "image_data": "base64-payload",
                    "_workspace": "default",
                }
            ]
        },
        "sources": [
            {
                "id": "1",
                "title": "report.pdf",
                "type": "document",
                "source_uri": "local://default/report.pdf",
                "workspace": "default",
                "document_id": "doc-report",
                "chunks": [],
            }
        ],
        "answer_images": [],
        "trace": {},
        "image_descriptions": [],
    }


async def test_get_capabilities_reports_answer_image_capability(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.answer.capability import AnswerImageCapability

    mock_mcp_application.answers.capabilities = answer_capability_view(
        AnswerImageCapability(
            status="supported",
            configured_ceiling=8,
            effective_max_images=6,
            provider="test",
            base_url=None,
            model="test-model",
            failure_kind=None,
        )
    ).read

    result = await mcp_server.mcp_app.call_tool("get_capabilities", {})

    cap = _tool_json(result)["answer_image_capability"]
    assert cap["status"] == "supported"
    assert cap["effective_max_images"] == 6
    assert cap["configured_ceiling"] == 8
    assert cap["model"] == "test-model"


async def test_mcp_v2_client_lists_and_calls_tools(mock_mcp_application: AsyncMock) -> None:
    mock_mcp_application.corpora.get_ingest_job = AsyncMock(
        return_value={"job_id": "job-1", "status": "running"}
    )

    async with Client(mcp_server.mcp_app) as client:
        listing = await client.list_tools()
        result = await client.call_tool("get_ingest_job", {"job_id": "job-1"})

    assert "retrieve" in {tool.name for tool in listing.tools}
    assert result.is_error is False
    assert _tool_json(result) == {"job_id": "job-1", "status": "running"}
    assert result.structured_content == {"job_id": "job-1", "status": "running"}


async def test_mcp_internal_errors_do_not_leak_details(mock_mcp_application: AsyncMock) -> None:
    mock_mcp_application.corpora.alist_workspace_records.side_effect = RuntimeError(
        "database-secret"
    )

    result = await mcp_server.mcp_app.call_tool("list_workspaces", {})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == "Error: internal tool failure"
    assert "database-secret" not in _tool_text(result)


async def test_mcp_protocol_errors_remain_protocol_errors() -> None:
    app = mcp_server.DlightRAGMCPServer("probe")

    @app.tool()
    async def reject() -> None:
        raise MCPError(INVALID_PARAMS, "invalid request")

    with pytest.raises(MCPError, match="invalid request"):
        await app.call_tool("reject", {})


async def test_mcp_lists_workspace_lifecycle_tools() -> None:
    tools = await mcp_server.mcp_app.list_tools()
    names = {tool.name for tool in tools}

    assert names == {
        "answer",
        "cancel_answer_run",
        "cancel_ingest_job",
        "create_workspace",
        "delete_files",
        "delete_workspace",
        "get_answer_run",
        "get_capabilities",
        "get_ingest_job",
        "ingest",
        "forget_memory",
        "list_answer_artifacts",
        "list_answer_runs",
        "list_memories",
        "list_files",
        "list_workspaces",
        "read_answer_artifact",
        "retrieve",
    }
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.input_schema["properties"]
    assert {
        "query",
        "history",
        "mode",
        "attachments",
        "filters",
        "chunk_top_k",
        "idempotency_key",
    } <= answer_props.keys()
    assert "query_images" not in answer_props
    # The answer tool starts work and returns; the follow-up tools read and stop it.
    assert answer_tool.annotations is not None
    assert answer_tool.annotations.read_only_hint is False
    for name in ("get_answer_run", "cancel_answer_run"):
        tool = next(item for item in tools if item.name == name)
        assert set(tool.input_schema["properties"]) == {"run_id"}
        assert tool.input_schema["properties"]["run_id"]["description"]
        assert tool.description and "answer run" in tool.description
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    ingest_props = ingest_tool.input_schema["properties"]
    assert {"source_type", "path", "url", "documents", "metadata"} <= ingest_props.keys()
    delete_files_tool = next(tool for tool in tools if tool.name == "delete_files")
    assert "dry_run" in delete_files_tool.input_schema["properties"]


def test_mcp_security_defaults_are_loopback_only() -> None:
    cfg = cast(Any, DlightragConfig)()

    assert cfg.mcp_allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert cfg.mcp_allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


async def test_mcp_rejects_unknown_mode_without_schema_wrapper(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x", "mode": "mix"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert "Error:" in _tool_text(result)
    assert "mode" in _tool_text(result)
    mock_mcp_application.answers.create.assert_not_awaited()


@pytest.mark.parametrize(
    ("tool_name", "payload", "error_fragment"),
    [
        (
            "retrieve",
            {
                "query": "x",
                "query_images": [_IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK],
            },
            "query_images",
        ),
        (
            "retrieve",
            {"query": "x", "query_images": ["data:image/png;base64,abc"]},
            "valid dictionary",
        ),
    ],
)
async def test_mcp_rejects_invalid_query_image_payloads(
    mock_mcp_application,
    tool_name: str,
    payload: dict[str, Any],
    error_fragment: str,
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        tool_name,
        payload,
    )

    assert "Error:" in _tool_text(result)
    assert error_fragment in _tool_text(result)
    mock_mcp_application.retrieval.retrieve.assert_not_awaited()


async def test_mcp_retrieve_forwards_chunk_top_k(mock_mcp_application) -> None:
    await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "top_k": 8, "chunk_top_k": 5},
    )

    await_args = mock_mcp_application.retrieval.retrieve.await_args
    assert await_args is not None
    request = await_args.args[0]
    assert request.top_k == 8
    assert request.chunk_top_k == 5


async def test_mcp_retrieve_uses_service_contract(mock_mcp_application: AsyncMock) -> None:
    result = await mcp_server.mcp_app.call_tool("retrieve", {"query": "x"})

    assert _tool_json(result)["contexts"] == {"chunks": [], "entities": [], "relationships": []}
    request = mock_mcp_application.retrieval.retrieve.await_args.args[0]
    assert request.workspaces == ("default",)
    assert request.projection.include_download_links is False


async def test_mcp_jwt_claims_access_control_denies_unmapped_workspace(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    test_config.auth_mode = "jwt"
    test_config.jwt_verification_key = "test-key"
    test_config.access_control = AccessControlConfig(
        mode="jwt_claims",
        rules=[
            AccessControlRuleConfig(
                claim="groups",
                value="finance-rag-readers",
                workspaces=["finance"],
                actions=["workspace.query"],
            )
        ],
    )

    with request_scope_context(
        RequestScope(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": ["legal-rag-readers"]},
        )
    ):
        result = await mcp_server.mcp_app.call_tool(
            "retrieve",
            {"query": "x", "workspaces": ["finance"]},
        )

    assert "Access denied" in _tool_text(result)
    mock_mcp_application.retrieval.retrieve.assert_not_awaited()


async def test_mcp_query_permission_does_not_imply_visual_asset_permission(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    test_config.access_control = AccessControlConfig(
        mode="jwt_claims",
        rules=[
            AccessControlRuleConfig(
                claim="groups",
                value="finance-rag-readers",
                workspaces=["default"],
                actions=["workspace.query"],
            )
        ],
    )
    mock_mcp_application.retrieval.retrieve.return_value = ServiceResponse(
        contexts={
            "chunks": [
                {
                    "chunk_id": "figure-1",
                    "reference_id": "1",
                    "file_path": "report.pdf",
                    "content": "Evidence",
                    "image_data": "bytes",
                    "_workspace": "default",
                    "metadata": {
                        "source_uri": "local://default/report.pdf",
                        "source_download_locator": "report.pdf",
                    },
                }
            ]
        },
        sources=(),
        trace={},
        image_descriptions=(),
    )

    with request_scope_context(
        RequestScope(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": ["finance-rag-readers"]},
        )
    ):
        result = await mcp_server.mcp_app.call_tool("retrieve", {"query": "x"})

    chunk = _tool_json(result)["contexts"]["chunks"][0]
    assert chunk["content"] == "Evidence"
    assert "image_url" not in chunk
    assert "thumbnail_url" not in chunk


async def test_mcp_retrieve_all_workspaces_uses_visible_records(mock_mcp_application) -> None:
    mock_mcp_application.corpora.alist_workspace_records.return_value = [
        {"workspace": "default"},
        {"workspace": "research_notes"},
    ]
    await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "all_workspaces": True},
    )

    request = mock_mcp_application.retrieval.retrieve.await_args.args[0]
    assert request.workspaces == ("default", "research_notes")


async def test_mcp_all_workspaces_rejects_empty_authorized_set(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    test_config.access_control = AccessControlConfig(mode="jwt_claims", rules=[])

    with request_scope_context(RequestScope(user_id="alice", auth_mode="jwt")):
        result = await mcp_server.mcp_app.call_tool(
            "answer",
            {"query": "x", "all_workspaces": True},
        )

    assert "No workspaces" in _tool_text(result)
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_all_workspaces_is_relative_to_query_authorization(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    registered = [f"ws_{index:02d}" for index in range(14)]
    allowed = registered[:10]
    test_config.access_control = AccessControlConfig(
        mode="jwt_claims",
        rules=[
            AccessControlRuleConfig(
                claim="groups",
                value="finance-rag-readers",
                workspaces=allowed,
                actions=["workspace.query"],
            )
        ],
    )
    mock_mcp_application.corpora.alist_workspace_records.return_value = [
        {"workspace": workspace} for workspace in registered
    ]
    mock_mcp_application.retrieval.retrieve.return_value = ServiceResponse(
        contexts={"chunks": [], "entities": [], "relationships": []},
        sources=(),
        trace={},
        image_descriptions=(),
    )

    with request_scope_context(
        RequestScope(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": ["finance-rag-readers"]},
        )
    ):
        await mcp_server.mcp_app.call_tool(
            "retrieve",
            {"query": "x", "all_workspaces": True},
        )

    request = mock_mcp_application.retrieval.retrieve.await_args.args[0]
    assert request.workspaces == tuple(allowed)


async def test_mcp_rejects_unknown_argument_without_echoing_the_url(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "url",
            "url": "https://fetch.example.com/file?signature=secret",
            "no_such_option": "loose",
        },
    )

    assert "Error:" in _tool_text(result)
    # A rejected call must not replay the caller's signed URL back to the model.
    assert "signature=secret" not in _tool_text(result)


async def test_mcp_rejects_mutually_exclusive_s3_key_and_prefix(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "s3", "bucket": "b", "s3_key": "a.pdf", "prefix": "docs/"},
    )

    assert "Error:" in _tool_text(result)
    assert "mutually exclusive" in _tool_text(result)


async def test_mcp_create_workspace_uses_corpus_catalog(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "create_workspace",
        {"workspace": "New Workspace", "display_name": "New Workspace"},
    )

    body = _tool_json(result)
    assert body == {
        "workspace": "new_workspace",
        "display_name": "New Workspace",
        "created": True,
    }
    mock_mcp_application.corpora.create_workspace.assert_awaited_once_with(
        "new_workspace",
        display_name="New Workspace",
    )


async def test_mcp_delete_workspace_resets_workspace(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "delete_workspace",
        {"workspace": "Old Workspace", "keep_files": True, "dry_run": True},
    )

    body = _tool_json(result)
    assert body["workspace"] == "old_workspace"
    assert body["deleted"] is False
    mock_mcp_application.corpora.reset.assert_awaited_once_with(
        workspace_ids=("old_workspace",),
        keep_files=True,
        dry_run=True,
    )


async def test_mcp_rejects_local_path_outside_input_dir(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "path": "/tmp/report.pdf"},
    )

    assert "relative to input_dir" in _tool_text(result)
    mock_mcp_application.corpora.start_ingest_job.assert_not_awaited()


async def test_mcp_rejects_local_path_traversal(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": "../default/report.pdf",
            "workspace": "finance",
        },
    )

    assert "relative to input_dir" in _tool_text(result)
    mock_mcp_application.corpora.start_ingest_job.assert_not_awaited()


async def test_mcp_remote_prefix_ingest_starts_background_job(mock_mcp_application) -> None:
    mock_mcp_application.corpora.start_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "s3",
            "bucket": "bucket",
            "prefix": "docs/",
            "workspace": "default",
        },
    )

    assert _tool_json(result) == {
        "job_id": "job-1",
        "workspace": "default",
        "source_type": "s3",
        "status": "queued",
    }
    mock_mcp_application.corpora.start_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(source_type="s3", bucket="bucket", prefix="docs/"),
    )


async def test_mcp_requests_stay_bound_to_running_application_config(
    mock_mcp_application,
    test_config: DlightragConfig,
    tmp_path,
) -> None:
    from dlightrag.config import set_config

    application_config = test_config.model_copy(
        update={
            "workspace": "Application Workspace",
            "working_dir": str(tmp_path / "application-storage"),
        }
    )
    global_config = test_config.model_copy(
        update={
            "workspace": "Global Workspace",
            "working_dir": str(tmp_path / "global-storage"),
        }
    )
    mock_mcp_application.config = application_config
    mock_mcp_application.corpora.start_ingest_job.return_value = {
        "job_id": "job-application-config",
        "workspace": "application_workspace",
        "source_type": "local",
        "status": "queued",
    }
    set_config(global_config)

    await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "path": "report.pdf"},
    )

    expected_path = str(
        (application_config.input_dir_path / "application_workspace" / "report.pdf").resolve()
    )
    mock_mcp_application.corpora.start_ingest_job.assert_awaited_once_with(
        "application_workspace",
        IngestSpec(source_type="local", path=expected_path),
    )


async def test_mcp_get_ingest_job_reads_corpus_job(mock_mcp_application) -> None:
    mock_mcp_application.corpora.get_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "status": "running",
            "processed_items": 64,
        }
    )

    result = await mcp_server.mcp_app.call_tool("get_ingest_job", {"job_id": "job-1"})

    assert _tool_json(result) == {
        "job_id": "job-1",
        "status": "running",
        "processed_items": 64,
    }
    mock_mcp_application.corpora.get_ingest_job.assert_awaited_once_with("job-1")


async def test_mcp_answer_returns_a_descriptor_without_waiting(
    mock_mcp_application: AsyncMock,
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {
            "query": "Follow up",
            "workspaces": ["default"],
            "top_k": 8,
            "chunk_top_k": 12,
            "attachments": [{"url": "https://example.com/report.pdf", "filename": "report.pdf"}],
            "filters": {"title": "Manual"},
            "semantic_highlights": True,
            "idempotency_key": "key-1",
        },
    )

    body = _tool_json(result)
    assert body == {
        "run_id": _RUN_ID,
        "status": "queued",
        "cancel_requested": False,
        "created_at": _CREATED_AT.isoformat(),
    }
    # The tool call never holds the run open, so no answer text is returned here.
    assert "answer" not in body

    answer_request = mock_mcp_application.answers.create.await_args.kwargs["request"]
    call_kwargs = mock_mcp_application.answers.create.await_args.kwargs
    assert answer_request.workspaces == ("default",)
    assert answer_request.top_k == 8
    assert answer_request.chunk_top_k == 12
    assert answer_request.semantic_highlights is True
    assert call_kwargs["idempotency_key"] == "key-1"
    assert call_kwargs["owner_id"] == _EXPECTED_OWNER
    assert answer_request.filters is not None
    assert answer_request.filters.title == "Manual"
    resources = answer_request.resources
    assert [resource.url for resource in resources] == ["https://example.com/report.pdf"]
    assert resources[0].filename == "report.pdf"
    assert resources[0].content is None


async def test_mcp_answer_reports_a_reused_key_with_different_input(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.runtime import IdempotencyKeyConflict

    mock_mcp_application.answers.create.side_effect = IdempotencyKeyConflict("reused")

    result = await mcp_server.mcp_app.call_tool(
        "answer", {"query": "x", "idempotency_key": "key-1"}
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert "idempotency_key" in _tool_text(result)


async def test_mcp_status_returns_the_canonical_result_and_sanitizes_contexts(
    mock_mcp_application: AsyncMock,
) -> None:
    mock_mcp_application.answers.get.return_value = _run_record(
        status="succeeded",
        result=_stored_result(),
    )

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_answer_run", {"run_id": _RUN_ID}))

    assert body["status"] == "succeeded"
    assert body["result"]["answer"] == "Answer [1-1]."
    assert body["result"]["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in body["result"]["contexts"]["chunks"][0]
    assert body["result"]["sources"][0]["source_uri"] == "local://default/report.pdf"
    assert body["result"]["sources"][0]["download_url"] is None
    assert {"workspace", "download_locator", "path", "url"}.isdisjoint(body["result"]["sources"][0])
    assert mock_mcp_application.answers.get.await_args.kwargs["owner_id"] == _EXPECTED_OWNER


async def test_mcp_status_keeps_the_recorded_answer_image_transport_state(
    mock_mcp_application: AsyncMock,
) -> None:
    """An image the answer model never received must not read as if it had."""
    stored = _stored_result()
    stored["answer_images"] = [
        {
            "id": "c1",
            "chunk_id": "c1",
            "workspace": "default",
            "source_ref": "1-1",
            "label": "Figure 1",
            "answer_image_sent": False,
        }
    ]
    mock_mcp_application.answers.get.return_value = _run_record(status="succeeded", result=stored)

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_answer_run", {"run_id": _RUN_ID}))

    assert body["result"]["answer_images"][0]["answer_image_sent"] is False


async def test_mcp_status_reports_a_failed_run_with_its_public_error(
    mock_mcp_application: AsyncMock,
) -> None:
    mock_mcp_application.answers.get.return_value = _run_record(
        status="failed",
        error_kind="answer_stream_failed",
        error_message="Service error.",
    )

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_answer_run", {"run_id": _RUN_ID}))

    assert body["status"] == "failed"
    assert body["error_kind"] == "answer_stream_failed"
    assert body["error_message"] == "Service error."
    assert body["result"] is None


@pytest.mark.parametrize("tool", ["get_answer_run", "cancel_answer_run"])
async def test_mcp_never_reveals_another_owners_run(
    mock_mcp_application: AsyncMock, tool: str
) -> None:
    mock_mcp_application.answers.get.return_value = None
    mock_mcp_application.answers.cancel.return_value = SimpleNamespace(outcome="unknown", run=None)

    result = await mcp_server.mcp_app.call_tool(tool, {"run_id": _RUN_ID})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == f"Error: Answer run not found: {_RUN_ID}"


async def test_mcp_cancel_reports_the_pending_request(mock_mcp_application: AsyncMock) -> None:
    running = _run_record(status="running", cancel_requested=True)
    mock_mcp_application.answers.cancel.return_value = SimpleNamespace(
        outcome="pending", run=running
    )

    body = _tool_json(await mcp_server.mcp_app.call_tool("cancel_answer_run", {"run_id": _RUN_ID}))

    assert body["status"] == "running"
    assert body["cancel_requested"] is True
    assert mock_mcp_application.answers.cancel.await_args.kwargs["owner_id"] == _EXPECTED_OWNER


async def test_mcp_answer_preserves_answer_input_error_kind(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.answer.errors import ANSWER_INPUT_OVERFLOW, AnswerInputOverflowError

    mock_mcp_application.answers.create.side_effect = AnswerInputOverflowError(
        "The answer input is too large."
    )

    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == f"Error [{ANSWER_INPUT_OVERFLOW}]: The answer input is too large."


async def test_mcp_answer_reports_tool_misconfiguration_as_a_server_failure(
    mock_mcp_application: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.answer.errors import (
        INVALID_TOOL_CONFIGURATION,
        InvalidToolConfigurationError,
    )

    mock_mcp_application.answers.create.side_effect = InvalidToolConfigurationError(
        ("read_resource",)
    )

    with caplog.at_level(logging.WARNING):
        result = await mcp_server.mcp_app.call_tool("answer", {"query": "x"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == (
        f"Error [{INVALID_TOOL_CONFIGURATION}]: Answer tooling is misconfigured."
    )
    assert "read_resource" not in _tool_text(result)
    assert [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.parametrize(
    "descriptor",
    [
        {"path": "/etc/passwd"},
        {"url": "https://example.com/x.pdf", "path": "/etc/passwd"},
        {"url": "https://example.com/x.pdf", "content": "aGVsbG8="},
        {"url": "http://example.com/x.pdf"},
    ],
)
async def test_mcp_answer_rejects_local_and_base64_attachments(
    mock_mcp_application, descriptor: dict[str, Any]
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {"query": "x", "attachments": [descriptor]},
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_answer_enforces_link_count_limit(mock_mcp_application) -> None:
    mock_mcp_application.config.answer.max_attachments = 2

    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {
            "query": "x",
            "attachments": [{"url": f"https://example.com/{index}.pdf"} for index in range(3)],
        },
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_answer_rejects_top_level_local_fields(mock_mcp_application) -> None:
    for field in ("path", "attachment_bytes", "attachment_base64"):
        result = await mcp_server.mcp_app.call_tool(
            "answer",
            {"query": "x", field: "value"},
        )
        assert isinstance(result, CallToolResult)
        assert result.is_error is True
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_delete_files_forwards_dry_run(mock_mcp_application) -> None:
    mock_mcp_application.corpora.delete_files = AsyncMock(return_value=[{"status": "would_delete"}])

    result = await mcp_server.mcp_app.call_tool(
        "delete_files",
        {"filenames": ["report.pdf"], "dry_run": True},
    )

    assert _tool_json(result)["results"] == [{"status": "would_delete"}]
    mock_mcp_application.corpora.delete_files.assert_awaited_once_with(
        "default",
        filenames=["report.pdf"],
        file_paths=None,
        dry_run=True,
    )


async def test_mcp_file_tools_canonicalize_display_workspace_before_access_and_manager(
    mock_mcp_application,
) -> None:
    mock_mcp_application.corpora.list_ingested_files.return_value = []
    mock_mcp_application.corpora.delete_files.return_value = []

    listed = await mcp_server.mcp_app.call_tool(
        "list_files",
        {"workspace": "Finance Reports"},
    )
    deleted = await mcp_server.mcp_app.call_tool(
        "delete_files",
        {"workspace": "Finance Reports", "filenames": ["report.pdf"]},
    )

    assert _tool_json(listed)["workspace"] == "finance_reports"
    assert _tool_json(deleted)["workspace"] == "finance_reports"
    mock_mcp_application.corpora.list_ingested_files.assert_awaited_once_with("finance_reports")
    mock_mcp_application.corpora.delete_files.assert_awaited_once_with(
        "finance_reports",
        filenames=["report.pdf"],
        file_paths=None,
        dry_run=False,
    )


async def test_mcp_ingest_job_tools_canonicalize_stored_workspace_before_access(
    mock_mcp_application,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = {
        "job_id": "job-1",
        "workspace": "Finance Reports",
        "source_type": "s3",
        "status": "running",
    }
    mock_mcp_application.corpora.get_ingest_job.return_value = job
    mock_mcp_application.corpora.cancel_ingest_job.return_value = job
    enforce = AsyncMock()
    monkeypatch.setattr(mcp_server, "_enforce_access", enforce)

    await mcp_server.mcp_app.call_tool("get_ingest_job", {"job_id": "job-1"})
    await mcp_server.mcp_app.call_tool("cancel_ingest_job", {"job_id": "job-1"})

    assert [call.args[1] for call in enforce.await_args_list] == [
        "finance_reports",
        "finance_reports",
    ]
