# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

import json
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from mcp import Client, MCPError
from mcp.types import INVALID_PARAMS, CallToolResult, InputRequiredResult, TextContent

from dlightrag.citations.schemas import SourceReference
from dlightrag.config import AccessControlConfig, AccessControlRuleConfig, DlightragConfig
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.scope import RequestScope, request_scope_context
from dlightrag.mcp import server as mcp_server
from dlightrag.models.schemas import Reference

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
def mock_mcp_manager(monkeypatch):
    manager = AsyncMock()
    manager.alist_workspaces = AsyncMock(return_value=["default"])
    manager.alist_workspace_records = AsyncMock(return_value=[{"workspace": "default"}])
    manager.acreate_workspace = AsyncMock()
    manager.areset = AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0})
    manager.aretrieve = AsyncMock()
    manager.aanswer = AsyncMock()
    manager.aingest = AsyncMock()
    manager.astart_ingest_job = AsyncMock()
    monkeypatch.setattr(mcp_server, "_ensure_manager", AsyncMock(return_value=manager))
    return manager


async def test_get_capabilities_reports_answer_image_capability(
    mock_mcp_manager: AsyncMock,
) -> None:
    from dlightrag.core.answer.capability import AnswerImageCapability

    mock_mcp_manager.answer_image_capability = AnswerImageCapability(
        status="supported",
        configured_ceiling=8,
        effective_max_images=6,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )

    result = await mcp_server.mcp_app.call_tool("get_capabilities", {})

    cap = _tool_json(result)["answer_image_capability"]
    assert cap["status"] == "supported"
    assert cap["effective_max_images"] == 6
    assert cap["configured_ceiling"] == 8
    assert cap["model"] == "test-model"


async def test_mcp_v2_client_lists_and_calls_tools(mock_mcp_manager: AsyncMock) -> None:
    mock_mcp_manager.aget_ingest_job = AsyncMock(
        return_value={"job_id": "job-1", "status": "running"}
    )

    async with Client(mcp_server.mcp_app) as client:
        listing = await client.list_tools()
        result = await client.call_tool("get_ingest_job", {"job_id": "job-1"})

    assert "retrieve" in {tool.name for tool in listing.tools}
    assert result.is_error is False
    assert _tool_json(result) == {"job_id": "job-1", "status": "running"}
    assert result.structured_content == {"job_id": "job-1", "status": "running"}


async def test_mcp_internal_errors_do_not_leak_details(mock_mcp_manager: AsyncMock) -> None:
    mock_mcp_manager.alist_workspace_records.side_effect = RuntimeError("database-secret")

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
        "cancel_ingest_job",
        "create_workspace",
        "delete_files",
        "delete_workspace",
        "get_capabilities",
        "get_ingest_job",
        "ingest",
        "list_files",
        "list_workspaces",
        "retrieve",
    }
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.input_schema["properties"]
    assert {"query", "history", "query_images", "filters", "chunk_top_k"} <= answer_props.keys()
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


async def test_mcp_rejects_unknown_mode_without_schema_wrapper(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x", "mode": "mix"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert "Error:" in _tool_text(result)
    assert "mode" in _tool_text(result)
    mock_mcp_manager.aanswer.assert_not_awaited()


@pytest.mark.parametrize(
    ("tool_name", "payload", "manager_method", "error_fragment"),
    [
        (
            "retrieve",
            {
                "query": "x",
                "query_images": [_IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK],
            },
            "aretrieve",
            "query_images",
        ),
        (
            "answer",
            {"query": "x", "query_images": [{"url": "data:image/png;base64,abc"}]},
            "aanswer",
            "image_url",
        ),
        (
            "retrieve",
            {"query": "x", "query_images": ["data:image/png;base64,abc"]},
            "aretrieve",
            "valid dictionary",
        ),
    ],
)
async def test_mcp_rejects_invalid_query_image_payloads(
    mock_mcp_manager,
    tool_name: str,
    payload: dict[str, Any],
    manager_method: str,
    error_fragment: str,
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        tool_name,
        payload,
    )

    assert "Error:" in _tool_text(result)
    assert error_fragment in _tool_text(result)
    getattr(mock_mcp_manager, manager_method).assert_not_awaited()


async def test_mcp_retrieve_forwards_chunk_top_k(mock_mcp_manager) -> None:
    mock_mcp_manager.aretrieve = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))

    await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "top_k": 8, "chunk_top_k": 5},
    )

    await_args = mock_mcp_manager.aretrieve.await_args
    assert await_args is not None
    call_kwargs = await_args.kwargs
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["chunk_top_k"] == 5


async def test_mcp_retrieve_uses_shared_executor(
    mock_mcp_manager: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))
    monkeypatch.setattr(mcp_server, "execute_retrieve", execute)

    result = await mcp_server.mcp_app.call_tool("retrieve", {"query": "x"})

    assert _tool_json(result)["contexts"] == {"chunks": [], "entities": [], "relationships": []}
    execute.assert_awaited_once()
    mock_mcp_manager.aretrieve.assert_not_awaited()


async def test_mcp_jwt_claims_access_control_denies_unmapped_workspace(
    mock_mcp_manager,
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
    mock_mcp_manager.aretrieve.assert_not_awaited()


async def test_mcp_retrieve_all_workspaces_uses_visible_records(mock_mcp_manager) -> None:
    mock_mcp_manager.alist_workspace_records.return_value = [
        {"workspace": "default"},
        {"workspace": "research_notes"},
    ]
    mock_mcp_manager.aretrieve.return_value = RetrievalResult(contexts={"chunks": []})

    await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "all_workspaces": True},
    )

    assert mock_mcp_manager.aretrieve.await_args.kwargs["workspaces"] == [
        "default",
        "research_notes",
    ]


async def test_mcp_all_workspaces_rejects_empty_authorized_set(
    mock_mcp_manager,
    test_config: DlightragConfig,
) -> None:
    test_config.access_control = AccessControlConfig(mode="jwt_claims", rules=[])

    with request_scope_context(RequestScope(user_id="alice", auth_mode="jwt")):
        result = await mcp_server.mcp_app.call_tool(
            "answer",
            {"query": "x", "all_workspaces": True},
        )

    assert "No workspaces" in _tool_text(result)
    mock_mcp_manager.aanswer.assert_not_awaited()


async def test_mcp_all_workspaces_is_relative_to_query_authorization(
    mock_mcp_manager,
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
    mock_mcp_manager.alist_workspace_records.return_value = [
        {"workspace": workspace} for workspace in registered
    ]
    mock_mcp_manager.aretrieve.return_value = RetrievalResult(contexts={"chunks": []})

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

    assert mock_mcp_manager.aretrieve.await_args.kwargs["workspaces"] == allowed


async def test_mcp_rejects_unknown_argument_without_echoing_the_url(mock_mcp_manager) -> None:
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
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_rejects_mutually_exclusive_s3_key_and_prefix(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "s3", "bucket": "b", "s3_key": "a.pdf", "prefix": "docs/"},
    )

    assert "Error:" in _tool_text(result)
    assert "mutually exclusive" in _tool_text(result)
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_create_workspace_uses_manager_registry(mock_mcp_manager) -> None:
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
    mock_mcp_manager.acreate_workspace.assert_awaited_once_with(
        "new_workspace",
        display_name="New Workspace",
    )


async def test_mcp_delete_workspace_resets_workspace(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "delete_workspace",
        {"workspace": "Old Workspace", "keep_files": True, "dry_run": True},
    )

    body = _tool_json(result)
    assert body["workspace"] == "old_workspace"
    assert body["deleted"] is False
    mock_mcp_manager.areset.assert_awaited_once_with(
        workspace="Old Workspace",
        keep_files=True,
        dry_run=True,
    )


async def test_mcp_rejects_local_path_outside_input_dir(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "path": "/tmp/report.pdf"},
    )

    assert "relative to input_dir" in _tool_text(result)
    mock_mcp_manager.astart_ingest_job.assert_not_awaited()
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_rejects_local_path_traversal(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": "../default/report.pdf",
            "workspace": "finance",
        },
    )

    assert "relative to input_dir" in _tool_text(result)
    mock_mcp_manager.astart_ingest_job.assert_not_awaited()
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_remote_prefix_ingest_starts_background_job(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
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
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(source_type="s3", bucket="bucket", prefix="docs/"),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_get_ingest_job_reads_manager_job(mock_mcp_manager) -> None:
    mock_mcp_manager.aget_ingest_job = AsyncMock(
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
    mock_mcp_manager.aget_ingest_job.assert_awaited_once_with("job-1")


async def test_mcp_answer_forwards_manager_answer_capabilities_and_sanitizes_contexts(
    mock_mcp_manager,
) -> None:
    mock_mcp_manager.aanswer = AsyncMock(
        return_value=RetrievalResult(
            answer="Answer [1-1].",
            contexts={
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
            references=[Reference(id="1", title="report.pdf")],
            sources=[
                SourceReference(
                    id="1",
                    title="report.pdf",
                    source_uri="local://default/report.pdf",
                    workspace="default",
                    document_id="doc-report",
                    download_locator="/private/report.pdf",
                )
            ],
        )
    )

    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {
            "query": "Follow up",
            "workspaces": ["default"],
            "top_k": 8,
            "chunk_top_k": 12,
            "query_images": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ],
            "filters": {"title": "Manual"},
            "semantic_highlights": True,
        },
    )

    body = _tool_json(result)
    assert body["answer"] == "Answer [1-1]."
    assert body["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in body["contexts"]["chunks"][0]
    assert body["sources"][0]["id"] == "1"
    assert body["sources"][0]["source_uri"] == "local://default/report.pdf"
    assert body["sources"][0]["download_url"] is None
    assert {"workspace", "download_locator", "path", "url"}.isdisjoint(body["sources"][0])

    call_kwargs = mock_mcp_manager.aanswer.call_args.kwargs
    assert call_kwargs["workspaces"] == ["default"]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["chunk_top_k"] == 12
    assert call_kwargs["query_images"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]
    assert "conversation_history" not in call_kwargs
    assert "session_id" not in call_kwargs
    assert "referenced_image_ids" not in call_kwargs
    assert call_kwargs["filters"].title == "Manual"
    assert call_kwargs["semantic_highlights"] is True


async def test_mcp_answer_uses_shared_executor(
    mock_mcp_manager: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = AsyncMock(return_value=RetrievalResult(answer="shared", contexts={"chunks": []}))
    monkeypatch.setattr(mcp_server, "execute_answer", execute)

    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x"})

    assert _tool_json(result)["answer"] == "shared"
    execute.assert_awaited_once()
    mock_mcp_manager.aanswer.assert_not_awaited()


async def test_mcp_delete_files_forwards_dry_run(mock_mcp_manager) -> None:
    mock_mcp_manager.adelete_files = AsyncMock(return_value=[{"status": "would_delete"}])

    result = await mcp_server.mcp_app.call_tool(
        "delete_files",
        {"filenames": ["report.pdf"], "dry_run": True},
    )

    assert _tool_json(result)["results"] == [{"status": "would_delete"}]
    mock_mcp_manager.adelete_files.assert_awaited_once_with(
        "default",
        filenames=["report.pdf"],
        file_paths=None,
        dry_run=True,
    )
