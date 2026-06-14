# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock

import pytest

from dlightrag.citations.schemas import SourceReference
from dlightrag.config import DlightragConfig
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.mcp import server as mcp_server
from dlightrag.models.schemas import Reference


@pytest.fixture
def mock_mcp_manager(monkeypatch):
    manager = AsyncMock()
    manager.list_workspaces = AsyncMock(return_value=["default"])
    manager.acreate_workspace = AsyncMock()
    manager.areset = AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0})
    monkeypatch.setattr(mcp_server, "_ensure_manager", AsyncMock(return_value=manager))
    return manager


async def test_mcp_lists_workspace_lifecycle_tools() -> None:
    tools = await mcp_server.list_tools()
    names = {tool.name for tool in tools}

    assert "create_workspace" in names
    assert "delete_workspace" in names
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.inputSchema["properties"]
    assert "conversation_history" in answer_props
    assert "query_images" in answer_props
    assert answer_props["query_images"]["items"]["anyOf"] == [
        {"type": "string"},
        {"type": "object"},
    ]
    assert "session_id" in answer_props
    assert "referenced_image_ids" in answer_props
    assert "filters" in answer_props
    retrieve_tool = next(tool for tool in tools if tool.name == "retrieve")
    retrieve_props = retrieve_tool.inputSchema["properties"]
    assert retrieve_props["query_images"]["items"]["anyOf"] == [
        {"type": "string"},
        {"type": "object"},
    ]
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    ingest_props = ingest_tool.inputSchema["properties"]
    assert "title" in ingest_props
    assert "author" in ingest_props
    assert "metadata" in ingest_props
    assert "wait" in ingest_props
    assert ingest_props["metadata_policy"]["enum"] == [
        "validate",
        "reject_unknown",
        "store_only",
    ]
    assert "ingest_job_status" in names


def test_mcp_streamable_http_uses_modern_transport_defaults() -> None:
    source = inspect.getsource(mcp_server.run_streamable_http)

    assert "StreamableHTTPSessionManager" in source
    assert "TransportSecuritySettings" in source
    assert "enable_dns_rebinding_protection=True" in source
    assert "json_response=True" in source
    assert "stateless=True" in source
    assert "MCPPathMiddleware" in source
    assert 'Mount("/mcp"' in source
    assert '"/sse"' not in source
    assert '"/messages"' not in source
    assert "StreamableHTTPServerTransport" not in source


def test_mcp_security_defaults_are_loopback_only() -> None:
    cfg = DlightragConfig()

    assert cfg.mcp_allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert cfg.mcp_allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


async def test_mcp_create_workspace_uses_manager_registry(mock_mcp_manager) -> None:
    result = await mcp_server.call_tool(
        "create_workspace",
        {"workspace": "New Workspace", "display_name": "New Workspace"},
    )

    body = json.loads(result[0].text)
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
    result = await mcp_server.call_tool(
        "delete_workspace",
        {"workspace": "Old Workspace", "keep_files": True, "dry_run": True},
    )

    body = json.loads(result[0].text)
    assert body["workspace"] == "old_workspace"
    assert body["deleted"] is False
    mock_mcp_manager.areset.assert_awaited_once_with(
        workspace="Old Workspace",
        keep_files=True,
        dry_run=True,
    )


async def test_mcp_ingest_forwards_document_metadata(mock_mcp_manager) -> None:
    mock_mcp_manager.aingest = AsyncMock(return_value={"ingested": 1})

    result = await mcp_server.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": "/tmp/report.pdf",
            "workspace": "finance",
            "title": "Quarterly Report",
            "author": "Ada",
            "metadata": {"department": "Finance"},
            "metadata_policy": "reject_unknown",
        },
    )

    assert json.loads(result[0].text) == {"ingested": 1}
    mock_mcp_manager.aingest.assert_awaited_once_with(
        "finance",
        source_type="local",
        path="/tmp/report.pdf",
        title="Quarterly Report",
        author="Ada",
        metadata={"department": "Finance"},
        metadata_policy="reject_unknown",
    )


async def test_mcp_remote_prefix_ingest_starts_background_job(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )

    result = await mcp_server.call_tool(
        "ingest",
        {
            "source_type": "s3",
            "bucket": "bucket",
            "prefix": "docs/",
            "workspace": "default",
        },
    )

    assert json.loads(result[0].text) == {
        "job_id": "job-1",
        "workspace": "default",
        "source_type": "s3",
        "status": "queued",
    }
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        source_type="s3",
        bucket="bucket",
        prefix="docs/",
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_local_directory_ingest_starts_background_job(mock_mcp_manager, tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
        }
    )

    result = await mcp_server.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": str(docs_dir),
            "workspace": "default",
        },
    )

    assert json.loads(result[0].text)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        source_type="local",
        path=str(docs_dir),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_single_s3_key_can_force_background_job(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )

    result = await mcp_server.call_tool(
        "ingest",
        {
            "source_type": "s3",
            "bucket": "bucket",
            "key": "docs/file.pdf",
            "workspace": "default",
            "wait": False,
        },
    )

    assert json.loads(result[0].text)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        source_type="s3",
        bucket="bucket",
        key="docs/file.pdf",
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_ingest_job_status_reads_manager_job(mock_mcp_manager) -> None:
    mock_mcp_manager.get_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "status": "running",
            "processed_items": 64,
        }
    )

    result = await mcp_server.call_tool("ingest_job_status", {"job_id": "job-1"})

    assert json.loads(result[0].text) == {
        "job_id": "job-1",
        "status": "running",
        "processed_items": 64,
    }
    mock_mcp_manager.get_ingest_job.assert_awaited_once_with("job-1")


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
            sources=[SourceReference(id="1", title="report.pdf", path="/private/report.pdf")],
        )
    )

    result = await mcp_server.call_tool(
        "answer",
        {
            "query": "Follow up",
            "workspaces": ["default"],
            "top_k": 8,
            "answer_candidate_top_k": 12,
            "answer_context_top_k": 4,
            "conversation_history": [{"role": "user", "content": "Previous"}],
            "query_images": ["data:image/png;base64,abc"],
            "session_id": "session-1",
            "referenced_image_ids": ["img_1"],
            "filters": {"doc_title": "Manual"},
        },
    )

    body = json.loads(result[0].text)
    assert body["answer"] == "Answer [1-1]."
    assert body["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in body["contexts"]["chunks"][0]
    assert body["sources"][0]["id"] == "1"

    call_kwargs = mock_mcp_manager.aanswer.call_args.kwargs
    assert call_kwargs["workspaces"] == ["default"]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["answer_candidate_top_k"] == 12
    assert call_kwargs["answer_context_top_k"] == 4
    assert call_kwargs["conversation_history"] == [{"role": "user", "content": "Previous"}]
    assert call_kwargs["query_images"] == ["data:image/png;base64,abc"]
    assert call_kwargs["session_id"] == "session-1"
    assert call_kwargs["referenced_image_ids"] == ["img_1"]
    assert call_kwargs["filters"].doc_title == "Manual"
