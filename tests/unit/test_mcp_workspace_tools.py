# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

import json
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import dlightrag
from dlightrag.citations.schemas import SourceReference
from dlightrag.config import AccessControlConfig, AccessControlRuleConfig, DlightragConfig
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.scope import RequestScope, request_scope_context
from dlightrag.mcp import server as mcp_server
from dlightrag.models.schemas import Reference

_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def _metadata_policy_enum(schema: dict, prop: dict) -> list[str]:
    if "enum" in prop:
        return prop["enum"]
    return next(
        resolved["enum"]
        for item in prop["anyOf"]
        if "enum" in (resolved := _resolve_ref(schema, item))
    )


def _resolve_ref(schema: dict, item: dict) -> dict:
    ref = item.get("$ref")
    if not ref:
        return item
    name = ref.removeprefix("#/$defs/")
    return schema["$defs"][name]


def _query_image_schema(schema: dict, prop: dict) -> dict:
    return _resolve_ref(schema, prop["items"])


def _tool_content(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def _tool_text(result) -> str:
    return _tool_content(result)[0].text


def _tool_json(result):
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


def test_mcp_server_info_uses_dlightrag_version() -> None:
    assert mcp_server.server.version == dlightrag.__version__


async def test_mcp_success_payloads_keep_fastmcp_structured_output(mock_mcp_manager) -> None:
    mock_mcp_manager.aget_ingest_job = AsyncMock(
        return_value={"job_id": "job-1", "status": "running"}
    )

    result = await mcp_server.mcp_app.call_tool("get_ingest_job", {"job_id": "job-1"})

    assert isinstance(result, tuple)
    assert _tool_json(result) == {"job_id": "job-1", "status": "running"}
    assert result[1] == {"job_id": "job-1", "status": "running"}


async def test_mcp_lists_workspace_lifecycle_tools() -> None:
    tools = await mcp_server.mcp_app.list_tools()
    names = {tool.name for tool in tools}

    assert hasattr(mcp_server, "mcp_app")
    assert "create_workspace" in names
    assert "delete_workspace" in names
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.inputSchema["properties"]
    assert "args" not in answer_props
    assert "query" in answer_props
    assert "conversation_history" not in answer_props
    assert "query_images" in answer_props
    image_block_schema = _query_image_schema(answer_tool.inputSchema, answer_props["query_images"])
    assert image_block_schema["type"] == "object"
    assert image_block_schema["properties"]["type"]["const"] == "image_url"
    assert set(image_block_schema["required"]) == {"type", "image_url"}
    assert "chunk_top_k" in answer_props
    assert "session_id" not in answer_props
    assert "referenced_image_ids" not in answer_props
    assert answer_props["semantic_highlights"]["default"] is False
    assert "filters" in answer_props
    assert answer_props["all_workspaces"]["default"] is False
    assert "bm25_query" not in answer_props
    retrieve_tool = next(tool for tool in tools if tool.name == "retrieve")
    retrieve_props = retrieve_tool.inputSchema["properties"]
    assert "semantic_highlights" not in retrieve_props
    assert "chunk_top_k" in retrieve_props
    assert "bm25_query" in retrieve_props
    assert "session_id" not in retrieve_props
    assert "referenced_image_ids" not in retrieve_props
    assert retrieve_props["all_workspaces"]["default"] is False
    retrieve_image_block_schema = _query_image_schema(
        retrieve_tool.inputSchema,
        retrieve_props["query_images"],
    )
    assert retrieve_image_block_schema["type"] == "object"
    assert retrieve_image_block_schema["properties"]["type"]["const"] == "image_url"
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    ingest_props = ingest_tool.inputSchema["properties"]
    assert "title" in ingest_props
    assert "all_workspaces" not in ingest_props
    assert "author" in ingest_props
    assert "metadata" in ingest_props
    assert "url" in ingest_props
    assert "urls" in ingest_props
    assert "s3_region" in ingest_props
    assert "filename" in ingest_props
    assert "source_uri" in ingest_props
    assert "source_uris" in ingest_props
    assert "download_uri" in ingest_props
    assert "download_uris" in ingest_props
    assert "download_url" not in ingest_props
    assert "download_urls" not in ingest_props
    assert "documents" in ingest_props
    assert "retain_source_file" in ingest_props
    assert "fetch" in ingest_props["url"]["description"].lower()
    assert "signed" in ingest_props["url"]["description"].lower()
    assert "identity" in ingest_props["source_uri"]["description"].lower()
    assert "queryless" in ingest_props["download_uri"]["description"].lower()
    assert "signed" in ingest_props["retain_source_file"]["description"].lower()
    delete_files_tool = next(tool for tool in tools if tool.name == "delete_files")
    assert "dry_run" in delete_files_tool.inputSchema["properties"]
    assert _metadata_policy_enum(ingest_tool.inputSchema, ingest_props["metadata_policy"]) == [
        "validate",
        "reject_unknown",
        "store_only",
    ]
    assert "get_ingest_job" in names


async def test_mcp_management_tool_descriptions_explain_contracts() -> None:
    tools = {tool.name: tool for tool in await mcp_server.mcp_app.list_tools()}

    expected_fragments = {
        "list_workspaces": [
            "visible",
            "records",
            "display_name",
            "embedding_model",
            "created_at",
            "updated_at",
        ],
        "create_workspace": ["display_name", "user-facing", "normalized", "created"],
        "delete_workspace": ["dry_run", "keep_files", "deleted", "result"],
        "ingest": ["durable", "job_id", "status", "workspace"],
        "get_ingest_job": ["job_id", "status", "workspace"],
        "list_files": ["files", "count", "workspace"],
        "delete_files": ["dry_run", "results", "workspace"],
    }

    for tool_name, fragments in expected_fragments.items():
        description = (tools[tool_name].description or "").lower()
        for fragment in fragments:
            assert fragment.lower() in description, f"{tool_name} missing {fragment!r}"


def test_mcp_security_defaults_are_loopback_only() -> None:
    cfg = cast(Any, DlightragConfig)()

    assert cfg.mcp_allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert cfg.mcp_allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


def test_mcp_dns_rebinding_protection_follows_auth_mode() -> None:
    none_cfg = cast(Any, DlightragConfig)()
    simple_cfg = cast(Any, DlightragConfig)(auth_mode="simple", api_auth_token="test-token")
    jwt_cfg = cast(Any, DlightragConfig)(auth_mode="jwt", jwt_verification_key="test-key")

    assert none_cfg.mcp_dns_rebinding_protection is True
    assert simple_cfg.mcp_dns_rebinding_protection is False
    assert jwt_cfg.mcp_dns_rebinding_protection is False


async def test_mcp_rejects_unknown_mode_without_schema_wrapper(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x", "mode": "mix"})

    content = _tool_content(result)
    assert content[0].type == "text"
    assert "Error:" in content[0].text
    assert "mode" in content[0].text
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


async def test_mcp_rejects_invalid_metadata_policy(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "metadata_policy": "loose"},
    )

    assert "Error:" in _tool_text(result)
    assert "metadata_policy" in _tool_text(result)
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


async def test_mcp_url_ingest_starts_background_job(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "url",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "url",
            "url": "https://api.bynder.com/docs/getting-started",
            "filename": "getting-started.html",
            "workspace": "default",
        },
    )

    assert _tool_json(result) == {
        "job_id": "job-1",
        "workspace": "default",
        "source_type": "url",
        "status": "queued",
    }
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(
            source_type="url",
            url="https://api.bynder.com/docs/getting-started",
            filename="getting-started.html",
        ),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_url_ingest_accepts_stable_source_uri(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "url",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "url",
            "url": "https://cdn.example.com/download?id=asset-1&signature=secret",
            "filename": "asset.pdf",
            "source_uri": "bynder://asset/asset-1",
            "workspace": "default",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(
            source_type="url",
            url="https://cdn.example.com/download?id=asset-1&signature=secret",
            filename="asset.pdf",
            source_uri="bynder://asset/asset-1",
        ),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_url_ingest_accepts_download_uris(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "url",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "url",
            "urls": [
                "https://fetch.example.com/download?id=a&signature=secret",
                "https://fetch.example.com/download?id=b&signature=secret",
            ],
            "download_uris": [
                "https://cdn.example.com/a.pdf",
                "https://cdn.example.com/b.pdf",
            ],
            "workspace": "default",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(
            source_type="url",
            urls=[
                "https://fetch.example.com/download?id=a&signature=secret",
                "https://fetch.example.com/download?id=b&signature=secret",
            ],
            download_uris=[
                "https://cdn.example.com/a.pdf",
                "https://cdn.example.com/b.pdf",
            ],
        ),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_local_directory_ingest_starts_background_job(
    mock_mcp_manager, test_config: DlightragConfig
) -> None:
    docs_dir = test_config.input_dir_path / "default" / "docs"
    docs_dir.mkdir(parents=True)
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": "docs",
            "workspace": "default",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(source_type="local", path=str(docs_dir)),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_single_s3_key_defaults_to_background_job(mock_mcp_manager) -> None:
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
            "s3_key": "docs/file.pdf",
            "workspace": "default",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        IngestSpec(source_type="s3", bucket="bucket", s3_key="docs/file.pdf"),
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
            "answer_context_top_k": 4,
            "query_images": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ],
            "filters": {"doc_title": "Manual"},
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
    assert call_kwargs["answer_context_top_k"] == 4
    assert call_kwargs["query_images"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]
    assert "conversation_history" not in call_kwargs
    assert "session_id" not in call_kwargs
    assert "referenced_image_ids" not in call_kwargs
    assert call_kwargs["filters"].doc_title == "Manual"
    assert call_kwargs["semantic_highlights"] is True


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
