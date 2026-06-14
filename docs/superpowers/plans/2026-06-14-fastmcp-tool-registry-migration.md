# FastMCP Tool Registry Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MCP hand-written tool list and central `call_tool` switch with FastMCP decorators and Pydantic input contracts while preserving the existing transport and security behavior.

**Architecture:** `src/dlightrag/mcp/server.py` owns the FastMCP app and the transport startup code. `src/dlightrag/mcp/contracts.py` owns tool input models and typed field aliases. A small `DlightRAGFastMCP` subclass enforces strict unknown-argument rejection before delegating to FastMCP, because upstream FastMCP currently ignores extra keys; successful dictionary payloads keep FastMCP's native structured output with text fallback.

**Tech Stack:** Python 3.12, MCP Python SDK `FastMCP`, Pydantic v2, pytest, ruff, pyright, Docker Compose.

---

## File Structure

- Create `src/dlightrag/mcp/contracts.py`
  - Pydantic models for each MCP tool input.
  - Typed helper aliases for query images, filters, and conversation turns.
  - Cross-field validators for mutually exclusive ingest selectors and image caps.
- Modify `src/dlightrag/mcp/server.py`
  - Replace `Server(...)` with `DlightRAGFastMCP(...)`.
  - Keep `server = mcp_app._mcp_server` as the low-level transport adapter.
  - Register all tools with `@mcp_app.tool(...)`.
  - Remove module-level `list_tools()` and `call_tool()`.
  - Keep `run_stdio()` and `run_streamable_http()` behavior unchanged except that they use the FastMCP-owned low-level server.
- Modify `tests/unit/test_mcp_workspace_tools.py`
  - Use `mcp_server.mcp_app.list_tools()` and `mcp_server.mcp_app.call_tool(...)`.
  - Add red tests for no legacy handlers, no `args` schema wrapper, and strict unknown-argument rejection.
  - Keep streamable-http/security assertions.

## Task 1: FastMCP Contract Tests

**Files:**
- Modify: `tests/unit/test_mcp_workspace_tools.py`

- [ ] **Step 1: Write failing tests for the new architecture**

Add tests that assert the MCP module exposes FastMCP as the tool registry, does not retain module-level tool wrappers, does not use low-level decorators for tools, and keeps top-level schemas:

```python
def test_mcp_uses_fastmcp_registry_without_legacy_wrappers() -> None:
    source = inspect.getsource(mcp_server)

    assert hasattr(mcp_server, "mcp_app")
    assert not hasattr(mcp_server, "list_tools")
    assert not hasattr(mcp_server, "call_tool")
    assert "@server.list_tools" not in source
    assert "@server.call_tool" not in source
    assert 'if name == "retrieve"' not in source
    assert 'if name == "answer"' not in source


async def test_mcp_tool_schemas_are_top_level_fastmcp_fields() -> None:
    tools = await mcp_server.mcp_app.list_tools()
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.inputSchema["properties"]

    assert "args" not in answer_props
    assert "query" in answer_props
    assert "conversation_history" in answer_props
    assert "query_images" in answer_props
```

- [ ] **Step 2: Write failing tests for strict contracts**

Add tests that verify unknown `mode`, too many query images, invalid metadata policy, and mutually exclusive ingest selectors return text errors:

```python
async def test_mcp_rejects_unknown_mode_without_schema_wrapper(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x", "mode": "mix"})

    assert result[0].type == "text"
    assert "Error:" in result[0].text
    assert "mode" in result[0].text
    mock_mcp_manager.aanswer.assert_not_awaited()


async def test_mcp_rejects_excess_query_images(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "query_images": ["1", "2", "3", "4"]},
    )

    assert "Error:" in result[0].text
    assert "query_images" in result[0].text
    mock_mcp_manager.aretrieve.assert_not_awaited()


async def test_mcp_rejects_invalid_metadata_policy(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "metadata_policy": "loose"},
    )

    assert "Error:" in result[0].text
    assert "metadata_policy" in result[0].text
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_rejects_mutually_exclusive_s3_key_and_prefix(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "s3", "bucket": "b", "key": "a.pdf", "prefix": "docs/"},
    )

    assert "Error:" in result[0].text
    assert "mutually exclusive" in result[0].text
    mock_mcp_manager.aingest.assert_not_awaited()
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_mcp_workspace_tools.py -q
```

Expected: failures because `mcp_app` does not exist and legacy `list_tools()` / `call_tool()` still exist.

## Task 2: Add Pydantic Tool Contracts

**Files:**
- Create: `src/dlightrag/mcp/contracts.py`
- Test: `tests/unit/test_mcp_workspace_tools.py`

- [ ] **Step 1: Add the contract module**

Create `contracts.py` with strict models:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetadataPolicy = Literal["validate", "reject_unknown", "store_only"]
SourceType = Literal["local", "azure_blob", "s3"]
QueryImage = str | dict[str, Any]


class MCPInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetadataFiltersInput(MCPInput):
    filename: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    custom: dict[str, Any] | None = None


class RetrieveInput(MCPInput):
    query: str
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    filters: MetadataFiltersInput | dict[str, Any] | None = None
    query_images: list[QueryImage] | None = Field(default=None, max_length=3)
    session_id: str | None = None
    referenced_image_ids: list[str] | None = None


class AnswerInput(RetrieveInput):
    answer_candidate_top_k: int | None = None
    answer_context_top_k: int | None = None
    conversation_history: list[dict[str, Any]] | None = None


class IngestInput(MCPInput):
    source_type: SourceType
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    bucket: str | None = None
    key: str | None = None
    prefix: str | None = None
    replace: bool | None = None
    workspace: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_policy: MetadataPolicy | None = None
    wait: bool | None = None

    @model_validator(mode="after")
    def validate_selectors(self) -> "IngestInput":
        if self.source_type == "azure_blob" and self.blob_path and self.prefix is not None:
            raise ValueError("'blob_path' and 'prefix' are mutually exclusive")
        if self.source_type == "s3" and self.key and self.prefix is not None:
            raise ValueError("'key' and 'prefix' are mutually exclusive")
        return self


class IngestJobStatusInput(MCPInput):
    job_id: str


class CreateWorkspaceInput(MCPInput):
    workspace: str
    display_name: str | None = None


class DeleteWorkspaceInput(MCPInput):
    workspace: str
    keep_files: bool = False
    dry_run: bool = False


class ListFilesInput(MCPInput):
    workspace: str | None = None


class DeleteFilesInput(MCPInput):
    filenames: list[str] | None = None
    file_paths: list[str] | None = None
    workspace: str | None = None
```

- [ ] **Step 2: Run contract-focused tests**

Run:

```bash
uv run pytest tests/unit/test_mcp_workspace_tools.py::test_mcp_rejects_invalid_metadata_policy -q
```

Expected: still fails until server handlers use the contracts.

## Task 3: Replace Tool Registry With FastMCP

**Files:**
- Modify: `src/dlightrag/mcp/server.py`
- Test: `tests/unit/test_mcp_workspace_tools.py`

- [ ] **Step 1: Replace low-level tool registration with FastMCP app**

In `server.py`, replace `Server(...)` construction and low-level tool decorators with:

```python
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any, Awaitable, Callable

from mcp.server.fastmcp import FastMCP
from mcp.types import ContentBlock, TextContent


class DlightRAGFastMCP(FastMCP):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[ContentBlock] | dict[str, Any]:
        try:
            self._reject_unknown_arguments(name, arguments)
            return await super().call_tool(name, arguments)
        except Exception as exc:
            logger.exception("MCP tool '%s' failed: %s", name, exc)
            return [TextContent(type="text", text=f"Error: {exc}")]

    def _reject_unknown_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        tool = self._tool_manager._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        allowed = set(tool.parameters.get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"Unexpected argument(s) for {name}: {', '.join(unknown)}")


mcp_app = DlightRAGFastMCP(
    "dlightrag",
    log_level="INFO",
    warn_on_duplicate_tools=True,
)
server = mcp_app._mcp_server
```

Use `server` only in `run_stdio()` and `run_streamable_http()`.

- [ ] **Step 2: Add typed tool handlers**

Convert each switch branch into one decorated function. Example shapes:

```python
@mcp_app.tool(name="retrieve", description="Query the RAG knowledge base for relevant information. Supports structured metadata filters for precise document lookups.")
async def retrieve_tool(
    query: str,
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    workspaces: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    query_images: list[str | dict[str, Any]] | None = None,
    session_id: str | None = None,
    referenced_image_ids: list[str] | None = None,
) -> dict[str, Any]:
    args = RetrieveInput.model_validate(locals())
    manager = await _ensure_manager()
    kwargs: dict[str, Any] = {}
    metadata_filters = metadata_filter_from_payload(args.filters)
    if metadata_filters is not None:
        kwargs["filters"] = metadata_filters
    if args.query_images:
        kwargs["query_images"] = args.query_images
    if args.session_id:
        kwargs["session_id"] = args.session_id
    if args.referenced_image_ids:
        kwargs["referenced_image_ids"] = args.referenced_image_ids
    scope = current_request_scope().for_workspaces(args.workspaces)
    result = await manager.aretrieve(
        args.query,
        workspaces=args.workspaces,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        scope=scope,
        **kwargs,
    )
    return retrieval_payload(result)
```

Repeat this pattern for `answer`, `list_workspaces`, `create_workspace`, `delete_workspace`, `ingest`, `ingest_job_status`, `list_files`, and `delete_files`.

- [ ] **Step 3: Remove legacy handlers**

Delete:

```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    ...

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    ...
```

Also remove unused imports: `json` if no longer needed, `Server`, and `Tool`.

- [ ] **Step 4: Run MCP tests**

Run:

```bash
uv run pytest tests/unit/test_mcp_workspace_tools.py -q
```

Expected: all MCP unit tests pass after test updates are complete.

## Task 4: Update Existing MCP Tests To FastMCP Calls

**Files:**
- Modify: `tests/unit/test_mcp_workspace_tools.py`

- [ ] **Step 1: Replace direct module wrapper calls**

Replace:

```python
tools = await mcp_server.list_tools()
result = await mcp_server.call_tool("tool_name", {...})
```

with:

```python
tools = await mcp_server.mcp_app.list_tools()
result = await mcp_server.mcp_app.call_tool("tool_name", {...})
```

- [ ] **Step 2: Adjust JSON parsing helper for FastMCP structured output**

FastMCP direct calls can return `(content, structuredContent)` for dictionary
results. Tests should read the content side without forcing production code to
unwrap structured output:

```python
def _tool_content(result):
    return result[0] if isinstance(result, tuple) else result


def _tool_json(result):
    return json.loads(_tool_content(result)[0].text)
```

No production wrapper should be added for test convenience.

- [ ] **Step 3: Run updated MCP tests**

Run:

```bash
uv run pytest tests/unit/test_mcp_workspace_tools.py -q
```

Expected: all tests pass.

## Task 5: Targeted Verification, Commit, Push, Restart

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run pytest tests/unit/test_mcp_workspace_tools.py tests/unit/test_config.py tests/unit/test_config_yaml.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run lint and type checks**

Run:

```bash
uv run ruff check src/dlightrag/mcp/server.py src/dlightrag/mcp/contracts.py tests/unit/test_mcp_workspace_tools.py
uv run ruff format --check src/dlightrag/mcp/server.py src/dlightrag/mcp/contracts.py tests/unit/test_mcp_workspace_tools.py
uv run pyright
git diff --check
```

Expected: no errors.

- [ ] **Step 3: Run stale-pattern scan**

Run:

```bash
rg -n "@server\\.list_tools|@server\\.call_tool|async def list_tools|async def call_tool|if name == \\\"retrieve\\\"|if name == \\\"answer\\\"" src/dlightrag/mcp tests/unit/test_mcp_workspace_tools.py -S
```

Expected: no matches.

- [ ] **Step 4: Commit**

Run:

```bash
git add src/dlightrag/mcp/server.py src/dlightrag/mcp/contracts.py tests/unit/test_mcp_workspace_tools.py
git commit -m "Migrate MCP tools to FastMCP registry"
```

Expected: pre-commit hooks pass and commit is created.

- [ ] **Step 5: Push and restart services**

Run:

```bash
git push origin main
docker compose up -d --no-deps --build dlightrag-api dlightrag-mcp
```

Expected: push succeeds and both containers restart.

- [ ] **Step 6: Verify runtime endpoints**

Run:

```bash
curl -fsS http://localhost:8100/health
curl -fsS -X POST http://127.0.0.1:8101/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"codex-check","version":"0"}}}'
docker compose ps dlightrag-api dlightrag-mcp
```

Expected: API health returns healthy, MCP initialize returns protocol `2025-11-25`, and both services are Up.
