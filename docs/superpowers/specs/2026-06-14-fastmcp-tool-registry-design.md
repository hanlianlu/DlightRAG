# FastMCP Tool Registry Migration Design

## Goal

Move the MCP tool layer from a hand-written `list_tools` registry plus
`call_tool` switch to FastMCP decorators with explicit Pydantic input contracts.
Keep the current streamable-http transport, endpoint shape, stateless behavior,
JSON response mode, bearer-auth middleware, request-scope propagation, and DNS
rebinding protection unchanged.

## Non-Goals

- Do not let FastMCP own the HTTP app or auth stack.
- Do not reintroduce legacy HTTP+SSE `/sse` or `/messages` routes.
- Do not keep Python compatibility wrappers for tests, such as module-level
  `list_tools()` or `call_tool()`.
- Do not change public MCP tool names, result payloads, workspace semantics, or
  REST API behavior.

## Architecture

`src/dlightrag/mcp/server.py` will expose one FastMCP app, named clearly enough
to be the only tool registry entry point. Tool handlers are registered with
`@mcp_app.tool(...)`; each handler owns one tool and delegates business logic to
the existing `RAGServiceManager` APIs.

The streamable-http and stdio runners continue to use the low-level MCP server
inside FastMCP because the existing transports need a low-level `Server`
instance. This transport adapter is runtime plumbing, not a legacy tool wrapper.

## Input Contracts

Each tool gets a Pydantic model:

- `RetrieveInput`
- `AnswerInput`
- `IngestInput`
- `IngestJobStatusInput`
- `CreateWorkspaceInput`
- `DeleteWorkspaceInput`
- `ListFilesInput`
- `DeleteFilesInput`

FastMCP generates top-level JSON schemas from function signatures, so handlers
will keep explicit keyword parameters instead of accepting a single `args`
model. The first line of each handler builds the Pydantic model from those
parameters. This keeps MCP client payloads unchanged while making validation and
defaults live in typed contracts.

Known validation rules move into models or small validators:

- `mode` remains rejected for `retrieve` and `answer`.
- `query_images` remains capped at 3 items.
- `metadata_policy` remains restricted to `validate`, `reject_unknown`, and
  `store_only`.
- `blob_path` and `prefix` remain mutually exclusive for Azure Blob ingest.
- `key` and `prefix` remain mutually exclusive for S3 ingest.
- Required fields stay required at the schema level where possible.

## Tool Outputs

Tool handlers return plain dictionaries or strings and let FastMCP convert them
to MCP tool results. Successful dictionary payloads should keep FastMCP's native
structured output instead of being forced back into text-only JSON; the text
content fallback remains available for broad client compatibility. Existing
payload builders stay in use:

- `retrieval_payload`
- `answer_payload`
- `metadata_filter_from_payload`

Error behavior remains client-friendly: tool exceptions are logged and returned
as text content beginning with `Error:` through a shared helper/decorator, so
transport-level failures are not introduced for ordinary tool validation errors.

## Tests

Tests should follow the new architecture instead of preserving old test-only
entry points:

- Call `mcp_app.list_tools()` and `mcp_app.call_tool(...)`.
- Assert source no longer contains `@server.list_tools`, `@server.call_tool`, or
  a tool-name `if`/`elif` switch.
- Assert generated schemas expose top-level fields and do not wrap every tool in
  an `args` object.
- Assert Pydantic validation covers invalid `mode`, excess `query_images`,
  invalid `metadata_policy`, and mutually exclusive ingest selectors.
- Keep the current streamable-http/security tests, including `/mcp` only,
  `json_response=True`, `stateless=True`, and `TransportSecuritySettings`.

## Migration Steps

1. Add FastMCP app and low-level transport adapter.
2. Add Pydantic input models and shared response/error helpers.
3. Convert tools one at a time from switch branches into decorators.
4. Remove old module-level `list_tools` and `call_tool`.
5. Update tests to use FastMCP registry calls.
6. Run MCP unit tests, targeted API/web regressions, type checking, linting, and
   local streamable-http initialize checks.

## Acceptance Criteria

- `retrieve`, `answer`, `ingest`, `ingest_job_status`, `list_workspaces`,
  `create_workspace`, `delete_workspace`, `list_files`, and `delete_files` are
  registered through FastMCP decorators.
- There is no hand-written MCP tool schema list.
- There is no central `call_tool` switch.
- There are no legacy Python wrappers kept only for old tests.
- Public MCP schemas and successful result payloads remain compatible.
- Streamable-http transport/security behavior is unchanged.
