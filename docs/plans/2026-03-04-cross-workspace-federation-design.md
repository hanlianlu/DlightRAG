# Cross-Workspace Federated Retrieval

## Problem

A single `RAGService` instance is bound to one workspace at initialization time. LightRAG's `aquery()` and `aquery_data()` accept no workspace parameter — the workspace is hardwired into every storage backend via `WHERE workspace=$1` (KV/vector) or AGE graph name (graph). Users need to search across multiple workspaces in a single query, with future RBAC support to control which workspaces each user can access.

## Solution

Add application-layer federation above `RAGService`, without modifying LightRAG. Three new components:

1. **WorkspacePool** (pool.py extension) — manages multiple `RAGService` instances keyed by workspace name, with lazy creation and shared PG connection pool.
2. **FederatedRetriever** (new: retrieval/federation.py) — orchestrates parallel queries across workspaces, merges results via round-robin interleaving.
3. **API/MCP extensions** — add optional `workspaces` parameter to retrieve/answer endpoints.

## Architecture

```
API/MCP Layer
  |  (workspaces=["a","b"] or None=current workspace)
  v
WorkspacePool          — manages RAGService lifecycle per workspace
  |
  v
FederatedRetriever     — parallel query + round-robin merge
  |--- RAGService(workspace="a").aretrieve()  ─┐
  |--- RAGService(workspace="b").aretrieve()  ─┤  asyncio.gather
  |                                             v
  └── round-robin merge ──> RetrievalResult (chunks tagged with _workspace)
```

### Component Responsibilities

| Component | Does | Does Not |
|-----------|------|----------|
| WorkspacePool (pool.py) | Manage RAGService lifecycle, lazy creation, list available workspaces | Retrieval logic |
| FederatedRetriever (new) | Parallel query, round-robin merge, workspace tagging | Workspace creation/destruction |
| RAGService (unchanged) | Single-workspace retrieval | Know about federation |

## WorkspacePool

Extends `pool.py` from a single-service singleton to a workspace-keyed dictionary:

- `get_workspace_service(workspace: str) -> RAGService` — get or create service for workspace
- `list_available_workspaces() -> list[str]` — query PG for distinct workspaces with data
- All workspace services share the same PG connection pool (LightRAG's `ClientManager` is a process-level singleton)
- Services created lazily on first access, not preloaded

The existing `get_shared_rag_service()` continues to work unchanged for backward compatibility (returns the default workspace service).

## FederatedRetriever

```python
@dataclass
class FederatedRetriever:
    workspace_pool: WorkspacePool
    workspace_filter: Callable[[list[str]], Awaitable[list[str]]] | None = None

    async def aretrieve(
        self,
        query: str,
        workspaces: list[str] | None = None,  # None = current workspace only
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs,
    ) -> RetrievalResult: ...

    async def aanswer(
        self,
        query: str,
        workspaces: list[str] | None = None,
        mode: str = "mix",
        **kwargs,
    ) -> RetrievalResult: ...
```

### Merge Strategy: Round-Robin Interleaving

Each workspace's results are already internally ranked (vector search → round-robin → rerank within LightRAG). The federation layer interleaves results to ensure fair representation:

1. Take rank-1 from workspace A, rank-1 from workspace B, rank-2 from A, rank-2 from B...
2. Tag each chunk/entity/relation with `_workspace` field
3. Truncate to requested `chunk_top_k`
4. Merge sources lists, each tagged with source workspace

No additional rerank at the federation level — each workspace's internal rerank is sufficient.

### Single-Workspace Fast Path

When `workspaces` has exactly one entry (or is None), bypass federation entirely and delegate directly to the single `RAGService`. Zero overhead for the common case.

### Error Handling

- `asyncio.gather(*tasks, return_exceptions=True)` — individual workspace failures don't block others
- Failed workspaces are logged and excluded from merged results
- If all workspaces fail, return error result

## API Changes

### Request Models

```python
class RetrieveRequest(BaseModel):
    query: str
    mode: str = "mix"
    workspaces: list[str] | None = None  # NEW — None = default workspace (backward compatible)
    top_k: int | None = None
    chunk_top_k: int | None = None

class AnswerRequest(BaseModel):
    query: str
    mode: str = "mix"
    workspaces: list[str] | None = None  # NEW
    top_k: int | None = None
    chunk_top_k: int | None = None
    conversation_history: list[dict[str, str]] | None = None
```

### Backward Compatibility

- `workspaces` omitted or `None` → behaves exactly as today (single default workspace)
- `workspaces=["a", "b"]` → federated retrieval across specified workspaces
- Existing clients see zero changes

### New Endpoint

- `GET /workspaces` — list all available workspaces (for UI workspace picker)

## RBAC — Interface Only (Not Implemented)

The `workspace_filter` parameter on `FederatedRetriever` is the RBAC extension point:

```python
WorkspaceFilter = Callable[[list[str]], Awaitable[list[str]]]
```

Future RBAC implementation passes a filter function that intersects requested workspaces with the user's accessible set. Default: `None` (all workspaces accessible).

No permission model, no user model, no auth changes in this design.

## What This Design Does NOT Do

- **No cross-workspace graph traversal** — AGE graphs are isolated by name per workspace; cannot traverse across graphs without modifying LightRAG
- **No global rerank** — each workspace's internal rerank is sufficient
- **No RBAC implementation** — only the interface hook
- **No workspace CRUD** — workspaces are created implicitly during ingest
- **No LightRAG modifications** — pure application-layer federation

## Files to Change

1. `src/dlightrag/pool.py` — extend to WorkspacePool
2. `src/dlightrag/retrieval/federation.py` — new: FederatedRetriever + merge logic
3. `src/dlightrag/api/server.py` — add `workspaces` param, `/workspaces` endpoint
4. `src/dlightrag/mcp/server.py` — add `workspaces` param to retrieve/answer tools
5. `tests/unit/test_federation.py` — new: federation merge tests

## Test Plan

- Single workspace (None) → delegates directly, no federation overhead
- Two workspaces, same query → results interleaved correctly with workspace tags
- One workspace fails → other workspace results still returned
- All workspaces fail → error result
- Empty workspaces list → error or empty result
- Round-robin merge preserves per-workspace ranking order
- Sources merged and tagged with workspace
- `GET /workspaces` returns available workspace list
