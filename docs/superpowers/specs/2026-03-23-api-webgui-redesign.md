# DlightRAG API & WebGUI Redesign

## Problem

DlightRAG's API layer is a 603-line monolithic `server.py` that mixes app
creation, auth, request/response models, all endpoints, error handlers, and
web mounting. Comparison with ArtRAG revealed 9 architectural gaps:

1. No CORS middleware — cannot be consumed cross-origin
2. No app factory pattern — module-level global app, untestable
3. Auth is bearer-token only — no JWT, no pluggable strategy
4. No structured error responses — plain string HTTPException details
5. DRY violations — query kwargs repeated 3x, workspace resolution repeated 5x
6. No response_model declarations — OpenAPI docs incomplete
7. No web route tests
8. Private field access from routes — `svc._metadata_index`, `manager._get_service()`
9. No progressive SSE streaming preview in WebGUI

Additionally: Snowflake sourcing is removed, S3 sourcing added (parity with ArtRAG).

## Solution

Incremental refactor of `server.py` into focused modules following ArtRAG's
proven decomposition, with improvements where DlightRAG can exceed ArtRAG.

## Architecture

### File Decomposition

Current:
```
api/
  server.py (603 lines — everything)
  __init__.py
```

New:
```
api/
  server.py    (~80 lines — factory, CORS, lifespan, error handlers, web mount)
  auth.py      (~75 lines — pluggable auth: none/simple/jwt, UserContext)
  models.py    (~130 lines — all Pydantic request + response models)
  routes.py    (~400 lines — all API endpoints via APIRouter, helpers)
  __init__.py
```

### Pluggable Auth (`auth.py`)

```python
class UserContext(BaseModel, frozen=True):
    user_id: str
    auth_mode: str  # "none" | "simple" | "jwt"

async def get_current_user(request: Request) -> UserContext:
    """FastAPI dependency — dispatches to auth strategy based on config."""
    mode = config.auth_mode
    if mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")
    if mode == "simple":
        # Validate bearer token from Authorization header
        return UserContext(user_id=user_id, auth_mode="simple")
    if mode == "jwt":
        # Decode + validate JWT, extract sub claim
        return UserContext(user_id=sub, auth_mode="jwt")
```

Config additions to `DlightragConfig`:
- `auth_mode: str = "none"` — replaces implicit "has token → auth enabled"
- Existing `api_auth_token` used by simple mode
- New `jwt_secret`, `jwt_algorithm` for jwt mode
- Backward compatible: if `api_auth_token` is set and `auth_mode` is unset,
  defaults to `"simple"`

Dependencies: `PyJWT` added to pyproject.toml.

### Request/Response Models (`models.py`)

**Request models** (migrated from server.py):
- `IngestRequest` — `source_type: Literal["local", "azure_blob", "s3"]`
  (Snowflake removed, S3 added). `@model_validator` validates required fields
  per source_type at the Pydantic layer (fail-fast with 422).
- `RetrieveRequest`, `AnswerRequest`, `DeleteRequest`, `ResetRequest`,
  `MetadataFilterRequest` — migrated as-is with improved docstrings.

**Response models** (all new):
- `ErrorDetail(detail: str, error_type: str)` — structured errors
- `HealthResponse(status, ready, degraded, warnings, version)`
- `IngestResponse(doc_id, status, page_count)`
- `RetrieveResponse(chunks, mode)`
- `FilesResponse(files, count)`
- `WorkspacesResponse(workspaces)`
- `ResetResponse(workspaces, total_errors)`
- `StatusResponse(status: str)` — generic success

### App Factory (`server.py`)

```python
def create_app(*, include_web: bool = True) -> FastAPI:
    config = get_config()

    @asynccontextmanager
    async def lifespan(_app):
        _app.state.manager = await RAGServiceManager.create()
        yield
        await _app.state.manager.close()

    app = FastAPI(title="dlightrag", version=..., lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # Exception handlers
    @app.exception_handler(RAGServiceUnavailableError)
    @app.exception_handler(ValueError)

    app.include_router(api_router)
    if include_web: mount web router + static
    return app

def get_app() -> FastAPI:
    """ASGI factory: uvicorn dlightrag.api.server:get_app --factory"""
    return create_app()

# Backward compat: module-level app for existing deployments
app = create_app()
```

### API Routes (`routes.py`)

DRY helpers:
```python
def _get_manager(request: Request) -> RAGServiceManager
def _resolve_workspace(ws: str | None, config) -> str
def _extract_query_kwargs(body: RetrieveRequest | AnswerRequest) -> dict
```

All endpoints use:
- `response_model=` declarations
- `Depends(get_current_user)` for auth
- Public manager methods only (no private field access)

### Remove Private Field Access

Add 3 public methods to `RAGServiceManager`:
- `aget_metadata(workspace, doc_id) -> dict`
- `aupdate_metadata(workspace, doc_id, data) -> None`
- `asearch_metadata(workspace, filters) -> list[dict]`

These delegate to `RAGService` which also gains the same 3 public methods,
encapsulating the `_metadata_index` access.

Routes change from:
```python
svc = await manager._get_service(ws)      # PRIVATE
meta = await svc._metadata_index.get(id)  # PRIVATE
```
To:
```python
meta = await manager.aget_metadata(ws, id)  # PUBLIC
```

### Sourcing: Remove Snowflake, Add S3

- Remove `"snowflake"` from `IngestRequest.source_type` literal
- Remove `snowflake` sourcing module if exists
- Add `sourcing/aws_s3.py` (port from ArtRAG — `aiobotocore` based)
- Add `aiobotocore` to pyproject.toml dependencies
- Update `core/service.py` and `core/servicemanager.py` ingest dispatch to
  handle `"s3"` source type (no `lifecycle.py` exists in DlightRAG)

### Progressive SSE Preview (WebGUI)

Add throttled markdown preview during streaming in `web/routes.py`:

```python
accumulated = ""
last_preview_ts = 0

async for token in token_iter:
    accumulated += token
    yield ServerSentEvent(data=json.dumps(token), event="token")

    now = time.monotonic()
    if now - last_preview_ts > 0.3:  # 300ms throttle
        html = render_markdown(accumulated)
        yield ServerSentEvent(data=html, event="preview")
        last_preview_ts = now
```

Frontend JS updates a preview div on `preview` events, replaces with final
rendered HTML on `done` event.

### Web Route Tests

New `tests/unit/test_web_routes.py`:
- `GET /web/` — HTML response with workspace selector
- `POST /web/answer` — SSE event sequence (meta → token → preview → done)
- `GET /web/files` — file list partial HTML
- `POST /web/files/upload` — file upload success
- `DELETE /web/files` — file deletion
- `GET /web/workspaces` — workspace list partial
- `POST /web/workspaces/switch` — cookie set + redirect

Uses same `TestClient` + mock `manager` pattern as existing API tests.

## Error Handling

All exception handlers return `ErrorDetail`:
```python
@app.exception_handler(RAGServiceUnavailableError)
async def _(request, exc):
    return JSONResponse(503, content=ErrorDetail(
        detail=exc.detail, error_type="unavailable"
    ).model_dump())

@app.exception_handler(ValueError)
async def _(request, exc):
    return JSONResponse(422, content=ErrorDetail(
        detail=str(exc), error_type="validation"
    ).model_dump())
```

## Files Changed

| File | Change |
|------|--------|
| `src/dlightrag/api/server.py` | Rewrite to factory pattern (~80 lines) |
| `src/dlightrag/api/auth.py` | New — pluggable auth with `get_current_user` dependency |
| `src/dlightrag/api/models.py` | New — all request/response models incl. `MetadataUpdateRequest` |
| `src/dlightrag/api/routes.py` | New — all API endpoints via APIRouter, incl. `/api/files/{path}` serve_file (S3 presigned URL replaces Snowflake 400) |
| `src/dlightrag/core/servicemanager.py` | Add 3 public metadata methods; update `aingest` source_type Literal to `"local" \| "azure_blob" \| "s3"` |
| `src/dlightrag/core/service.py` | Add 3 public metadata methods; update `aingest` source_type Literal |
| `src/dlightrag/web/routes.py` | Add progressive SSE preview |
| `src/dlightrag/web/templates/index.html` | Add `preview` event handler in JS |
| `src/dlightrag/sourcing/__init__.py` | Update source registry: remove snowflake, register s3 |
| `src/dlightrag/sourcing/snowflake.py` | Delete |
| `src/dlightrag/sourcing/aws_s3.py` | New — port from ArtRAG (aiobotocore-based) |
| `src/dlightrag/config.py` | Add `auth_mode`, `jwt_secret`, `jwt_algorithm`; remove 6 Snowflake fields; add S3 fields if needed. Backward compat: `@model_validator` defaults `auth_mode="simple"` when `api_auth_token` is truthy and `auth_mode` not explicitly set |
| `pyproject.toml` | Add PyJWT, aiobotocore; remove snowflake-connector-python if present |
| `tests/unit/test_api_server.py` | Update imports: `app` stays (backward compat), `_get_config`→ use `app.dependency_overrides` with new auth dep. Update Snowflake tests → S3 |
| `tests/unit/test_web_routes.py` | New — web route tests |

**Out of scope:** `src/dlightrag/mcp/server.py` may reference Snowflake — address
in a separate MCP cleanup task if needed.

## What Stays Unchanged

- `web/deps.py` — Jinja filters, template setup
- `core/servicemanager.py` — existing public methods untouched (only additions)
- `config.yaml` — add auth_mode, remove snowflake fields
