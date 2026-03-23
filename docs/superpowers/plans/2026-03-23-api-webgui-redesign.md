# DlightRAG API & WebGUI Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor DlightRAG's monolithic 603-line `server.py` into a clean, modular API layer with pluggable auth, structured errors, response models, CORS, and app factory pattern — matching ArtRAG's architecture.

**Architecture:** Incremental decomposition: extract models.py → auth.py → routes.py → rewrite server.py as factory. Each step keeps existing tests passing. Also: add public metadata methods (remove private access), swap Snowflake→S3 sourcing, add SSE preview, add web tests.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, PyJWT, aiobotocore, pytest-asyncio, httpx

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/dlightrag/api/models.py` | All Pydantic request + response models |
| `src/dlightrag/api/auth.py` | Pluggable auth: none/simple/jwt. UserContext model |
| `src/dlightrag/api/routes.py` | All API endpoints via APIRouter + DRY helpers |
| `src/dlightrag/api/server.py` | App factory, CORS, lifespan, error handlers, web mount |
| `src/dlightrag/core/servicemanager.py` | +3 public metadata methods |
| `src/dlightrag/core/service.py` | +3 public metadata methods, S3 source_type |
| `src/dlightrag/sourcing/aws_s3.py` | S3 data source (port from ArtRAG) |
| `src/dlightrag/web/routes.py` | +progressive SSE preview |
| `tests/unit/test_web_routes.py` | Web route tests |

---

### Task 1: Extract models.py (request + response models)

**Files:**
- Create: `src/dlightrag/api/models.py`
- Modify: `src/dlightrag/api/server.py` — update imports to use models.py
- Test: `tests/unit/test_api_server.py` — existing tests must still pass

- [ ] **Step 1: Create `models.py` with all request models moved from server.py**

Move `MetadataFilterRequest`, `IngestRequest`, `RetrieveRequest`, `AnswerRequest`, `DeleteRequest`, `ResetRequest` from `server.py` (lines 79-145) into new `src/dlightrag/api/models.py`.

Change `IngestRequest.source_type` from `Literal["local", "azure_blob", "snowflake"]` to `Literal["local", "azure_blob", "s3"]`.

Add `@model_validator(mode="after")` to `IngestRequest` that validates required fields per source_type:
- `"local"` requires `path`
- `"azure_blob"` requires `container_name`
- `"s3"` requires `bucket` and `key` (new fields)

Add all response models: `ErrorDetail`, `HealthResponse`, `IngestResponse`, `RetrieveResponse`, `FilesResponse`, `WorkspacesResponse`, `ResetResponse`, `StatusResponse`, `MetadataUpdateRequest`.

- [ ] **Step 2: Update server.py imports to use models.py**

Replace model class definitions in `server.py` with imports from `dlightrag.api.models`.

- [ ] **Step 3: Run existing tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && uv run pytest tests/unit/test_api_server.py -v`
Expected: ALL 30 tests PASS

- [ ] **Step 4: Run ruff check + format**

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/api/models.py src/dlightrag/api/server.py
git commit -m "refactor(api): extract request/response models to models.py"
```

---

### Task 2: Extract auth.py (pluggable auth)

**Files:**
- Create: `src/dlightrag/api/auth.py`
- Modify: `src/dlightrag/config.py` — add auth_mode, jwt_secret, jwt_algorithm
- Modify: `src/dlightrag/api/server.py` — replace `_verify_auth` with `get_current_user`
- Modify: `pyproject.toml` — add PyJWT dependency
- Test: `tests/unit/test_api_server.py` — update auth tests

- [ ] **Step 1: Add auth config fields to DlightragConfig**

Add `auth_mode`, `jwt_secret`, `jwt_algorithm` fields. Add `@model_validator` for backward compat: auto-detect `auth_mode="simple"` when `api_auth_token` is set.

- [ ] **Step 2: Create `auth.py` with pluggable auth**

Three modes: none, simple (bearer token), jwt. Returns `UserContext(user_id, auth_mode)`.

- [ ] **Step 3: Update server.py to use `get_current_user`**

Replace `_verify_auth` dependency. Each endpoint gets `user: UserContext = Depends(get_current_user)`.

- [ ] **Step 4: Update tests + add JWT tests**

- [ ] **Step 5: Add PyJWT to pyproject.toml**

- [ ] **Step 6: Run all tests, ruff, commit**

```bash
git commit -m "feat(auth): pluggable auth with none/simple/jwt strategies"
```

---

### Task 3: Add public metadata methods to service + servicemanager

**Files:**
- Modify: `src/dlightrag/core/service.py`
- Modify: `src/dlightrag/core/servicemanager.py`

- [ ] **Step 1: Add 3 public methods to RAGService**

`aget_metadata(doc_id)`, `aupdate_metadata(doc_id, data)`, `asearch_metadata(filters)` — encapsulate `_metadata_index` access.

- [ ] **Step 2: Add 3 delegate methods to RAGServiceManager**

`aget_metadata(workspace, doc_id)`, `aupdate_metadata(workspace, doc_id, data)`, `asearch_metadata(workspace, filters)`.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(service): add public metadata methods (remove private access need)"
```

---

### Task 4: Extract routes.py + rewrite server.py as factory

The core decomposition task.

**Files:**
- Create: `src/dlightrag/api/routes.py`
- Rewrite: `src/dlightrag/api/server.py`
- Modify: `tests/unit/test_api_server.py`

- [ ] **Step 1: Create `routes.py` with all endpoints via APIRouter**

Move all endpoints from server.py into `APIRouter`. Key changes:
- All endpoints use `response_model=` declarations
- All endpoints use `Depends(get_current_user)` for auth
- Metadata endpoints use public `manager.aget_metadata()` etc. (no private access)
- DRY helpers: `_get_manager()`, `_resolve_workspace()`, `_extract_query_kwargs()`
- `serve_file` endpoint: replace Snowflake 400 with S3 presigned URL

- [ ] **Step 2: Rewrite server.py as app factory**

~80 lines: `create_app()`, `get_app()`, CORS, lifespan, exception handlers, web mount. Keep `app = create_app()` at module level for backward compat.

- [ ] **Step 3: Update test imports**

Update `test_api_server.py` for new module structure. Snowflake tests → S3.

- [ ] **Step 4: Run all tests, ruff, commit**

```bash
git commit -m "refactor(api): decompose server.py into factory + routes + structured errors"
```

---

### Task 5: Remove Snowflake sourcing, add S3

**Files:**
- Delete: `src/dlightrag/sourcing/snowflake.py`
- Create: `src/dlightrag/sourcing/aws_s3.py`
- Modify: `src/dlightrag/sourcing/__init__.py`, `core/service.py`, `core/servicemanager.py`, `config.py`, `pyproject.toml`

- [ ] **Step 1: Port `aws_s3.py` from ArtRAG**
- [ ] **Step 2: Update `sourcing/__init__.py`** — remove Snowflake, add S3
- [ ] **Step 3: Update `service.py` + `servicemanager.py`** — change Literal, update dispatch
- [ ] **Step 4: Remove Snowflake config fields** from config.py (6 fields)
- [ ] **Step 5: Delete `sourcing/snowflake.py`**
- [ ] **Step 6: Update pyproject.toml** — remove snowflake-connector-python, add aiobotocore
- [ ] **Step 7: Update tests** — Snowflake tests → S3
- [ ] **Step 8: Run tests, ruff, commit**

```bash
git commit -m "feat(sourcing): replace Snowflake with S3 (aiobotocore-based)"
```

---

### Task 6: Progressive SSE preview in WebGUI

**Files:**
- Modify: `src/dlightrag/web/routes.py`
- Modify: `src/dlightrag/web/static/citation.js` (SSE event handler)

- [ ] **Step 1: Add throttled preview events to web SSE stream**

In `web/routes.py` `event_generator()`, add 300ms-throttled `preview` events with rendered markdown during token iteration.

- [ ] **Step 2: Add preview event handler in frontend JS**

In `citation.js`, add handler for `preview` event that updates a preview container using sanitized HTML (DlightRAG already uses nh3 for HTML sanitization in deps.py).

Hide preview container when `done` event arrives.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(web): progressive SSE preview with 300ms throttle"
```

---

### Task 7: Web route tests

**Files:**
- Create: `tests/unit/test_web_routes.py`

- [ ] **Step 1: Write web route tests**

Tests for: `GET /web/`, `GET /web/files`, `DELETE /web/files`, `GET /web/workspaces`, `POST /web/workspaces/switch`.

Use `create_app(include_web=True)` + mock `manager` on `app.state`.

- [ ] **Step 2: Run web tests + full suite**

- [ ] **Step 3: Commit**

```bash
git commit -m "test(web): add web route tests"
```

---

### Task 8: Global quality checks

- [ ] **Step 1: Run pyright** — `uv run pyright src/`
- [ ] **Step 2: Run ruff check** — `uv run ruff check src/ tests/ --fix`
- [ ] **Step 3: Run ruff format** — `uv run ruff format src/ tests/`
- [ ] **Step 4: Run full test suite** — `uv run pytest tests/ -v`
- [ ] **Step 5: Verify no remaining private accesses**

```bash
grep -rn "_metadata_index" src/dlightrag/api/
grep -rn "_get_service" src/dlightrag/api/
grep -rn "snowflake" src/dlightrag/ --include="*.py"
```

- [ ] **Step 6: Final commit if needed**

```bash
git commit -m "chore: post-redesign quality fixes"
```
