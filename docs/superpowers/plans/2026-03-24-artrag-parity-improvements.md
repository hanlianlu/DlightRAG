# ArtRAG Parity Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 5 proven improvements from ArtRAG to DlightRAG: streaming batch ingestion, per-workspace backoff, LightRAG patch pre-check, PG pool singleton, and cascade deletion.

**Architecture:** Each improvement is independent — they can be implemented and committed in any order. All are backward-compatible ports of ArtRAG patterns adapted to DlightRAG's codebase structure.

**Tech Stack:** Python 3.12, asyncio, asyncpg, pypdfium2, Pillow, pytest-asyncio

---

## File Structure

| File | Change |
|------|--------|
| `src/dlightrag/unifiedrepresent/renderer.py` | Add `render_file_batched()` async generator |
| `src/dlightrag/unifiedrepresent/engine.py` | Batch loop + prefetch + shared JPEG/PNG encoding |
| `src/dlightrag/config.py` | Add `ingestion_batch_pages`, `api_image_quality` |
| `src/dlightrag/core/servicemanager.py` | Per-workspace `_backoff` dict |
| `src/dlightrag/core/_lightrag_patches.py` | Pre-check `ag_catalog.ag_graph` before `create_graph()` |
| `src/dlightrag/storage/pool.py` | New — PG pool singleton |
| `src/dlightrag/storage/pg_jsonb_kv.py` | Use `pg_pool` instead of LightRAG pool |
| `src/dlightrag/core/service.py` | Close `pg_pool` in `close()` |
| `src/dlightrag/core/ingestion/cleanup.py` | Add `cascade_delete()` function |
| `src/dlightrag/unifiedrepresent/lifecycle.py` | Call `cascade_delete()` |

---

### Task 1: Streaming Batch Ingestion

Port ArtRAG's streaming batch pattern to DlightRAG's unified mode.

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/renderer.py`
- Modify: `src/dlightrag/unifiedrepresent/engine.py`
- Modify: `src/dlightrag/config.py`
- Test: `tests/unit/test_unified_renderer.py`, `tests/unit/test_unified_engine.py`

**Reference:** ArtRAG `src/artrag/core/ingestion/renderer.py` and `engine.py`

- [ ] **Step 1: Add config fields**

In `src/dlightrag/config.py`, after `ingestion_replace_default` (~line 228), add:
```python
    ingestion_batch_pages: int = Field(
        default=20,
        description="Pages per batch during streaming ingestion.",
    )
    api_image_quality: int = Field(
        default=95,
        description="JPEG quality for VLM/embedding API calls.",
    )
```

- [ ] **Step 2: Add `render_file_batched()` to PageRenderer**

In `src/dlightrag/unifiedrepresent/renderer.py`:

Add `from collections.abc import AsyncIterator` to imports.

Add shared `_render_pdfium_doc_range(source, start, end, total) -> RenderResult` method (renders a range of pages).

Refactor existing `_render_pdfium_doc()` to delegate to `_render_pdfium_doc_range(source, 0, len(doc), len(doc))`.

Add `_get_page_count(source) -> int` and `_get_page_count_from_bytes(data) -> int` helpers.

Add `render_file_batched(path, batch_size=20) -> AsyncIterator[RenderResult]`:
- Dispatches by extension (same as `render_file`)
- PDF: `_render_pdf_batched()` — get page count, loop in ranges, yield batches
- Office: `_render_office_batched()` — convert to bytes, then batch render
- Image: yield single batch

Add `_render_pdf_batched()` and `_render_office_batched()` that both use `_render_pdfium_doc_range()` (no code duplication).

- [ ] **Step 3: Add renderer tests**

Add `TestRenderFileBatched` class to `tests/unit/test_unified_renderer.py`:
- `test_single_image_yields_one_batch`
- `test_pdf_batches_correct_size` (7 pages / batch_size=3 → 3 batches of 3,3,1)
- `test_batched_page_indices_are_global` (0,1,2,3,4 across batches)
- `test_batched_metadata_on_all_batches`
- `test_file_not_found_raises`

- [ ] **Step 4: Add encoding utilities to engine.py**

Add at top of `engine.py`:
```python
import base64
import io
from PIL import Image

def encode_jpeg_b64(image: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def encode_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

async def _safe_anext(aiter: Any) -> Any:
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return None
```

- [ ] **Step 5: Rewrite `aingest()` as batch loop**

Replace the body of `aingest()` with streaming batch pattern:
- Manual `__aiter__`/`__anext__` with prefetch via `asyncio.create_task(_safe_anext())`
- Per batch: encode JPEG → parallel embed+extract → encode PNG + release PIL → write stores
- Use chunk_ids from extractor's page_infos for visual_chunks keys
- Accumulate page_infos across batches
- PROCESSING checkpoint already exists (previous commit); keep it
- Write PROCESSED + metadata after all batches

- [ ] **Step 6: Update engine tests**

Update `tests/unit/test_unified_engine.py`:
- Mock `renderer.render_file_batched` (async generator) instead of `render_file`
- Update `test_aingest_writes_doc_status` for PROCESSING + PROCESSED (2 calls)
- Add `test_multi_batch_accumulates_chunk_ids`

- [ ] **Step 7: Run tests, ruff, pyright, commit**

```bash
uv run pytest tests/ -x -q
uv run pyright src/
uv run ruff check src/ tests/ --fix && uv run ruff format src/ tests/
git commit -m "feat(unified): streaming batch ingestion with prefetch and shared encoding"
```

---

### Task 2: Per-Workspace Exponential Backoff

**Files:**
- Modify: `src/dlightrag/core/servicemanager.py`
- Test: `tests/unit/test_servicemanager.py`

- [ ] **Step 1: Replace global backoff fields with per-workspace dict**

In `__init__()` (~line 44-58), replace:
```python
self._last_error: str | None = None
self._last_error_ts: float | None = None
self._retry_after: float = 30.0
```
With:
```python
self._backoff: dict[str, tuple[float, float]] = {}
# workspace -> (last_error_ts, retry_interval)
```

- [ ] **Step 2: Update `_get_service()` for per-workspace backoff**

Replace global backoff checks with per-workspace:
```python
# Before lock:
if workspace in self._backoff:
    last_ts, interval = self._backoff[workspace]
    if time.time() - last_ts < interval:
        raise RAGServiceUnavailableError(
            detail=f"Workspace '{workspace}' in backoff (retry in {interval:.0f}s)"
        )

# After lock (double-check):
# Same check again

# On success:
self._backoff.pop(workspace, None)

# On failure:
_, prev_interval = self._backoff.get(workspace, (0, 7.5))
new_interval = min(prev_interval * 2, _MAX_RETRY_INTERVAL)
self._backoff[workspace] = (time.time(), new_interval)
```

- [ ] **Step 3: Update `get_error_info()` to report per-workspace backoff**

```python
def get_error_info(self) -> dict[str, Any]:
    return {
        "backoff_workspaces": {
            ws: {"retry_after": interval, "since": ts}
            for ws, (ts, interval) in self._backoff.items()
        },
    }
```

- [ ] **Step 4: Update tests**

In `tests/unit/test_servicemanager.py`:
- Update any tests that reference `_last_error_ts` or `_retry_after`
- Add `test_per_workspace_backoff_isolation` — verify workspace A failing doesn't block workspace B
- Add `test_backoff_clears_on_success`

- [ ] **Step 5: Run tests, ruff, pyright, commit**

```bash
git commit -m "feat(servicemanager): per-workspace exponential backoff"
```

---

### Task 3: LightRAG Patch Pre-Check

**Files:**
- Modify: `src/dlightrag/core/_lightrag_patches.py`
- Test: `tests/unit/test_lightrag_patches.py`

**Reference:** ArtRAG `src/artrag/core/_lightrag_patches.py`

- [ ] **Step 1: Replace wrap-based configure_age with pre-check version**

In `_patch_configure_age()`, replace the wrapped function with:
```python
@staticmethod
async def patched_configure_age(
    connection: asyncpg.Connection, graph_name: str
) -> None:
    await connection.execute('SET search_path = ag_catalog, "$user", public')
    exists = await connection.fetchval(
        "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1)",
        graph_name,
    )
    if not exists:
        try:
            await connection.execute(f"select create_graph('{graph_name}')")
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
            asyncpg.exceptions.DuplicateSchemaError,
        ):
            pass  # race condition — another worker created it

PostgreSQLDB.configure_age = patched_configure_age
```

This is a **replace** (not wrap), since the original's behavior is entirely superseded.

Update `_needs_patch()` check: look for the pre-check query string instead:
```python
def _needs_patch(method):
    try:
        source = inspect.getsource(method)
        return "ag_catalog.ag_graph" not in source
    except (OSError, TypeError):
        return True
```

- [ ] **Step 2: Update tests**

Update `tests/unit/test_lightrag_patches.py`:
- `test_skips_create_when_graph_exists` — mock `fetchval` returning True, verify `create_graph` NOT called
- `test_creates_graph_when_not_exists` — mock `fetchval` returning False, verify `create_graph` called
- `test_race_condition_caught` — `fetchval` returns False but `create_graph` raises DuplicateSchemaError
- `test_other_exceptions_propagate`

- [ ] **Step 3: Run tests, ruff, pyright, commit**

```bash
git commit -m "fix(patches): pre-check ag_catalog before create_graph (eliminates PG ERROR log spam)"
```

---

### Task 4: PG Pool Singleton

**Files:**
- Create: `src/dlightrag/storage/pool.py`
- Modify: `src/dlightrag/storage/pg_jsonb_kv.py`
- Modify: `src/dlightrag/core/service.py`
- Test: `tests/unit/test_pool.py` (new)

**Reference:** ArtRAG `src/artrag/storage/pool.py`

- [ ] **Step 1: Create `storage/pool.py`**

Port ArtRAG's `PGPool` class, adapting imports:
```python
"""Process-wide asyncpg connection pool for DlightRAG domain stores.

Independent from LightRAG's internal ClientManager pool to avoid contention.
"""
import asyncio
import logging

import asyncpg

from dlightrag.config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SIZE = 2
_DEFAULT_MAX_SIZE = 10


class PGPool:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._get_lock():
            if self._pool is not None:
                return self._pool
            config = get_config()
            self._pool = await asyncpg.create_pool(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_user,
                password=config.postgres_password,
                database=config.postgres_database,
                min_size=_DEFAULT_MIN_SIZE,
                max_size=_DEFAULT_MAX_SIZE,
            )
            logger.info("DlightRAG domain store pool created (min=%d, max=%d)",
                       _DEFAULT_MIN_SIZE, _DEFAULT_MAX_SIZE)
            return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("DlightRAG domain store pool closed")


pg_pool = PGPool()
```

- [ ] **Step 2: Update `pg_jsonb_kv.py` to use `pg_pool`**

In `PGJsonbKVStorage.initialize()`, replace:
```python
from lightrag.kg.postgres_impl import ClientManager
db = await ClientManager.get_client()
self._pool = db.pool
```
With:
```python
from dlightrag.storage.pool import pg_pool
self._pool = await pg_pool.get()
```

- [ ] **Step 3: Close pool in service.py**

In `RAGService.close()`, add at the end:
```python
from dlightrag.storage.pool import pg_pool
await pg_pool.close()
```

- [ ] **Step 4: Add pool tests**

Create `tests/unit/test_pool.py`:
- `test_get_returns_pool` — mock asyncpg.create_pool, verify called once
- `test_double_get_reuses_pool` — call get() twice, verify create_pool called once
- `test_close_sets_none` — close(), verify pool.close() called

- [ ] **Step 5: Run tests, ruff, pyright, commit**

```bash
git commit -m "feat(storage): dedicated PG pool singleton for domain stores"
```

---

### Task 5: Cascade Deletion

**Files:**
- Modify: `src/dlightrag/core/ingestion/cleanup.py`
- Modify: `src/dlightrag/unifiedrepresent/lifecycle.py`
- Test: `tests/unit/test_cleanup.py`

**Reference:** ArtRAG `src/artrag/core/ingestion/cleanup.py`

- [ ] **Step 1: Add `cascade_delete()` to cleanup.py**

Add after `collect_deletion_context()`:
```python
async def cascade_delete(
    ctx: DeletionContext,
    lightrag: Any,
    visual_chunks: Any | None = None,
    hash_index: HashIndexProtocol | None = None,
    metadata_index: Any | None = None,
) -> dict[str, Any]:
    """5-layer cascade deletion with per-layer fault isolation."""
    stats: dict[str, Any] = {"docs_deleted": 0, "errors": []}

    for doc_id in ctx.doc_ids:
        # Layer 1: LightRAG cross-backend cleanup
        try:
            await lightrag.adelete_by_doc_id(doc_id)
            stats["docs_deleted"] += 1
        except Exception as exc:
            stats["errors"].append(f"Layer 1 LightRAG ({doc_id}): {exc}")
            logger.warning("cascade_delete Layer 1 failed for %s: %s", doc_id, exc)

        # Layer 2: Visual chunks
        if visual_chunks is not None:
            try:
                await visual_chunks.delete_by_doc_id(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Layer 2 visual_chunks ({doc_id}): {exc}")
                logger.warning("cascade_delete Layer 2 failed for %s: %s", doc_id, exc)

        # Layer 3: Metadata index
        if metadata_index is not None:
            try:
                await metadata_index.delete(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Layer 3 metadata ({doc_id}): {exc}")
                logger.warning("cascade_delete Layer 3 failed for %s: %s", doc_id, exc)

    # Layer 4: Hash index (by content_hash)
    if hash_index is not None:
        for h in ctx.content_hashes:
            try:
                await hash_index.remove(h)
            except Exception as exc:
                stats["errors"].append(f"Layer 4 hash ({h}): {exc}")
                logger.warning("cascade_delete Layer 4 failed for hash %s: %s", h, exc)

    # Layer 5: File source cleanup (placeholder for future S3/Azure cleanup)

    return stats
```

- [ ] **Step 2: Update `unified_delete_files()` to call `cascade_delete()`**

In `src/dlightrag/unifiedrepresent/lifecycle.py`, replace the inline 3-layer deletion (lines ~430-471) with:
```python
from dlightrag.core.ingestion.cleanup import cascade_delete

stats = await cascade_delete(
    ctx=ctx,
    lightrag=lightrag,
    visual_chunks=engine.visual_chunks,
    hash_index=hash_index,
    metadata_index=metadata_index,
)
```

Keep the pre-delete context collection logic (`collect_deletion_context`) unchanged.

- [ ] **Step 3: Add cascade_delete tests**

In `tests/unit/test_cleanup.py`:
- `test_cascade_delete_all_layers_called` — verify all 4 layers called
- `test_cascade_delete_layer1_failure_continues` — Layer 1 raises, Layers 2-4 still called
- `test_cascade_delete_empty_context` — no doc_ids → stats show 0 deleted
- `test_cascade_delete_skips_none_stores` — visual_chunks=None → Layer 2 skipped

- [ ] **Step 4: Run tests, ruff, pyright, commit**

```bash
git commit -m "refactor(cleanup): 5-layer cascade deletion with fault isolation"
```

---

### Task 6: Global Quality Checks

- [ ] **Step 1: Run pyright** — `uv run pyright src/`
- [ ] **Step 2: Run ruff check** — `uv run ruff check src/ tests/ --fix`
- [ ] **Step 3: Run ruff format** — `uv run ruff format src/ tests/`
- [ ] **Step 4: Run full test suite** — `uv run pytest tests/ -v`
- [ ] **Step 5: Final commit if needed**
