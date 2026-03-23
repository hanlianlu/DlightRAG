# ArtRAG Parity Improvements for DlightRAG

## Problem

5 areas where ArtRAG's architecture is ahead of DlightRAG in shared scope:

1. **Ingestion loads all pages at once** — OOM on large PDFs (200+ pages)
2. **Global error backoff** — one failing workspace blocks all workspaces
3. **LightRAG patch logs PG errors** — `create_graph()` called unconditionally
4. **Domain stores share LightRAG's PG pool** — contention under load
5. **3-layer inline deletion** — orphans possible, no entity-level cleanup

## Solution

Port proven patterns from ArtRAG. Each improvement is independent and
backward-compatible.

## 1. Streaming Batch Ingestion

### renderer.py

Add `render_file_batched(path, batch_size=20) -> AsyncIterator[RenderResult]`:
- PDF: opens document once, yields N pages at a time via `_render_pdfium_doc_range()`
- Office: converts to PDF bytes first, then batched rendering
- Image: single batch (one page)
- Keep `render_file()` for backward compat

Shared `_render_pdfium_doc_range(source, start, end, total)` used by both
PDF and Office paths (no code duplication).

### engine.py

Rewrite `aingest()` as batch loop with double-buffered prefetch:

```
batch_aiter = renderer.render_file_batched(path, batch_size=N)
current_batch = await _safe_anext(batch_aiter)

while current_batch is not None:
    1. Encode JPEG q95 -> api_b64 (shared by embedder + extractor)
    2. Parallel: embedder(api_b64) + extractor(api_b64)
    3. Use chunk_ids from page_infos for visual_chunks keys
    4. Encode PNG per page -> upsert visual_chunks, release PIL
    5. Start prefetch after PIL release
    6. Write chunks_vdb + text_chunks
    7. Accumulate metadata across batches
    current_batch = await prefetch_task

After all batches:
    Write doc_status (PROCESSED), metadata_index
```

Early checkpoint: write `doc_status = PROCESSING` before store writes
(already implemented in previous commit). On exception: `FAILED` with
error_msg.

### config.py

Internal defaults only (not in config.yaml):
- `ingestion_batch_pages: int = 20`
- `api_image_quality: int = 95`

## 2. Per-Workspace Exponential Backoff

### servicemanager.py

Replace:
```python
self._last_error_ts: float | None = None
self._retry_after: float = 30.0
```

With:
```python
self._backoff: dict[str, tuple[float, float]] = {}
# workspace -> (last_error_ts, retry_interval)
```

In `_get_service(workspace)`:
- Check `self._backoff[workspace]` before attempting creation
- On success: `self._backoff.pop(workspace, None)`
- On failure: `self._backoff[workspace] = (time.time(), min(prev * 2, 300.0))`
- Initial interval: 7.5s, max: 300s

Other workspaces unaffected by one workspace's failures.

## 3. LightRAG Patch Pre-Check

### _lightrag_patches.py

Replace wrap-based `_patch_configure_age()` with pre-check pattern:

```python
async def patched_configure_age(connection, graph_name):
    await connection.execute('SET search_path = ag_catalog, "$user", public')
    exists = await connection.fetchval(
        "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1)",
        graph_name,
    )
    if exists:
        return  # skip create_graph — no PG ERROR logged
    try:
        await connection.execute(f"select create_graph('{graph_name}')")
    except (InvalidSchemaNameError, UniqueViolationError, DuplicateSchemaError):
        pass  # race condition: another worker created it between check and create
```

Result: zero PG ERROR log spam on normal startups. Race condition still
handled by exception catch.

Keep `_patch_execute()` wrapper unchanged (defense-in-depth).

## 4. PG Pool Singleton

### storage/pool.py (new file)

Port ArtRAG's `PGPool` singleton:

```python
class PGPool:
    """Process-wide asyncpg connection pool for DlightRAG domain stores."""
    async def get(self) -> asyncpg.Pool:
        # Lazy creation with double-check locking
        # Independent from LightRAG's ClientManager pool
    async def close(self) -> None:
        # Graceful shutdown

pg_pool = PGPool()  # module-level singleton
```

Default sizing: min=2, max=10. Reads PG credentials from DlightragConfig.

### pg_jsonb_kv.py

Change `initialize()`:
```python
# Old: borrows from LightRAG
db = await ClientManager.get_client()
self._pool = db.pool

# New: uses dedicated pool
from dlightrag.storage.pool import pg_pool
self._pool = await pg_pool.get()
```

### service.py

Add `await pg_pool.close()` in `RAGService.close()`.

## 5. Cascade Deletion

### cleanup.py

Add standalone `cascade_delete()` function:

```python
async def cascade_delete(
    ctx: DeletionContext,
    lightrag: Any,
    visual_chunks: Any | None = None,
    hash_index: Any | None = None,
    metadata_index: Any | None = None,
) -> dict[str, Any]:
    """5-layer cascade deletion with per-layer fault isolation."""
    stats = {"layers_completed": 0, "errors": []}

    for doc_id in ctx.doc_ids:
        # Layer 1: LightRAG (adelete_by_doc_id)
        try:
            await lightrag.adelete_by_doc_id(doc_id)
            stats["layers_completed"] += 1
        except Exception as exc:
            stats["errors"].append(f"Layer 1 ({doc_id}): {exc}")

        # Layer 2: visual_chunks
        if visual_chunks is not None:
            try:
                # Read doc_status chunks_list BEFORE layer 1 deletes it
                # (or use ctx.chunk_ids if pre-collected)
                await visual_chunks.delete_by_doc_id(doc_id)
                stats["layers_completed"] += 1
            except Exception as exc:
                stats["errors"].append(f"Layer 2 ({doc_id}): {exc}")

        # Layer 3: metadata_index
        if metadata_index is not None:
            try:
                await metadata_index.delete(doc_id)
                stats["layers_completed"] += 1
            except Exception as exc:
                stats["errors"].append(f"Layer 3 ({doc_id}): {exc}")

    # Layer 4: hash_index (by content_hash, not doc_id)
    if hash_index is not None:
        for h in ctx.content_hashes:
            try:
                await hash_index.remove(h)
            except Exception as exc:
                stats["errors"].append(f"Layer 4 ({h}): {exc}")
        stats["layers_completed"] += 1

    return stats
```

Note: Layer 2 must read `chunks_list` from doc_status BEFORE Layer 1
deletes it (pre-collect in `collect_deletion_context()`).

### lifecycle.py

Replace inline 3-layer deletion in `unified_delete_files()` with:
```python
stats = await cascade_delete(ctx, lightrag, visual_chunks, hash_index, metadata_index)
```

## Files Changed

| File | Change |
|------|--------|
| `src/dlightrag/unifiedrepresent/renderer.py` | Add `render_file_batched()` async generator |
| `src/dlightrag/unifiedrepresent/engine.py` | Batch loop + prefetch + shared encoding |
| `src/dlightrag/config.py` | Add `ingestion_batch_pages`, `api_image_quality` |
| `src/dlightrag/core/servicemanager.py` | Per-workspace `_backoff` dict |
| `src/dlightrag/core/_lightrag_patches.py` | Pre-check before `create_graph()` |
| `src/dlightrag/storage/pool.py` | New — PG pool singleton |
| `src/dlightrag/storage/pg_jsonb_kv.py` | Use `pg_pool` instead of LightRAG pool |
| `src/dlightrag/core/ingestion/cleanup.py` | Add `cascade_delete()` function |
| `src/dlightrag/unifiedrepresent/lifecycle.py` | Call `cascade_delete()` |
| `src/dlightrag/core/service.py` | Close `pg_pool` in `close()` |
| Tests | Update existing + new tests per improvement |
