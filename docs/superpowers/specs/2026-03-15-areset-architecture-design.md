# Workspace Reset Architecture

**Goal:** Move storage reset logic from the standalone `scripts/reset.py` into the service layer (`RAGService.areset()` + `RAGServiceManager.areset()`), so all interface layers can invoke it through a single code path with proper workspace isolation.

---

## Architecture

### Calling Chain

```
reset.py (CLI)   →  RAGServiceManager.areset(workspace, ...)
API POST /reset  →  RAGServiceManager.areset(workspace, ...)
                          │
                          ├── single workspace: _get_service(ws) → service.areset()
                          └── all workspaces:   list_workspaces() → each service.areset()
                                                    │
                                                    └── evict from _services cache
```

### Interface Exposure

| Layer | Exposed | Reason |
|-------|---------|--------|
| RAGService | `areset()` | Core implementation |
| RAGServiceManager | `areset()` | Workspace routing + cache eviction |
| reset.py | CLI wrapper | Human-operated, `--yes` confirmation |
| API | `POST /reset` | Admin endpoint, Bearer auth |
| MCP | Not exposed | Too dangerous for agent automation |
| WebUI | Not exposed | Too dangerous for user UI |

### RAGService.areset() — 5-Phase Reset

```
Phase 1: LightRAG stores
  └── Iterate _STORAGE_ATTRS, call storage.drop() on each

Phase 2: DlightRAG stores
  ├── _hash_index.clear()
  ├── _metadata_index.clear()     (if not None)
  └── _visual_chunks.drop()       (if not None)

Phase 3: PG orphan table cleanup  (PG backend only)
  └── Scan lightrag_* / dlightrag_* tables
      DELETE WHERE workspace = $current_workspace

Phase 4: Workspace metadata
  └── DELETE FROM dlightrag_workspace_meta WHERE workspace = $1

Phase 5: Local files              (skip if keep_files=True)
  └── Remove working_dir/* except sources/
```

**Constraints:**
- Phase 3 only runs when `config.kv_storage.startswith("PG")`.
- All PG operations use `WHERE workspace = $1` — never touch other workspaces.
- Each phase has independent error handling; one failure does not block the next.
- `dry_run=True` collects stats without executing any deletions.

### RAGServiceManager.areset()

```python
async def areset(
    self,
    *,
    workspace: str | None = None,
    keep_files: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
```

- `workspace=None` → `list_workspaces()` → reset each.
- `workspace="xxx"` → reset that one.
- After each successful reset, evict from `self._services` cache (stale state).
- Returns `{"workspaces": {"ws1": {...stats...}}, "total_errors": 0}`.

### RAGService.areset()

```python
async def areset(
    self,
    *,
    keep_files: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
```

Returns:
```python
{
    "workspace": "default",
    "storages_dropped": 10,
    "dlightrag_cleared": ["hash_index", "metadata_index", "visual_chunks"],
    "orphan_tables_cleaned": 2,   # 0 if non-PG
    "local_files_removed": 15,    # 0 if keep_files
    "errors": [],
}
```

### API Endpoint

```
POST /reset
Authorization: Bearer <token>
Body: {"workspace": "default", "keep_files": false, "dry_run": false}
```

Protected by existing Bearer auth. No additional confirmation field needed since the caller is programmatic and authenticated.

## What Gets Removed

- `_STORAGE_ATTRS` constant moves from `reset.py` to `service.py` (service owns storage knowledge).
- `_drop_storages()`, `_clean_pg_orphan_tables()`, `_reset_local()` logic moves into `RAGService.areset()`.
- `reset.py` reduces to CLI argument parsing + `manager.areset()` call + output formatting.

## Files Changed

| File | Change |
|------|--------|
| `src/dlightrag/core/service.py` | Add `areset()`, move `_STORAGE_ATTRS` here |
| `src/dlightrag/core/servicemanager.py` | Add `areset()` with workspace routing + cache eviction |
| `src/dlightrag/api/server.py` | Add `POST /reset` endpoint |
| `scripts/reset.py` | Simplify to CLI wrapper calling `manager.areset()` |

## Testing

- Unit test: `service.areset()` calls `.drop()` on all LightRAG stores and `.clear()` on DlightRAG stores
- Unit test: non-PG backend skips Phase 3 orphan cleanup
- Unit test: `dry_run=True` returns stats without executing deletions
- Unit test: `keep_files=True` skips Phase 5
- Unit test: `manager.areset(workspace=None)` iterates all workspaces
- Unit test: manager evicts service from cache after reset
- Unit test: errors in one phase do not block subsequent phases
