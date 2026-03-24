# PostgreSQL Maintenance Notes

## LightRAG Upstream Bug Patches

DlightRAG (and ArtRAG) maintain monkey-patches for two critical LightRAG
PostgreSQL bugs that remain unfixed as of **LightRAG v1.4.11** (2026-03-24).

### Bug 1: `configure_age()` missing `DuplicateSchemaError`

**Location:** `lightrag/kg/postgres_impl.py`, `configure_age()` static method

**Problem:** When Apache AGE's `create_graph()` is called on an
already-existing graph, it raises `asyncpg.exceptions.DuplicateSchemaError`.
LightRAG's `configure_age()` only catches `InvalidSchemaNameError` and
`UniqueViolationError` — `DuplicateSchemaError` is in a completely different
exception hierarchy (`SyntaxOrAccessError`, not `InvalidSchemaNameError`).

**Impact:** `configure_age()` is called on EVERY query via `_run_with_retry()`.
After the first successful `create_graph()`, all subsequent calls crash with
the uncaught exception. This aborts graph label creation (`create_vlabel`,
`create_elabel`), leaving the graph in a half-initialized state. Ingestion
then fails with `UndefinedTableError: relation "<workspace>.base" does not
exist`.

**Our fix (`_lightrag_patches.py`):**
- Pre-check `ag_catalog.ag_graph` before calling `create_graph()` — avoids
  both the exception and PG ERROR log spam on normal startups
- Catch `DuplicateSchemaError` for race conditions (another worker creates
  the graph between check and create)
- Wrap `execute()` to also catch `DuplicateSchemaError` (defense-in-depth)

### Bug 2: `execute()._operation` missing `DuplicateSchemaError`

**Location:** `lightrag/kg/postgres_impl.py`, `execute()` method, inner
`_operation` closure

**Problem:** The exception tuple in `_operation` catches
`UniqueViolationError`, `DuplicateTableError`, `DuplicateObjectError`, and
`InvalidSchemaNameError` — but NOT `DuplicateSchemaError`. When
`ignore_if_exists=True` or `upsert=True`, `DuplicateSchemaError` propagates
instead of being silently handled.

**Our fix:** Wrap `execute()` to catch `DuplicateSchemaError` at the outer
level, respecting `ignore_if_exists` and `upsert` flags.

### Auto-detection

Both patches use `inspect.getsource()` to check if the upstream code has
been fixed. If LightRAG adds the pre-check or exception handling, our
patches automatically skip themselves:

```python
def _needs_patch(method):
    source = inspect.getsource(method)
    return "ag_catalog.ag_graph" not in source  # pre-check sentinel
```

### Post-Init Graph Verification

In addition to the patches, `RAGService._verify_graph_labels()` runs after
`initialize_storages()` to detect corrupted graphs (schema exists but
`base` vlabel table missing). If found, it drops the graph and re-runs
initialization. This handles stale state from previous failed inits.

## PG Pool Architecture

DlightRAG uses **two independent asyncpg pools**:

| Pool | Owner | Purpose | Sizing |
|------|-------|---------|--------|
| LightRAG ClientManager pool | LightRAG | KV, vector, graph, doc_status storage | LightRAG defaults |
| `pg_pool` singleton | DlightRAG (`storage/pool.py`) | Domain stores (visual_chunks via PGJsonbKV) | min=2, max=10 |

The dedicated pool avoids contention between LightRAG's internal operations
and DlightRAG's domain store writes (especially visual_chunks with large
BYTEA blobs).

## Version Compatibility Log

| Date | LightRAG Version | Patches Needed | Notes |
|------|-----------------|----------------|-------|
| 2026-03-23 | 1.4.11rc2 | Yes (both bugs) | Initial patch created |
| 2026-03-24 | 1.4.11 | Yes (both bugs) | Verified: upstream unchanged |

Update this table when LightRAG releases new versions. Check by running:
```bash
uv run python -c "
from lightrag.kg.postgres_impl import PostgreSQLDB
import inspect
src = inspect.getsource(PostgreSQLDB.configure_age)
print('Pre-check:', 'ag_catalog.ag_graph' in src)
print('DuplicateSchemaError:', 'DuplicateSchemaError' in src)
"
```

If both print `True`, the patches can be removed.
