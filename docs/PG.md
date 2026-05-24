# PostgreSQL Maintenance Notes

DlightRAG's supported core storage ecosystem is PostgreSQL 18 with:

- `pgvector` for vector search
- Apache AGE for LightRAG graph storage
- `pg_textsearch` for BM25

`pg_trgm` is intentionally not required. Metadata filtering is normalized
exact/pattern matching over declared fields, not fuzzy matching.

## Required Version

Startup checks require PostgreSQL 18 or newer. Workspaces should not mix
embedding models or dimensions after data has been indexed; changing
`embedding.dim` requires clearing the workspace and rebuilding vector indexes.

Default vector storage is `VECTOR(dim)` with HNSW. `HNSW_HALFVEC` is an
explicit opt-in index type for deployments that have chosen the precision
tradeoff and rebuilt indexes accordingly.

## LightRAG Upstream Compatibility Patches

DlightRAG keeps defensive monkey-patches around LightRAG PostgreSQL AGE graph
initialization. They self-disable through source inspection if upstream adds
equivalent handling.

### Patch 1: `configure_age()` and Existing Graphs

**Location:** `lightrag/kg/postgres_impl.py`, `PostgreSQLDB.configure_age()`

Apache AGE can raise `asyncpg.exceptions.DuplicateSchemaError` when
`create_graph()` races with an existing graph. The patch pre-checks
`ag_catalog.ag_graph`, catches `DuplicateSchemaError` for races, and avoids
normal startup error noise.

### Patch 2: `execute()` and Idempotent DDL

**Location:** `lightrag/kg/postgres_impl.py`, `PostgreSQLDB.execute()`

The patch wraps idempotent DDL calls so `DuplicateSchemaError` is handled when
`ignore_if_exists=True` or `upsert=True`.

### Auto-Detection

Both patches inspect the upstream methods and skip themselves when the required
checks are present:

```python
def _configure_age_needs_patch(method):
    source = inspect.getsource(method)
    return not (
        "ag_catalog.ag_graph" in source
        and "DuplicateSchemaError" in source
    )

def _execute_needs_patch(method):
    source = inspect.getsource(method)
    return "DuplicateSchemaError" not in source
```

## PG Pool Architecture

DlightRAG uses role-aware PostgreSQL endpoints:

```text
dlightrag-ingest / admin API -> primary PostgreSQL
dlightrag-query workers      -> hot standby / read replica PostgreSQL
primary                      -> physical streaming replication -> replica
```

Set `runtime_role: ingest|admin|query`. Ingest/admin workers attach to the
primary endpoint and may run DDL, write sidecar tables, update metadata, and
perform resets. Query workers attach to `postgres_replica_*`, skip advisory
locks and write-time initialization, verify that required LightRAG/DlightRAG
tables already exist, and reject mutating APIs.

This is process-level separation, not dynamic in-process read/write routing.
Strong read-after-write flows can set `read_after_write_mode: wait_for_replay`
on ingest/admin workers so write acknowledgements wait for replica WAL replay.
Flows that cannot wait should call an admin/ingest surface explicitly; ordinary
query workers remain read-only and never dynamically route to primary.

DlightRAG uses two asyncpg pools per process target:

| Pool | Owner | Purpose |
|---|---|---|
| LightRAG ClientManager pool | LightRAG | KV, vector, graph, doc status |
| `pg_pool` singleton | DlightRAG | Metadata registry, document artifacts, chunk provenance, visual side tables |

The dedicated DlightRAG pool avoids contention between LightRAG internals and
domain side-table reads/writes. In query role, both pools attach to the replica
and only read or verify schema.

### Replica Requirements

- PostgreSQL extension versions must match on primary and replica:
  `pgvector`, Apache AGE, and `pg_textsearch`.
- DDL and migrations run primary-only. Query workers never self-heal missing
  tables or indexes.
- `pg_textsearch` BM25 index creation is primary-only; query role verifies
  `idx_lightrag_doc_chunks_bm25` exists.
- Hot standby queries can conflict with WAL replay. Tune
  `max_standby_streaming_delay` and `hot_standby_feedback` deliberately.

## Version Compatibility Log

| Date | LightRAG Baseline | Patches Needed | Notes |
|---|---|---|---|
| 2026-05-24 | `main` commit `9d1910bb63dd5f492844fef9bf91ed228e88a4f7` | Yes, defensive | Locked through `uv.lock`; verify again before release |

Check upstream by running:

```bash
uv run python -c "
from lightrag.kg.postgres_impl import PostgreSQLDB
import inspect
configure_age = inspect.getsource(PostgreSQLDB.configure_age)
execute = inspect.getsource(PostgreSQLDB.execute)
print('configure_age pre-check:', 'ag_catalog.ag_graph' in configure_age)
print('configure_age DuplicateSchemaError:', 'DuplicateSchemaError' in configure_age)
print('execute DuplicateSchemaError:', 'DuplicateSchemaError' in execute)
"
```

If all three checks print `True`, the patches can be removed after the unit
tests covering graph initialization are updated.
