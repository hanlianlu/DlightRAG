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

The checked-in Docker Compose stack builds `dlightrag-postgres:pg18` from the
local `postgres/` image definition and preloads `age,pg_textsearch`. The
optional `replica` profile uses the same image and physical streaming
replication, so vector, graph, and BM25 indexes replicate through WAL rather
than application-level copy jobs.

Default vector storage is `VECTOR(dim)` with HNSW. `HNSW_HALFVEC` is an
explicit opt-in index type for deployments that have chosen the precision
tradeoff and rebuilt indexes accordingly.

## Tuning Boundaries

DlightRAG splits PostgreSQL tuning into two layers:

- **Server-level settings** (`shared_buffers`, `work_mem`,
  `maintenance_work_mem`, WAL settings, preload libraries) belong to the
  PostgreSQL deployment. The checked-in Docker compose stack carries a local
  single-node profile; production primary/replica deployments should tune
  these in their own Postgres configuration.
- **Docker shared memory** is separate from PostgreSQL memory GUCs. The
  checked-in compose stack sets `shm_size: 8gb` on both primary and replica so
  HNSW index builds and rebuilds have enough `/dev/shm` headroom. This should
  be kept in proportion to corpus size and concurrent index maintenance.
- **Session-level settings** belong to DlightRAG config. `pg_hnsw_ef_search`
  becomes `hnsw.ef_search`, and `postgres_session_settings` can add additional
  per-connection GUCs. DlightRAG applies the same session settings to both
  LightRAG's PostgreSQL pool and the DlightRAG domain-store `pg_pool`.

Example:

```yaml
pg_hnsw_ef_search: 256
postgres_session_settings:
  application_name: dlightrag
  statement_timeout: "60000"
postgres_statement_cache_size: 256
postgres_connection_retries: 10
postgres_connection_retry_backoff: 3.0
postgres_connection_retry_backoff_max: 30.0
postgres_pool_close_timeout: 5.0
postgres_ssl_mode: require  # disable | allow | prefer | require | verify-ca | verify-full
postgres_ssl_root_cert: /etc/postgresql/ca.crt
```

PostgreSQL SSL config is shared by the active primary/replica endpoint and is
bridged to LightRAG's `POSTGRES_SSL_*` environment contract. DlightRAG's
domain-store pool, reset helpers, replication checks, and status probes use
the same `pg_connection_kwargs()` path, so managed PostgreSQL deployments do
not need a second SSL configuration surface.

## DlightRAG Schema Migrations

DlightRAG-owned PostgreSQL tables use `dlightrag_schema_migrations` as a small
ledger for domain schema changes. This applies to DlightRAG tables such as
`dlightrag_doc_metadata` and `dlightrag_workspace_meta`; LightRAG-owned tables
and AGE graph schemas remain managed by LightRAG.

Writable runtimes ensure these idempotent DDL migrations on startup and record
their versions in the ledger. Query read-only runtimes only verify that the
required migration versions already exist, so run an ingest/admin process
against the primary before starting replica-only query workers after an upgrade.

## LightRAG AGE Compatibility Patches

DlightRAG keeps two narrowly scoped patches around LightRAG PostgreSQL AGE
graph initialization. They are not general backward-compatibility shims: they
exist because LightRAG 1.5.0 still lacks the guards below.
They self-disable through source inspection if upstream adds equivalent
handling, and should be deleted once `required_patch_names(PostgreSQLDB)`
returns an empty tuple.

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
checks are present. The public audit helper is:

```python
from dlightrag.core._lightrag_patches import required_patch_names
from lightrag.kg.postgres_impl import PostgreSQLDB

print(required_patch_names(PostgreSQLDB))
```

Implementation equivalent:

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
primary endpoint and may run DDL, ingest LightRAG data, update metadata, and
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
| `pg_pool` singleton | DlightRAG | Metadata index, BM25, workspace/role metadata |

The dedicated DlightRAG pool avoids contention between LightRAG internals and
metadata/BM25 reads and writes. In query role, both pools attach to the replica
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

Local commands:

```bash
make postgres-replica-prepare
make postgres-replica-start
make postgres-replica-smoke
make postgres-replica-reset
```

## Version Compatibility Log

| Date | LightRAG Baseline | Patches Needed | Notes |
|---|---|---|---|
| 2026-06-03 | `lightrag-hku==1.5.0` | `configure_age`, `execute` | Locked through `uv.lock`; remove this patch module when upstream covers both |

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
