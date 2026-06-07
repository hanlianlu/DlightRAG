# PostgreSQL Maintenance Notes

DlightRAG's supported core storage ecosystem is PostgreSQL 18 with:

- `pgvector` for vector search
- Apache AGE for LightRAG graph storage
- `pg_textsearch` for BM25
- `pg_jieba` for the Chinese `public.jiebacfg` BM25 profile

No fuzzy-search or separate Chinese-parser extension is required. Metadata
filtering is normalized exact/pattern matching over declared fields.

## Required Version

Startup checks require PostgreSQL 18 or newer. Workspaces should not mix
embedding models or dimensions after data has been indexed; changing
`embedding.dim` requires clearing the workspace and rebuilding vector indexes.

The checked-in Docker Compose stack builds `dlightrag-postgres:pg18` from the
local `postgres/` image definition and preloads `age,pg_textsearch,pg_jieba`.

Default vector storage is `HALFVEC(dim)` with HNSW. Plain `HNSW` over
`VECTOR(dim)` remains available as an explicit fallback for deployments that
prefer full-precision storage and have rebuilt indexes accordingly.

## Tuning Boundaries

DlightRAG splits PostgreSQL tuning into two layers:

- **Server-level settings** (`shared_buffers`, `work_mem`,
  `maintenance_work_mem`, WAL settings, preload libraries) belong to the
  PostgreSQL deployment. The checked-in Docker compose stack carries a local
  single-node profile; production deployments should tune these in their own
  Postgres configuration.
- **Docker shared memory** is separate from PostgreSQL memory GUCs. The
  checked-in compose stack sets `shm_size: 8gb` so HNSW index builds and
  rebuilds have enough `/dev/shm` headroom. This should be kept in proportion
  to corpus size and concurrent index maintenance.
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
postgres_lightrag_pool_max_size: 16
postgres_pool_min_size: 2
postgres_pool_max_size: 10
postgres_connection_retries: 10
postgres_connection_retry_backoff: 3.0
postgres_connection_retry_backoff_max: 30.0
postgres_pool_close_timeout: 5.0
postgres_ssl_mode: require  # disable | allow | prefer | require | verify-ca | verify-full
postgres_ssl_root_cert: /etc/postgresql/ca.crt
```

PostgreSQL SSL config is bridged to LightRAG's `POSTGRES_SSL_*` environment
contract. DlightRAG's domain-store pool, reset helpers, and status probes use
the same `pg_connection_kwargs()` path, so managed PostgreSQL deployments do
not need a second SSL configuration surface.

Connection budgets are split deliberately:

- `postgres_lightrag_pool_max_size` controls LightRAG's PostgreSQL backend
  pool and is bridged to `POSTGRES_MAX_CONNECTIONS`.
- `postgres_pool_min_size` / `postgres_pool_max_size` control DlightRAG-owned
  domain stores such as metadata, workspaces, checkpoints, and BM25.
- Docker Compose defaults `max_connections` to `80`, enough for the checked-in
  API + MCP local profile with headroom. Production deployments should size the
  server limit from the number of DlightRAG processes and their two pool caps.

## DlightRAG Schema Migrations

DlightRAG-owned PostgreSQL tables use `dlightrag_schema_migrations` as a small
ledger for domain schema changes. This applies to DlightRAG tables such as
`dlightrag_doc_metadata` and `dlightrag_workspace_meta`; LightRAG-owned tables
and AGE graph schemas remain managed by LightRAG.

DlightRAG ensures these idempotent DDL migrations on startup and records their
versions in the ledger. All application surfaces use the same configured
PostgreSQL endpoint.

## LightRAG AGE Contract Patches

DlightRAG keeps two narrowly scoped patches around LightRAG PostgreSQL AGE
graph initialization. They exist because LightRAG 1.5.0 still lacks the guards
below.
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

DlightRAG uses one configured PostgreSQL endpoint per service process. That
endpoint must be write-capable because startup may apply DlightRAG schema
migrations, LightRAG may initialize storage, and operational APIs can mutate
workspace state. LightRAG's staged pipeline already supports ingest and query
in the same process; local query-while-ingest behavior should be tuned through
parser/analyze/insert/model concurrency before changing database topology.

DlightRAG uses two asyncpg pools:

| Pool | Owner | Purpose |
|---|---|---|
| LightRAG ClientManager pool | LightRAG | KV, vector, graph, doc status |
| `pg_pool` singleton | DlightRAG | Metadata index, BM25, workspace metadata |

The dedicated DlightRAG pool avoids contention between LightRAG internals and
metadata/BM25 reads and writes. Both pools use the same endpoint, SSL settings,
and session-level PostgreSQL tuning.

If production infrastructure needs read replicas, configure them outside
DlightRAG through managed PostgreSQL, a proxy, or platform-level routing.
DlightRAG itself does not accept separate replica credentials, runtime roles,
or read-after-write replay policies.

## Version Support Log

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
