# PostgreSQL

This page is for operators deploying or tuning DlightRAG's database layer. It
owns PostgreSQL version requirements, extensions, pool sizing, HNSW tuning, AGE
patches, schema migrations, and deployment notes. Runtime ownership lives in
[architecture.md](architecture.md); config fields live in
[configuration.md](configuration.md); rebuild procedures live in
[operations.md](operations.md).

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
postgres_pool_max_size: 16
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
  domain stores such as metadata, workspaces, Web conversations, and BM25.
- Docker Compose defaults `max_connections` to `80` for the local profile.
  Production deployments should size the server limit from the number of
  DlightRAG processes and their two pool caps.

At startup, DlightRAG logs a connection sanity line using the connected
server's real `max_connections`. If common process-count env vars such as
`WEB_CONCURRENCY`, `UVICORN_WORKERS`, or `GUNICORN_WORKERS` are set, it
multiplies the per-process pool budget by that count and warns when the
estimated pool budget consumes the server after a small admin headroom.

Concurrency knobs affect different bottlenecks:

| Setting | Controls | First bottleneck |
|---|---|---|
| `postgres_lightrag_pool_max_size` | LightRAG PostgreSQL connections | PostgreSQL `max_connections` |
| `postgres_pool_max_size` | DlightRAG metadata/BM25/job connections | PostgreSQL `max_connections` |
| `max_parallel_insert` | staged insert/vector/KG write workers | PostgreSQL writes and vector indexes |
| `max_parallel_parse_native` | native parser workers | CPU and file I/O |
| `max_parallel_parse_mineru` | External parser workers for the MinerU-compatible route | Parser service, CPU/GPU, OCR latency |
| `max_parallel_analyze` | visual/multimodal analysis workers | VLM endpoint limits |
| `max_async` | LightRAG LLM request concurrency | LLM endpoint limits |
| `embedding_func_max_async` | embedding request concurrency | embedding endpoint and vector writes |

For a single DlightRAG process, reserve roughly
`postgres_lightrag_pool_max_size + postgres_pool_max_size` PostgreSQL
connections. Multiply that by API worker count before comparing it with
PostgreSQL `max_connections`, leaving room for migrations, admin sessions,
health checks, and managed-service maintenance.

## DlightRAG Schema Migrations

DlightRAG-owned PostgreSQL tables use `dlightrag_schema_migrations` as a small
ledger for domain schema changes. This applies to DlightRAG tables such as
`dlightrag_doc_metadata` and `dlightrag_workspace_meta`; LightRAG-owned tables
and AGE graph schemas remain managed by LightRAG.

DlightRAG ensures these idempotent DDL migrations on startup and records their
versions in the ledger.

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

## PG Pool Architecture

DlightRAG uses one configured PostgreSQL endpoint per service process, selected
by `service_role`. A **writer** process (the default) targets a write-capable
endpoint because startup may apply DlightRAG schema migrations, LightRAG may
initialize storage, and operational APIs can mutate workspace state. A
**reader** process targets an infra-provided read endpoint and never writes (see
[Reader role and read replicas](#reader-role-and-read-replicas)). LightRAG's
staged pipeline already supports ingest and query in the same writer process;
local query-while-ingest behavior should be tuned through
parser/analyze/insert/model concurrency before changing database topology.

DlightRAG uses two asyncpg pools:

| Pool | Owner | Purpose |
|---|---|---|
| LightRAG ClientManager pool | LightRAG | KV, vector, graph, doc status |
| `pg_pool` singleton | DlightRAG | Metadata index, BM25, workspace metadata |

The dedicated DlightRAG pool avoids contention between LightRAG internals and
metadata/BM25 reads and writes. Both pools use the same endpoint, SSL settings,
and session-level PostgreSQL tuning.

## Reader role and read replicas

To scale read/query load horizontally, run additional processes with
`service_role: reader` (or `DLIGHTRAG_SERVICE_ROLE=reader`) whose `postgres_host`
points at an infra-managed **read endpoint**. Replica routing, health, and
failover stay entirely in the infrastructure layer; DlightRAG accepts no replica
credentials, no app-level routing, and no read-after-write replay policy.

A reader:

- attaches to the already-provisioned schema without any DDL, migration, or AGE
  graph creation, and verifies the required tables/labels exist (fail-fast if
  the writer has not created them yet);
- forces `default_transaction_read_only=on` on every connection (both pools and
  direct probes) as defense-in-depth, so a reader accidentally pointed at a
  writable endpoint still cannot write;
- disables the LightRAG response cache and rejects mutating REST/MCP/SDK calls
  with HTTP 403; and
- serves only stateless query/read APIs; the bundled Web conversation surface
  runs on writer instances.

Infrastructure requirements:

- Use **physical streaming replication** (it copies AGE, pgvector, and DDL
  byte-for-byte; logical replication does not propagate DDL and is unfit for the
  AGE/extension stack).
- Keep replication **asynchronous**; never make ingest wait on a replica.
- Bring up a writer to create/migrate schema first, let replication ship it,
  then start readers. On schema-changing releases, migrate on the writer first,
  then roll readers.
- Route traffic only to instances whose unauthenticated `GET /ready` probe
  returns HTTP 200. Reader readiness also verifies the DlightRAG pool is in a
  read-only transaction mode; `/health` remains a diagnostic HTTP 200 response.
- Readers observe **eventual consistency**: a document ingested on the writer is
  retrievable on readers only after replication catches up.
- Physical replication does not copy the `working_dir`. If readers must serve
  visual evidence or retained local source downloads, mount the writer's
  `working_dir` read-only at the **same absolute path**; otherwise prefer remote
  (S3/Azure/HTTPS) source locators.
- Tune `hot_standby_feedback` and `max_standby_streaming_delay` to reduce
  standby query cancellations, and monitor replication slot / WAL retention so a
  dead replica cannot fill the primary's disk.

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
