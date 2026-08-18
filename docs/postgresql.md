# PostgreSQL

This page is for operators deploying or tuning DlightRAG's database layer. It
owns PostgreSQL version requirements, extensions, pool sizing, HNSW tuning,
schema migrations, and deployment notes. Runtime ownership lives in
[architecture.md](architecture.md); config fields live in
[configuration.md](configuration.md); rebuild procedures live in
[operations.md](operations.md).

DlightRAG's supported core storage ecosystem is PostgreSQL 18 with:

- `pgvector` for vector search
- `pg_textsearch` for BM25
- `pg_jieba` for the Chinese `public.jiebacfg` BM25 profile

No fuzzy-search or separate Chinese-parser extension is required. Metadata
filtering compares `LOWER(TRIM(...))` on both sides, over the built-in columns
and over any key of the `custom_metadata` JSONB column.

## Required Version

Startup checks require PostgreSQL 18 or newer. Workspaces should not mix
embedding models or dimensions after data has been indexed; changing
`embedding.dim` requires clearing the workspace and rebuilding vector indexes.

The checked-in Docker Compose stack builds `dlightrag-postgres:pg18` from the
local `postgres/` image definition and preloads `pg_textsearch,pg_jieba`.

Default vector storage is `HALFVEC(dim)` with HNSW. Plain `HNSW` over
`VECTOR(dim)` remains available as an explicit fallback for deployments that
prefer full-precision storage and have rebuilt indexes accordingly.

## External and Managed Endpoints

Set `DLIGHTRAG_POSTGRES_*` in `.env` (see `.env.example`). `config.yaml` is
tracked and carries no endpoint; under Compose `.env` outranks it anyway.

Three capabilities are gated independently, so missing one does not force the
others down:

| Requirement | If unavailable |
| --- | --- |
| PostgreSQL 18 | Hard stop, no fallback |
| pgvector ≥ 0.7 | `pg_vector_index_type: HNSW` |
| `pg_textsearch` | `bm25_enabled: false` (vector-only) |

`pg_textsearch` refuses to install unless the server preloads it, which managed
providers rarely expose — that, not the extension catalog, usually decides
whether BM25 is available. `pg_jieba` installs and tokenizes without preloading,
and is needed only for the `public.jiebacfg` BM25 profile.

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
```

SSL belongs with the endpoint in `.env`
(`DLIGHTRAG_POSTGRES_SSL_MODE`, `_SSL_ROOT_CERT`, `_SSL_CERT`, `_SSL_KEY`,
`_SSL_CRL`). It is bridged to LightRAG's `POSTGRES_SSL_*` environment
contract once, when the root PostgreSQL corpus adapter is constructed.
DlightRAG's domain-store pool, maintenance adapter, and readiness adapter use the
same `pg_connection_kwargs()` path, so managed PostgreSQL deployments do not
need a second SSL configuration surface. Constructing configuration alone does
not mutate LightRAG's process environment.

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
| `max_parallel_parse_docling` | External parser workers for the Docling route | Parser service, CPU/GPU, OCR latency |
| `max_parallel_analyze` | visual/multimodal analysis workers | VLM endpoint limits |
| `max_async` | Process-wide AI provider request concurrency | model endpoint throughput |
| `runtime.answer_worker_concurrency` | Durable Answer runs executed per process | run throughput, CPU, and memory |
| `rag_pipeline_max_async` | LightRAG pipeline LLM request concurrency | LLM endpoint limits |
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
remain managed by LightRAG.

DlightRAG ensures the current idempotent DDL baseline on writer startup and
records its versions in the ledger; readers validate the same versions without
issuing DDL. Because the project is pre-release, a ledger version not declared
by the running revision is incompatible: both roles fail startup and require a
full development-data reset rather than attempting an old-data migration. See
[ADR 0001](adr/0001-reset-development-data-for-breaking-schema-changes.md).
Run `uv run scripts/reset_development.py --mode docker` (or `--mode native`)
to perform that reset; it also recreates the required PostgreSQL extensions
and verifies the empty checkpoint. See
[operations.md](operations.md#full-development-reset).

## Durable Answer Run State

Every answer is one durable run. Four DlightRAG-owned tables hold that state
under the `answer_runs` migration scope:

| Table | Key | Holds |
| --- | --- | --- |
| `dlightrag_answer_runs` | `(owner_id, run_id)` | status, phase, completed turns, stop reason, cancellation request, lease owner/expiry, fencing epoch, consecutive-recovery count, next event sequence, event-trim timestamp, latest checkpoint, canonical result or terminal error |
| `dlightrag_answer_run_events` | `(owner_id, run_id, event_sequence)` | gap-free `progress` / `token` / `reset` / `done` / `error` events |
| `dlightrag_answer_artifacts` | `(owner_id, digest)` | immutable content-addressed bytes within one owner |
| `dlightrag_answer_run_artifacts` | `(owner_id, run_id, resource_id)` | ordered run inputs and fetched resources, referencing a digest |

`run_id` is a UUIDv7. A partial unique index makes one idempotency key unique per
owner, and a second one allows exactly one terminal event per run. The
run-artifact join carries `ON DELETE CASCADE` to the run and `ON DELETE RESTRICT`
to the blob, so adopting a digest takes the key-share lock that serializes
against cleanup. Deleting a run removes its events and references, never shared
bytes; a blob is deleted only once no reference for that owner survives.

Web conversation turns link to a run with `(principal_id, answer_run_id)` and
`ON DELETE CASCADE`. The turn carries conversation order and the run link only:
request content, answer text, sources, and uploaded bytes all live in the run, so
nothing about one answer is stored twice. The baseline schema creates only this
run-link representation; no duplicated-answer or Web-owned attachment tables
exist.

### Retention

Retention is fixed internal maintenance with no operator knob. Every run-owning
process runs it hourly in bounded `SKIP LOCKED` batches, so it is safe on every
host at once and needs no leader election:

- **Event logs** are deleted 30 days after `finished_at` for every terminal run,
  even one a conversation still shows. That transaction sets `events_trimmed_at`,
  after which the run's event endpoint returns HTTP 410 and clients read the
  canonical result from the status endpoint instead.
- **Terminal run rows** are pruned 30 days after `finished_at`, except a
  `succeeded` run a committed Web turn still references — its lifetime belongs to
  that conversation. Failed and cancelled linked runs prune normally and their
  visible terminal turn cascades away with them.
- **Blobs** are released in the same transaction once no run-artifact row
  references the digest for that owner. A digest a concurrent run adopted is left
  alone and released when that run is itself deleted.

Conversation deletion, conversation-TTL pruning, and window trimming all delete
the linked runs before the conversation rows, which is the same lock order run
retention takes; the reverse order deadlocks against a concurrent prune.

## Graph Storage

LightRAG's knowledge graph uses `PGTableGraphStorage`: two ordinary PostgreSQL
tables, no extension.

| Table | Key |
| --- | --- |
| `lightrag_graph_nodes` | `(workspace, namespace, id)` |
| `lightrag_graph_edges` | `(workspace, namespace, src_id, tgt_id)` |

Node and edge attributes live in a `properties JSONB` column, and traversal is
plain recursive SQL over an index on `(workspace, namespace, tgt_id)`. Edges are
undirected: LightRAG canonicalizes each pair in Python before writing, never
with SQL `LEAST`/`GREATEST`, so endpoint ordering cannot drift with the
database collation.

Keeping the graph in ordinary tables is why no compiled graph extension,
`shared_preload_libraries` entry, or per-workspace schema DDL appears anywhere
in the operational surface, and why the graph alone would run on stock
PostgreSQL 14. Upstream measures `get_knowledge_graph` at 39ms against the
former Apache AGE backend's 1099ms on the same data.

The tables are created by `initialize()` under an advisory lock, so any process
may be first. Workspace isolation is a column, not a schema, so resetting a
workspace is a `DELETE`, and orphaned workspaces leave no schemas behind.

## PG Pool Architecture

DlightRAG uses one configured PostgreSQL endpoint per service process, selected
by `service_role`. Both roles target the **same primary endpoint**: a writer
applies DlightRAG schema migrations and mutates the corpus, and a reader still
writes DlightRAG operational state (see
[Service roles and shared artifacts](#service-roles-and-shared-artifacts)).
LightRAG's staged pipeline already supports ingest and query in the same writer
process; local query-while-ingest behavior should be tuned through
parser/analyze/insert/model concurrency before changing database topology.

DlightRAG uses two asyncpg pools:

| Pool | Owner | Purpose |
|---|---|---|
| LightRAG ClientManager pool | LightRAG | KV, vector, graph, doc status |
| `pg_pool` singleton | DlightRAG | Metadata index, BM25, workspace metadata |

The dedicated DlightRAG pool avoids contention between LightRAG internals and
metadata/BM25 reads and writes. Both pools use the same endpoint, SSL settings,
and session-level PostgreSQL tuning.

All concrete implementations live under `dlightrag.adapters.postgres`. RAG owns
the storage-neutral `WorkspaceCorpusBackend` bundle, `CorpusCoordination`, and
`CorpusMaintenanceStore` interfaces. Their PostgreSQL implementations own
version and extension checks,
initialization and pipeline-recovery advisory locks, read-only corpus attach,
workspace catalog cleanup, and readiness probing without exposing asyncpg
connections or exception classes to RAG, reset, Web, API, or MCP code.

## Service roles and shared artifacts

`service_role: reader` (or `DLIGHTRAG_SERVICE_ROLE=reader`) means
**corpus-read-only, not process-read-only**. A reader may create and execute
answer runs and may write DlightRAG operational state: runs, events, artifacts,
and Web conversations. Web is enabled on readers.

A reader:

- uses a **writable** DlightRAG domain session, while the LightRAG pool keeps
  `default_transaction_read_only=on` and the no-DDL attach path;
- **validates** the migrated domain and LightRAG schemas at startup and issues no
  DDL; a missing or incompatible schema fails startup with a diagnostic and
  serves no traffic, and a runtime schema mismatch answers HTTP 503;
- keeps the LightRAG LLM response cache disabled and skips ingest-job recovery;
  and
- still rejects ingestion, workspace creation/reset, metadata mutation,
  failed-document retry, and deletion through `CorpusAdmin` (HTTP 403).

DlightRAG makes no physical-standby or read-endpoint promise: both roles use the
same primary endpoint. Read-replica routing would need a separate corpus endpoint
and is outside this design.

Deployment requirements:

- Apply writer migrations first, then roll readers onto the new revision. A
  reader started against an older schema fails fast rather than degrading.
- An operator-set `default_transaction_read_only=on` on the **domain** session
  fails `/ready` for **both** roles, because both write operational state.
- Route traffic only to instances whose unauthenticated `GET /ready` returns HTTP
  200. `/ready` probes the database and short-caches its verdict; `GET /health`
  is liveness only and never touches PostgreSQL.
- Every process serving KB images or retained source downloads must see the same
  POSIX artifact tree at the **same absolute `working_dir` path**. Single host:
  the existing volume or a shared named volume. Multi-host: one shared POSIX
  mount such as EFS, NFS, or Azure Files. DlightRAG emulates no object store:
  LightRAG writes `file://` sidecar URIs under `INPUT_DIR/__parsed__` and its
  installed resolver returns `None` for remote schemes.
- All Answer workers sharing one database must run a compatible software revision
  and the same effective model-role, Answer image-policy, and agent-limit
  configuration. Drain or cancel active and queued runs before an incompatible
  rolling change.

## Version Support Log

`lightrag-hku>=1.5.6` is required: `PGTableGraphStorage` first ships there.
DlightRAG carries no patches against LightRAG's PostgreSQL layer.
