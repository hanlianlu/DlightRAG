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

Startup checks require PostgreSQL 18 or newer and `lightrag-hku>=1.5.7`.
DlightRAG carries no patches against LightRAG's PostgreSQL layer. Workspaces
should not mix embedding models or dimensions after indexing; changing
`models.embedding.dim` requires clearing/rebuilding vectors.

The checked-in Docker Compose stack builds `dlightrag-postgres:pg18` from the
local `postgres/` image definition, pins `pg_textsearch` to v1.4.0, and preloads
`pg_textsearch,pg_jieba`.

Default vector storage is `HALFVEC(dim)` with HNSW. Plain `HNSW` over
`VECTOR(dim)` remains available as an explicit fallback for deployments that
prefer full-precision storage and have rebuilt indexes accordingly.

## External and Managed Endpoints

Keep the PostgreSQL password in `.env` locally or an orchestrator Secret. The
bundled Compose manifest owns `DLIGHTRAG_STORAGE__POSTGRES__HOST=postgres`
because that hostname is created by Compose service discovery; it inherits the
remaining non-secret local defaults. For an external or managed endpoint, choose
one owner for the non-secret connection fields: place stable values under
`storage.postgres` in the deployment's `config.yaml`, or inject topology-created
values from its deployment manifest. Do not duplicate them in both. See
[Configuration](configuration.md#configuration-ownership) and `.env.example`.

Three capabilities are gated independently, so missing one does not force the
others down:

| Requirement | If unavailable |
| --- | --- |
| PostgreSQL 18 | Hard stop, no fallback |
| pgvector ≥ 0.7 | `storage.lightrag.vector_index_type: HNSW` |
| `pg_textsearch` | `corpus.retrieval.bm25_enabled: false` (vector-only) |

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
  Postgres configuration. The compose profile reads its overrides from
  `COMPOSE_POSTGRES_SHARED_BUFFERS`, `COMPOSE_POSTGRES_WORK_MEM`,
  `COMPOSE_POSTGRES_MAINTENANCE_WORK_MEM`, `COMPOSE_POSTGRES_EFFECTIVE_CACHE_SIZE`,
  `COMPOSE_POSTGRES_MAX_CONNECTIONS`, and `COMPOSE_POSTGRES_SHM_SIZE`, set in the
  shell or the `.env` beside `docker-compose.yml`. Compose-only inputs stay out
  of the application-reserved `DLIGHTRAG_*` namespace, where the application
  rejects any name that is neither a configuration field nor a documented
  client or test variable.
- **Docker shared memory** is separate from PostgreSQL memory GUCs. The
  checked-in compose stack sets `shm_size: 8gb` so HNSW index builds and
  rebuilds have enough `/dev/shm` headroom. This should be kept in proportion
  to corpus size and concurrent index maintenance.
- **Session-level settings** belong to DlightRAG config.
  `storage.lightrag.hnsw_ef_search` becomes `hnsw.ef_search`, and
  `storage.postgres.session_settings` can add additional
  per-connection GUCs. DlightRAG applies the same session settings to both
  LightRAG's PostgreSQL pool and the DlightRAG domain-store `pg_pool`.

Example:

```yaml
storage:
  lightrag:
    hnsw_ef_search: 256
  postgres:
    session_settings:
      application_name: dlightrag
      statement_timeout: "60000"
    statement_cache_size: 256
    lightrag_pool_max_size: 16
    pool_min_size: 2
    pool_max_size: 16
    connection_retries: 10
    connection_retry_backoff: 3.0
    connection_retry_backoff_max: 30.0
    pool_close_timeout: 5.0
```

SSL belongs with the endpoint in `.env`
(`DLIGHTRAG_STORAGE__POSTGRES__SSL_MODE`, `__SSL_ROOT_CERT`, `__SSL_CERT`,
`__SSL_KEY`, `__SSL_CRL`). It is bridged to LightRAG's `POSTGRES_SSL_*`
environment contract once, when the root PostgreSQL corpus adapter is
constructed; that adapter owns the whole LightRAG environment bridge
(PostgreSQL, Milvus, parser sidecars, parser rules, input directory). The
session settings both pools use are rendered in one place, the PostgreSQL
adapter core (`adapters/postgres/core/_session_settings.py`): the domain pool
applies them directly and the bridge hands them to LightRAG. Every connection
DlightRAG opens itself (the domain-store pool, the notification listener, the
maintenance and readiness adapters, and the workspace write gate) takes its
endpoint, credentials, and TLS context from one adapter-core helper,
`pg_connection_kwargs()` in `adapters/postgres/core/_connection.py`, which the
bridge also reads its endpoint from, so managed PostgreSQL deployments do not
need a second SSL configuration surface. Configuration only holds the validated
settings: constructing it builds no TLS context and does not mutate LightRAG's
process environment.

Connection budgets are split deliberately:

- `storage.postgres.lightrag_pool_max_size` controls LightRAG's PostgreSQL
  backend pool and is bridged to `POSTGRES_MAX_CONNECTIONS`.
- `storage.postgres.pool_min_size` / `storage.postgres.pool_max_size` control DlightRAG-owned
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
| `storage.postgres.lightrag_pool_max_size` | LightRAG PostgreSQL connections | PostgreSQL `max_connections` |
| `storage.postgres.pool_max_size` | DlightRAG metadata/BM25/Run connections | PostgreSQL `max_connections` |
| `corpus.ingestion.pipeline.max_parallel_insert` | staged insert/vector/KG write workers | PostgreSQL writes and vector indexes |
| `corpus.ingestion.pipeline.max_parallel_parse_native` | native parser workers | CPU and file I/O |
| `corpus.ingestion.pipeline.max_parallel_parse_mineru` | External parser workers for the MinerU-compatible route | Parser service, CPU/GPU, OCR latency |
| `corpus.ingestion.pipeline.max_parallel_parse_docling` | External parser workers for the Docling route | Parser service, CPU/GPU, OCR latency |
| `corpus.ingestion.pipeline.max_parallel_analyze` | visual/multimodal analysis workers | VLM endpoint limits |
| `models.max_concurrency` | Process-wide AI provider request concurrency | model endpoint throughput |
| `runtime.query.worker_concurrency` | Query-lane runs executed per process | run throughput, CPU, and memory |
| `runtime.query.max_nonterminal_runs` | Atomic deployment-wide Query-lane nonterminal admission limit | durable backlog growth |
| `corpus.ingestion.pipeline.max_concurrency` | LightRAG pipeline LLM request concurrency | LLM endpoint limits |
| `models.embedding.max_concurrency` | embedding request concurrency | embedding endpoint and vector writes |

For a single DlightRAG process, reserve roughly
`storage.postgres.lightrag_pool_max_size + storage.postgres.pool_max_size` PostgreSQL
connections, plus the one the notification hub holds. Multiply that by API worker
count before comparing it with PostgreSQL `max_connections`, leaving room for
migrations, admin sessions, health checks, and managed-service maintenance.

## Filtered BM25 Top-K

DlightRAG issues one explicit pg_textsearch top-K scan per selected language
profile. The checked-in retrieval defaults are:

- `corpus.retrieval.top_k: 40` for LightRAG graph/entity breadth;
- `corpus.retrieval.chunk_top_k: 20` for text candidates, including the BM25
  SQL `LIMIT`; and
- `corpus.retrieval.direct_visual_top_k: 20` for the independent visual leg.

The BM25 query filters by workspace and may additionally filter by language and
metadata scope before returning those 20 candidates. Publication is an
independent predicate: `_dlightrag_finalization_complete IS TRUE`. Without a
user filter, BM25 ranks a bounded over-fetch window and uses a correlated
metadata `EXISTS`; it never materializes the set of all visible document IDs.
pg_textsearch v1.4.0 uses
planner selectivity to seed the internal scan limit for this query shape,
avoiding repeated score-and-filter passes for selective filters. Compose makes
the upstream defaults explicit:

```text
pg_textsearch.filtered_seed=on
pg_textsearch.filtered_seed_margin=3.0
```

The approximate initial internal budget is
`ceil(margin * chunk_top_k / estimated_filter_selectivity)`; the SQL filter and
`LIMIT` still determine the exact result. The optimization therefore changes
work performed, not result correctness. Override the two server settings with
`COMPOSE_POSTGRES_PG_TEXTSEARCH_FILTERED_SEED` and
`COMPOSE_POSTGRES_PG_TEXTSEARCH_FILTERED_SEED_MARGIN` only after comparing
representative `EXPLAIN (ANALYZE, BUFFERS)` plans and latency. External
PostgreSQL deployments should set the equivalent GUCs in their own server or
session configuration.

## pg_jieba Caveats

pg_jieba v2.0.1 has two defects that DlightRAG works around rather than patches,
so any pg_jieba build serves Chinese BM25 correctly:

- **Whitespace becomes a term.** Each run of spaces, tabs, newlines, or
  ideographic spaces is indexed as its own lexeme, so a spaced query would match
  nearly every chunk through the space alone. Every query DlightRAG sends to a
  `public.jiebacfg` index replaces whitespace runs with a full-width comma, which
  jieba splits on and drops as a stopword: segmentation is unchanged and no
  whitespace term remains. Indexed whitespace still counts toward document
  length, a small uniform bias in BM25 length normalization.
- **Listing jieba token types crashes the server.** `jieba_lextype` writes one
  entry past its allocation, and PostgreSQL restarts every backend to recover.
  DlightRAG never calls it: user text reaches pg_jieba only to be tokenized,
  when chunks and memory bodies are indexed and when `to_bm25query` reads a
  query, and tokenizing never lists token types. Operators must not run `ts_debug` or `ts_token_type` on a
  jieba parser or configuration, `\dF+ jiebacfg` or `\dFp+ jieba` in psql,
  `ALTER TEXT SEARCH CONFIGURATION ... MAPPING FOR` on a jieba configuration, or
  `pg_dump` of a custom configuration built on the jieba parser. Tokenizing with
  `to_tsvector('public.jiebacfg', ...)` is safe. `pg_upgrade` dumps extension
  members in binary-upgrade mode and hits the same function, so patch pg_jieba,
  or drop and later recreate it, before a major-version upgrade. Between the drop
  and the recreate no jieba BM25 index accepts writes, so do it with every
  writer stopped.

## DlightRAG Schema Migrations

DlightRAG-owned PostgreSQL tables use `dlightrag_schema_migrations` as a small
ledger for domain schema changes. This applies to DlightRAG tables such as
`dlightrag_doc_metadata` and `dlightrag_workspace_meta`; LightRAG-owned tables
remain managed by LightRAG. One explicitly derived exception is the
DlightRAG-owned partial index
`idx_dlightrag_file_panel_processed_updated_id` on the LightRAG-owned
`LIGHTRAG_DOC_STATUS` table. It covers the bounded Files presentation order
`(workspace, updated_at DESC NULLS FIRST, id ASC) WHERE status = 'processed'`.
A writer creates it only after LightRAG has established that table; readers
issue no DDL. The metadata migration normalizes legacy NULL publication markers
to false (never true), makes the marker non-null with a false default, installs
a partial visible-document index, and recomputes planner field statistics from
true rows only. Its trigger handles false-to-true, true-to-false, ordinary
updates, and deletes exactly. Reader startup requires the new migration,
constraint, and index. During a rolling upgrade, start an upgraded writer before
readers serve file pages that rely on these objects.

DlightRAG ensures the current idempotent DDL baseline on writer startup and
records its versions in the ledger; readers validate the same versions without
issuing DDL. Because the project is pre-release, a ledger version not declared
by the running revision is incompatible: both roles fail startup and require a
full development-data reset rather than attempting an old-data migration.
Run `uv run scripts/reset_development.py --mode docker` (or `--mode native`)
to perform that reset; it also recreates the required PostgreSQL extensions
and verifies the empty database. See
[operations.md](operations.md#full-development-reset).

A ledger version alone does not prove a schema, since it survives an object
dropped afterwards, so each scope also declares the objects its revision needs
(columns, keys, foreign keys, checks, indexes, and guard triggers) and readers
read each one back from the catalog. The Run scope declares each index once:
its baseline and the migration that introduced it both create the index from
that declaration, and readers verify it by the same name. The Run baseline is
the complete current schema; each later Run migration only brings a database
created before it up to date. Integration tests require a freshly migrated Run
catalog to contain exactly what is declared, no more and no less, and require
every later Run migration to leave a fresh baseline unchanged down to each
definition. A database that still holds `dlightrag_answer_runs` (created by
releases 2.0.0 through 2.0.5 and never started by a later release) is not
migrated: writers and readers both refuse to start on it and name the full
development reset above as the remedy.

## Durable Run State

Every top-level Retrieval and Answer is one durable Run. DlightRAG-owned tables
under the `runs` migration scope separate common lifecycle from Answer-owned
routing, Session, control, child, and blob-reference state:

| Table | Key | Holds |
| --- | --- | --- |
| `dlightrag_runs` | `(owner_id, run_id)` plus globally unique `run_id` | kind, lane, submitter/access scope, submission key, status, retry/checkpoint, retention, cancellation, fenced lease, Prepared Input, Corpus Mutation handoff/repair state, result or terminal error |
| `dlightrag_run_events` | `(owner_id, run_id, event_sequence)` | gap-free executor-owned events, including Answer `progress` / `token` / `reset` / tool / terminal events |
| `dlightrag_blobs` | `(owner_id, digest)` | immutable content-addressed blob metadata within one owner |
| `dlightrag_answer_run_artifacts` | `(owner_id, run_id, resource_id)` | ordered request attachments and Published Artifact bytes |
| `dlightrag_answer_artifact_attachments` | `(owner_id, run_id, relative_path)` | settled Root Artifact Attachment authority: label, raw digest/size, presentation, Effect provenance, and settlement order |
| `dlightrag_answer_run_routing` | `(owner_id, run_id)` | requested/valid/resolved mode and canonical Agent Session/Lane mapping |
| `dlightrag_agent_sessions` | `(owner_id, session_id)` | Session commit sequence, Entry sequence, current run owner and fencing epoch |
| `dlightrag_agent_session_entries` | `(owner_id, session_id, sequence)` | immutable parent-linked User/Assistant/ToolResult/Control/Compaction Entries |
| `dlightrag_agent_session_registers` | `(owner_id, session_id, kind, key)` | exact-CAS Lane heads/state, total OperationState, Plan metadata, request/tool snapshots, bounded inputs and Fast reservation |
| `dlightrag_answer_evidence` / resource tables | run/session/intent/result identity | atomic durable Evidence, fetched resources, workspace inventory, spills, and blobs; a Tool settlement stores the Session's Evidence ledger only when it changed, and recovery reads the latest snapshot; a workspace rescan writes its inventory in two statements that delete removed paths and rewrite only changed rows |
| `dlightrag_answer_child_sessions` | parent run + child Session id | parent/call/intent lineage, ContextSnapshot, depth, independent lease/epoch, pinned plan/budget/tools/Host state, status and usage |
| `dlightrag_answer_child_operations` | parent run + child Session + operation sequence | same-Session continuation Operations, idempotency, origin, status, cancellation origin, usage/outcome |
| `dlightrag_answer_child_guidance` | parent run + request id | correlated `ask_parent` questions, expiry, reply origin, and status `pending` / `replied` / `expired` / `cancelled` |
| `dlightrag_agent_controls` | run + control sequence | ordered steer inbox for the parent Run or a targeted child Session/Operation; origin `user` or `parent`; append-before-ack; a worker locks and reads at most 100 pending controls at a time, in sequence order |

Independent critique reuses those Child Session, Operation, control, and
guidance rows. There is no council, supervisor, or budget table.

`run_id` is a UUIDv7. A partial unique index makes one idempotency key unique per
owner, and a second one allows exactly one terminal event per run. The
run-artifact join carries `ON DELETE CASCADE` to the run and `ON DELETE RESTRICT`
to the blob, so linking a digest takes the key-share lock that serializes
against cleanup. Deleting a run removes its events and references, never shared
bytes; a blob is deleted only once no reference for that owner survives.
The Query and Corpus Mutation claim paths share the bounded
`idx_dlightrag_runs_claim` index; Workspace mutation eligibility also uses
`idx_dlightrag_runs_mutation_fifo`, and event reconnect uses the event primary
key. `EXPLAIN (ANALYZE, BUFFERS)` evidence at representative bounded backlogs
is recorded with the [RunRuntime targets](run-runtime-and-scaling-target.md#captured-local-load-evidence)
and regenerated by `make load-runtime`. A
compact sequential scan chosen for a 1,000-row admission count is not by itself an index
regression; the structural integration test separately proves the ordered
indexes remain usable.

Web conversation turns link to a run with `(principal_id, answer_run_id)` and
`ON DELETE CASCADE`. The turn carries conversation order and the run link only:
request content, answer text, sources, and uploaded bytes all live in the run, so
nothing about one answer is stored twice. The baseline schema creates only this
run-link representation; no duplicated-answer or Web-owned attachment tables
exist.

### Retention Implementation

Every run-owning process sweeps hourly in bounded `SKIP LOCKED` batches, so no
leader or cron job is required. Row locks, cascades, Session reference checks,
and the run-artifact/blob foreign key serialize pruning against new references.
Conversation deletion follows the same Run-first lock order. Answer uses the
configured retention floor; top-level Retrieval and Corpus Mutation select seven days. Lifecycle
and HTTP 410 semantics are defined in
[RunRuntime and durable query execution](durable-answer-runs.md); the
field/default is in [Configuration](configuration.md).

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

Ordinary tables require no compiled graph extension,
`shared_preload_libraries` entry, or per-workspace schema DDL.

The tables are created by `initialize()` under an advisory lock, so any process
may be first. Workspace isolation is a column, not a schema, so resetting a
workspace is a `DELETE`, and orphaned workspaces leave no schemas behind.

## PG Pool Architecture

DlightRAG uses one configured PostgreSQL endpoint per service process, selected
by `deployment.service_role`. Both roles target the **same primary endpoint**: a writer
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

Cross-process wake-ups use LISTEN/NOTIFY through one notification hub per
process. The hub holds one connection of its own, outside `pg_pool` but with its
session settings, LISTENs every channel declared in
`adapters/postgres/core/_channels.py` on it once, and fans each channel out to
its subscribers (the Connections scheduler and OAuth inbox, the model catalogue
reload, the run-cancellation rescan, and Answer child-guidance waits, however
many are waiting); joining and leaving never touch the connection. A
notification is only a wake hint: after every (re)connect, and after each
30-second keepalive passes, the hub tells every subscriber to re-read its
authoritative rows, and it tells a subscriber that joins a live hub once on its
own. A connection that is lost, or whose LISTENs or keepalive fail or outlast
five seconds, is terminated and replaced after one second, doubling with each
further loss up to 30 seconds until a connection passes a keepalive; a
subscriber that joins during that wait cuts it short. Since the hub
resynchronizes only while connected, a waiting child also re-reads its guidance
row every 30 seconds on its own, and an idle Connections refresh loop scans
again when its next refresh falls due and at least every 30 seconds. A
cancellation request's NOTIFY only wakes every process to rescan the
cancel-pending leases it holds, and a process starts claiming Runs only once
such a rescan after its channel went live has signalled everything it found,
however long after startup that is.

All concrete implementations live under `dlightrag.adapters.postgres`. RAG owns
the storage-neutral `WorkspaceCorpusBackend` bundle, `CorpusCoordination`, and
`CorpusMaintenanceStore` interfaces. Their PostgreSQL implementations own
version and extension checks,
initialization and pipeline-recovery advisory locks, read-only corpus attach,
workspace catalog cleanup, and readiness probing without exposing asyncpg
connections or exception classes to RAG, reset, Web, API, or MCP code.

## Service roles and shared artifacts

`deployment.service_role: reader` (or `DLIGHTRAG_DEPLOYMENT__SERVICE_ROLE=reader`) means
**corpus-read-only, not process-read-only**. A reader may create and execute
Retrieval and Answer Runs and may write DlightRAG operational state: Runs,
events, Artifacts, and Web conversations. Web is enabled on readers.

A reader:

- uses a **writable** DlightRAG domain session, while the LightRAG pool keeps
  `default_transaction_read_only=on` and the no-DDL attach path;
- **validates** the migrated domain and LightRAG schemas at startup and issues no
  DDL; a missing or incompatible schema fails startup with a diagnostic and
  serves no traffic, and a runtime schema mismatch answers HTTP 503;
- keeps the LightRAG LLM response cache disabled; and
- **refuses corpus writes at acceptance, before staging a byte**: it registers no
  corpus-mutation executor, so an upload, retry, delete, or reset is answered
  immediately with HTTP 503 and the remedy (send it to a writer) instead of a Run
  no process here can execute. Workspace creation, metadata updates, and
  runtime model-catalogue changes are refused the same way. Web's Files panel
  keeps its read surface: the file list, the failed list, and source download
  where the corpus root is mounted.

DlightRAG makes no physical-standby or read-endpoint promise: both roles use the
same primary endpoint. Read-replica routing would need a separate corpus endpoint
and is outside this design.

A domain session forced read-only fails `/ready` for both roles because both
write operational state. Migration order, probes, shared mounts, homogeneous
worker requirements, and rollout commands are in
[Operations](operations.md#runruntime-and-durable-query-and-corpus-mutation-runs).
