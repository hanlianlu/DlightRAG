# PostgreSQL

This page is for operators deploying or tuning DlightRAG's database layer. It
owns PostgreSQL version requirements, extensions, pool sizing, HNSW tuning,
schema ownership and migrations, and the service roles. Runtime ownership lives
in [architecture.md](architecture.md); config fields live in
[configuration.md](configuration.md); rebuild procedures live in
[operations.md](operations.md).

DlightRAG's supported core storage ecosystem is PostgreSQL 18 with:

- `pgvector` for vector search
- `pg_textsearch` for BM25
- `pg_jieba` for Chinese BM25: the corpus `public.jiebacfg` profile and
  Profile Memory
- `pg_trgm` for the filename-substring index on document metadata

Metadata filtering compares `LOWER(TRIM(...))` on both sides, over the built-in
columns and over any key of the `custom_metadata` JSONB column.

## Required Version

Startup requires PostgreSQL 18 or newer and checks the LightRAG API surface that
DlightRAG uses; the dependency floor is `lightrag-hku>=1.5.8`. DlightRAG carries
no patches against LightRAG's PostgreSQL layer. Workspaces should not mix
embedding models or dimensions after indexing; changing `models.embedding.dim`
requires clearing/rebuilding vectors.

The checked-in Docker Compose stack builds `dlightrag-postgres:pg18` from the
local `postgres/` image definition, pins `pg_textsearch` to v1.5.1 and
`pg_jieba` to v2.0.1, and preloads `pg_textsearch,pg_jieba`. Its
`postgres/init.sql` creates the four extensions when a data volume is first
initialized.

Default vector storage is `HALFVEC(dim)` with HNSW. Plain `HNSW` over
`VECTOR(dim)` is available as an explicit fallback for deployments that prefer
full-precision storage and have rebuilt indexes accordingly.

## Startup Recovery

A Compose PostgreSQL that restarts forever with `FATAL: bogus data in lock file` never
reached a database: an unclean shutdown of the Docker Desktop VM, or of the host, left
the lock files it had just created on disk with their length but not their bytes, so
every byte of them reads as NUL. The data directory is otherwise intact — PostgreSQL
reports the previous shutdown time cleanly and needs no WAL recovery — but it refuses to
start on a lock file it cannot parse:

- the volume's `postmaster.pid` (`/var/lib/postgresql/18/docker/`), reported as
  `FATAL: bogus data in lock file "postmaster.pid": ""`, and
- `/var/run/postgresql/.s.PGSQL.5432.lock` in the container's own layer, reported the
  same way once the first one is gone.

`docker logs` can be empty at the same time, because the same shutdown corrupts the
container's log file: `docker start -a dlightrag-postgres-1` prints the FATAL.
`make postgres-recover` repairs both. It takes the container out of its restart loop,
backs up the volume's lock file under `/tmp` and removes it when it carries no readable
PID, recreates the container so the layer-held socket lock file goes with the old layer,
and waits for the health check. It reads the data volume only through the image's own
tools and rebuilds nothing, so it is safe on a healthy stack, where it reports and exits.
A different FATAL, or one PostgreSQL repeats after the repair, is not this failure: it
points at the data directory, which only the destructive [full development
reset](operations.md#full-development-reset) rebuilds.

Stopping the stack with `docker compose stop` before quitting Docker Desktop or
restarting the host makes the unclean shutdown behind this failure rare.

## External and Managed Endpoints

Keep the PostgreSQL password in `.env` locally or an orchestrator Secret. The
bundled Compose manifest owns `DLIGHTRAG_STORAGE__POSTGRES__HOST=postgres`
because that hostname is created by Compose service discovery; it inherits the
remaining non-secret local defaults. For an external or managed endpoint, choose
one owner for the non-secret connection fields: place stable values under
`storage.postgres` in the deployment's `config.yaml`, or inject topology-created
values from its deployment manifest. Do not duplicate them in both. See
[Configuration](configuration.md#configuration-ownership) and `.env.example`.

Each requirement is gated on its own, so missing one does not force the others
down:

| Requirement | If unavailable |
| --- | --- |
| PostgreSQL 18 | Hard stop, no fallback |
| `pg_trgm` | Hard stop: every corpus needs it |
| pgvector ≥ 0.7 | `storage.lightrag.vector_index_type: HNSW` |
| `pg_textsearch` | `corpus.retrieval.bm25_enabled: false` (vector-only) |
| `pg_jieba` | `corpus.retrieval.bm25_profiles` without a jieba profile |

`pg_textsearch` installs only on a server that preloads it, which managed
providers rarely allow, so preloading usually decides whether BM25 is available.
Neither `pg_textsearch` nor `pg_jieba` is a trusted extension, so only a
superuser can create them. Profile Memory never needs either to start: it
creates `pg_textsearch` and then `pg_jieba` where the server allows, skips
`pg_jieba` without `pg_textsearch`, and logs a warning for what it lacks.
Without `pg_jieba` its Chinese BM25 index runs under `simple`; without
`pg_textsearch` a fact is recalled only when restated exactly or, with an
embedding model that has a calibrated relevance floor, by similarity. An
extension installed later takes effect as each process restarts, the writer
first; running processes keep recalling with the index they found.

`CREATE EXTENSION IF NOT EXISTS` never updates an extension that is already installed,
so after an image moves `pg_textsearch` forward the writer also runs
`ALTER EXTENSION pg_textsearch UPDATE` at startup when the installed release is lower than
the one the server carries, and Profile Memory does the same in its own database. A refusal
(a role that does not own the extension) is a logged warning, not a failed start; see
[Updating An Installed Extension](operations.md#updating-an-installed-extension).

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
  shell or the `.env` beside `docker-compose.yml`. These Compose-only inputs use
  `COMPOSE_*` names because the application rejects unknown `DLIGHTRAG_*` names
  ([Configuration](configuration.md)).
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

TLS settings live with the endpoint under `storage.postgres` (`ssl_mode`,
`ssl_root_cert`, `ssl_cert`, `ssl_key`, `ssl_crl`), typically through
`DLIGHTRAG_STORAGE__POSTGRES__SSL_*` in `.env`. Every connection DlightRAG opens
itself takes its endpoint, credentials, and TLS context from one adapter-core
helper, and DlightRAG bridges the same values to LightRAG's `POSTGRES_SSL_*`
environment whenever it builds a PostgreSQL corpus backend, so managed
PostgreSQL deployments need no second SSL configuration. Configuration only
holds the validated settings: constructing it builds no TLS context and does not
mutate LightRAG's process environment.

Connection budgets are split deliberately:

- `storage.postgres.lightrag_pool_max_size` controls LightRAG's PostgreSQL
  backend pool and is bridged to `POSTGRES_MAX_CONNECTIONS`.
- `storage.postgres.pool_min_size` / `storage.postgres.pool_max_size` control
  DlightRAG's domain pool, `pg_pool`
  ([PG Pool Architecture](#pg-pool-architecture)).
- Workspace writes hold dedicated write-gate connections, at most
  `pool_max_size` per process, outside both pools.
- Docker Compose defaults `max_connections` to `80` for the local profile.
  Production deployments should size the server limit from the number of
  DlightRAG processes and their connection budgets.

For one DlightRAG process, reserve
`lightrag_pool_max_size + 2 × pool_max_size` connections (both pools and the
write gate), plus the one the notification hub holds. Multiply that by the
process count before comparing it with PostgreSQL `max_connections`, leaving
room for migrations, admin sessions, health checks, and managed-service
maintenance. At startup, DlightRAG logs the pools-and-gate part of this budget
against the connected server's real `max_connections`; when `WEB_CONCURRENCY`,
`UVICORN_WORKERS`, or `GUNICORN_WORKERS` is set, it multiplies by that count
and warns when the estimate leaves less than a small admin headroom.

Concurrency knobs affect different bottlenecks:

| Setting | Controls | First bottleneck |
|---|---|---|
| `storage.postgres.lightrag_pool_max_size` | LightRAG PostgreSQL connections | PostgreSQL `max_connections` |
| `storage.postgres.pool_max_size` | DlightRAG domain-pool and write-gate connections | PostgreSQL `max_connections` |
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
pg_textsearch seeds its internal scan limit for this query shape from planner
selectivity, avoiding repeated score-and-filter passes for selective filters.
Compose sets the two server settings explicitly:

```text
pg_textsearch.filtered_seed=on
pg_textsearch.filtered_seed_margin=3.0
```

The SQL filter and `LIMIT` still determine the exact result, so these settings
change work performed, not result correctness. Override them with
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
  jieba index (`jiebacfg` or `jiebaqry`) replaces whitespace runs with a
  full-width comma, which
  jieba splits on and drops as a stopword: segmentation is unchanged and no
  whitespace term remains. Indexed whitespace still counts toward document
  length, a small uniform bias in BM25 length normalization.
- **Listing jieba token types crashes the server.** `jieba_lextype` writes one
  entry past its allocation, and PostgreSQL restarts every backend to recover.
  DlightRAG never calls it: user text reaches pg_jieba only to be tokenized,
  when chunks and memory bodies are indexed or scored and when `to_bm25query`
  reads a query, and tokenizing never lists token types. Operators must not
  run `ts_debug` or `ts_token_type` on a jieba parser or configuration,
  `\dF+ jiebacfg` or `\dFp+ jieba` in psql,
  `ALTER TEXT SEARCH CONFIGURATION ... MAPPING FOR` on a jieba configuration, or
  `pg_dump` of a custom configuration built on the jieba parser. Tokenizing with
  `to_tsvector('public.jiebacfg', ...)` is safe. `pg_upgrade` dumps extension
  members in binary-upgrade mode and hits the same function, so patch pg_jieba,
  or drop and later recreate it, before a major-version upgrade. Between the drop
  and the recreate no jieba BM25 index accepts writes, so do it with every
  writer stopped.

## DlightRAG Schema Migrations

DlightRAG-owned PostgreSQL tables use `dlightrag_schema_migrations` as a small
ledger of the schema baseline each scope has applied. This applies to DlightRAG tables such as
`dlightrag_doc_metadata` and `dlightrag_workspace_meta`; LightRAG-owned tables
remain managed by LightRAG. A writer changes some of them only after LightRAG
has created them; readers issue no DDL:

- `LIGHTRAG_DOC_STATUS` gets the partial indexes
  `idx_dlightrag_file_panel_processed_updated_id` and
  `idx_dlightrag_file_panel_failed_updated_id`, which cover the bounded Files
  order `(workspace, updated_at DESC NULLS FIRST, id ASC)` for processed and
  failed documents;
- with BM25 enabled, `LIGHTRAG_DOC_CHUNKS` gets the `dlightrag_bm25_language`
  column and its indexes; and
- `LIGHTRAG_DOC_CHUNKS`, and with `PGVectorStorage` LightRAG's chunk-vector
  table, become workspace-partitioned parents.

`dlightrag_doc_metadata` holds the publication marker
`_dlightrag_finalization_complete`, non-null and false by default, with a
partial index over visible documents. A trigger keeps
`dlightrag_metadata_field_stats`, the planner's field statistics, counted over
visible rows only through every insert, update, marker change, and delete.

Each scope (`runs`, `web_conversations`, `connections`, `doc_metadata`,
`promotion_jobs`, `workspace_registry`, `model_catalogue`) declares one `baseline`
version: the whole current schema as idempotent DDL. A writer ensures it on startup
and records it in the ledger; readers validate the same version without issuing
DDL. There is no upgrade path. A ledger version the running revision does not
declare, which is every database an earlier revision created, is incompatible
before any DDL runs: both roles refuse to start, and the remedy is a full
development reset. A ledger version alone does not prove a schema, since it
survives an object dropped afterwards, so each scope also declares the objects its
revision needs (columns, keys, foreign keys, checks, indexes, and guard triggers).
Readers read each one back from the catalog, and writers do the same after
applying the baseline, refusing a database that lacks one. Run
`uv run scripts/reset_development.py --mode docker` (or `--mode native`) to
perform that reset; it also recreates the required PostgreSQL extensions and
verifies the empty database. See
[operations.md](operations.md#full-development-reset).

## Durable Run State

Every top-level Retrieval, Answer, and Corpus Mutation is one durable Run
([RunRuntime](run-runtime.md)). DlightRAG-owned tables under the `runs`
migration scope separate common lifecycle from Answer-owned routing, Session,
control, child, and blob-reference state:

| Table | Key | Holds |
| --- | --- | --- |
| `dlightrag_runs` | `(owner_id, run_id)` plus globally unique `run_id` | kind, lane, submitter/access scope, submission key, status, retry/checkpoint, retention, cancellation, fenced lease, Prepared Input, Corpus Mutation handoff/repair state, result or terminal error |
| `dlightrag_run_events` | `(owner_id, run_id, event_sequence)` | gap-free executor-owned events, including Answer `progress` / `token` / `reset` / tool / terminal events |
| `dlightrag_corpus_mutation_windows` | `(run_id, window_number)` | each ingest window's document and chunk counts, recorded once so workspace totals and promotion count it once |
| `dlightrag_blobs` | `(owner_id, digest)` | immutable content-addressed blob metadata within one owner |
| `dlightrag_blob_chunks` | `(owner_id, digest, chunk_index)` | the blob's bytes, in ordered chunks |
| `dlightrag_answer_run_artifacts` | `(owner_id, run_id, resource_id)` | ordered request attachments and Published Artifact bytes |
| `dlightrag_answer_artifact_attachments` | `(owner_id, run_id, relative_path)` | settled Root Artifact Attachment authority: label, raw digest/size, presentation, the files the root was built from, Effect provenance, and settlement order |
| `dlightrag_answer_run_routing` | `(owner_id, run_id)` | requested/valid/resolved mode and canonical Agent Session/Lane mapping |
| `dlightrag_answer_run_stages` | `(owner_id, run_id, stage_intent_id)` | settled Fast `planner`, `retrieval`, and `final_generation` stage state that recovery reuses |
| `dlightrag_agent_sessions` | `(owner_id, session_id)` | Session commit sequence, Entry sequence, current run owner and fencing epoch |
| `dlightrag_agent_session_entries` | `(owner_id, session_id, sequence)` | immutable parent-linked User/Assistant/ToolResult/Control/Compaction Entries, each naming the Run that wrote it, which the writer states. An Assistant or Compaction Entry's payload also holds a `call`, written in the same transaction: the model, parameters, prompt digests and timings (how long it waited for a model slot and how long it took) of the model call that made it, and no prompt text, option, header, credential or URL ([ADR 0043](adr/0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md)). A User Entry's payload names its `origin` when the owner did not write it: `child_notification` marks a Child's settled result or guidance question that the Run handed its agent as the next input |
| `dlightrag_agent_session_registers` | `(owner_id, session_id, register_kind, register_key)` | exact-CAS Lane heads/state, total OperationState, Plan metadata, request/tool snapshots, bounded inputs and Fast reservation |
| `dlightrag_answer_goals` | `(owner_id, session_id)` | the Agent Session's one current Goal: objective, phase, a positive-integer round cap with no product ceiling, and rounds started, and the Run that last changed it (empty when a human goal command wrote the row). Every write moves the revision it read, so a round driver and a human turn cannot overwrite each other; the row cascades with its Session ([ADR 0060](adr/0060-goal-mode-is-a-harness-feature-skills-declare-checkpoints.md)) |
| `dlightrag_schedules` | `(owner_id, schedule_id)` | one owner's Schedule ([ADR 0103](adr/0103-a-schedule-starts-a-run-on-the-clock-and-the-database-is-its-queue.md)): its prompt, mode, workspaces, target and conversation, cadence (a five-field recurrence or one instant, with its IANA zone), daily token ceiling, the round cap of a goal Schedule (null for any other), `active` / `paused` / `completed` state with the pause reason, and `next_fire_at`; and the identity (user id, auth mode, claims) it acts as, which is its owner when a Run starts or the Fire is refused, with when its owner last confirmed it and when the Schedule last resumed. A partial index over the active rows is the look's claim |
| `dlightrag_schedule_fires` | `(owner_id, fire_id)`, unique `(schedule_id, slot)` | what one slot did: its state and reason, whether the owner asked for it, the last error that stopped an attempt to start it, the Run and conversation it started, and for a gap of missed slots their count and last one. The unique slot is what lets any number of processes look and take a slot once. Fires cascade with their Schedule and are trimmed to `schedules.fires_kept`; the Run is linked by id with no foreign key, since a Run is pruned by its own retention |
| `dlightrag_answer_session_notes` | `(owner_id, session_id, relative_path)` | the Agent Session's notes, attributed to the Run that wrote each |
| `dlightrag_answer_feedback` | `(owner_id, run_id, agent, sequence)` | one owner's judgement of one answered message: `up` or `down`, and for a `down` the reasons it gives (from a closed set) or the words of its note — at least one of the two, which the table's own check holds it to — the note being up to 2000 characters, with when it was first given and last changed. `agent` is `main` or a Child Session id and `sequence` the message's own place in that Session, which never moves, so judging again replaces the row. A row is written only for an assistant message the Run itself wrote, checked in the statement that writes it, and cascades with its Run |
| `dlightrag_runtime_policies` | `(key)` | one JSON document per [Runtime Policy](interfaces.md#runtime-policies), replaced whole by the administrator's last write, with who wrote it and when. The key is one the application registers and the table only requires an object under a key of 1-64 characters; the application owns what each document holds. A policy with no row holds its code default. A request reads its row by key every time, so no process caches one and none is told of a change |
| `dlightrag_answer_evidence` / resource tables | run/session/intent/result identity | atomic durable Evidence, fetched resources, Agentspace Inventory, spills, and blobs; a Tool settlement stores the Session's Evidence ledger only when it changed, and recovery reads the latest snapshot; an Agentspace rescan writes its inventory in two statements that delete removed paths and rewrite only changed rows. The maintenance loop compacts a finished Run's Evidence in place: the newest row of each Session stays whole, and every other row keeps its chunk ids, source metadata and digests while dropping the passage text of chunks the answer did not cite, and `content_slimmed` names the rows that lost it, so the payload a reader loads stays the document |
| `dlightrag_answer_child_sessions` | parent run + child Session id | parent/call/intent lineage, ContextSnapshot, depth, independent lease/epoch, pinned plan/budget/tools/Host state, status and usage; the maintenance loop clears a finished Run's Host state, which no reader surfaces and no control can reach once the Run is over |
| `dlightrag_answer_child_operations` | parent run + child Session + operation sequence | same-Session continuation Operations, idempotency, origin, status, cancellation origin, usage/outcome; the maintenance loop clears a finished Run's operation outcomes the same way, since a retry cannot reach them |
| `dlightrag_answer_child_guidance` | parent run + request id | correlated `ask_parent` questions, expiry, reply origin, and status `pending` / `replied` / `expired` / `cancelled` |
| `dlightrag_agent_browser_leases` | `endpoint` | one row per configured Agent Browser endpoint, registered at first use, with the Run (owner, run, lease owner, fencing epoch) that holds it or none, and no expiry of its own; whether a held row is live follows the Run's lease ([Agent Browser](architecture.md#agent-browser)) |
| `dlightrag_agent_accounts` | `(owner_id, site)` | one row per owner and site: the account id its envelope is bound to, its email and username (one of them at least), and the password's envelope with the id of the key ring key that sealed it, so a retired key's count shows what is left to re-seal, with when it was registered (kept by a reset), when its credentials last changed, and when a login last filled its password; no password in clear and no Run's Child account ([Agent Accounts](security.md#agent-accounts)) |
| `dlightrag_agent_account_settings` | `(owner_id)` | the owner's switch for the Agent's new sign-ups, which an owner with no row has on ([Agent Accounts](retrieval-answer.md#agent-accounts-and-the-agent-mailbox)) |
| `dlightrag_privacy_settings` | `(owner_id)` | the owner's Conversation Contribution ([ADR 0041](adr/0041-a-conversation-improves-the-product-only-when-its-owner-asked.md)): whether their conversations may be used to evaluate and train models, and the moment that permission began. An owner with no row has never chosen and contributes nothing. A check refuses a row whose flag and moment disagree, so turning it off clears the moment and turning it on again begins a new one. One statement decides the moment from the stored row, reading the database's clock as it writes (`clock_timestamp()`, not the start of its transaction, so a turn-on that waited behind a turn-off is dated after it). The trace export compares it with the start of a Run's Agent Session: the earlier of `dlightrag_agent_sessions.created_at` and the earliest `created_at` among the Runs whose `dlightrag_answer_run_routing.agent_session_id` is that Session |
| `dlightrag_agent_controls` | run + control sequence | ordered steer inbox for the parent Run or a targeted child Session/Operation; origin `user` or `parent`; append-before-ack; a worker locks and reads at most 100 pending controls at a time, in sequence order |

Every pass of the maintenance loop is a bounded SQL batch that a finished
Run waits out for the configured grace, so live Runs are never touched, and each
pass drops data no reader needs rather than moving bytes around: what a finished
Run's Evidence log no longer grounds, and the working state its children ran on.
Reusing and returning the space a table churns is PostgreSQL's own job, through
autovacuum, which neither pass duplicates. The image also runs with
`default_toast_compression=lz4`, which stores the same text slightly smaller than
pglz and compresses several times faster. When a table vacuums, and how much dead
space it holds between runs, is PostgreSQL's to decide: no table here carries its
own autovacuum settings.

Independent critique reuses those Child Session, Operation, control, and
guidance rows.

`run_id` is a UUIDv7. The unique index `idx_dlightrag_runs_submission` makes a
submission key unique per Run kind and submitter, and a partial unique index
allows exactly one terminal event per run. The run-artifact join carries
`ON DELETE CASCADE` to the run and `ON DELETE RESTRICT` to the blob, so linking
a digest takes the key-share lock that serializes against cleanup. Deleting a
run removes its events and references, never shared bytes; a blob is deleted
only once no reference for that owner survives. The Query and Corpus Mutation
claim paths share the bounded `idx_dlightrag_runs_claim` index; Workspace
mutation eligibility also uses `idx_dlightrag_runs_mutation_fifo`, and event
reconnect uses the event primary key. An owner's [active-run
limit](run-runtime.md#admission-limits) counts that owner's queued and running Answers
through `idx_dlightrag_runs_owner_active`, a partial index on `(owner_id)` over
`run_kind = 'answer' AND status IN ('queued', 'running')`: the count runs under the
lane's acceptance lock, and without the index it walks the lane's active rows, or the
owner's whole history, which in a deployment with one owner is the whole table.
`idx_dlightrag_answer_run_routing_session` on `(owner_id, agent_session_id)` finds a Session's
Answer Runs by the Session, which the [Session Memory sweep](adr/0045-profile-memory-is-maintained-per-session-and-owned-across-them.md)
asks of each Session whose pass is behind (its newest succeeded Run, and whether any is live).
The claim-plan `EXPLAIN` output is part of the [load evidence](run-runtime.md#load-evidence).

Web conversation turns link to a run with `(principal_id, answer_run_id)` and
`ON DELETE CASCADE`. The turn carries conversation order and the run link only:
request content, answer text, sources, and uploaded bytes all live in the run, so
nothing about one answer is stored twice.

Retention prunes these rows in bounded `SKIP LOCKED` batches; row locks,
cascades, Session reference checks, and the run-artifact/blob foreign key
serialize pruning against new references, and conversation deletion follows the
same Run-first lock order. What retention selects, and when, is in
[RunRuntime](run-runtime.md#retention).

## Graph Storage

LightRAG's knowledge graph uses `PGTableGraphStorage`: two ordinary PostgreSQL
tables, no extension.

| Table | Key |
| --- | --- |
| `lightrag_graph_nodes` | `(workspace, namespace, id)` |
| `lightrag_graph_edges` | `(workspace, namespace, src_id, tgt_id)` |

Node and edge attributes live in a `properties JSONB` column, and traversal is
LightRAG's frontier-capped iterative breadth-first search over an index on
`(workspace, namespace, tgt_id)`. Edges are undirected: LightRAG canonicalizes
each pair in Python before writing, never with SQL `LEAST`/`GREATEST`, so
endpoint ordering cannot drift with the database collation.

Ordinary tables require no compiled graph extension,
`shared_preload_libraries` entry, or per-workspace schema DDL.

A writer creates the tables in `initialize()` under an advisory lock, so any
writer may be first; readers verify them without DDL. Workspace isolation is a
column, not a schema, so resetting a workspace is a `DELETE`, and orphaned
workspaces leave no schemas behind.

## PG Pool Architecture

Every DlightRAG process connects to the one endpoint `storage.postgres` names.
Both service roles target the **same primary endpoint**: a writer applies
DlightRAG schema migrations and mutates the corpus, and a reader still writes
DlightRAG operational state (see
[Service roles and shared artifacts](#service-roles-and-shared-artifacts)).
LightRAG's staged pipeline already supports ingest and query in the same writer
process; local query-while-ingest behavior should be tuned through
parser/analyze/insert/model concurrency before changing database topology.

DlightRAG uses two asyncpg pools:

| Pool | Owner | Purpose |
|---|---|---|
| LightRAG ClientManager pool | LightRAG | KV, vector, graph, doc status |
| `pg_pool` singleton | DlightRAG | Metadata index, BM25, workspace metadata, Runs, Web conversations, Connections, the model catalogue, and Profile Memory |

The dedicated DlightRAG pool avoids contention between LightRAG internals and
DlightRAG's own reads and writes. Both pools use the same endpoint, SSL
settings, and session-level PostgreSQL tuning; a reader's LightRAG pool also
runs with `default_transaction_read_only=on`.

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
write operational state. Startup order, probes, and homogeneous worker
requirements are in
[Operations](operations.md#runruntime-and-durable-query-and-corpus-mutation-runs).
