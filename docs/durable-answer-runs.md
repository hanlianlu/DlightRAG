# Durable Answer Runs

This document defines durable Answer behavior across two owners. The neutral
`dlightrag.runtime` package owns lifecycle records, the store port, fenced
sessions, subscriptions, durable progress, the cancellation listener, and
`RunCoordinator`. The Answer executor owns retrieval/synthesis, converts
product failures to `RunExecutionError`, and drives Research through the
product-neutral Agent Session Runtime. `AnswerService` gives REST, MCP, Web, and
in-process Python callers the same coordinator-backed lifecycle.

`dlightrag.adapters.postgres.answer_runs.PGAnswerRunStore` implements the store
port. Runtime never imports PostgreSQL, Answer implementation, RAG
implementation, or a transport package; the root adapter imports the records it
persists from Runtime. No `dlightrag.storage` compatibility package exists.

## Goals

- Every answer is one durable run with one identifier and one lifecycle.
- Client disconnect does not cancel the run.
- A process restart opens the durable Agent Session and restores complete typed
  OperationState. Committed effects never execute again, and pending effects
  replay only under their pinned AgentRunPlan policy.
- REST, MCP, Web, and Python use the same run state, artifacts, events, and
  final result.
- Durability does not change retrieval, citation, workspace, or round-robin
  semantics.
- Durability adds no workflow framework. Research hosts the product-neutral
  `AgentSessionRuntime` over an immutable parent-linked Entry Tree and typed
  registers. Fast shares that Tree through a Host reservation but has no Agent
  Operation.

## Non-Goals

- Exactly-once execution for an in-progress read-only tool batch.
- Exactly-once token generation across a process crash.
- LangGraph, LangChain, or Pi runtime adoption.
- Detached/background agents, workflows, missions, councils, worktrees, tool
  approvals, or a permission platform.
- A bundled sandbox backend, MCP registry/OAuth platform, or object-storage shim.

## Query-Time Core

The durable runtime rests on these query-time rules.

### Model image capabilities

Answer, VLM, and chat-rerank image capability are separate facts because they
may resolve to different model roles. Application startup probes each distinct model
endpoint once at startup and caches the result by resolved model configuration.
When two roles resolve to the same configuration they share one probe result.

- Answer capability controls current-image admission and final answer images.
- VLM capability controls query-image description and `inspect`.
- Rerank capability controls image-bearing chat rerank requests.

`supported` and `unsupported` are terminal for the process. Only `unknown`
uses the existing cooldown-bounded, single-flight lazy re-probe.

### Agent termination

Tool errors are not convergence. An invalid, unavailable, or failed tool result
is replayed to the model for correction. Research ends when the model writes the
answer and makes no tool call.

Every model-visible field has a description. Tool names are checked for
uniqueness after per-run composition. Each tool execution emits one observation
with its name, duration, cache status, and safe outcome metadata.

### Async boundaries

The event loop owns orchestration and async I/O only. These operations run in a
worker thread:

- DNS resolution used by public-URL SSRF validation;
- resource charset detection, text window construction, and focused BM25;
- research context/evidence packing and token estimation;
- sidecar provenance index loading;
- existing image, PDF, conversion, embedding-payload, and Markdown work.

There is no executor or concurrency setting for them; they share the default
executor.

### Answer image policy

One frozen Answer image policy owns the answer transport geometry, byte limits,
and quality limits. It creates fresh mutable budgets for a run.
`AnswerSynthesizer`, `QueryImageDescriber`, and `ResourceInspector` consume the
policy instead of repeating scalar constructor parameters and default literals.

Embedding and rerank image limits remain separate policies: they target
different providers and request contracts.

## Public API

`POST /answer` always creates a durable run and returns HTTP 202:

```json
{
  "run_id": "019...",
  "status": "queued",
  "status_url": "/answer/019...",
  "events_url": "/answer/019.../events"
}
```

An idempotent replay also returns 202 with the run's current status rather than
forcing it back to `queued`.

There is no `stream` request field and no temporary Answer mode.

### Execution

An answer execution slot is one of the
`answer.runtime.answer_worker_concurrency` runs that a process may execute
concurrently. It is not a token, context, image, storage, PostgreSQL, or model
request budget. A process reserves a local slot before it claims a row, so a
worker never owns a lease while waiting for local capacity.

AI provider requests use the independent process-wide `models.max_concurrency` scheduler.
LightRAG corpus processing uses `corpus.ingestion.pipeline.max_concurrency`. Changing either value
does not change durable Answer worker admission; a run may issue multiple model
requests, and those requests compete fairly with requests from other runs.

The creating process wakes its worker after committing the run. A worker with a
free slot claims the oldest eligible row with `FOR UPDATE SKIP LOCKED`, ordering
by creation time and `run_id`. It may claim a `queued` row or a `running` row
whose lease expired. Workers also sweep in bounded batches at startup, after a
local run finishes, and on an internal one-second cadence so another host's
queue and expired leases are recovered without relying on process-local
signals. Claiming increments a fencing epoch and starts the existing
lease-heartbeat pattern.

The sweeper does not reserve an execution slot to inspect or terminally update
rows. It finalizes any cancel-pending nonterminal row with no live lease and
fails a run past its recovery bound with error kind `run_abandoned`, even when
all execution slots are busy. Only a claim that will start or resume model/tool
execution reserves a slot.

Fencing and crash recovery use separate counters. The fencing epoch increments
monotonically on every claim and never resets. Durable progress replaced the
old turn and recovery counters: `durable_progress_version` advances
only on fenced model turn appends, compaction appends, effect settlements, and
Fast stage settlements; `reclaims_without_progress` counts consecutive
expired-lease reclaims without such progress, and four of them abandon the run
with `run_abandoned` and its `error` event. A long run that commits Session
progress between reclaims survives more total process restarts. The bound is
not configurable.

A worker renews only an unexpired lease by comparing its owner and fencing
epoch. If a heartbeat or ordinary lease-guarded write affects no row, the
worker rereads the row: an expired lease or owner/epoch divergence means
ownership is lost, so it cancels and joins its in-process work, performs no run
transition, releases the local execution slot, and leaves recovery to the
current or next owner. It never tries to revive an expired lease. Blocking
CPU work remains cancellation-cooperative at its existing settlement boundary;
the stale worker cannot persist that result.

A durable Answer run has no wall-clock execution deadline.
`corpus.retrieval.timeout` bounds one caller-awaited inline Retrieval operation and does not wrap Answer
execution. Each external LLM, embedding, rerank, URL-fetch, resource, and
parser-sidecar call keeps its own provider, request, or stream-idle timeout, so
one stalled awaited call cannot hold a slot forever. A run releases its slot on
terminal transition, lease loss, or process shutdown; there is no run-timeout
setting.

On graceful shutdown the coordinator stops claiming rows first. Active workers
may finish an in-flight terminal transition or Session settlement during the
application's shutdown grace; remaining work is then cancelled and joined.
Each owned cancel-pending run is finalized as `cancelled` with its terminal
`done` event. Every other owned nonterminal run receives a fenced `running` to
`queued` transition that clears its lease while preserving its durable
progress and cancellation field. It is immediately reclaimable as ordinary
queued work and does not increment the reclaim counter. Only reclaim after
lease expiry counts as crash recovery. A run interrupted during generation
emits `reset` when next claimed, whether the interruption was a crash or
graceful shutdown.

If every process is busy, an accepted row remains `queued` until a worker has a
slot or the caller cancels it. Queue age never turns an accepted run into a
capacity failure, and queue depth is not capped. POST may fail before acceptance
for validation, authorization, or persistence errors, but execution-slot
exhaustion is not one of them. No queue timeout, queue-depth, retry, or
sweep-interval setting exists.

`GET /answer/{run_id}` returns lifecycle status, cancellation, phase, durable
progress, continuation lineage, terminal error, and the final payload. The
payload includes canonical answer material plus transport-neutral usage and
Evidence counts. Additional owner-scoped operations are:

- `POST /answer/{run_id}/steer` for an ordered live Research instruction;
- `POST /answer/{run_id}/follow-up` and `/fork` for new lineage-linked runs;
- `POST /answer/{run_id}/resume` to reattach to current state;
- `GET /answer/{run_id}/transcript` and `/children` for bounded transcript and
  foreground child roster; and
- `DELETE /answer/{run_id}` for explicit cancellation.

`GET /answer/{run_id}/events` is reconnectable SSE. Each durable sequence is the
SSE `id`. The `Last-Event-ID` header or integer `after` query parameter resumes
after that sequence; supplying both with different values returns 400. Without
either cursor, replay starts at sequence 1. While a queued or quiet running run
has no new event, the connection sends an SSE comment keepalive every ten
seconds; comments do not consume sequence numbers. Unknown, pruned, and
other-owner runs return 404. An authorized terminal run whose event log has
expired returns 410, after which clients read its canonical result from the
status endpoint.

The same owner-scoped 404 rule applies to status and cancellation; transports
do not reveal whether another owner has a matching run id.

`DELETE /answer/{run_id}` requests cancellation. A queued run becomes
`cancelled` in that transaction. A running run records `cancel_requested_at`;
its worker observes the request after each awaited planning, retrieval, and tool
batch phase, between control turns, and at each coalesced token batch, then
performs the terminal transition. If its lease expires first, the sweeper
finalizes cancellation instead of reclaiming it. Cancelling a terminal run is
an idempotent no-op that returns the terminal state.
DELETE returns 200 when it completed or found a terminal transition and 202
when a running worker still must observe the cancellation request.

Every claim and terminal transition locks the run row. A pending cancellation
takes precedence over reclaim and failure: the claimer finalizes `cancelled`
instead of resuming or abandoning the run. Successful finalization is allowed
only while `cancel_requested_at` is null. If finalization wins the row lock
first, later cancellation is the terminal no-op; if cancellation wins first,
the worker commits `cancelled` instead of `succeeded`.

`AnswerService` keeps convenience methods:

- `answer()` creates a run and waits for its result;
- `answer_stream()` creates a run and subscribes to its events;
- `create()`, `get()`, `subscribe()`, and `cancel()` expose the durable contract.

Cancelling a waiting convenience call or closing any event subscriber detaches
that caller only. Explicit run cancellation is the sole client action that sets
`cancel_requested_at`.

`POST /web/api/answer` creates a core run and returns its descriptor. For a first
submission, an omitted `conversation_id` causes the server to create the
conversation, turn, uploaded artifacts, and run in that same transaction; a
failed admission leaves no empty conversation. The browser then subscribes to
its own owner-scoped `/web/api/answer/{run_id}/events`. That
stream follows the same durable event log with the same sequence, resume, 410,
and detach semantics as the REST stream, and differs only in projection: a
browser `done` frame embeds the same typed `AnswerPresentation` used by
conversation history and Primary Report reads. It contains sanitized semantic
answer HTML plus structured sources/images rather than the canonical stored
result REST serves.
Disconnecting the browser only closes that subscriber, and reconnecting resumes
from the durable event sequence.

One async helper, `dlightrag.sdk.AnswerRunClient`, owns REST create-and-wait behavior. It
creates the run, follows durable events, and falls back to status reads after a
reconnect or an expired event log. The synchronous CLI invokes that helper
through `asyncio.run`; the async evaluation script awaits it directly. Neither
retains a synchronous endpoint nor implements an independent polling loop. The
`AnswerService.answer()` and `answer_stream()` use the same coordinator
semantics in process.

MCP `answer` is deliberately descriptor-only and returns immediately. The
separate MCP status tool returns the canonical result after success, and the
cancel tool requests cancellation; MCP does not hold one tool call open for a
tens-of-minutes run.

Every transport derives the owner through one transport-neutral principal
projection in Access. `access.auth_mode="none"` and
`access.auth_mode="simple"` intentionally collapse callers into one deployment
owner; `access.auth_mode="jwt"` is the tenant
boundary. Trusted in-process callers pass an explicit owner id to
`AnswerService` and may use the deployment owner when no tenant boundary exists.

### Reader role

`deployment.service_role="reader"` means corpus-read-only, not process-read-only. A reader
may create and execute answer runs and may write DlightRAG operational state,
including runs, events, artifacts, and Web conversations.

Reader safeguards remain at the corpus boundary:

- `CorpusAdmin` rejects ingestion, workspace creation/reset,
  metadata mutation, failed-document retry, and deletion;
- the LightRAG PostgreSQL pool continues to use read-only sessions and the
  no-DDL attach path;
- LightRAG LLM cache writes remain disabled;
- ingest pipeline recovery remains writer-only.

The DlightRAG domain pool is writable for both roles, so the supported reader
topology uses the same primary PostgreSQL endpoint. A reader process does not
point its pools at a physical standby; routing corpus reads elsewhere would
require a separate corpus endpoint and is outside this contract.

Concretely:

- domain-pool connection setup does not apply session read-only mode to a
  reader, while LightRAG pool setup does;
- writer startup owns schema migrations; reader startup validates that the
  current domain and LightRAG schemas already exist without issuing DDL;
- readiness permits Answer and Web operational writes on a reader and still
  rejects corpus-mutating operations at its writer boundary;
- Web is enabled for readers.

A reader with a missing or incompatible schema fails startup with a diagnostic
and serves no traffic. It does not retry DDL or run partially ready. Deployment
order applies writer migrations before starting readers on the new revision.

## PostgreSQL State

### `dlightrag_answer_runs`

One row owns the run:

- owner identity and `run_id`;
- bounded prepared input JSON (`prepared_input_json`), required for queued and
  running rows and cleared on every terminal transition;
- status, phase, and stop reason;
- cancellation request time, lease owner, lease expiration, and fencing epoch;
- durable progress: `durable_progress_version`,
  `last_reclaim_progress_version`, and `reclaims_without_progress`;
- next durable event sequence;
- event-log trim timestamp;
- final result JSON or terminal error;
- created, updated, started, and finished timestamps.

Queued and expired-running rows are reclaimable. Active workers renew a lease.
Terminal rows and event logs follow the single
`answer.runtime.answer_run_retention_days` floor (default 365) from
`finished_at`, using bounded `SKIP LOCKED` batches. Conversation turns are read
windows over those runs, not a separate inactivity clock. Run deletion also
reclaims candidate Agent Sessions when no remaining routing row names them.
A Web Conversation row is navigation identity, not a second durable-history
reference; it may remain briefly after its final turn and Session are reclaimed.
Sessions shared by another routed Answer Run survive. Child Session trees are
candidates when their owning run is pruned. When a run row still exists but its
events were trimmed, `events_trimmed_at` makes the event endpoint return 410 and
the canonical result remains available from status.

`run_id` is a UUIDv7. Run creation takes one optional idempotency key unique per
owner. REST uses the `Idempotency-Key` header, MCP and Python expose an
`idempotency_key` argument, and Web passes its existing `submission_id`; there
is no second Web replay mechanism. This intentionally changes Web's current
conversation-scoped submission uniqueness to the same owner-wide namespace as
all other transports. Recreating the same normalized request under the same key
returns the existing run, while reusing a key with different input returns 409.
Creation without a key always creates a new run. The key expires with the run
row.

Idempotency hashes canonical JSON over the normalized public request before
model resolution: query, exact authorized workspace set, retrieval and answer
options, caller history, ordered resource descriptors, and uploaded artifact
digests after validation. It excludes transport headers, temporary paths,
authorization-dependent URLs, secrets, and every resolved model fact. A keyed
replay checks that fingerprint and returns the accepted run before repeating
profile resolution, URL fetches, image description, or history projection.

The run's stored request is the immutable resolved execution input. In addition
to the public fields it carries the selected history projection, each role's
endpoint fingerprint and effective `ModelProfile`, context-policy and catalog
revisions, and accepted image descriptions. Workers use those pinned profile
values for request arithmetic and never substitute current catalog facts.
Workers must use the pinned model endpoint facts and compatible session/schema
contracts; provider credentials remain deployment state. Capacity is recalculated
from the immutable pinned profile instead of rejecting a run solely because the
global arithmetic-policy revision changed.

The run row is the sole authority for lifecycle status, phase, durable
progress, stop reason, cancellation, lease, final result, and terminal error.
Research state lives in the Agent Session's immutable parent-linked Entries and
closed typed registers. Before each provider call the Runtime commits the exact
RequestSnapshot and attempt identity. An Assistant settlement commits the
complete response plus a ToolBatchPlan covering every source position; tool
clearance, effect settlement, ToolResult placement, HostDelta, and durable
progress then commit in source order under the live lease/epoch predicate.
Recovery plans from the same total OperationState: explicit `replayable` effects
may reconcile under an unchanged contract, `never` effects close as
`outcome_unknown`, and changed contracts settle `tool_contract_changed`.

### `dlightrag_answer_run_events`

Events use a monotonically increasing sequence per run. Durable event types are:

- `progress`;
- `token`;
- `reset`;
- `tool_start`;
- `tool_progress`;
- `tool_end`;
- `done`;
- `error`.

Tool lifecycle events contain metadata only: name, status, elapsed time, output
byte count, spill state, call identity, and attachment count. Raw stdout and
stderr remain transient tool data and are never written to the event log or
displayed by Web.

Token writes are coalesced into bounded text chunks rather than one row per
provider token. A successful finalization transaction stores the canonical
result, appends `done`, and changes the run to `succeeded` under one
lease/fencing check. Successful `done` carries `status="succeeded"` and embeds
the complete canonical result: answer, contexts, references, sources,
answer-image metadata, trace, and image descriptions. Cancellation and failure
likewise append their terminal event in the same transaction as their terminal
state transition. Cancelled `done` carries `status="cancelled"` with no result;
`error` is used only for `failed` and carries the public error kind and message.
Exactly one terminal event is committed per run. SSE closes after replaying
that event.

The per-run event sequence is gap-free. The append transaction locks the run
row, consumes its next sequence, and increments that value, so an `after`
cursor replays every committed event exactly once. Worker appends are permitted
only when the lease owner, unexpired lease, and fencing epoch all match. The
queued-cancellation transaction and sweeper use the same row lock and terminal
transition predicate, preventing a cancellation race from producing two
terminal events. Deleting a run cascades its events.

These eight are the only durable event types. Intermediate contexts are not
published because research may still change them; clients receive the
authoritative contexts and all other metadata in successful `done`. Status GET
and create-and-wait return that same canonical result without a required
follow-up read on the normal SSE path.

Stored events and results contain transport-neutral source identities and never
store authorization-dependent download URLs. Each authenticated status or SSE
read projects fresh URLs while preserving the durable event type, sequence, and
payload ordering. This projection does not append or modify an event.

Web derives previews from token text and requests semantic highlights after
`done`; neither preview nor highlight output is stored. `progress` uses the core
phases `routing`, `planning`, `searching`, `researching`, and `generating`; Web-local
presentation phases are not durable.

Every sub-turn phase transition uses one small transaction that, under the
current unexpired lease and fencing epoch, updates the run row's `phase` and
appends the matching `progress` event with the next durable sequence. It does
not modify durable progress. Recovery may re-enter an earlier phase;
`progress` is a durable last-writer-wins state notification, not a monotonic
workflow history, so this does not emit `reset`. `reset` is reserved for
clearing a partial token draft before regenerated output.

### `dlightrag_blobs` and `dlightrag_blob_chunks`

Immutable raw bytes are content-addressed within one owner namespace and
chunked into exactly 1,048,576 bytes per non-final chunk. Metadata existence
means complete: one transaction writes every chunk and inserts the blob row
last, so a partial blob is never visible. Lifetime is derived only from run and
resource references; there is no independent artifact-retention state. Caller
attachments may be reused across runs without cross-owner deduplication.

Accepted attachments and fetched Web bytes share this blob contract. Accepted
inputs register atomically with run acceptance; fetched Web bytes settle
through their effect's `FetchedResourceSettlementUpdate` after the existing
HTTPS, redirect, DNS, SSRF, and byte validation passes, so the resource id is
permanently bound to those validated bytes and recovery never silently
re-fetches a changed page. Recovery reads the stored bytes without live DNS
revalidation because no network request occurs; ordinary live URL reads retain
their per-read validation.

### `dlightrag_answer_run_artifacts`

This join table records ordered run inputs and discovered resources without
duplicating bytes. Each reference owns its safe filename, MIME type, ordinal,
resource kind, and any deterministic transform locator needed to regenerate a
page or image window. It distinguishes current attachments, historical
attachments, and run-scoped fetched resources.

Its `(owner, digest)` columns have a composite foreign key to
`dlightrag_blobs` with `ON DELETE RESTRICT`. Inserting a reference therefore
takes the PostgreSQL key-share lock that serializes against blob deletion;
reusing a blob is not protected merely by checking both tables in one READ
COMMITTED transaction.

Deterministic attachment conversion is recomputed from stored bytes on resume;
no conversion-result table is introduced. VLM inspection prose remains run
evidence, not a cross-run cache.

Deleting a run removes that run's references, not shared bytes. A blob is
deleted only when no run/resource reference keeps its digest for that owner.
Reference checks and deletion occur in one transaction, and the foreign key
rejects deletion if a concurrent run has linked the digest. A deferred cleanup
pass may retry a blob that remained because of that race.

## Agent Session Recovery

Each Answer Run routing row authorizes one Agent Session/Lane. The immutable
Entry Tree is canonical conversation ancestry across Fast and Research runs;
LaneHead and LaneState registers select branches without copying shared Entries.
Research restores OperationMeta plus the closed total OperationState and invokes
the same pure NextAction interpreter used live. Historical Compaction Entries
remain audit facts; one branch-local ContextProjection controls model context and
never becomes Evidence. There is no legacy schema reader.

Steer controls enter an ordered Answer inbox. Runtime consumes a steer only at a
stable checkpoint, appends a ControlMessage Entry, then acknowledges its durable
sequence. PendingInput is an unaccepted bounded FIFO; dequeue creates a fresh
Operation acceptance and immutable Plan. A steer observed after terminal commit
also becomes a fresh linked Operation.

There is no per-turn checkpoint JSON, restored exact-call cache, inferred phase
recovery, or `checkpoint_*` error kind.

Image blocks stay as Resource Handle and corpus sidecar identities, never data
URIs. Claim-time rehydration restores those blocks before the first model call.
A missing corpus visual drops the image and keeps the text and citation identity;
a missing attachment blob fails the run.

Durable progress advances only on fenced live Session/HostDelta settlements and
Fast stage commits. Register-only recovery bookkeeping never advances it.
Changed tool contracts settle `tool_contract_changed` without dispatch.

Fast shares the Agent Session Entry Tree but never enters the interpreter. Host
acceptance atomically appends UserMessage plus HostTurnReservation. Before the
Assistant settlement, the Answer Host stages the complete canonical result on
the run's deterministic final-generation stage. Success appends AssistantMessage
with the same acceptance identity and clears the reservation. A crash after that
Assistant commit reloads the staged result and terminalizes the run without
retrieval, generation, or publication work. Failure before result staging clears
only the reservation and preserves the unanswered user entry; after staging,
failure or cancellation preserves the reservation so replay can commit the exact
Assistant without lane interleaving. Interrupted pre-settlement generation appends
`reset`; DlightRAG does not claim exactly-once token generation before a result is
staged.

`spawn_agent` is replayable because each child id derives from the parent Effect
intent and each terminal roster row stores the exact parent-visible Child outcome,
including its Evidence state and stable citation handles. Replay re-merges that
state into the existing parent ledger and returns the persisted outcome without
persistence, claim, or drive re-entry. A nonterminal child may restore under its
deterministic id and fenced child epoch.

If the process dies during another replayable read-only tool batch, that batch may
run again.

## Web Conversation Adapter

Web conversations remain principal-scoped navigation and history. They do not
own execution state.

Web run creation validates the conversation and, in the same PostgreSQL
transaction that inserts or reuses the run, inserts or reuses one conversation
turn identified by that submission id and referencing its `answer_run_id`. The
turn keeps conversation order; request content lives in the immutable run
input. Uploaded blobs and run-artifact references are inserted or reused in
that same transaction. Reusing an idempotency key with different normalized
input returns HTTP 409. The link is created before the 202 response, so no SSE
subscriber, request finalizer, or browser reconnect is responsible for
committing history. Non-Web run creation uses the same atomic run, input-blob,
and run-artifact transaction without a conversation turn.

Conversation reads return every linked turn in conversation order. Queued and
running turns are pending entries carrying `answer_run_id`, status, and
cancellation-request state, which lets a reloaded browser resubscribe without
remembering the original 202 response. Failed and cancelled turns remain
visible until their run reaches the configured retention floor; they are never
fed back to the model as conversation history.

The conversation-turn reference uses `ON DELETE CASCADE`, so run pruning cannot
leave a dangling entry. Pruning the final routed run also removes its Agent
Session tree even while an empty Conversation row still names that identity;
the row contains no hidden model history. If another turn is accepted before the
empty-row sweep, the adapter rebases it to a fresh `main` Lane and starts a new
Session tree. A linked turn becomes model history only when its run succeeds;
user input, answer, usage, Evidence summary, and source snapshot are projected
from the run instead of copied into another execution record. Follow-up adds a
turn to the current conversation. Fork atomically creates a new conversation
branch whose first run carries parent lineage.

Conversation deletion deletes its linked runs in one transaction; lease-fenced
workers can no longer append after the run row disappears. Cascades remove
events and run-artifact references, and unreferenced artifact blobs are removed
by the same ownership-safe cleanup path.

Web conversations own no raw attachment table and no duplicated turn answer
payload; both are read from the run. The baseline schema creates only that
representation.

## Artifact Topology

LightRAG creates parser artifacts under `INPUT_DIR/__parsed__` and stores
`file://` sidecar URIs. Its installed resolver returns `None` for remote schemes;
`s3://` is documented upstream as future support, not an active storage backend.

The supported topology is therefore:

- default: the local `deployment.working_dir` volume;
- multi-process, one host: a shared named volume;
- multi-host: one shared POSIX mount such as EFS, NFS, or Azure Files mounted at
  the same configured `deployment.working_dir` path in every process.

`deployment.working_dir` is configurable, so there is no separate storage switch. Direct
object-storage support would require a LightRAG sidecar resolver or a DlightRAG
materialization cache and is outside this contract.

Every process serving KB images or source downloads sees the same POSIX artifact
tree at the same absolute path; `postgresql.md#service-roles-and-shared-artifacts`
is the operational reference.

## Failure And Security Rules

- Owner identity scopes every run, event, and artifact lookup.
- Workspace authorization is evaluated once before the run-creation transaction.
  The immutable run input stores only the resulting workspace set, not a JWT or
  mutable claims. A later policy change does not revoke an already accepted
  run; its owner may cancel it, and normal retention or conversation deletion
  later removes it.
- Attachments retain the existing count, per-item, total-byte, and pixel limits.
- Indefinite queueing deliberately has no aggregate DlightRAG byte quota. A
  deployment must bound and monitor PostgreSQL storage and apply ingress rate
  limits appropriate to its auth mode. If artifact persistence cannot commit,
  POST fails before run acceptance.
- Public URLs retain HTTPS-only, redirect, DNS, SSRF, and byte validation.
- Model-visible tool errors are sanitized; operator traces retain the exception
  class and traceback.
- Duplicate names in a composed per-run tool set fail that run with
  `invalid_tool_configuration` before any model call.
- A lost lease stops the old worker from appending events to, committing, or
  deleting state owned by the new worker.

## Verification

This contract is held by:

- unit tests for every status transition and recovery boundary named in this
  document;
- PostgreSQL integration tests for claim, lease loss, Session recovery, event replay,
  cancellation, pruning, and artifact ownership;
- transport contract tests for REST, MCP, Web, and Python;
- a process-restart test that restores OperationState or resumes Fast stages;
- reconnect tests that replay events without duplicate sequence numbers;
- the full local GitHub Actions equivalent (`make ci`).