# Durable Answer Runs

This document defines the target Answer runtime. It is a product contract, not
a Web-only workflow: REST, MCP, Web, and the Python manager all use the same
durable run coordinator.

## Goals

- Every answer is one durable run with one identifier and one lifecycle.
- Client disconnect does not cancel the run.
- A process restart resumes from the latest completed research control turn.
- REST, MCP, Web, and Python use the same run state, artifacts, events, and
  final result.
- Durability does not change retrieval, citation, workspace, or round-robin
  semantics. The query-time cleanup below is the only intentional answer
  behavior change.
- The implementation adds no orchestration framework and reuses the existing
  ingest-job lease, recovery, sweep, and prune mechanics. DlightRAG keeps its
  custom agent loop, `EvidenceLedger`, `RunEpisode`, and tools-disabled final
  synthesis.

## Non-Goals

- Exactly-once execution for an in-progress read-only tool batch.
- Exactly-once token generation across a process crash.
- LangGraph, LangChain, or pi runtime adoption.
- Human-in-the-loop, steering queues, sub-agents, or a global tool registry.
- An object-storage shim for LightRAG parser sidecars.

## Runtime Cleanup Before Persistence

The durable runtime builds on a smaller query-time core.

### Model image capabilities

Answer, VLM, and chat-rerank image capability are separate facts because they
may resolve to different model roles. The manager probes each distinct model
endpoint once at startup and caches the result by resolved model configuration.
When two roles resolve to the same configuration they share one probe result.

- Answer capability controls current-image admission and final answer images.
- VLM capability controls query-image description and `inspect_resource`.
- Rerank capability controls image-bearing chat rerank requests.

`supported` and `unsupported` are terminal for the process. Only `unknown`
uses the existing cooldown-bounded, single-flight lazy re-probe.

### Agent termination

Tool errors are not convergence. A control turn may stop for no new evidence
only when it contains no tool errors. An invalid, unavailable, or failed tool
result is replayed to the model for correction; the existing
`max_agent_turns` bound remains the only error-loop safety cap.

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

No new executor or concurrency setting is introduced unless measurement shows
the default executor is saturated after these duplicate computations are
removed.

### Answer image policy

One frozen Answer image policy owns the answer transport geometry, byte limits,
quality limits, and context window. It creates fresh mutable budgets for a run.
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

The old `stream` request field is removed. There is no temporary Answer mode.

### Execution

An answer execution slot is one of the `max_async` runs that a process may
execute concurrently. It is not a token, context, image, storage, or PostgreSQL
budget. A process reserves a local slot before it claims a row, so a worker
never owns a lease while waiting for local capacity.

`max_async` retains its existing dual role: it also remains LightRAG's
`llm_model_max_async`. One deployment setting therefore bounds both concurrent
Answer runs and LightRAG LLM calls. Operators choosing a small value are
choosing durable queueing rather than capacity rejection; no second Answer-run
concurrency setting is introduced.

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
monotonically on every claim and never resets. A consecutive-recovery counter
increments when an expired lease is reclaimed and resets after each committed
control-turn checkpoint. A fixed internal maximum permits four consecutive
reclaims without durable progress. The next reclaim instead atomically fails
the run with `run_abandoned` and appends its `error` event. This prevents a
deterministic process-killing run at the head of the queue from monopolizing
every worker while allowing a long run that checkpoints progress to survive
more total process restarts. The bound is not configurable.

A worker renews only an unexpired lease by comparing its owner and fencing
epoch. If a heartbeat or ordinary lease-guarded write affects no row, the
worker rereads the row: an expired lease or owner/epoch divergence means
ownership is lost, so it cancels and joins its in-process work, performs no run
transition, releases the local execution slot, and leaves recovery to the
current or next owner. It never tries to revive an expired lease. The
indeterminate checkpoint-commit procedure below is the only zero-row retry
exception. Blocking CPU work remains cancellation-cooperative at its existing
result-commit boundary; the stale worker cannot persist that result.

A durable Answer run has no wall-clock execution deadline. The existing
`request_timeout` continues to bound non-durable Retrieve and Query operations
but no longer wraps Answer execution. Each external LLM, embedding, rerank,
URL-fetch, resource, and parser-sidecar call retains its existing provider,
request, or stream-idle timeout, so one stalled awaited call cannot hold a slot
forever. A run releases its slot on terminal transition, lease loss, or process
shutdown; no new run-timeout setting is introduced.

On graceful shutdown the coordinator stops claiming rows first. Active workers
may finish an in-flight terminal transition or control-turn checkpoint during
the application's shutdown grace; remaining work is then cancelled and joined.
Each owned cancel-pending run is finalized as `cancelled` with its terminal
`done` event. Every other owned nonterminal run receives a fenced `running` to
`queued` transition that clears its lease while preserving its latest
checkpoint, turn count, and cancellation field. It is immediately reclaimable
as ordinary queued work and does not increment the consecutive-recovery
counter. Only reclaim after lease expiry counts as crash recovery. A run
interrupted during generation emits `reset` when next claimed, whether the
interruption was a crash or graceful shutdown.

If every process is busy, an accepted row remains `queued` until a worker has a
slot or the caller cancels it. Queue age never turns an accepted run into a
capacity failure, and queue depth is not capped in this design. POST may fail
before acceptance for validation, authorization, or persistence errors, but
execution-slot exhaustion is not one of them. The obsolete
`answer_acquire_timeout` setting and capacity error are removed; no queue
timeout, queue-depth, retry, or sweep-interval setting replaces them.

`GET /answer/{run_id}` returns:

- `queued`, `running`, `succeeded`, `failed`, or `cancelled`;
- whether cancellation has been requested for a running run;
- current phase and completed control-turn count;
- the final answer payload for `succeeded`;
- one public error kind and message for terminal failures.

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

The Python manager keeps convenience methods:

- `aanswer()` creates a run and waits for its result;
- `aanswer_stream()` creates a run and subscribes to its events;
- explicit run start/status/events/cancel methods expose the durable contract.

Cancelling a waiting convenience call or closing any event subscriber detaches
that caller only. Explicit run cancellation is the sole client action that sets
`cancel_requested_at`.

The Web POST creates a core run and returns its descriptor; the browser then
subscribes to the same owner-scoped core `GET /answer/{run_id}/events` endpoint.
Disconnecting the browser only closes that subscriber, and reconnecting resumes
from the durable event sequence.

One new async helper in `dlightrag.client` owns REST create-and-wait behavior. It
creates the run, follows durable events, and falls back to status reads after a
reconnect or an expired event log. The synchronous CLI invokes that helper
through `asyncio.run`; the async evaluation script awaits it directly. Those
callers do not retain a legacy synchronous endpoint or implement independent
polling loops. The manager's `aanswer()` and `aanswer_stream()` convenience
methods use the same coordinator semantics in process.

MCP `answer` is deliberately descriptor-only and returns immediately. The
separate MCP status tool returns the canonical result after success, and the
cancel tool requests cancellation; MCP does not hold one tool call open for a
tens-of-minutes run.

Every transport derives the owner through one transport-neutral principal
projection moved from the current Web adapter into core. `auth_mode="none"` and
`auth_mode="simple"` intentionally collapse callers into one deployment owner;
`auth_mode="jwt"` is the tenant boundary. Direct in-process manager calls
without a user use that same deployment owner.

### Reader role

`service_role="reader"` means corpus-read-only, not process-read-only. A reader
may create and execute answer runs and may write DlightRAG operational state,
including runs, events, artifacts, and Web conversations.

Reader safeguards remain at the corpus boundary:

- `require_writer()` continues to reject ingestion, workspace creation/reset,
  metadata mutation, failed-document retry, and deletion;
- the LightRAG PostgreSQL pool continues to use read-only sessions and the
  no-DDL attach path;
- LightRAG LLM cache writes remain disabled;
- ingest pipeline recovery remains writer-only.

The DlightRAG domain pool is writable for both roles. The supported reader
topology therefore uses the same primary PostgreSQL endpoint; the old promise
that a reader process may point every pool at a physical hot standby is removed.
Read-replica routing would require a separate corpus endpoint and is outside
this design.

This changes the current reader implementation and documentation explicitly:

- domain-pool connection setup no longer applies session read-only mode to a
  reader, while LightRAG pool setup still does;
- writer startup owns schema migrations; reader startup validates that the
  current domain and LightRAG schemas already exist without issuing DDL;
- readiness permits Answer and Web operational writes on a reader and still
  rejects corpus-mutating operations through `require_writer()`;
- Web is enabled for readers; and
- configuration, PostgreSQL, operations, and architecture documentation remove
  physical-standby and whole-process-read-only claims.

A reader with a missing or incompatible schema fails startup with a diagnostic
and serves no traffic. It does not retry DDL or run partially ready. Deployment
order applies writer migrations before starting readers on the new revision.

## PostgreSQL State

### `dlightrag_answer_runs`

One row owns the run:

- owner identity and `run_id`;
- normalized immutable request input;
- status, phase, stop reason, and completed turn count;
- cancellation request time, lease owner, lease expiration, fencing epoch, and
  consecutive-recovery count;
- next durable event sequence;
- event-log trim timestamp;
- latest checkpoint JSON;
- final result JSON or terminal error;
- created, updated, started, and finished timestamps.

Queued and expired-running rows are reclaimable. Active workers renew a lease.
Terminal rows are retained for 30 days after `finished_at` and then pruned in
bounded, `SKIP LOCKED` batches. A run referenced by a committed Web turn is
exempt: its lifetime is owned by the conversation and it is deleted only when
that conversation is deleted or expires. Pruning checks the turn reference in
the same transaction.

Event rows have their own retention. All events for any terminal run are
deleted 30 days after `finished_at`, even when a successful run row remains for
a Web conversation. The pruning transaction sets `events_trimmed_at` on the run
after deleting its event rows. The canonical result remains on that run; its
event endpoint returns 410 only when `events_trimmed_at` is set. Before trimming,
a cursor equal to or greater than the terminal event sequence opens and closes
immediately with no replay rather than returning 410. This prevents token and
derived-final payloads from becoming a second conversation-lifetime copy.

`run_id` is a UUIDv7. Run creation takes one optional idempotency key unique per
owner. REST uses the `Idempotency-Key` header, MCP and Python expose an
`idempotency_key` argument, and Web passes its existing `submission_id`; there
is no second Web replay mechanism. This intentionally changes Web's current
conversation-scoped submission uniqueness to the same owner-wide namespace as
all other transports. Recreating the same normalized request under the same key
returns the existing run, while reusing a key with different input returns 409.
Creation without a key always creates a new run. The key expires with the run
row.

The normalized request is canonical JSON over the query, exact authorized
workspace set, retrieval and answer options, history, ordered resource
descriptors, and uploaded artifact digests after validation. It excludes
transport headers, temporary paths, authorization-dependent URLs, and secrets.

All Answer workers sharing a database must run a compatible software revision
and the same effective model-role, Answer image-policy, and agent-limit
configuration. These values are deployment state, not copied into every run.
Operators drain or cancel active and queued runs before an incompatible rolling
change; heterogeneous execution is unsupported.

The run row is the sole authority for lifecycle status, phase, completed turn
count, stop reason, cancellation, lease, final result, and terminal error. The
checkpoint is restorable agent state, not a second lifecycle record. A control
turn transaction writes the checkpoint and advances the row's completed-turn
count atomically. Its update predicate requires the current lease owner,
unexpired lease, fencing epoch, and expected completed-turn count. Recovery
rejects a checkpoint whose copied turn number does not equal the authoritative
row value.

### `dlightrag_answer_run_events`

Events use a monotonically increasing sequence per run. Durable event types are:

- `progress`;
- `token`;
- `reset`;
- `done`;
- `error`.

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

These five are the only durable event types. Intermediate contexts are not
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
phases `planning`, `searching`, `researching`, and `generating`; Web-local
presentation phases are not durable.

Every sub-turn phase transition uses one small transaction that, under the
current unexpired lease and fencing epoch, updates the run row's `phase` and
appends the matching `progress` event with the next durable sequence. It does
not modify the checkpoint or completed-turn count. Recovery may re-execute an
uncheckpointed turn and append an earlier phase again; `progress` is a durable
last-writer-wins state notification, not a monotonic workflow history, so this
does not emit `reset`. `reset` is reserved for clearing a partial token draft
before regenerated output.

### `dlightrag_answer_artifacts`

Immutable raw bytes are content-addressed within one owner namespace. The table
stores the owner, SHA-256 digest, byte size, bytes, and creation timestamp.
Lifetime is derived only from run references; there is no independent
artifact-retention state. Caller attachments may be reused across runs without
cross-owner deduplication.

Run-scoped fetched Web bytes use the same blob contract. They are stored only
after the existing HTTPS, redirect, DNS, SSRF, and byte validation passes.
Storing the blob and its run-artifact reference is one transaction performed
before the tool result may enter a checkpoint. Once that turn is checkpointed,
its resource id is permanently bound to those validated bytes and recovery
never silently re-fetches a changed page. Work from an uncheckpointed tool batch
may execute again after a crash and carries no same-bytes guarantee, consistent
with the read-only-tool replay non-goal.

### `dlightrag_answer_run_artifacts`

This join table records ordered run inputs and discovered resources without
duplicating bytes. Each reference owns its safe filename, MIME type, ordinal,
resource kind, and any deterministic transform locator needed to regenerate a
page or image window. It distinguishes current attachments, historical
attachments, and run-scoped fetched resources.

Its `(owner, digest)` columns have a composite foreign key to the artifact
table with `ON DELETE RESTRICT`. Inserting a reference therefore takes the
PostgreSQL key-share lock that serializes against blob deletion; reusing a blob
is not protected merely by checking both tables in one READ COMMITTED
transaction.

Deterministic attachment conversion is recomputed from stored bytes on resume;
no conversion-result table is introduced. VLM inspection prose remains run
evidence, not a cross-run cache.

Deleting a run removes that run's references, not shared bytes. An artifact row
is deleted only when no run-artifact row references its digest for that owner.
Reference checks and deletion occur in one transaction, and the foreign key
rejects deletion if a concurrent run has adopted the digest. A deferred cleanup
pass may retry a blob that remained because of that race.

## Checkpoint Contract

A checkpoint is written atomically after every completed agent control turn. It
contains only JSON-safe state:

- `EvidenceLedger` contexts and citation identities;
- `RunEpisode` exchanges including provider-native state;
- completed exact-call cache results;
- resource catalog and cursor state restored verbatim, including the exact
  resource ids and cursor tokens already named in episode messages;
- the copied completed-turn number used only to validate it against the run
  row.

Episode and evidence image blocks are serialized as stable references rather
than data URIs or raw bytes. Attachment images use owner, artifact digest, and
ordinal; corpus images use workspace, chunk, and sidecar identity. Claim-time
rehydration restores those blocks in place before the first model call while
preserving provider message, tool-call, and block order. Missing corpus visuals
follow the degradation rule below; missing attachment blobs fail the run as
`checkpoint_corrupt` durable state.

Each checkpoint has a schema version and an 8 MiB bound measured on its compact
UTF-8 JSON representation after image-reference substitution. Before writing,
older provider-native reasoning is discarded using the existing `RunEpisode`
replay policy. If the compacted state still exceeds the bound, the worker fails
the run with `checkpoint_too_large`; it does not retry the same deterministic
turn forever.
A worker that cannot read a checkpoint version, or whose checkpoint turn number
does not match the row, fails the run with `checkpoint_incompatible` or
`checkpoint_corrupt` instead of guessing at state.

Serialization and size validation happen before the checkpoint transaction. A
definite database rollback leaves the previous checkpoint and authoritative
turn count intact, so lease recovery may re-execute the uncommitted turn. After
an indeterminate commit result, the worker opens a new transaction and locks the
run row with `FOR UPDATE`, which waits for the original transaction to resolve.
If the authoritative turn count is the expected value plus one and its
checkpoint has that copied value, the commit succeeded and execution continues.
If the count is still the expected value and owner, unexpired lease, and fencing
epoch still match, the worker retries the same compare-and-set transaction. If
the lease expired or owner/epoch differs, it has lost the lease. Any other
row/checkpoint combination fails the run as `checkpoint_corrupt`. A compare-and-
set retry that affects zero rows therefore resolves through this locked reread
rather than being treated immediately as lease loss.

Large binary image payloads are not copied into checkpoint JSON. Current
attachments refer to core artifacts; KB visuals refer to workspace/chunk
identity and are rehydrated from shared sidecars. A KB visual that no longer
resolves is dropped from the rehydrated image evidence while its text and
citation identity remain; a missing visual never fails the run.

If the process dies during a read-only tool batch, the batch may execute again.
The next checkpoint makes completed turns durable.

A resumed run continues its recorded turn count; `max_agent_turns` bounds the
whole run, not one process lifetime. A fast-path answer has no control turn and
therefore no intermediate checkpoint; recovery emits `reset` and re-executes it
from immutable input.

If the process dies during final generation, recovery appends a `reset` event
and regenerates the final answer from the latest checkpoint. Subscribers clear
the partial draft after `reset`. DlightRAG does not claim exactly-once token
generation.

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
visible with their public terminal error or status until their normal 30-day
run pruning; they are not fed back to the model as conversation history.

The conversation-turn reference to a run uses `ON DELETE CASCADE`. Pruning a
failed or cancelled run therefore deletes its visible terminal turn in the same
transaction; no dangling history entry remains after day 30.

A linked turn becomes committed conversation history only when its run is
`succeeded`. Its user input, answer text, and canonical source snapshot are read
from the run instead of copied into a second execution record. Only succeeded
turns are included in model history. A successful linked turn exempts its run
from 30-day pruning for the conversation's lifetime.

Conversation deletion deletes its linked runs in one transaction; lease-fenced
workers can no longer append after the run row disappears. Cascades remove
events and run-artifact references, and unreferenced artifact blobs are removed
by the same ownership-safe cleanup path.

The current raw Web attachment table and duplicated turn answer payload are
superseded. The migration resets existing Web conversations; no compatibility
view or dual-write path is retained.

## Artifact Topology

LightRAG currently creates parser artifacts under `INPUT_DIR/__parsed__` and
stores `file://` sidecar URIs. Its installed resolver returns `None` for remote
schemes; `s3://` is documented upstream as future support, not an active storage
backend.

Therefore the supported topology is:

- default: the existing local `working_dir` volume;
- multi-process, one host: the existing shared named volume;
- multi-host: one shared POSIX mount such as EFS, NFS, or Azure Files mounted at
  the same configured `working_dir` path in every process.

`working_dir` is already configurable, so no new public storage switch is
needed. Direct object-storage support would require a LightRAG sidecar resolver
or a DlightRAG materialization cache and is outside this design.

Startup and operations documentation must state that every process serving KB
images or source downloads sees the same POSIX artifact tree at the same
absolute path. The implementation replaces the current reader-replica section
with `postgresql.md#service-roles-and-shared-artifacts` and updates README,
configuration, interfaces, retrieval/answer, operations, security, and
architecture documentation to the durable API and topology.

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

Implementation is complete only when it has:

- red/green unit tests for every status transition and recovery boundary named
  in this document;
- PostgreSQL integration tests for claim, lease loss, checkpoint, event replay,
  cancellation, pruning, and artifact ownership;
- transport contract tests for REST, MCP, Web, and Python;
- a process-restart test that resumes after a completed control turn;
- reconnect tests that replay events without duplicate sequence numbers;
- the full local GitHub Actions equivalent (`make ci`);
- a final read-only architecture review with no Critical, High, or Medium
  findings.