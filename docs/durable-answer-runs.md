# Durable Answer Runs

This document owns answer lifecycle, fencing, recovery, event persistence,
resource blobs, and the Web conversation adapter. Public endpoint shapes live in
[Interfaces](interfaces.md); retrieval and generation behavior lives in
[Retrieval and Answer](retrieval-answer.md); PostgreSQL deployment details live
in [PostgreSQL](postgresql.md).

`dlightrag.engine.runtime` owns storage-neutral lifecycle records, store ports,
fenced sessions, subscriptions, cancellation listening, and `RunCoordinator`.
Engine Answer owns product execution and maps product failures to
`RunExecutionError`. `PGAnswerRunStore` implements the runtime port without
creating an Engine dependency on PostgreSQL.

## Guarantees And Limits

- Every answer on every interface is one durable run with one ID/lifecycle.
- Disconnecting a client never cancels its run.
- Recovery restores complete typed Agent Operation state or exact Fast stages.
- Committed effects do not execute again; pending effects replay only under the
  pinned plan and contract.
- All transports see the same events, Artifacts, and canonical result.
- Retrieval, citation, and federation semantics do not change for durability.

DlightRAG does **not** promise exactly-once execution for an interrupted
read-only tool batch or exactly-once token generation before final result
staging. It adds no external workflow framework, detached/background agents,
permission platform, sandbox backend, MCP registry/OAuth service, or object
storage shim.

## Lifecycle

`POST /answer` always validates and persists a run before returning HTTP 202.
There is no temporary answer mode or `stream` request field.

```text
accept  -> run + routing + pinned input + blobs (one transaction)
claim   -> oldest eligible row; fencing epoch++; lease heartbeat
execute -> durable phases/events and Agent or Fast settlements
finish  -> canonical result + exactly one terminal event (one transaction)
recover -> reclaim expired lease and restore total durable state
```

An execution slot is one of
`answer.runtime.answer_worker_concurrency` local runs. The coordinator reserves a
slot **before** claiming a row, so a worker never holds a lease while waiting for
local capacity. Model-provider and ingestion concurrency are independent.

A free worker claims the oldest eligible queued or expired-running row with
`FOR UPDATE SKIP LOCKED`. It sweeps bounded batches at startup, after local
completion, and once per second so work from another host does not depend on a
process-local wakeup.

Accepted work queues indefinitely while all slots are busy. Queue age, queue
depth, and slot exhaustion never become capacity failures. There is no answer
wall-clock timeout. Individual LLM, embedding, rerank, URL, resource, and parser
calls retain their own timeouts.

## Leases, Fencing, And Recovery

Every claim increments a monotonic fencing epoch. A heartbeat renews only an
unexpired lease matching owner and epoch. If a guarded write updates no row, the
worker rereads it; expiration or owner/epoch divergence means lease loss. The
stale worker cancels and joins local work, performs no transition, and releases
its slot. It never revives a lease.

Recovery uses two distinct counters:

- `durable_progress_version` advances only on fenced model-turn, compaction,
  effect, or Fast-stage settlements.
- `reclaims_without_progress` counts consecutive expired-lease reclaims with no
  such progress.

Four consecutive no-progress reclaims fail the run as `run_abandoned`. A long
run that settles progress between crashes may survive more than four restarts.
The bound is not configurable.

The sweeper needs no execution slot to finalize a cancel-pending row without a
live lease or abandon a no-progress run. Cancellation takes precedence over
reclaim and abandonment.

### Graceful Shutdown

The coordinator first stops claiming. Active workers may finish a settlement or
terminal transaction during shutdown grace. Then remaining work is cancelled
and joined:

- owned cancel-pending runs become terminal `cancelled` with `done`;
- other owned nonterminal runs return from `running` to `queued`, clearing the
  lease while preserving durable state and cancellation fields.

This handoff does not increment crash-reclaim counters. Generation interrupted
by shutdown or crash emits `reset` when reclaimed.

## Cancellation And Controls

Deleting a queued run terminalizes it immediately. Deleting a running run sets
`cancel_requested_at`; the worker observes it after awaited planning,
retrieval/tool batches, between control turns, and at token-batch boundaries.
If the lease expires first, the sweeper terminalizes it.

Cancellation and successful finalization serialize on the run-row lock.
Success is allowed only while no cancellation is pending. Cancelling a terminal
run is an idempotent no-op. REST returns 200 when terminal and 202 while a live
worker still must observe cancellation.

Steer instructions enter an ordered inbox and are consumed at stable checkpoints
as durable `ControlMessage` entries. A terminal-race steer/follow-up creates a
fresh linked Operation. Fork creates a new Lane in the same Agent Session.
Endpoint details are in [Interfaces](interfaces.md#answer-run-endpoints).

Closing an SSE subscriber or cancelling `AnswerService.answer()` /
`answer_stream()` only detaches the caller. Explicit cancellation is the sole
client action that sets `cancel_requested_at`.

## Idempotency And Pinned Input

Run IDs are UUIDv7. One optional idempotency key is unique per owner:

- REST: `Idempotency-Key`
- MCP/Application: `idempotency_key`
- Web: `submission_id`

A matching normalized replay returns the existing run with current status;
conflicting input returns 409. No key always creates a new run. The key expires
with the run row.

The fingerprint hashes canonical normalized public input: query, authorized
workspace set, options, bounded history, ordered resource descriptors, and
validated upload digests. It excludes headers, temporary paths,
authorization-dependent URLs, secrets, and later-resolved model facts. A replay
returns before profile resolution, URL fetches, image descriptions, or history
projection are repeated.

The immutable prepared input stores accepted history/resources/scope plus each
role's endpoint fingerprint and effective model profile, catalogue/context-policy
revisions, and accepted image descriptions. Recovery uses these pinned facts;
provider credentials remain deployment state. A changed global arithmetic
revision alone does not invalidate the run.

## Durable Events

Events have gap-free, monotonically increasing per-run sequences:

- `progress`
- `token`
- `reset`
- `tool_start`, `tool_progress`, `tool_end`
- `done`
- `error`

Appending locks the run row, consumes its next sequence, and checks live lease
owner/epoch. Token text is coalesced into bounded chunks. Tool events store only
name, status, elapsed time, output byte count, spill state, call identity, and
attachment count—never stdout/stderr.

A terminal transaction stores status/error/result and appends exactly one
terminal event:

- success: `done` with complete canonical result;
- cancellation: `done` with `status="cancelled"`, no result;
- failure: `error` with public kind/message.

SSE closes after replaying the terminal event. Intermediate contexts are not
published because Research may change them. `progress` is last-writer-wins and
may move backward after recovery. `reset` invalidates all previously streamed
draft text before a tool-bearing turn, provider retry or failure, continuing
follow-up/correction, interrupted regeneration, or canonical citation/Artifact
rewrite. Only a successful `done.result` is terminal answer authority.

Stored results/events contain transport-neutral source identities. Each
authorized read projects fresh download URLs without modifying stored events.
Event retention may end before the run row: then SSE returns 410 while status
continues to expose the result.

## PostgreSQL State

### `dlightrag_answer_runs`

One row owns:

- owner, UUIDv7 run ID, and optional idempotency key/fingerprint;
- bounded `prepared_input_json` while queued/running;
- status, phase, stop reason, cancellation time;
- lease owner/expiration and fencing epoch;
- durable progress/reclaim counters and next event sequence;
- routing/continuation lineage;
- final result or terminal error; and
- created/updated/started/finished/event-trim timestamps.

The row is the sole authority for lifecycle. Research state lives in the Agent
Session's immutable parent-linked entries and closed typed registers. Fast stage
state is stored under deterministic stage identities.

Terminal runs and event logs follow the configured retention floor from
`finished_at` in bounded `SKIP LOCKED` batches. Conversation turns do not extend
it. Deleting the last routed run makes its Agent Session tree eligible for
cleanup; shared Sessions and child trees still referenced by runs survive.

### `dlightrag_answer_run_events`

Rows are keyed by run and sequence and cascade with the run. Worker writes
require the active lease/epoch. Queued cancellation and sweeper transitions use
the same lock/predicate, preventing duplicate terminal events.

### `dlightrag_blobs` And `dlightrag_blob_chunks`

Raw bytes are content-addressed inside one owner namespace. Every non-final
chunk is exactly 1,048,576 bytes. One transaction writes all chunks, then inserts
the blob metadata; metadata therefore means complete. There is no independent
blob-retention clock and no cross-owner deduplication.

Accepted uploads link atomically during run acceptance. A fetched Web Resource
links only after HTTP(S)/redirect/DNS/SSRF/byte validation settles its effect.
Its resource catalog row retains the canonical locator, provenance capabilities,
and replay ordinal. Recovery reads those stored bytes without live DNS validation
because it performs no network request; a settled Resource remains one fixed
snapshot, while an acquisition that admitted no Evidence may retry later.

### `dlightrag_answer_run_artifacts`

This join table records ordered request attachments and terminal Published
Artifacts: safe filename, MIME type, ordinal, resource kind, and deterministic
transform locator. Fetched Web snapshots use `dlightrag_answer_resources` plus
the shared Blob tables instead. The artifact table references `(owner, digest)` with
`ON DELETE RESTRICT`, so insertion takes the PostgreSQL key-share lock that
serializes against blob deletion.

Deleting a run removes only its references. Blob deletion occurs in one
transaction only when no references survive; the foreign key protects a
concurrent reuse. Deterministic conversion is recomputed from stored bytes.
VLM inspection prose remains run Evidence rather than a cross-run cache.

### `dlightrag_answer_artifact_attachments`

One row per `(owner_id, run_id, relative_path)` is the durable Root Artifact
Attachment authority. It records the display label, raw SHA-256 digest and byte
size, presentation capability, originating Session/Effect Intent, attachment
time, and monotonic settlement order. It is distinct from
`dlightrag_answer_run_artifacts`: attachment rows authorize workspace roots;
run-artifact rows reference owner-visible published bytes and other run
resources.

The row cascades with its run. Reattaching the same path replaces the authority
and assigns the latest settlement order. Workspace inventory refresh or deletion
does not erase the attachment: terminal publication must instead compare it
with the current raw bytes and fail closed when the file is missing or stale.

## Agent Session Recovery

Each routing row authorizes one Agent Session/Lane. Immutable entries preserve
parent-linked ancestry; Lane registers select branches without copying shared
history. Research restores complete `OperationState` and invokes the same pure
`NextAction` interpreter used live.

Before a provider call, Runtime commits the exact request snapshot and attempt.
Assistant settlement records the complete response and ordered Tool Batch Plan.
Tool clearance, effect settlement, ToolResult placement, Host deltas, and
progress then commit under the lease/epoch predicate.

Recovery treats effects by contract:

- `replayable`: reconcile or dispatch again under the unchanged contract;
- `never`: settle as `outcome_unknown`;
- changed contract: settle `tool_contract_changed` without dispatch.

`attach_artifact` is replayable because it only validates current workspace
bytes and produces authority through settlement. Its model-visible `ToolResult`
and `ArtifactAttachmentUpdate` commit atomically; a crash cannot commit one
without the other.

Image state stores resource/corpus identities, never data URIs. A missing corpus
visual drops only its image while preserving text/citation; a missing attachment
blob fails the run.

`spawn_agent` is replayable because child IDs derive from parent effect intent
and terminal roster rows persist parent-visible outcome/Evidence. Replay
re-merges that state without creating or driving another child. An interrupted
ordinary read-only batch may execute again.

### Fast Recovery

Fast shares the Session Entry Tree but never enters the Agent interpreter.
Acceptance atomically appends `UserMessage` plus `HostTurnReservation`. Before
assistant settlement, the Host stages the complete canonical result at a
deterministic final-generation stage. If the process crashes after the assistant
commit, recovery terminalizes from the staged result without retrieval or
generation.

Failure before staging clears the reservation and preserves unanswered input.
After staging, failure/cancellation preserves it so replay can commit the exact
assistant without Lane interleaving. Interrupted pre-staging generation emits
`reset`; token generation is not exactly-once.

## Web Conversation Adapter

A Web conversation owns principal-scoped navigation/history, not execution.
The run-creation transaction inserts/reuses the run, input blobs/artifact
references, and one conversation turn keyed by `submission_id`. Admission
failure leaves no empty conversation, and history exists before HTTP 202.

History returns chronological keyset pages (newest 40 by default, maximum 100
per request). Queued/running turns remain resubscribable pending entries; failed
and cancelled turns remain until run retention. Only succeeded turns become
model history, projected from the run rather than copied.

A follow-up adds a linked turn. Fork atomically opens a conversation branch with
parent lineage. Conversation deletion removes linked runs in one transaction;
workers cannot append after the fenced rows disappear. Cascades remove events
and references, then ownership-safe cleanup removes unreferenced blobs.

If retention removes the last turn/routed run before the empty conversation row
is swept, a later reuse starts a fresh `main` Lane and Session. Conversation
identity never preserves hidden model history.

Browser SSE uses the same event sequence, resume, 410, and detach semantics as
REST; only terminal projection differs (`AnswerPresentation` rather than stored
canonical JSON). The browser reconciles ambiguous acceptance through the
owner-scoped submission lookup and never blindly repeats POST.

## Reader Role And Artifact Topology

A `reader` is corpus-read-only, not operationally read-only. It can execute
answers and write runs/events/artifacts/conversations, while CorpusAdmin and the
LightRAG pool reject corpus mutation/DDL. Both roles use the same writable
primary; writers migrate before readers validate schema and serve traffic.

LightRAG parser artifacts use `file://` paths under the configured working
directory. Multi-process/multi-host deployments therefore mount one shared
POSIX artifact tree at the same absolute path. Direct object-storage resolution
is outside this contract.

## Failure And Security Rules

- Owner scope applies to every run, event, and Artifact lookup.
- Workspace authorization is resolved before acceptance; pinned workspace scope
  is not retroactively revoked by later policy changes.
- Attachment count, item, total-byte, and pixel limits apply before acceptance.
- Deployments must monitor PostgreSQL/blob growth and enforce ingress limits;
  DlightRAG intentionally has no aggregate queue byte quota.
- Public URLs retain scheme, redirect, DNS, SSRF, and byte checks.
- Tool errors shown to models are sanitized; operator traces retain detail.
- Duplicate run-local tool names fail before a model call.
- A stale worker cannot append, commit, or delete after lease loss.

The contract is verified by unit state-transition tests; PostgreSQL integration
tests for claiming, fencing, recovery, cancellation, pruning, and ownership;
transport/reconnect tests; process-restart tests; and the full `make ci` gate.
