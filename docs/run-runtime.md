# RunRuntime

Every top-level Retrieval, Answer, and Corpus Mutation is one durable Run: one
identity, lifecycle, event log, cancellation path, and terminal result, shared
by REST, MCP, the Web, the Python Application, and evaluation. This document
owns the Run lifecycle, lanes, admission limits, the Schedule look, Corpus Mutation
semantics, cancellation and repair, retention, and worker scaling.
[Domain Language](domain-language.md) defines the terms,
[Interfaces](interfaces.md#run-lifecycle-and-answer-endpoints) the wire
contracts, [Configuration](configuration.md#runruntime-lanes-and-retention) the
fields and defaults, and [PostgreSQL](postgresql.md#durable-run-state) the
tables. [ADR 0007](adr/0007-unified-durable-runs-and-execution-lanes.md)
records the decision.

## Lifecycle

```text
accept  -> Run row + bounded immutable prepared input
claim   -> oldest lane-eligible row; fencing epoch + 1; lease
execute -> executor-owned phases, checkpoints, and durable events
finish  -> result or error + exactly one terminal event, in one transaction
recover -> reclaim an expired lease and continue from durable state
```

A Run's status is `queued`, `running`, `succeeded`, `failed`, or `cancelled`;
its `phase` is a label its executor owns. `run_kind` (`retrieval`, `answer`, or
`corpus_mutation`) and `lane` are data on the one Run record. An Answer's
internal Retrieval Stage executes under that Answer's lease and capacity; it is
never a nested Run.

Application use cases authenticate, authorize, normalize, and bound a request,
then hand RunRuntime an opaque prepared envelope; one executor per run kind owns
the operation's behavior and result. Acceptance stores the Run before it returns
a descriptor, and nothing executes inline. A process accepts only while its own
coordinator runs. The prepared input is at most 8 MiB of canonical JSON; the
terminal transition clears it and keeps a bounded accepted envelope for status
and history.

Run ids are UUIDv7. Every Run has a submission key, unique per run kind and
submitter: the caller's idempotency key, or the Run id when the caller sends
none. A repeated key with the same normalized input returns the existing Run;
the same key with different input is refused as a conflict. The key lives as
long as the Run row. The fingerprint covers only the normalized public request,
such as the query, Workspace set, options, history, and ordered Resource
descriptors, never headers, temporary paths, projected URLs, secrets, or
execution output.

The prepared input pins what execution needs to behave the same after a crash.
Retrieval pins its normalized request, authorized Workspace set, model
fingerprints and profiles, and model-catalogue and context-policy revisions. It
keeps query-image bytes only while nonterminal; its terminal envelope keeps
their count and SHA-256 digests. Answer also pins routing, its Agent Session and
Lane, history, Resources, image descriptions, its Agent Plan, and a random
resource identity that mints every Resource handle and cursor of the Run.
Credentials take no part in a handle, so handles survive credential rotation.
Execution rechecks the pins and fails closed rather than run with changed
semantics.

An executor ends each attempt with success, failure, a deferral, a wait for
repair (Corpus Mutations only), or a terminal it committed itself. A transient
interruption of corpus storage, the parser, a model provider, or the Agentspace
volume defers the Run:
it checkpoints, releases its execution slot, and becomes claimable again at
`next_attempt_at`, with backoff from five seconds (two for Corpus Mutations)
doubling to 60. A Run has ten deferrals across all dependencies, about six to
seven minutes of backoff; the next outage fails it as `dependency_unavailable`.
A Corpus Mutation waiting behind a Workspace's partition-promotion write fence
or for LightRAG's pipeline to settle spends none of them. Invalid input, an
unsupported capability, rejected credentials, a deterministic operation
failure, and a misconfigured endpoint (a TLS certificate that fails
verification, or a TLS protocol mismatch) are terminal. Executors classify
their outcomes; an exception no executor classified fails the Run as
`run_execution_failed` rather than being retried.

Answer has no run-level timeout. A top-level Retrieval's
`corpus.retrieval.timeout` starts when it is claimed and bounds planning and
search, not queue residence; expiry fails it as `retrieval_timeout`. Model,
embedding, rerank, URL, resource, and parser calls keep their own timeouts.

Disconnecting a client, closing an event stream, or abandoning a caller-awaited
Application call only detaches that observer; explicit cancellation is the only
client action that cancels a Run. RunRuntime does not promise exactly-once
execution of an interrupted read-only tool batch, or exactly-once token
generation before a final result is staged.

### Events

Each Run's events carry a gap-free, increasing sequence. An append locks the
Run row, takes its next sequence, and succeeds only for the lease owner at its
epoch; a database trigger also refuses a nonterminal event for a Run that is not
running with a lease owner, and a terminal event that does not match the stored
status and result.
Retrieval and Corpus Mutation emit `progress` and one terminal event. Answer
also emits tokens, model reasoning, `reset`, Tool, and memory events, and commits
token or reasoning text in batches once the buffered stream reaches 512
characters or 250 ms after its first batch; a switch from one stream to the other
commits the first before the second is buffered, so a reader sees the order the
model wrote.
[Interfaces](interfaces.md#run-lifecycle-and-answer-endpoints) defines every
event and payload.

The terminal transaction stores the status with the result or error and appends
exactly one terminal event, which a unique index enforces: `done` with the
result, `done` with `status: cancelled`, or `error` with a public kind and
message. A subscriber replays committed events after its cursor and follows the
Run until that event: a commit in the same process wakes it at once, and it
polls PostgreSQL every second for commits made elsewhere. The event log, not
the stream connection, is authoritative. `progress` is last-writer-wins and may
move back after recovery. Only a successful `done` result is terminal result
authority: Retrieval publishes no contexts before it, and Answer `reset`
invalidates the draft text streamed so far. An open Agent
traces dock draws a live Run from this log: the batch a subscriber receives is
what tells it the Run stated something — a tool step, a phase, the end — and the
pages it reads for those carry the Entries, while a batch of tokens or reasoning
costs it no read at all, because the chat draws those words and a row already
counts its own time. The log states no child agent's work: a Run's own agent streams, and a
child's pane comes from the pages again until a child has a stream of its own. An Answer that resumes after it
committed events emits `reset` just before its first new token, so a recovery
that streams nothing emits none.

Stored results and events hold transport-neutral identities, never projected
URLs or image bytes; every read projects URLs for the caller's current
permissions and never rewrites what is stored.

### Agent Session Recovery

An Answer's routing row maps the Run to one Agent Session and Lane. Session
Entries are immutable and parent-linked, and Lane registers select a branch head
without copying shared history. Research recovery restores the persisted typed
`OperationState` and drives the same pure `next_action` interpreter as live
execution; only the transient flag that a compaction was declined is not
persisted, so a recovery right after one asks again.
Recovery and Fork rebuild from the selected local Context Projection, never from
a remote response or conversation id.

A Lane is held while it has an active Operation, and its Lane state names the
holder, the Run that accepted it; a Fast reservation names its Run the same way.
A Run's Operations hold the Lane one at a time, so the held one is the Run's
program counter: a recovered attempt drives it to its end before it takes new
input from pending follow-ups, controls, or Child results, takes no input twice,
and never succeeds while an Operation it accepted is open. A Run that finds its
Lane held by another Run that is still nonterminal fails. A holder that has
ended, whether cancelled while queued, swept, abandoned, out of deferrals, or
refused as incompatible, can never release the Lane, so the next Run on it
closes what that holder left: its Operation through the typed close a
cancellation takes, or its reservation, keeping the unanswered question.

Before each provider call the Runtime commits the exact request snapshot and
attempt. Assistant settlement records the complete response and its ordered Tool
Batch Plan, and in the same transaction the `call` that made it: the model,
parameters, prompt digests and timings of the attempt that produced the response, which
a Compaction Entry records for its summarizer call in the same way
([ADR 0043](adr/0043-a-step-records-the-model-parameters-and-prompt-digests-it-ran-on.md)).
That record takes no part in usage, cost, a Plan or acceptance digest, or any
check that decides whether an active Run may resume. Tool clearance, effect
settlement, ToolResult placement, Host deltas, and progress then commit under
the lease and epoch. A model call interrupted before settlement, compaction
included, runs again and records the attempt that settles. Compaction keeps
or removes whole assistant and Tool exchanges. A Research compaction summary
carries the twenty newest committed spills and up to eight published-Artifact
handles, read from their rows, then Evidence citation handles from the Run's
ledger, forty handles at most in all, plus up to eight note paths from the Run's
Agentspace Inventory; a Fast compaction carries only the notes
([ADR 0018](adr/0018-run-notes-and-one-continuation-narrative.md)).

A call runs alone unless its Tool declares it read-only. Up to sixteen adjacent
read-only calls run at once and settle in source order, so Evidence, citations,
and Host deltas match one-at-a-time execution
([ADR 0029](adr/0029-read-only-calls-run-at-once-and-settle-in-source-order.md)).
Provider-native items of a complete Assistant Entry are replayed only to the
same pinned provider, model, endpoint, and API family; any other invocation
receives canonical text and calls
([ADR 0027](adr/0027-api-family-selects-the-provider-wire.md),
[ADR 0030](adr/0030-gemini-uses-the-stateless-interactions-api.md)).

A model that re-issues an identical `bash` or `search_web` call after it
returned is refused: the duplicate settles as a synthetic `repeated_call`
ToolResult against the previous two turns' calls instead of executing again, so
one retry cannot burn a Run. The guard is also bounded: after three repeated calls
refused in a row, whichever calls they were (one call re-issued, or several
alternating), a fourth fails the Operation
(`kind: repeated_call_loop`) rather than being refused again. The bound is
needed because the guard's own window includes the refused turns, so a model
that keeps re-issuing the call would otherwise be refused for ever.

Recovery settles each pending effect by its contract: a `replayable` effect
reconciles or dispatches again when its contract is unchanged and settles
`tool_contract_changed` without dispatch when it changed, while a pending
`never` effect settles as `outcome_unknown` whatever its contract. Pending
read-only calls close one by one in source order, positions never dispatched
close as `interrupted`, and cancellation settles every pending call of the group
as `outcome_unknown`. `attach_artifact` is replayable, and its ToolResult and
attachment authority commit together. `spawn_agent` is replayable because child
ids derive from the parent's effect intent and roster rows persist before
handles are visible; replay re-merges that state without starting another
child, and a parent reclaim rebuilds running children from their stored
envelopes. Entries, Evidence rows, and child pins hold identities, never data
URIs (the transient request snapshot holds them until the assistant settles): a
recovered Run re-renders no corpus pixels while its text and citations survive,
and a missing or mismatched attachment blob fails the Run.

A Fork opens its Lane at the parent Run's recorded Fork Point, not at the source
Lane's current head, and a parent with no recorded Fork Point refuses the Fork
with a remedy ([ADR 0019](adr/0019-turn-accurate-forking-and-the-carry-point.md)).
A Run's first bind materializes the Session's notes into its fresh Agentspace
Epoch, and the handoff records them as that epoch's Agentspace Inventory in the transaction
that sets the Run's epoch; a recovered attempt instead copies the recorded epoch,
verified, and records the copied manifest, never materializing again, and
unrecorded epochs below the current fencing epoch are discarded
([ADR 0022](adr/0022-session-owned-memory-and-run-owned-products.md)).
A volume that does not answer while a Run binds its epoch (an I/O error, a
source that changes while it is copied, or no room for one maximum epoch copy)
defers the Run as an `agent_workspace` outage: nothing recorded changes before
a handoff commits, so the next claim copies the recorded epoch again, and a copy
no handoff recorded is discarded. Content recovery refuses (a link or special
file, a copy unlike its stable source, or a committed spill missing or unlike
its record) fails the Run as `workspace_integrity_error`.

Fast never enters the Agent interpreter. Acceptance appends the user message
and a Host turn reservation. Before the assistant settles, the Host stages the
complete canonical result at a deterministic stage, so a crash after the
assistant commit ends the Run from that stage without retrieval or generation.
A failure before staging clears the reservation and keeps the unanswered input.

## Lanes

| Lane | Run kinds | Workers |
|---|---|---|
| Query | `retrieval`, `answer` | every process |
| Corpus Mutation | `corpus_mutation`: `ingest`, `replace`, `delete`, `retry`, `reset`, `delete_workspace` | writer processes |

Each lane has its own per-process worker bound and its own deployment-wide
[admission limit](#admission-limits). A worker reserves a local slot before it
claims a Run, so it never holds a lease while it waits for capacity. Claims take
the oldest eligible Run; one deferred to a later time or with cancellation
pending is skipped, so it never blocks the rest of its lane.

Corpus Mutations are FIFO within one Workspace and concurrent across
Workspaces. A mutation is eligible only when no earlier nonterminal mutation of
its Workspace exists, so across the deployment at most one Corpus Mutation owns
a Workspace at a time. A deferred or waiting-for-repair mutation releases its
slot but keeps later mutations of its Workspace queued. The owning Run holds its
lease and one slot while LightRAG's in-process pipeline runs; LightRAG owns
concurrency inside the pipeline. This durable order is unrelated to the Agent's
process-local `AccessScheduler`, which is conflict-based and not FIFO.

Query Runs execute while a mutation runs and see LightRAG's eventual
consistency: mutation success promises that every projection converges, not an
atomic snapshot, and no gate separates Queries from mutations. A document
becomes visible only once its finalization completes
([Retrieval And Answer](retrieval-answer.md#product-document-visibility-and-metadata-in-filtering)).
Provider concurrency (`ModelScheduler`, per process) and LightRAG's pipeline
stages have bounds of their own, independent of both lanes.

## Admission Limits

Each lane caps its nonterminal Runs across the deployment with
`runtime.query.max_nonterminal_runs` and
`runtime.corpus_mutation.max_nonterminal_runs`
([defaults](configuration.md#runruntime-lanes-and-retention)). Acceptance counts
the lane's `queued` and `running` Runs under a lane-wide lock, so deferred Runs
and Runs waiting for repair count. At the limit a new Run is refused before
anything is stored ([Interfaces](interfaces.md#health-and-errors) gives the
response); accepted Runs stay durable, and a repeated submission key still
returns its existing Run. The limits keep a dependency outage from growing an
unbounded durable queue. Queue residence has no timeout.

An administrator may also bound how many Answers one owner has in progress, with the
`active_answer_runs` [Runtime Policy](interfaces.md#runtime-policies)
([ADR 0042](adr/0042-runtime-policies-are-read-at-the-door-and-pinned-into-the-run.md)).
It is not a lane limit. At acceptance, under the same lane-wide lock and after a repeated
submission key has been answered, the store counts the owner's `answer` Runs that are
`queued` or `running`, deferred ones included, and refuses the new Run with HTTP 429
`active_run_limit` when the count has reached the limit. Racing submissions of one owner
therefore admit exactly the limit, a keyed retry of an accepted Run is still answered, an
accepted Run is never cancelled because a limit dropped, and a Run that ended makes room. The
policy is read once and the limit is not stored with the Run. The count is served by
`idx_dlightrag_runs_owner_active`, a partial index over the queued and running Answers, so
it reads the owner's active Runs and not the lane's or the owner's history. Under `none` and
`simple` authentication every caller is one owner, so the limit there is the deployment's.

DlightRAG has no aggregate byte quota or tenant quota, and no fairness between owners beyond
that opt-in limit. Monitor PostgreSQL and blob growth, and rate-limit acceptance at ingress
before either limit is reached ([Security](security.md#ingress-responsibilities)).

## Schedules

A Schedule starts an Answer Run at times its owner named, through the admission and
conversation path a typed turn takes
([ADR 0103](adr/0103-a-schedule-starts-a-run-on-the-clock-and-the-database-is-its-queue.md);
[settings](configuration.md#schedules)). The database is the queue, so no process leads and no
cron runs outside the product.

Every process that hosts Web conversations and has `deployment.schedules` on looks about every
`schedules.look_interval_seconds`, in this order:

1. **Retry.** A Fire that was written but never started (the process died between the Fire and
   the Run, or an attempt hit an error the rules do not name) is taken by one process after two
   look intervals. It first asks whether its Run already exists under the Fire's idempotency key,
   and if so records that, before any rule is asked and whatever the owner's access is now. Only
   when there is none is the Fire judged and started, under the same key. A Fire still unstarted
   once its slot is past the grace below is `missed`, with the last error that stopped it as its
   reason, and a Fire missed after an error counts as a failure toward the streak.
2. **Settle.** A Fire still `started` five minutes after it was last looked at, whose Run has in
   fact ended, is settled from the Run. The settlement observer does this at once in the process
   that ran the Run; the look covers an observer that never ran, and a Run still working is
   looked at once per five minutes, so a few long Runs cannot keep the others out of a sweep.
3. **Claim.** `SELECT ... WHERE state = 'active' AND next_fire_at <= now ... FOR UPDATE SKIP
   LOCKED`, twenty Schedules at a time. In that one transaction each due Schedule's Fires are
   written under a unique `(schedule_id, slot)` and its `next_fire_at` moves to the first slot
   after now, which is why a slot is taken once in any number of processes. Slots that passed
   while nothing was looking are one `missed` Fire; only the latest slot may still run, and only
   within half its interval (at most six hours; an hour for a single time). The Schedule's Fires
   beyond `schedules.fires_kept` are trimmed in the same transaction. `now` is the database's
   clock, not the process's, so replicas whose clocks differ agree on what is due.

Then each pending Fire is decided, and the first rule that holds ends it:

| Rule | Fire ends | Schedule |
|---|---|---|
| The Schedule was paused or deleted after the Fire was written (a Fire the owner asked for is exempt) | `cancelled` | as it was |
| The identity it acts as is no longer its owner, or, for a signed-in identity, was last confirmed more than `schedules.principal_max_age_days` ago (creating, editing or resuming the Schedule renews it), or the owner reads none of its workspaces (unreadable ones are dropped from the turn) | `refused` | paused with the reason |
| The previous Fire's Run is still working, or the goal that Run started is still `active`, or an earlier Fire of the Schedule is still starting | `skipped_busy` | stays active |
| The Runs of its Fires, their Children and their goal rounds used `max_tokens_per_day` in the last 24 hours | `refused` | paused with the reason |
| A goal Schedule's conversation holds a goal that is `active` | `skipped_busy` | stays active |
| A goal Schedule's conversation holds a goal that is `paused` or `blocked`, which waits for its owner | `refused` | stays active; the failure streak pauses it |
| The target conversation already has a Run in flight | `skipped_busy` | stays active |
| The owner's active-Run limit or the deployment's admission limit is reached | `skipped_limit` | stays active |
| The turn is refused for something only its owner can change (the conversation is gone, the request is invalid) | `refused` | paused with the reason |
| A dependency is briefly out, or the Connections changed while the turn was admitted | stays `pending` for the retry | stays active |
| Otherwise the Run starts | `started` | |

The Run's user message is one line naming the Schedule and the slot's local time, then the owner's
prompt, and the turn is accepted as the Schedule's identity, so the Runtime Policies are read for
it at the door like any turn's. A Fire's Run that fails ends `failed` with its error; nothing
retries it. Three consecutive Fires that ended `failed` or `refused`, counted from the last time
the Schedule was resumed, pause the Schedule; a success or a cancellation ends the count.

Run now is a Fire of its own, flagged manual, whose slot is the moment it was asked for. It goes
through the rules above as the owner who asked, on a paused or completed Schedule too, and it does
not resume the Schedule or change what is waiting. It is refused while the Schedule has any Fire
still starting or working, and a manual Fire is worth starting for at least five minutes.

A single time is done only when its Fire started a Run. One whose Fire started nothing (refused,
missed, skipped, cancelled) leaves the Schedule paused with that reason, with no time to wait for:
the owner edits the time and resumes it.

A **goal Schedule** (one with a round cap, `goal_max_rounds`) makes a Goal at each fire whose
objective is its prompt, in a conversation made for the purpose when it has none, replacing a
finished goal and never an unfinished one. The Fire's Run is that Goal's round 1: a
`goal_round` Run admitted through the goal service like the `/goal` command's first round, with
the same round prose under the harness line, so the goal driver settles it as any round (a
failed round blocks the goal and a cancelled one pauses it; neither starts another). The Fire
follows round 1 only: later rounds are the goal's, show in the conversation, and count toward
the ceiling. The goal's id comes from the Fire's key, so a retried Fire finds its goal and
admits no second round; a goal whose round 1 never started is cleared, so goal recovery does not
start it outside the Schedule's rules.

The ceiling is counted from what each Run states for itself: the Run Inspector's total, which
includes its Children, and the goal rounds the Session ran after it. A provider that states no
total is read as its prompt tokens, cache reads and writes included, plus its output tokens. A Run
that failed or was cancelled states no total, so it is counted from the model calls it made, each
stated by the assistant message that call wrote. It is checked before a Run starts and does not
cut a running Run short.

## Corpus Mutations

A Corpus Mutation Run is scoped to one Workspace and records its submitter. It
shows to its submitter and, through `workspace.list_files`, to others as
[Security](security.md#workspace-creators) describes; cancelling or resuming it
needs the Run's own action on the Workspace at that time.

Every mutation needs a Workspace the catalog lists, and only a writer accepts
one: a reader refuses corpus writes at acceptance, before staging a byte
([PostgreSQL](postgresql.md#service-roles-and-shared-artifacts)). A writer
accepts while its coordinator runs and the lane has room, without reaching
LightRAG, the parser, or a model provider; an unavailable dependency defers the
Run later. When it runs, every action except Workspace Delete checks the catalog
again, so a Run queued behind a Workspace Delete fails as `workspace_not_found`
before any effect.

Each Run keeps one stable LightRAG `track_id` (`dlightrag-corpus-<run id>`),
checkpoints its phases, and commits `handoff_started_at` before its first
destructive or otherwise non-idempotent LightRAG effect. Ingest, replace, and
retry first read LightRAG's documents for their `track_id`. An ingest that
resumes after its handoff reconciles the upstream documents it finds instead of
duplicating them, or, finding none, runs again under the same `track_id`; a
staged source that is gone fails it as `corpus_source_unavailable`. A replace
that resumes with upstream documents reconciles them the same way. Any other
destructive action that resumes after its handoff without a checkpointed
outcome waits for [repair](#repair).

Sources are staged per Run, and documents keep their source copies in the
Workspace's corpus directory; [Interfaces](interfaces.md#ingestion) gives the
layout and stage lifecycle. Requested physical deletion is a completion
condition, not best effort: a delete or reset that cannot remove its files waits
for repair. A multi-document mutation with any failed document is `failed` and
keeps every per-document outcome; there is no partial success.

### Ingest And Replace

Ingest commits the handoff and runs LightRAG's pipeline under the Run's
`track_id` while it holds its slot. Each document is finalized, and so becomes
visible, as soon as LightRAG settles it, one at a time while the rest of the
batch is still in the pipeline. A replacement that retires another document's
identity finalizes after the pipeline call ends instead, since undoing its
failure deletes through LightRAG. A provider failure inside LightRAG's
per-document pipeline fails that document. A parser outage instead defers the
Run once every attempted document has settled; resumed, it retries only the
documents that did not become ready
([Parser Services](operations.md#parser-services)). A document that became
ready stays ready and is not processed again.

Replace is non-atomic and resumable. It places the new parser input, hides the
old document, and deletes it through LightRAG's public delete before it enqueues
the replacement. It removes the old document's parser files only after LightRAG
confirms the delete, and never reconstructs LightRAG document status from a
snapshot.

### Delete

Delete hides each document, by setting its finalization marker false, before it
calls LightRAG's public `adelete_by_doc_id`. `success` or `not_found` completes
the upstream leg; DlightRAG then removes its projections and the document's
source and parser files. A `not_allowed` rejection, which LightRAG guarantees
wrote nothing, restores the previous visibility and fails that document. Any
other or uncertain result waits for repair, so an ambiguous delete can lose
recall but never leaves a half-deleted document visible.

### Retry

Retry takes explicit document ids or `selector: all_retryable` and seals its
cohort in the checkpoint on first execution, so recovery never absorbs documents
that fail later. The cohort holds LightRAG `FAILED` documents and `PROCESSED`
documents DlightRAG has not finalized; the latter replay only their missing
finalization. A failed document is hidden, deleted through LightRAG's public
delete, and enqueued again from its retained source under the Retry Run's
`track_id`. Documents replay in shared pipeline passes of up to 64, with
outcomes that match a one-at-a-time retry. A document whose source is gone or
whose source metadata is incomplete fails at once. An uncertain outcome ends the
retry and waits for repair; resuming retries the same sealed cohort, which
converges.

### Corpus Reset

Corpus Reset (`reset`) is a FIFO barrier for its Workspace: earlier mutations
finish first, and later ones run against an empty corpus. It drops the
Workspace's LightRAG storage, DlightRAG metadata, remaining corpus rows,
promotion jobs, and ingest counters, and its corpus files except Run stages. It
keeps the Workspace's identity and access, Run history, Conversations, Agent
Sessions, and historical Artifacts. A reset that reports any error waits for
repair rather than succeeding.

### Workspace Delete

Workspace Delete (`delete_workspace`) is a Workspace's final mutation: a full
Corpus Reset, then removal of its catalog identity, then cancellation of every
mutation queued behind it. Removing the identity first stops new submissions
from joining the queue. Each step is idempotent, so recovery after the settled
reset repeats the rest; an ambiguous reset waits for repair with the identity
still registered. The deployment's default Workspace cannot be deleted.
Owner-scoped history (Runs, Conversations, Agent Sessions, historical Artifacts)
remains, and a Workspace created later under the same name starts empty.

## Cancellation, Repair, And Supersession

### Cancellation

Cancelling a queued Run ends it `cancelled` at once. Cancelling a running Run
sets `cancel_requested_at` and notifies the owning worker through a PostgreSQL
channel; the worker interrupts its executor, which also checks at its stable
boundaries, and commits the cancelled terminal. If the lease expires first, the
sweeper ends the Run. Cancelling a terminal Run changes nothing. A success or
failure commits only while no cancellation is pending; otherwise the Run ends
`cancelled`. Settling an Answer Run cancels its running Child Sessions and
retires its pending guidance requests.

Steer instructions enter an ordered inbox and are consumed at stable checkpoints
as durable `ControlMessage` entries. A steer or follow-up that races the
terminal transition starts a fresh linked Operation, and a Fork opens a new Lane
in the same Agent Session
([Interfaces](interfaces.md#run-lifecycle-and-answer-endpoints)).

A Corpus Mutation's cancellation closes at its handoff. Cancellation and the
handoff race on the Run row, and the first to commit wins; once
`handoff_started_at` is set, cancellation is rejected, including while the Run
is deferred or waiting for repair, and nothing rolls back. Cancelling a queued
mutation removes its stage.

### Repair

A destructive action (`replace`, `delete`, `retry`, `reset`, or
`delete_workspace`) waits for repair when its upstream outcome is ambiguous
after the handoff, or when it resumes after the handoff without a checkpointed
outcome, unless it is a replace that finds its upstream documents. A parser
outage needs no repair: ingestion reports it only once every attempted document
has settled, so the deferred Run resumes on its own.

A Run waiting for repair stays `running` with `phase=waiting_for_repair` and
exposes bounded `repair_reason` and `repair_remedy`. It releases its lease and
slot, keeps its Workspace's mutation barrier, and counts toward the admission
limit; it is neither failed nor replaced. An authorized operator repairs the
upstream state and resumes the same Run through REST, the Web, or the
Application `RunService`. Resuming records the confirmation and requeues the
Run, with the same id and `track_id`, at its place in the Workspace FIFO
([Operations](operations.md#repairing-an-ambiguous-mutation)).

### Supersession

When repair is inappropriate, a Corpus Reset or a Workspace Delete may name the
waiting Run as `supersedes_run_id`. Accepting the reset, in the same
transaction, fails the waiting Run with `repair_superseded` and records its
`superseded_by_run_id`; the acceptance is refused unless the named Run is that
Workspace's mutation waiting for repair. The superseding Run removes the
superseded Run's stage when it runs. Nothing unlocks a waiting Run while keeping
unknown corpus state. A Workspace Delete that names no Run waits behind the
waiting one like any later mutation.

## Retention

A Run's retention starts when it finishes: the terminal transition sets
`purge_after` from the retention the Run was accepted with.

| Run kind | Kept after finishing |
|---|---|
| Answer | `runtime.run_retention_days` ([default](configuration.md#runruntime-lanes-and-retention)) |
| Retrieval | seven days |
| Corpus Mutation | seven days |

Nonterminal Runs are never pruned. Every process with a coordinator runs the
retention pass hourly, after a small random start delay, in bounded
`SKIP LOCKED` batches that need no leader or cron. A pass trims the event logs
of expired Runs, then deletes their rows; in between, a trimmed Run answers its
event stream with 410 while its status still carries the result.

Deleting a Run cascades its events, turns, and artifact references. A blob goes
only when no reference of its owner survives, so blobs have no retention clock
of their own ([ADR 0010](adr/0010-separate-run-blob-plane-from-operational-state.md)).
An Agent Session goes when no routed Run references it. A pruned Answer Run's
Agentspace root (`workspace_root/<owner shard>/<run id>`) goes with it, and
an orphan sweep removes roots whose Run row is gone. The sweep follows the
configured root, not the execution mode, so turning execution off still
reclaims trees earlier Runs left.

The tree itself has a shorter clock than the Run. Nothing reads a finished Run's
Agentspace tree: its notes are the Session's rows, which the next Run materializes; a
published Artifact is a Blob its row pins, so the Answer and a later Run reach the
bytes without the tree; a spilled Tool result is read only by the Run that wrote
it, and a terminal Run never runs again. What is left is what the Run's processes
generated on the way — a script, a chart preview, a downloaded page, a fetched
library — so the same sweep that removes orphan roots removes the root of a Run
that has been over for seven days, while the row and its events keep the
retention above. A Run still working keeps its tree however long it has been at
it, and the Agentspace quota is what bounds one that grows: past
`WORKSPACE_MAX_ENTRIES` or `WORKSPACE_MAX_BYTES` a write is refused rather than a
directory emptied under an agent that is using it.

On a writer the same pass removes Corpus
Mutation stages no Run will read again: one whose Run has ended, and one with no
Run a day after its request began.

A Web conversation turn lives and dies with its Answer Run; conversations do
not extend retention. An hourly sweep deletes conversations left with no turns,
and a conversation reused after retention emptied it starts a fresh `main` Lane
and Agent Session. Deleting a conversation deletes its Runs in the same
transaction, so no worker can append to them afterwards.

## Workers And Scaling

Each process runs one coordinator. Every process runs Query workers; only a
writer registers the Corpus Mutation executor and runs its workers. Each lane's
scheduler reserves a local slot, then claims the oldest eligible Run with
`FOR UPDATE SKIP LOCKED`. With nothing to claim it waits up to one second or
until a local acceptance or completion wakes it, so a free slot picks up work
another process accepted within about a second. Processes add their slots
together, and row locks keep two processes from claiming one Run. A process
starts claiming only once its cancellation listener is listening
([Operations](operations.md#runruntime-and-durable-query-and-corpus-mutation-runs)).

Before it starts claiming, a process checks every active Retrieval and Answer
Run against its own configuration. An active Retrieval whose pinned models,
model catalogue, or context policy no longer match stops startup until it is
drained or cancelled. An incompatible Answer does not: it fails when claimed,
before any effect, and Research re-checks its pins before every provider and
Tool effect.

### Leases And Fencing

A claim sets a 300-second lease and increments the Run's fencing epoch. The
worker renews the lease every 30 seconds, and each fenced write extends it.
Every write a worker makes (event, checkpoint, settlement, terminal transition)
requires its own lease owner and epoch on a running Run, whether or not the lease
window has lapsed: only a reclaim, which bumps the epoch, takes a Run from its
owner. The first such write that matches no row means the lease is lost: the
worker stops, writes nothing more, and frees its slot. A heartbeat the store
fails to answer is retried, not taken as lease loss.

A free worker reclaims a `running` Run whose lease expired.
`durable_progress_version` advances with each fenced settlement: a model turn,
a compaction, an effect, a Fast stage, or a Corpus Mutation checkpoint. The
fourth consecutive reclaim without progress in between fails the Run as
`run_abandoned`, so a long Run that keeps settling progress survives more
restarts; the bound is fixed. The sweeper runs every second without a slot: it
ends cancel-pending Runs whose lease expired and abandons Runs past the bound.
Cancellation takes precedence over both reclaim and abandonment.

### Graceful Shutdown

On shutdown the coordinator stops claiming, interrupts its running work, and
waits up to five seconds for fenced writes already in flight. An owned Run with
cancellation pending ends `cancelled`; every other owned Run returns to
`queued` with its lease cleared and its durable state kept. This requeue does
not count as a reclaim. A Corpus Mutation never abandons an in-process LightRAG
call: shutdown waits for the call to return and records its outcome.

### Scaling Out

Deployment configuration owns the process count, and so total active capacity:
worker bounds limit each process, and admission limits the deployment. Leases,
claims, and Workspace order live in PostgreSQL, so processes on several hosts
share one primary. They share one POSIX `deployment.working_dir` mounted at the
same absolute path, and every process executing Research shares one
`answer.agent.workspace_root`, because a reclaimed Run resumes wherever it is
claimed ([Architecture](architecture.md#deployment-and-storage)).

DlightRAG owns Run lifecycle, admission, recovery, Workspace mutation order, and
health signals. Deployment infrastructure owns database, parser, provider,
ingress, and filesystem capacity; replica count, placement, and autoscaling;
TLS, rate limits, and DDoS protection; and backup and disaster recovery.
DlightRAG does not inspect an orchestrator or infer capacity.

### Load Evidence

`make load-runtime`, part of `make validate-runtime` and outside CI, submits
10,000 Query Runs (Retrieval over 1, 10, 50, and 100 Workspaces, Fast, and
Research) and 1,000 Corpus Mutation Runs spread over 100 Workspaces to fake
executors, in one process against one database, with 16 Query and two Corpus
Mutation slots. It passes only when every accepted Run is stored and drains with
exactly one terminal event, both lanes fill their slots, Workspace FIFO holds,
and the Corpus Mutation admission limit refuses a Run before inserting it,
within loose ceilings (900 s in total, 300 s to drain, 2 GiB of memory growth)
that catch collapse rather than set latency targets. It writes its measurements
and claim-plan `EXPLAIN` output to `.test-results/load-runtime/`. It does not
exercise the Query admission limit, several processes or hosts, provider
throughput, or real parsers.
