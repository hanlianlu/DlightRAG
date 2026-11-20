# DlightRAG Domain Language

DlightRAG is a durable agent — frontier RAG and memory in one runtime — joining reusable AI, agent, and corpus capabilities without merging their ownership.

## Product Surfaces

**Application**:
The local in-process DlightRAG product surface that exposes Answer, retrieval, and corpus-administration capabilities and owns their shared lifetime.
_Avoid_: Manager, service locator

**Answer**:
The default owner-visible terminal prose of an Answer Run, shown in the conversation surface. It directly satisfies the request unless a separate deliverable is warranted; when a Published Artifact carries the complete deliverable, the Answer is a concise orientation and handoff rather than a second copy.
_Avoid_: Answer Run, Published Artifact, parallel report body

**Run**:
A durable submitted operation with one common identity, lifecycle, event log, cancellation path, and terminal result. `run_kind` and `lane` are data on this record rather than subclasses.
_Avoid_: job, task, `QueryRun`, per-operation lifecycle record

**RunRuntime**:
The single operation-neutral lifecycle coordinator and store contract. It dispatches Retrieval and Answer executors on the Query Lane and Corpus Mutation executors on the Corpus Mutation Lane by `run_kind`.
_Avoid_: Query runtime, Answer runtime, Ingest Job coordinator, generic workflow engine

**Query Lane**:
The execution and capacity lane for retrieval and answer kinds. It is not an aggregate, class, module, or separate runtime.
_Avoid_: QueryRun, query service

**Corpus Mutation Lane**:
The bounded execution and admission lane for Workspace-scoped ingest, replace,
exact delete, retry, reset, and Workspace Delete Runs. Runs are FIFO within one
Corpus Workspace and may execute concurrently across Workspaces; the lane has a
separate deployment-wide nonterminal admission limit from the Query Lane.
_Avoid_: Ingest Job queue, per-action coordinator

**Corpus Mutation Run**:
A Workspace-scoped Run with `run_kind=corpus_mutation` and
`lane=corpus_mutation`. It keeps a stable `track_id`, records the upstream
handoff before non-idempotent corpus effects, and retains its terminal row for
at least seven days.
_Avoid_: Ingest Job, corpus task, per-action lifecycle

**Answer Run**:
An owner-scoped Run with `run_kind=answer` and `lane=query`.
_Avoid_: separate Answer lifecycle, request

**Answer Service**:
The product capability that validates and accepts Answer input and owns Answer-specific steer, follow-up, fork, transcript, child roster, and terminal projection. Common status, events, cancellation, and listing belong to the Run service.
_Avoid_: Answer manager, transport-specific runtime service

**Continuation**:
A new durable Answer Run with `parent_run_id` and kind `follow_up` or `fork`. Follow-Up is a line operation and appends to the selected Lane's tip; Fork is a tree operation and opens a new Lane and Web Conversation at its Fork Point. Neither re-injects context its branch already carries.
_Avoid_: in-place run mutation, session checkout, hidden conversation copy, rewind to an arbitrary Entry

**Fork Point**:
The exact state a Fork branches from: the Lane head one Run settled at, plus the Context Projection active at that head, both recorded at that Run's terminal settlement whatever its outcome. It is neither the Lane's current head nor a Stable Checkpoint, so forking an earlier turn branches from that turn rather than from the conversation as it now stands.
_Avoid_: current Lane head, Stable Checkpoint, latest conversation state, Agentspace Epoch

**Answer Feedback**:
One owner's judgement of one message of one agent of an Answer Run: a thumb up or down, and for a thumb down why it was given — the reasons it gives, from a closed set, or a note in the owner's own words. It hangs off the Run and is keyed by the agent (`main` for the Run's own, otherwise a Child Session) and the message's own place in that agent's Session, which never moves, so judging a message again replaces the earlier judgement instead of adding a version. Only an assistant message the Run itself wrote can be judged, and deleting the Run deletes its feedback. It is the owner's own state, like history and Memory, so no Access Rule gates it; whether an owner is offered it at all is the `answer_feedback` Runtime Policy.
_Avoid_: rating history, review, per-answer version

**Conversation Contribution**:
An owner's permission for their conversations to be used to evaluate and train models (ADR 0041). It is off until they turn it on, and it is one fact, the moment it began: turning it off clears the moment and ends the permission at once, and turning it on again begins a new one. Its unit is the Agent Session, because the Session carries the earlier text of every Run in it. The owner's word for a Session stays "conversation": a Web Conversation and everything forked or followed up from it, on the Web or over REST or MCP, share one Session, and a Run submitted without a parent starts a new one. A Session starts at the earlier of its own creation and the acceptance of its first remaining Run, so that neither the wait before a Run is claimed nor retention pruning its oldest Runs moves the start later; the Run's Session is the one its routing row names, never a Child's. A Run is covered when its owner contributes now and its Agent Session started at or after that moment, so the permission covers no conversation begun earlier, even one that goes on or is forked afterwards, and none begun across a turn-off and a turn-on. A covered conversation is covered with what it brings in: an attachment, the history a REST caller sends with a request made after the moment, and a Profile Memory the agent recalls. It is the owner's own state, like Answer Feedback and Memory, so no Access Rule gates it. The operator's trace export honours it, reading only Runs of owners who contribute and only Sessions that started at or after the moment. It is not a monitoring switch: monitoring has no off state and its traces keep bounded conversation text, which this setting does not change (ADR 0039).
_Avoid_: Context Contribution (what a Run adds to the model's context), per-Run permission, per-Web-Conversation permission, monitoring or telemetry consent, opt-out, the `answer_feedback` Runtime Policy

**Training Dataset**:
What the operator's export packs from the conversations their owners contributed (ADR 0044): `train.jsonl`, `eval.jsonl` and a `datasheet.json` that says what is in them and what is not. It is a full re-export of whole Agent Sessions, never an increment, so an owner who turns Conversation Contribution off is absent from the next one. A line is one Run, an episode: its steps, its Children, the sources its answer cites, the owner's judgements of its answers as they gave them, and the context it ran on that no exported Run wrote (`carried`). Every owner, Session, Run, Entry and Child id in it, and the ids minted for a tool call, a compaction, a steer and a fork's lane, is an HMAC-SHA256 under the operator's key; the text is not scrubbed. A Session goes to `eval` or `train` by a hash of its stored id, so exporting more never moves one.
_Avoid_: backup, archive of deleted conversations, the raw export (one record per Run, ids as stored, for the operator to inspect), preference pairs

**Retrieval**:
An owner-scoped Run with `run_kind=retrieval` and `lane=query` that returns corpus Evidence without generating an Answer. Top-level Retrieval is durable; retrieval inside an Answer is an internal Retrieval Stage, never a nested Run.
_Avoid_: Fast Answer, inline Retrieval, `QueryRun`

**Fast Answer**:
A durable Answer Run that plans, retrieves, and generates without an Agent Operation, tools, skills, or publication. It shares Context Contribution, Evidence, citation, the Agent Session, an inert Agentspace (to hold the Session's notes), and Profile Memory recall. Its Host turn still commits User and Assistant Entries to the routed Agent Session.
_Avoid_: Retrieval, non-durable fast path

**Web Conversation**:
The browser conversation surface that creates Answer Runs; it is a transport, not a research capability. The thread shows the run's terminal answer and opens its Published Artifacts in Artifact Canvas.
_Avoid_: Web Search, Web, Web Channel when meaning public-Web retrieval; chat column as the report

**Web Search**:
The optional capability that discovers open-Web sources and admits their passages as Evidence inside an Answer Run. It is not the direct reading of a known Web Resource.
_Avoid_: Web UI, Web Conversation, Web Channel, URL read

**Answer Mode**:
The caller-facing selector `auto | fast | research`. Omitted means `auto` on REST, MCP, Web Conversation, and Python.
_Avoid_: research bool, inferred path, Web Search as a mode

**Agent Effort**:
The caller-facing three-level selector `low | high | max` for one Answer Run's own
answering agent. Omitted means the deployment's configured level for that role, and
a subagent keeps the level configured for its own role, under the Run's **Effort
Ceiling**. The Web control offers only the levels the answering profile can express and
the caller's ceiling leaves; a chosen level below that model's
ladder is clamped by reasoning resolution rather than refused, and a model that
names no non-off level ignores it. A chosen effort never fails a Run: the accepted
Run's stored `effort` is the choice, and the run trace's `agent_effort` states the
level that ran or why the choice was ignored.
_Avoid_: reasoning level (the deployment setting), Web Search effort (`fast | balanced`), Answer Mode

**Effort Ceiling**:
The highest effort any agent of one Answer Run may run at: `low`, `high` or `max`, where `max` is no ceiling. An administrator sets it for the callers the `agent_effort_ceiling` Runtime Policy names, it is read once when the Run is accepted, and the Run carries it as `effort_ceiling` beside the caller's own `effort`, so a later edit never reaches a Run already accepted. Every agent of the Run, its own and each subagent, runs at the lower of the level it would have and the ceiling, on its model's ladder: the highest level that model offers at or below the ceiling. A choice above the ceiling is lowered, never refused.
_Avoid_: effort limit, effort quota, maximum reasoning, the Agent Effort the caller chose

**Active Run Limit**:
How many Answer Runs one owner may have queued or running at once, set by the `active_answer_runs` Runtime Policy and counted when a new Answer is accepted. A Run deferred for a dependency is queued and counts; a repeated submission key is answered even at the limit; and lowering the limit cancels nothing. Under `none` and `simple` authentication every caller is one owner, so the limit is the deployment's.
_Avoid_: admission limit (the lane's deployment-wide cap, `runtime.*.max_nonterminal_runs`), rate limit, concurrency limit

**Valid Mode Set**:
The capability-derived subset of `{fast, research}` that one Prepared Input may legally resolve to.
_Avoid_: requested mode, router guess, heuristic from Web Search being configured

**Resolved Mode**:
The durable `fast` or `research` value written once after routing; crash recovery does not recompute it.
_Avoid_: Answer Mode when the request was `auto`

**Corpus Administration**:
The product capability for Corpus Workspace lifecycle, ingestion, files, metadata, visual assets, and reset. Product mutation acceptance belongs to the Corpus Mutation Service; common observation and cancellation belong to Run Service.
_Avoid_: Workspace manager, direct WorkspaceRag call from an interface

**Upstream Handoff**:
The durable Corpus Mutation boundary committed before a destructive or otherwise
non-idempotent LightRAG effect. Recovered work after handoff proceeds only after
authoritative public reconciliation or explicit repair confirmation.
_Avoid_: checkpoint-only hint, private status mailbox

**Waiting For Repair**:
The nonterminal `waiting_for_repair` phase of the same Corpus Mutation Run when
public reconciliation cannot prove a safe repeat. Public `repair_reason` and
`repair_remedy` guide an authorized operator, who repairs upstream state and
explicitly resumes that same Run.
_Avoid_: failed replacement Run, automatic destructive replay, new retry job

**Reset Supersession**:
An authorized full Corpus Reset may explicitly supersede one
`waiting_for_repair` mutation. The old Run becomes terminal with its superseding
Run identity; reset preserves Corpus Workspace identity and history rather than
recreating the Workspace. A Workspace Delete is a full reset too and may
supersede the same way before it removes the Workspace.
_Avoid_: partial reset, hidden repair abandonment

**Workspace Delete**:
The Workspace's final Corpus Mutation (`delete_workspace`): a full Corpus Reset
that retains no sources, then removal of the Workspace's catalog identity and
cancellation of every mutation queued behind it. Owner-scoped history (Runs,
Conversations, Agent Sessions, historical Artifacts) remains. The deployment's
configured default Workspace cannot be deleted.
_Avoid_: Corpus Reset, identity-preserving reset, synchronous registry delete

**Authorized Workspace Set**:
The concrete canonical Corpus Workspace ids a caller may use after authentication and access expansion are complete.
_Avoid_: Workspace selector, raw claims

**Access Rule**:
One deployment-configured allow mapping from a verified JWT claim value to
Workspace patterns and Action patterns. Rules are not durable user memberships.
_Avoid_: Database role, Workspace membership, deny policy

**Workspace Creator**:
The owner that created a Workspace, recorded with its catalog identity. Once
Access Rules apply it holds the `common_user` actions, reset, and delete on that
Workspace beyond what they grant; a Workspace the deployment registers itself has
none.
_Avoid_: Workspace owner (an owner scopes Runs and history), Workspace membership

**Action Preset**:
A named expansion (`common_user` or `admin`) into product Actions inside an
Access Rule. There are two, so an Access Rule states a place in the deployment
rather than a role; a grant narrower than `common_user` names the Actions
themselves. A preset is not a stored role assignment and never bypasses the
rule's Workspace match.
_Avoid_: Role record, user role, permission group

## Corpus And Research

**Corpus Workspace**:
A named, authorized corpus scope whose ingestion and retrieval state is owned by one corpus runtime.
_Avoid_: Agentspace (the agent's own files), agent filespace, agent workspace; bare Workspace (工作区) is this term's reader-facing word

**Product Document**:
One directly visible corpus consistency unit: a LightRAG document plus required
DlightRAG metadata/source locator, BM25 labels, retained source/sidecar effects,
and enabled visual fusion. It is published only while
`_dlightrag_finalization_complete` is exactly true. A LightRAG `PROCESSED` row
whose product finalizers are incomplete stays upstream `PROCESSED`, remains
hidden through this marker, and retries only the same idempotent finalization
path; DlightRAG does not rewrite upstream status. Shared graph summaries remain
eventually consistent rather than document-snapshot isolated.
_Avoid_: LightRAG processed row, metadata row, graph snapshot

**Agent Session**:
The owner-scoped durable parent-linked Entry Tree shared by routed Fast and Research Answer Runs. Stable Lane heads select branches, and the tree remains durable only while at least one Answer Run routing row names it; a Web Conversation identity alone is not a second history authority.
_Avoid_: Answer Run, linear journal, thread

**Agent Session Repository**:
The narrow Host-facing read/transaction seam for immutable Session snapshots. Runtime mutations go through exact-register CAS transactions; Hosts do not receive append/fork/archive storage shortcuts.
_Avoid_: generic store, lifecycle hook registry, mutable transcript

**Child Session**:
A first-class Agent Session inside one parent Answer Run. `spawn_agent` durably accepts it into the Subagent Roster and returns its handle at once; the Child then runs asynchronously beside its parent until it settles, and the parent follows it through status, wait, read, and roster observation. It has explicit parent/call lineage, ContextSnapshot, bounded depth and concurrency, pinned plan/model/tool allowlist/budget/Host state, and an independently renewed lease. Its tool allowlist is its parent's composition minus the Run's authority — roster controls, durable owner memory writes, and publication — while an explicit `tools` list only narrows it and never fails the spawn (ADR 0025). Targeted steer applies only to its current Operation; continue starts a new Operation on the same Child Session; `ask_parent` is one correlated guidance wait until reply, expiry, or cancellation; observation is the public transcript, control, question, and result projection.
_Avoid_: child run, detached job, mission, background swarm

**Subagent Roster**:
The owner-visible status, observation, targeted steer/continue/cancel, and guidance-question projection for Child Sessions created by `spawn_agent`. It carries when each Child's current Operation started and finished and how many of its questions still wait. Children run asynchronously after dispatch; Child Evidence is admitted into the parent ledger before the parent effect settles.
_Avoid_: Delegate Research, workflow engine, child queue

**Prepared Input**:
The immutable, bounded execution description accepted for a new Answer Run after authorization, capability resolution, and profile pinning. Resolve never rewrites it.
_Avoid_: Request payload, checkpoint

**Routing Record**:
The Answer-owned one-to-one durable mapping from an Answer Run to requested/valid/resolved mode and its canonical Agent Session/Lane. It is the transcript authorization anchor and does not make Answer Run and Agent Operation identities equivalent.
_Avoid_: Prepared Input, second policy source, inferred research bool

**Public Request Fingerprint**:
The digest of the normalized caller request used for owner-scoped idempotency before Prepared Input is built. Canonical Answer Mode is part of that digest: omitted equals `auto`.
_Avoid_: Prepared-input hash

## Goal Mode

**Goal**:
The one durable objective an Agent Session carries across goal rounds. A goal is `active` while the
platform may continue it, `paused` when a human stopped it (a pause is immediate: it cancels the goal
round already running), `blocked` when a concrete condition stops it, and `complete` when the Agent
has achieved the objective; one conversation
carries one goal at a time, and a completed goal may be replaced by a new one. Whenever the Session
is idle with an `active` goal and rounds remaining, the harness starts the next round itself. A
process restart does not park it: in the shipped deployment the process that serves the conversation
runs goal recovery, which reads active goals back from their durable records and continues them
(startup and about once a minute), including a goal with no answer Run to continue from. Continuation
therefore does not wait for a human re-arm, and there is no activation state separate from the
durable phase: `active` is the whole permission to continue. The Agent decides completion and no evaluator checks the
analysis; the one machine gate is delivery evidence: a loaded sealed Skill whose required read was
skipped, whose verifier did not pass, or whose seal was refused makes `complete` a refusal and
leaves the goal active ([ADR 0060](adr/0060-goal-mode-is-a-harness-feature-skills-declare-checkpoints.md), [ADR 0077](adr/0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md)).
_Avoid_: task, todo, plan, objective stack, armed goal, activation state

**Goal Round**:
One Answer Run that continues an active Goal: the harness admits it when the Session is idle and the
round cap has room, and it carries the objective, the round number and the cap. A round is a budget
boundary, not a context reset: it continues the Session's committed Agentspace, its Session Notes,
its published products and its history, so a round starts by inspecting state rather than redoing an
earlier round's work. The round that records `complete` or `blocked` answers with the closing
message ([ADR 0071](adr/0071-a-goal-round-is-a-same-lane-continuation.md), [ADR 0077](adr/0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md)).
_Avoid_: auto-continue, retry, loop, fresh context, new run for the same work

**Goal Bar**:
The conversation surface that shows a Goal's phase, rounds, objective and blocked reason, and offers
the phase's controls: pause while active, resume for paused or blocked, edit, clear. Pause is
immediate: it stops the goal now and cancels the goal round already running; a human turn already
running is not cancelled, and a paused goal waits for a human resume. A complete or blocked Goal
shows one neutral terminal line: complete
means no further round will start, blocked says to resume (or to raise the round cap when the cap
stopped it) and keeps its reason. A refused goal command shows the server's specific reason rather
than a generic failure.
_Avoid_: goal widget, status chip, inactive goal

**Goal Command**:
A `/goal` message the reader writes in the composer: it changes the conversation's Goal without
being sent to the model as a turn. Creating one starts round 1 at once, so it works in a fresh
conversation with no earlier Run.
_Avoid_: slash command, admin action

## Long-Term Memory

**Memory Record**:
An owner-scoped, non-citable remembered item that may be recalled across separate conversations. Its `kind` says whose statement it is and therefore how it is recalled: a `preference` is what the owner wants of an answer, a `fact` is what the owner is like, and an `experience` is what the agent has learned to be true about its own work. A Session Memory Pass may contribute one without a tool call and without the owner having asked. A record the pass wrote may also carry its **quote**: a short verbatim excerpt (at most 120 characters, one line) of what the owner said in the part of the conversation the pass judged, in the owner's own language, which recall shows on a second line under the memory (`  owner said: "…"`) so the Agent reads the memory by the words it was drawn from. The host keeps a quote only if it is, apart from whitespace, a stretch of one of the owner's turns in that part, so a record never carries words the owner did not say. A record drawn from something the owner did not say (an experience the Agent learned alone), and every record the Agent's tools, REST or MCP wrote, has none. A quote belongs to the write that made the record's current form: an update or a fuse carries its own, not its sources'. It is never Evidence and never a citation source.
_Avoid_: Compaction Summary, Journal Entry, PriorTurns, Evidence, Artifact, remembered citation

**Memory Subject**:
The owner identity a Memory store scopes every operation to; bound by the host (DlightRAG: the JWT owner, or the deployment owner under `none` and `simple`; stdio MCP: `--subject`) and never accepted from a model tool argument.
_Avoid_: caller-selected namespace, conversation id, Agent Session

**RecallResult**:
The structured outcome of one query-aware Memory recall, in two sections sharing one character budget. The relevant section — records of any kind that the query's search found, at most `RECALL_TOP_K` and at most half the budget — is filled first; the standing section — preferences and experiences, newest first — takes the rest, leaves out any record the relevant section already shows, and reports how many standing records it could not show. Each section reads oldest first, and the budget counts rendered lines, labels included. A lookup also carries the newest other memories, so the Agent can correct a record recall did not return. A record appears at most once; duplication of meaning is repaired at write time, never filtered at read time. Never a prompt fragment.
_Avoid_: prompt text, standing block, packed string

**Memory Operation**:
One host-bound, owner-scoped remember, forget, undo, or fuse request carrying a stable idempotency key and trusted provenance. The storage seam validates, settles, and journals it atomically; a host may impose an atomic mutation limit under its own scope value without changing the package interface.
_Avoid_: Silent promotion, transcript scan, model aside, adapter-owned mutation

**Memory Fusion**:
One atomic Memory Operation that marks several active records superseded and writes one record replacing all of them, the whole group naming the operation that produced it. It exists because a semantic merge is not a supersession: `supersedes_id` names one predecessor, and a merge has several. Its body is supplied, never invented by the store: a model words it inside a Session Memory Pass, where the verdict names the records and returns the replacing sentence, and a person words it when merging from Settings or a management call. Its compensating Undo restores its sources, and is refused when another active record has since taken one of their bodies. It is never an Agent tool — that is, a model is never handed the move of collapsing several records at once; the Session Memory Pass applies a verdict, and the owner may ask for a merge.
_Avoid_: supersession chain, chained forget, last-writer-wins overwrite, confidence score

**Memory Operation Vocabulary**:
The one name each Memory Operation carries across every surface, so a reader maps `remember`, `forget`, `undo`, `fuse`, `recall`, `lookup`, `browse` and `clear` once rather than once per host. A tool takes the name of the package method it calls, prefixed `memory_`, and the Agent's tools and the MCP server's tools use the same names: `memory_remember`, `memory_forget`, `memory_undo`, `memory_lookup`, and on MCP alone `memory_fuse`. The read tool is `memory_lookup` because `lookup`, not `recall`, is what returns the identities a correction needs. REST names resources rather than methods. Which surfaces expose which operation is a separate question from what it is called, and an operation a surface withholds is recorded as withheld rather than renamed.
_Avoid_: per-host synonym, aliased tool, second name for one operation

**Memory Undo Addressing**:
How an Undo names its target. A caller that holds a change id names it; the Agent holds record identities rather than changes, so its `memory_undo` names an active record by `memory_id` or exact body and the store resolves that to the change that produced the record's current form. Undo compensates one change, not a history: an undo cannot itself be undone, and a compensation writes the restored sentence as a new record. A named record that is no longer active is a conflict rather than a guess about which earlier change was meant.
_Avoid_: undo stack, history rewind, revision number, change id in a tool result

**Memory Operation Receipt**:
The replay-stable result of one Memory Operation: change identity, action, outcome, affected record identities, safe body, provenance, and supersede/undo links. Answer projects it into a durable product event; Agent Core never interprets it.
_Avoid_: parsed tool prose, telemetry payload, UI-only notification

**Memory Operation Journal**:
The package-owned owner-scoped ledger that makes operation replay, changed-input rejection, mutation caps, optimistic conflicts, and compensating Undo one transaction with record transitions.
_Avoid_: Answer Event Log, Agent Journal, browser cache, out-of-band undo snapshot

**Memory Record Lifecycle**:
`active → superseded` or `active → forgotten`. Forget is idempotent and leaves a non-recallable tombstone; Undo is a new compensating operation rather than a reverse state transition. Non-active history follows the shared retention floor. Clear Profile Memory is the explicit exception: it physically removes that owner's package records and operation journal without rewriting Answer or Conversation history.
_Avoid_: state rollback, silent transcript promotion, confidence score

**Retention Floor**:
The selected per-Run clock bounding terminal Run rows and event logs. Answer uses the configured selection (365 days by default), which also bounds routed Agent Session history and superseded Memory history; top-level Retrieval and Corpus Mutation each use seven days. Reclamation may happen later, never earlier.
_Avoid_: deadline, SLA, per-aggregate TTL, inactivity expiry

**Conversation History Page**:
One sealed turn-number keyset page from Web history. It bounds one read, not retention or exact model recovery.
_Avoid_: snapshot, max_turns, trim window, retention window

**File Panel Page**:
One workspace-bound sealed keyset page of finalized Product Documents whose
LightRAG status is processed, in durable server order. It bounds
presentation/read work, not inventory or retention; failed-file administration
is a separate repair view.
_Avoid_: full file snapshot, OFFSET page, workspace catalog scan, retention window

**Session Memory Pass**:
The background consolidation of one Agent Session's conversation into Session-contributed Memory Records. A successful Answer Run of the Session — committed by the coordinator or, for Fast, by the executor itself — triggers it; it reads the main-Session entries past the Session's watermark, however few, and reads on in batches under its one claim until it has caught up with the Session, so no new text waits for a later Run. For each batch a `query`-role call at the reasoning level `memory_consolidation.reasoning` sets (low by default) decides add, update, fuse, or drop against the owner's existing records, and one such call adds at most `session_record_limit` records; updating or fusing records the owner already holds never counts, whichever Session wrote them. Each call is also shown, as context only, the last five rounds of the Session just before its batch (user turns whole, assistant turns cut to their first 500 characters), which an earlier pass already judged, and never text from a period when the owner's Profile Memory was off. It reads only the Session's own user and assistant turns, never a tool result, a Child Session, a Child's notification (the result or question a Run hands its agent as a user message), a Session Note, or an injected recall block. It is not the Agent: it holds no tools and produces a verdict, which it stores before applying so that a retry resumes the verdict rather than asking the model again. Zero records is its ordinary outcome, and it never fails the Run that triggered it.
_Avoid_: automatic injection hook, transcript promotion, periodic summarizer, second transcript, Session Note, memory daemon

**Session Memory Sweep**:
The low-frequency look, by each process's pass runner, for Agent Sessions whose Session Memory Pass was lost: shortly after the process starts and every ten minutes after, it asks the Run store for at most twenty Sessions whose watermark is behind their own unread text and hands each to the same path a successful Run takes, attributed to the Session's newest succeeded Answer Run. The database is the queue: a Session is pending when it is a main Session with a succeeded Answer Run and no live one, has a user or assistant turn with text (not a Child's notification) past its watermark, belongs to an owner whose Profile Memory is on, and is neither held by a pass nor backing off after failures. It recovers what a restart cancelled, a failed model call, a failed or cancelled Run's text and a pass that stopped at its batch bound; a pass a successful Run schedules does not wait for it, and ignores the backoff.
_Avoid_: retry queue, dead-letter queue, reaper, cron job, dirty flag

**Session Memory Pass Provenance**:
The `origin_kind` value, `session_pass`, that marks a record the background consolidation wrote rather than one the Agent's tools or the owner's management calls wrote. It exists so an inferred record is distinguishable from one the owner asked for, in the Settings list, in the operation journal and in any export, and it is what the Agent's `answer_run` provenance is not. The record also names the Run whose success scheduled the pass (`run_id`), so the conversation can show it under that Run's answer beside what the Agent itself wrote.
_Avoid_: agent memory, inferred flag, background memory, confidence score

**Session Memory Watermark**:
The one progress row a Session Memory Pass keeps per Agent Session, cascading with it: the entry `sequence` it has read through, the claim that lets one pass at a time run across processes, a stored plan awaiting application together with the Run it is attributed to, and the context floor, the entry at which the owner last turned Profile Memory back on, below which no turn is ever shown to a model as context. The row also counts the passes that failed since it last moved the watermark and the time before which a Session Memory Sweep leaves the Session alone (ten minutes after the first failure, doubling to six hours), both cleared when a pass moves the watermark. Turning memory back on moves every Session of the owner's watermark and floor to its latest entry together, never back, leaving a held claim and a stored plan alone, so nothing written while memory was off is read at all. What lies past it is read from the entries each time, never kept as a count, and a pass moves it batch by batch to the end of the Session. It bounds what a pass judges, so no text is judged twice (the turns before a batch are shown again, as context only); because entries are removed only with their whole Session, it never points into a gap.
_Avoid_: cursor, checkpoint, offset, compaction position

## Web Conversation Streaming

**Event Cursor**:
The client's consumed position in one Answer Run's event stream, carried across reconnects. It bounds replay, not retention, and never outlives its run.
_Avoid_: Conversation History Page, offset, bookmark, retention marker

**Attachment Lease**:
The exactly-once settlement of a locally attached file's ownership when a submission is handed to an Answer Run: accept keeps it with the stored turn, restore returns it to the composer, discard releases it. A lease is never shared between submissions.
_Avoid_: upload, blob registry, attachment copy

**Optimistic Turn**:
The local turn a Web Conversation shows while a submission handshake is in flight. It is swapped for the stored turn on acceptance and dropped on failure.
_Avoid_: placeholder message, fake reply, client-side Answer Run

**Turn Projection**:
The browser-side fold of one Answer Run's event stream into a presentable turn view. Replaying the same events yields the same view.
_Avoid_: answer parser, incremental rendering state, second transcript

**Tool Activity**:
The ordered viewer-side rows describing the Tool calls of one Answer Run while it runs: one verb, the Tool Subject the call reports, its state, and its elapsed time. It is a live affordance folded from the run's Tool events, not a durable record of its own; a reader that arrives after the run keeps no rows.
_Avoid_: Tool log, execution history, run transcript

**Agent Activity**:
One agent's steps as the Web shows them in Agent traces: what it said, each Tool it called beside what came back, what it was told, and — where its transcript states it — the model's own reasoning for the message behind a step. It is projected from the agent's durable Session transcript, so unlike Tool Activity it survives reloads and reaches the whole past. The Web reads it a page at a time, newest page first, each message carrying the Session sequence that is the cursor of the page before it; a reader holding a transcript asks instead for what landed after its last message, so an open pane appends rather than read the same page again. A Run's main agent has one for the Run it answers; a conversation's Lane, earlier turns included, is what the conversation projects, and each Child Session has its own. A step's reasoning is how the step was reached rather than the step: it is masked through the reader content gate and bounded before the page states it.
_Avoid_: Tool Activity, execution history, second transcript

**Tool Subject**:
The single bounded thing a Tool call reports it is acting on, so a viewer can name the call while it runs: a query, an Agentspace path, a command, a pattern, a skill name, an anonymous public URL, or a run-scoped Resource Handle. It is a producer-reported fact rather than a display string, it is one line and bounded by construction, and a Tool whose only candidate is prose rather than a name reports none.
_Avoid_: arguments, Tool Label, display name, `object`

**Wire Format**:
The snake_case field spelling of REST and SSE payloads at the transport boundary. It is translated exactly once at the consuming edge and never appears beyond it.
_Avoid_: DTO leak, dual naming convention, camelCase server contract

## Session Entries And Effects

**Session Entry**:
An immutable semantic fact with one physical `parent_entry_id` in an Agent Session Tree. The closed variants are UserMessage, AssistantMessage, ToolResult, ControlMessage, and Compaction; transaction sequence records commit order but never defines ancestry.
_Avoid_: generic journal row, custom entry, mutable message

**Agent Session Tree**:
The immutable Entry set plus stable Lane heads and Lane state. Branch ancestry follows parent links; Fork creates a new Lane at one Run's Fork Point without copying or deleting shared Entries.
_Avoid_: linear log, DAG merge, navigation operation

**Stable Checkpoint**:
The one property a Lane head must have before a branch may open on it: no provider Tool Call anywhere in its ancestry lacks its Tool Result. It says a branch never starts inside an unfinished Tool batch; it says nothing about where the branch starts.
_Avoid_: Fork Point, Lane tip, turn boundary, compaction boundary

**Context Projection**:
The bounded model-facing projection of one selected Lane ancestry. Exactly one active branch-local compaction summary precedes the retained suffix; historical summaries remain immutable audit facts and never Evidence. Evidence text admitted before the retained suffix, and every committed spill or Session Note, survive as their recorded handles rather than as re-rendered content.
_Avoid_: authority, checkpoint, transcript snapshot

**Context Contribution**:
A typed model-ready contribution with its source, authority and messages. It unifies projection ordering without merging the contributor's storage ownership.
_Avoid_: global prompt registry, second state authority

**Compaction Summary**:
Typed continuation memory for one contiguous branch-ancestry prefix, validated and rendered by the framework but never treated as citable Evidence.
_Avoid_: Free-form checkpoint, evidence summary, Memory Record

**Effect Intent**:
A durable declaration of one validated tool operation and the replay contract under which it may execute.
_Avoid_: Tool call when referring to durable recovery identity

**Effect Settlement**:
The atomic durable outcome of an Effect Intent, including its ordered result and any host-owned updates.
_Avoid_: Tool result when referring to the full commit boundary

**Read-only Tool**:
A Tool whose own declaration says a call changes nothing outside its Run's record of what it read. Adjacent read-only calls of one Tool batch run at once, then admit their evidence and settle in source order. Every other call runs alone and is a barrier between them. Nothing infers the declaration from a name or from a remote server's hint.
_Avoid_: parallel-safe tool, readOnlyHint, side-effect-free when referring to the declared contract

**Durable Progress**:
A monotonically increasing run fact advanced only by live fenced execution boundaries that recovery must not repeat.
_Avoid_: Heartbeat, phase, turn count, recovery interrupt, Agentspace Epoch handoff

**Fast Stage**:
A deterministic Answer Run progress boundary inside Fast. Fast shares the Agent Session Entry Tree through a HostTurnReservation but creates no Agent Operation.
_Avoid_: Agent step, retrieval request, Session replacement

## Evidence And Resources

**Evidence**:
Citable, run-scoped source material with durable identity and content/locator integrity. Within one Answer Run it reaches the model inside the Tool result that admitted it, frozen at that position; source handles recorded by a compaction keep text Evidence re-readable.
_Avoid_: Summary, agent prose

**Web Resource**:
A run-scoped public HTTP(S) source admitted from a caller, Web Search, or an Agent-selected URL and represented by a Resource Handle. Admission origin and acquisition method are independent provenance facts; the Agent Browser's acquisitions are `browser_render`, `browser_capture`, and `browser_download`. Within one Answer Run, the same normalized URL resolves to its first successfully admitted durable snapshot rather than silently refetching mutable content. A Rendered Read appends a second representation to it without replacing that snapshot, while a Browser Capture or a browser download is a Resource of its own. Raw Bash output is model context, not Web Evidence.
_Avoid_: Web Search result, URL attachment, raw URL, Bash output

**Held Image**:
Pixels an Answer Run itself holds as a Resource -- a picture the Agent viewed on a page it fetched --
as distinct from a visual asset retrieval selected inside a cited corpus chunk. A
held image is placed in the Answer only where the Model names it, with `![label](evidence:<resource id>)`;
placement is therefore the Model's judgement about what a reader should see, while the address is the
transport's, composed from the Run and the Resource and never from anything the Model wrote. Its pixels
are an observation of what the Agent looked at, not a citable source: a claim still cites the page's own
text, and placing a picture does not make it evidence for a sentence.
_Avoid_: screenshot when the image is a viewed picture, Evidence Image, citation

**Rendered Read**:
A `read` of a Web Resource through the Agent Browser. Its representation is appended to the same Web Resource, with acquisition `browser_render`, and never replaces an admitted snapshot; the Resource keeps one handle, and a cursor names the representation it continues. The model asks for one with `rendered=true`, or the Extract chain's browser step makes one when the direct fetch failed or held no text. Its text is the browser's assertion of the page after its scripts ran, not an attestation that an anonymous GET serves the same page.
_Avoid_: screenshot, page capture, snapshot replacement, hosted extraction

**Browser Capture**:
A Web Resource admitted from the Agent Browser's current page (acquisition `browser_capture`). Browser downloads are admitted as Resources with acquisition `browser_download`. Each capture is its own Resource and never rebinds its URL's snapshot, because the page it holds is the one the Agent drove to, which a fetch of its URL may not reproduce. Its citation is the page's final public URL, or its Resource Handle when the page has none.
_Avoid_: screenshot, Rendered Read, URL snapshot, page cache

**Resource Handle**:
An owner/run-scoped identity through which prepared, fetched, evidence-backed, spilled, or published content remains addressable across recovery; it is opaque except for a Published Artifact's deterministic address (`artifact-<hash of its Artifact path>`), which is derived on purpose so the Tool can name it before the publication exists. A later Run on the same Agent Session may adopt an earlier Run's Resource on first use; the adopting Run then holds the canonical handle, the earlier handle stays readable only as its alias, and the adopting Run never converts newly adopted bytes: a convertible document reads text only through the conversion view stored with it, and other formats are decoded from the adopted bytes. What a later Run may adopt is declared once, as (capability, resource kind) pairs, so a new re-readable kind is added where it is written rather than in each reader.
_Avoid_: File path, URL, blob id

**Blob**:
Owner-scoped immutable content addressed by digest and stored independently from the references that keep it reachable.
_Avoid_: Artifact when referring only to stored bytes

**Orphan Blob**:
A complete Blob that currently has no run or Resource Handle reference and is therefore eligible for grace-delayed cleanup.
_Avoid_: Failed artifact

**Spill**:
Private full tool output retained for the life of an active Answer Run and addressed only through a Resource Handle.
_Avoid_: Artifact, report, Journal Entry, Blob when referring to the handle

**Root Artifact Attachment**:
A run-scoped publication authorization settled by the parent Research Session for one normalized Agentspace path at an exact raw-content digest. It carries the root's display label, its settlement order, and the files the root declares it was built from, which publish with it as its dependencies and return with it when a later Run reopens the published product ([ADR 0052](adr/0052-a-deliverable-carries-what-it-was-built-from.md)); answer links may place it but cannot create the authorization.
_Avoid_: Published Artifact, Agentspace file, `artifact:` link, upload attachment

**Published Artifact**:
An optional owner-visible, run-scoped output descriptor created by fenced publication of an authorized Root Artifact Attachment or its validated dependency bytes. It is a separate reading, presentation, or download surface for a user-requested file, an impractically long or structurally rich deliverable, or content materially improved by that surface. It carries stable resource identity, validated media type, presentation capability, publication status, and an owner-scoped data plane without exposing an Agentspace path. Publication also registers it as a Resource of its Agent Session, so a later turn of that conversation reads the version it published with the same handle the answer references: a conversation can iterate on one deliverable without the product becoming a mutable file, because each publication is a new version that keeps its own authoring Run, digest, and evidence.
_Avoid_: Answer, parallel Answer, routine Research by-product, Spill, Blob when referring to the reference rather than the bytes, unattached Agentspace file

**Artifact Canvas**:
The Web Feature that opens any presentable Published Artifact and owns loading, safe renderer selection, side/wide/fullscreen layout, focus restoration, and preview teardown.
_Avoid_: Report Pane, universal Panel abstraction, same-DOM active HTML

**External Video Playback**:
A reader-activated viewing of a linked video's official player within the Answer surface. It is independent of the link's preview metadata, does not publish or ingest the remote video, and gives way when the reader selects another external video on the page.
_Avoid_: Published Artifact, video Evidence, Active HTML Preview, video ingestion

**Active HTML Preview**:
An explicit opt-in rendering mode for a self-contained HTML Artifact inside an opaque-origin, script-enabled sandboxed iframe. It is isolated from DlightRAG credentials and DOM; its CSP blocks normal external loads, but it is not a server execution sandbox or an absolute network-egress guarantee.
_Avoid_: trusted content, same-origin iframe, Sandbox service, zero-egress claim

**Publication**:
The fenced terminal transaction that makes staged Agentspace files owner-visible as Published Artifacts.
_Avoid_: Staging, Spill settlement, a second model call

**Agent Loop**:
The product-neutral turn cycle driven by one durable Agent Session: it stops when the model emits no tool call or cancellation is observed, and a provider error produces an error stop. Its only lifecycle observations are Agent Session events — ephemeral while a turn runs, committed at each settlement — because one loop means one event vocabulary. Research hosts it through durable boundaries; Fast does not enter it.
_Avoid_: Agent Event, Tool Turn Executor, workflow engine, max-agent-turn policy, READY protocol, Fast Answer

## Execution And Agentspace

### Personal Connections

The terms in this subsection describe the owner-bound implementation in [ADR 0012](adr/0012-personal-connections-and-hot-plug.md) and [ADR 0062](adr/0062-mcp-tools-are-declared-whole-and-resources-read-on-demand.md), including OAuth refresh, retention, resources, and server instructions.

**Connection**:
An owner's logical relationship to one remote MCP endpoint and external account context. Enabling a Connection authorizes all of its current and future discovered tools for that owner's future Research Runs. It also states the endpoint's non-secret headers, which are configuration and never a credential.
_Avoid_: Server configuration, plugin, per-tool integration

**Credential Grant**:
An owner's authorization to one external account and resource audience for a Connection. It holds the sealed credential: a token, and — when the credential travels in a header of its own rather than in `Authorization: Bearer` — the header name and prefix that travel with it. Refresh can preserve a Grant; credential/account replacement or consented scope expansion creates another Grant.
_Avoid_: DlightRAG identity, bearer field, permission proof

**Capability Catalogue**:
The complete validated set of remote tool definitions observed for one Connection at publication time. It is untrusted remote description data, not a tool-by-tool authorization list.
_Avoid_: Allowlist, marketplace, global registry

**Connection Generation**:
An immutable published version of a Connection's endpoint, Credential Grant identity, and Capability Catalogue. It fixes local definitions only, not remote code, data, availability, or side effects.
_Avoid_: Live MCP session, mutable connection record

**Run Connection Binding**:
An owner-scoped association between an accepted Research Run and one Connection Generation at a specific Connection Activation Epoch. It preserves the Run's selected definitions without overriding a later disable or Grant revocation.
_Avoid_: Per-conversation selection, invocation permit, current Connection lookup

**Connection Activation Epoch**:
The identity of one continuous enabled period of a Connection. A later re-enable has a different epoch and cannot revive an older Run Connection Binding.
_Avoid_: Connection Generation, Fencing Epoch

**MCP Resource**:
A document one Connection's server publishes through the MCP resources protocol, addressed by URI and read on demand rather than declared. It is live external content: the Run's pinned generation authorizes the read, but the content is not a Capability Catalogue entry, is never pinned, and may differ on a later read. A binary MCP Resource is described rather than returned.
_Avoid_: Resource (the Run's Resource), Capability Catalogue entry, pinned definition

**Resource Read**:
One call of `mcp_read_resource`, `mcp_list_resources`, or `mcp_list_resource_templates`, resolved against a Run's pinned Connection token and admitted by that Connection's live authority gate. It is read-only and replayable, so it may run beside other read-only calls and a recovery may run a pending read again; its content enters the Run as the call's tool result.
_Avoid_: Resource acquisition, Rendered Read, Connection dispatch

**Server Instructions**:
The `initialize` result's `instructions` text a Connection's server publishes about itself, bounded by `max_instruction_bytes` and stored with its immutable Connection Generation. A Run pins the generation and carries the instructions in its prompt, attributed to the Connection and framed as the external server's own guidance — data to judge, never an instruction from DlightRAG and never authority.
_Avoid_: system prompt, DlightRAG instruction, capability grant

### Current execution and Agentspace concepts

**Execution Environment**:
The adapter behind exactly two modes: `disabled` and `trust`. Trust runs the Agent's processes in the host user's authority **confined to its Agentspace**: the corpus, the deployment's configuration, the project tree, and other Runs' Agentspaces are outside the process view, while the toolchain's runtime stays readable. The allow-list is code, capabilities declare the layers they need, and one that overlaps the corpus or the project tree fails composition. Network egress is the deployment's to enforce, because a path list cannot enforce it.
_Avoid_: implicit downgrade, permission catalog, approval prompt, shell-command filtering as a security boundary

**Agent Browser**:
The Run-scoped browser capability: one browser per Research Run, leased from a pool the deployment runs, with a short-lived anonymous context for each Rendered Read and an Agent Page for each Agent Session that opens one. It is a deployment capability reached through tools, never a Connection or a process of the Agent, and the deployment's network, not DlightRAG, confines what it reaches. Fast never has one.
_Avoid_: Connection, Chromium in the answering container, browsing session, browser profile

**Agent Page**:
An Agent Session's anonymous context in its Run's leased Agent Browser, which the model drives as one active page; a popup or new tab becomes the active page. It ends with its Agent Session or Run.
_Avoid_: browser session, browsing session, context (in prose)

**BrowserProvider**:
The port that leases a Run's browser endpoint. It is an ordinary adapter boundary with one implementation, a pool of Playwright run-servers; see [Architecture](architecture.md#agent-browser) for when a lease is live.
_Avoid_: Connection provider, browser service, MCP server

**Agent Account**:
An account the Agent registered on a third-party site under its own identity. DlightRAG generates its password, fills it by ref, and seals it under the deployment key ring; a parent's account persists for its owner, a Child's lasts for its Run.
_Avoid_: owner account, saved login, browser profile

**Agent Mailbox**:
An optional deployment capability that delivers mail sent to Mailbox Aliases, read from an S3-compatible bucket the deployment fills.
_Avoid_: inbox provider, temporary mail

**Mailbox Alias**:
An address DlightRAG mints on the Agent Mailbox's domain for one account: deterministic per owner and site for a parent's account, random for a Child's.
_Avoid_: alias on its own (a Resource Handle also has aliases), temporary address, the owner's email

**Agent Skill**:
A progressively disclosed `SKILL.md` package discovered from packaged built-ins, the operator-global root, or one owner's published skills. A name belongs to the tier that owns it: a built-in keeps its name, a global Skill keeps any name a built-in does not hold, and an owner's Skill keeps what is left, so a Skill whose name is already served is a conflict and is not served itself ([ADR 0047](adr/0047-a-skill-name-belongs-to-the-tier-that-owns-it.md)). Metadata is projected first; the framework reads contained references only through `load_skill` and never executes Skill code. What an Agent's own processes may read is the roots the capability declares to the Execution Environment (ADR 0024): the operator-global root and the Run owner's own shard, never the shared parent and never a sibling owner's. Users write their own skills only through the validated `publish_skill` tool. An owner can turn one of their own Skills off without deleting it, through `set_skill_enabled` or Settings, which also lists, reads and deletes them: it stays stored and counts toward the quota, a `.disabled` file in its directory marks it, and a republish keeps it off. A Skill that is off is not served, so a name it was holding is free for a lower tier, `load_skill` refuses it as disabled, and when nothing else serves its name the catalog still names it, without its description, as disabled by its owner. Only the owner's own tier can be turned off, never a built-in or global Skill. What lives under a skill root is picked up without a restart: `load_skill` resolves names as they are when it is called, so a published Skill is loadable at once and one turned off is not, while the catalog the Run was started with, a stable part of its prompt prefix (ADR 0015), lists either change from the next Run. A Skill's frontmatter may declare `required_reads`: the Skill-relative documents a Run that loads it must read (one `path`, or an `any_of` list where reading any one satisfies the entry, each with a `when`), and a declaration naming a file the Skill does not carry is refused like an unhonourable `attestation`. Whether a Run read each one to the end is derived from the `load_skill` read fact on its durable events and published as evidence beside the Artifact's attestation, never as a gate ([ADR 0057](adr/0057-a-skill-names-what-a-run-must-read.md)).
_Avoid_: owner Profile Memory, marketplace plugin, arbitrary extension

**Outbound MCP Tool**:
An owner-bound remote tool invoked through a bounded foreground Streamable-HTTP MCP session after the Run/Child pending-effect gate. Deployment-global tool declarations and stdio are not accepted.
_Avoid_: Personal Connection, MCP registry, marketplace, OAuth platform

**Agentspace**:
The model-visible filesystem rooted at the active Agentspace Epoch's directory. It lives while its Run works and for seven days after the Run finishes, since nothing reads a finished Run's tree: a sweep removes it then, or with the row when retention prunes it, or as soon as its row is gone, guarded by the row rather than by directory age. A Resource is never a file in it until Resource Materialization copies one. It is the conversation's file space, carried from one Run to the next, and each settled Run leaves a restorable snapshot of it ([ADR 0075](adr/0075-workspace-names-the-corpus-and-agentspace-names-the-agents-files.md)).
_Avoid_: Agent Workspace, agent workspace, Agent Space, agent-space, Corpus Workspace, working_dir, working directory, sandbox, bare workspace (that is the corpus's word)

**Agentspace Epoch**:
One filesystem generation of an Agentspace; recovery copies a stable observation of the recorded epoch and never executes in the old one.
_Avoid_: Durable Progress, Fencing Epoch, checkpoint

**Agentspace Inventory**:
The current Agentspace Epoch's path, type, size, and digest observation of an Agentspace.
_Avoid_: Journal Entry, checkpoint, historical epoch listing

**Resource Materialization**:
Copying a Resource's admitted original bytes into the Agentspace with the `materialize` tool. The Resource stays the immutable, citable original; the copy is mutable work with no provenance of its own, so a citation names the Resource, never the file. `ResourceRegistry.materialize`, which loads a Resource's bytes into the registry, lineage adoption, and the binding of a Session's notes into an Agentspace are not this term.
_Avoid_: adoption, re-download

**Session Note**:
A file the answering agent writes under its Agentspace's reserved notes path, outside `artifacts/`, so that work product survives one Run's compaction and the next turn of the same Session reads it. Memory belongs to the Agent Session, not to the Run: one authoritative note set per Session holds each note's bytes, size, digest, revision, and writing Run, and every Run of that Session materializes a working copy of it when it binds and promotes its changes back at Tool settlement under its own lease. A session note therefore outlives the Run that wrote it, and per-Run reclamation cannot take it. The Context Projection names each note by the `read(path=…)` call that reads it again. Memory never fails a Run: a plane that cannot be read, a note the plane refuses for budget, or a promotion that fails is recorded on the Run's trace and the Run proceeds with the working copy it has. A Session Note is never citable, never a second statement of what happened, and never a Memory Record.
_Avoid_: scratchpad, agent journal, second transcript, Memory Record, Artifact, Evidence

## Configuration And Deployment

**API Family**:
The per-model provider wire of one invocation: `chat_completion` or `response` for
an OpenAI-compatible model, and `interactions`, Gemini's only wire. It neither
selects the provider or model nor grants a provider's remote state authority over
the Agent Session, and it states no retention policy.
_Avoid_: Provider, Model Profile, Responses dialect, ZDR mode

**Application Configuration**:
The non-secret operator choices that govern DlightRAG product behavior and integrations independently of how a process is packaged or scheduled.
_Avoid_: credentials, container topology, duplicated environment override

**Runtime Policy**:
An admin-edited setting of the deployment, stored in PostgreSQL and read on every request, so a change applies on the next request, with no restart. A **targeted** policy has a default and ordered rules that match a caller's verified JWT claims; the first rule that matches decides, otherwise the default holds. A caller whose claims are not verified only meets the default. There are three, and they decide at different moments: `answer_feedback` when a reader rates, and `agent_effort_ceiling` and `active_answer_runs` once, when an Answer is accepted, after which the Run carries what they said.
_Avoid_: feature-flag service, config reload

**Deployment Binding**:
A value created by the runtime topology or process role to connect DlightRAG to its environment, such as service discovery or a listener binding. It adapts Application Configuration to the deployment without becoming a second owner of product policy.
_Avoid_: application default, product setting, secret

## Operations

**Fencing Epoch**:
The monotonically increasing write-generation of one Run lease; every durable write is predicated on the current epoch and the lease owner. A lapsed lease window only makes the Run reclaimable; a reclaim is what bumps the epoch.
_Avoid_: Agentspace Epoch, Durable Progress

**Full Development Reset**:
The explicit replacement of development database and local runtime/corpus data; see [Operations](operations.md#full-development-reset).
_Avoid_: Migration, Workspace reset, compatibility cutover

## Concept Map

| Concept | Meaning | Reference |
|---|---|---|
| Corpus Workspace | Isolation unit for indexed data, metadata, files, and queries | this document |
| Ingestion | One durable contract for local files, uploads, object storage, URLs, and SDK sources | [Interfaces](interfaces.md#ingestion) |
| Retrieval | One durable Query-lane Run returning LightRAG mix plus metadata, BM25, visual fusion, and rerank evidence | [Retrieval and Answer](retrieval-answer.md) |
| Run | Common durable lifecycle for Retrieval, Answer, and Corpus Mutation across REST, MCP, Web, Python, and evaluation | [Run runtime](run-runtime.md) |
| Answer Run | A Query-lane Run that resolves Fast or Research and generates an Answer | [Retrieval and Answer](retrieval-answer.md#answer-orchestration) |
| Resource | Answer attachment or public link, read as bounded text or viewed as images on demand; later Runs in the same Session can adopt it | [Resource reading](resource-reading.md) |
| Published Artifact | Owner-visible Research output authorized by a settled root attachment and validated at publication | this document |
| Goal | One durable objective a Session carries; the platform continues it automatically and the Agent decides completion | [Goal mode](adr/0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md) |
| Source | Durable provenance and download contract for an ingested document | [Interfaces](interfaces.md#sources) |

