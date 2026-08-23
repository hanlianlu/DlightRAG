# DlightRAG Domain Language

DlightRAG is a durable research-and-retrieval product that joins reusable AI, agent, and corpus capabilities without merging their ownership.

## Product Surfaces

**Application**:
The local in-process DlightRAG product surface that exposes Answer, retrieval, and corpus-administration capabilities and owns their shared lifetime.
_Avoid_: Manager, service locator

**Answer Run**:
An owner-scoped durable execution accepted once and observable through status, events, cancellation, and a terminal result.
_Avoid_: Request, job, task

**Answer Service**:
The product capability that owns start, status, events, steer, follow-up, cancel, resume, fork, transcript tail, child roster, and terminal projection through one transport-neutral interface.
_Avoid_: Answer manager, transport-specific runtime service

**Continuation**:
A new durable Answer Run with `parent_run_id` and kind `follow_up` or `fork`. Follow-up includes the selected answer as context; fork starts a sibling branch from its accepted context.
_Avoid_: in-place run mutation, session checkout, hidden conversation copy

**Retrieval**:
A caller-awaited corpus-evidence use case implemented with asynchronous I/O that returns one result without creating an Answer Run.
_Avoid_: Fast Answer

**Fast Answer**:
A durable Answer Run that plans, retrieves, and generates without creating an Agent Session or research workspace.
_Avoid_: Retrieval, non-durable fast path

**Web Conversation**:
The browser conversation surface that creates Answer Runs; it is a transport, not a research capability. The thread shows the run's terminal answer; a Primary Report is a separate reading surface the owner opens from that turn.
_Avoid_: Web Search, Web, Web Channel when meaning Exa or open-web retrieval; chat column as the report

**Web Search**:
The optional open-web evidence capability used inside an Answer Run.
_Avoid_: Web UI, Web Conversation, Web Channel

**Answer Mode**:
The caller-facing selector `auto | fast | research`. Omitted means `auto` on REST, MCP, Web Conversation, and Python.
_Avoid_: research bool, inferred path, Web Search as a mode

**Valid Mode Set**:
The capability-derived subset of `{fast, research}` that one Prepared Input may legally resolve to.
_Avoid_: requested mode, router guess, heuristic from Web Search being configured

**Resolved Mode**:
The durable `fast` or `research` value written once after routing; crash recovery does not recompute it.
_Avoid_: Answer Mode when the request was `auto`

**Corpus Administration**:
The product capability for corpus workspace lifecycle, ingestion, files, metadata, visual assets, and reset.
_Avoid_: Workspace manager

**Authorized Workspace Set**:
The concrete canonical Corpus Workspace ids a caller may use after authentication and access expansion are complete.
_Avoid_: Workspace selector, raw claims

## Corpus And Research

**Corpus Workspace**:
A named, authorized corpus scope whose ingestion and retrieval state is owned by one corpus runtime.
_Avoid_: Workspace when it could mean an agent filesystem

**Agent Session**:
A durable research conversation within one Answer Run, reconstructed from its source journal.
_Avoid_: Run, thread

**Child Session**:
A first-class foreground Agent Session inside the same Answer Run. It has durable parent/call lineage, bounded depth and concurrency, selected context/model/tools, inclusive usage, and no independent lease.
_Avoid_: child run, detached job, mission, background swarm

**Subagent Roster**:
The status/wait/cancel projection for Child Sessions created by `spawn_agent`. Child Evidence is admitted into the parent ledger before the parent effect settles.
_Avoid_: Delegate Research, workflow engine, child queue

**Prepared Input**:
The immutable, bounded execution description accepted for a new Answer Run after authorization, capability resolution, and profile pinning. Resolve never rewrites it.
_Avoid_: Request payload, checkpoint

**Routing Record**:
The Answer-owned one-to-one durable fact of requested mode, valid modes, and nullable resolved mode for one Answer Run. An `auto`→research Agent Session id is issued here, not by mutating Prepared Input.
_Avoid_: Prepared Input, second policy source, inferred research bool

**Public Request Fingerprint**:
The digest of the normalized caller request used for owner-scoped idempotency before Prepared Input is built. Canonical Answer Mode is part of that digest: omitted equals `auto`.
_Avoid_: Prepared-input hash

## Long-Term Memory

**Memory Record**:
An owner-scoped, non-citable remembered preference or fact that may be recalled across separate conversations. It is never Evidence and never a citation source.
_Avoid_: Compaction Summary, Journal Entry, PriorTurns, Evidence, Artifact, remembered citation

**Memory Subject**:
The owner identity a Memory store scopes every operation to; bound by the host (DlightRAG: JWT owner or stable local single-user owner; stdio MCP: `--subject`) and never accepted from a model tool argument. Shared simple-auth callers are not personal Memory subjects.
_Avoid_: caller-selected namespace, conversation id, Agent Session

**RecallResult**:
The structured outcome of one query-aware Memory recall: selected records (exact matches pinned first, then chronological), the raw leg candidates the fusion consumed, degradation flags, and the recalled body character cost (rendering overhead excluded). Never a prompt fragment.
_Avoid_: prompt text, standing block, packed string

**Memory Write**:
The only durable create, supersede, or forget of a Memory Record: a named remember or forget channel that passed a closed policy check.
_Avoid_: Silent promotion, transcript scan, model aside, journal side effect

**Memory Record Lifecycle**:
`active → superseded` or `active → forgotten`. Forget is idempotent and leaves a non-recallable tombstone; superseded history follows the shared retention floor. Formation may be proposed without mutation and committed idempotently by proposal id.
_Avoid_: hard-delete forget, silent transcript promotion, confidence decay

**Retention Floor**:
The single deployment clock (`answer.runtime.answer_run_retention_days`, default 365) that bounds how long terminal Answer runs, their event logs, and superseded Memory history stay durable. The sweep is best-effort: it may reclaim later, never earlier.
_Avoid_: deadline, SLA, per-aggregate TTL, inactivity expiry

**Conversation Read Window**:
The bounded number of recent turns a Web Conversation snapshot and the history endpoint return. It is a UI and payload bound, not retention: older turns stay durable until the Retention Floor reclaims their runs.
_Avoid_: max_turns, trim window, retention window

## Journal And Effects

**Journal Entry**:
An immutable source fact in an Agent Session whose ordered fold reconstructs active model context.
_Avoid_: Message when the fact may be an intent, result, compaction, or terminal outcome

**Context Projection**:
The bounded model-facing projection of the selected head in the canonical linear Agent Session journal. The parent-linked graph is a derived view; 3.0 does not persist alternate heads. Exactly one active compaction summary precedes the retained non-compaction suffix; historical summaries remain audit facts.
_Avoid_: authority, checkpoint, transcript snapshot

**Context Contribution**:
A typed model-ready contribution with source, authority, citable and compressible facts. It unifies projection ordering without merging the contributor's storage ownership.
_Avoid_: global prompt registry, second state authority

**Compaction Summary**:
Typed continuation memory for one contiguous journal prefix, validated and rendered by the framework but never treated as citable Evidence.
_Avoid_: Free-form checkpoint, evidence summary, Memory Record

**Effect Intent**:
A durable declaration of one validated tool operation and the replay contract under which it may execute.
_Avoid_: Tool call when referring to durable recovery identity

**Effect Settlement**:
The atomic durable outcome of an Effect Intent, including its ordered result and any host-owned updates.
_Avoid_: Tool result when referring to the full commit boundary

**Durable Progress**:
A monotonically increasing run fact advanced only by live fenced execution boundaries that recovery must not repeat.
_Avoid_: Heartbeat, phase, turn count, recovery interrupt, Workspace Epoch handoff

**Fast Stage**:
A deterministic durable progress boundary inside a Fast Answer, used for recovery without creating an Agent Session.
_Avoid_: Agent step, retrieval request

## Evidence And Resources

**Evidence**:
Citable, run-scoped source material with durable identity and content/locator integrity.
_Avoid_: Summary, agent prose

**Resource Handle**:
An opaque owner/run-scoped identity through which prepared, fetched, evidence-backed, or spilled content remains addressable across recovery.
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

**Published Artifact**:
An owner-visible output reference created by fenced publication of staged Agent Workspace bytes.
_Avoid_: Spill, Blob when referring to the reference rather than the bytes

**Primary Report**:
The optional published Markdown document taken from `artifacts/report.md` after citation finalization; its bytes live in the Blob store and the result names it by Resource Handle. On Web Conversation it is read in the document panel, not as a second chat body.
_Avoid_: answer body, chat column, Spill, Compaction Summary, required report

**Publication**:
The fenced terminal transaction that makes staged Agent Workspace files owner-visible as Published Artifacts.
_Avoid_: Staging, Spill settlement, a second model call

**Agent Loop**:
The product-neutral event-driven turn cycle that stops when the model emits no tool call or cancellation is observed; provider errors produce an error stop. Research hosts it through durable boundaries; Fast does not enter it.
_Avoid_: workflow engine, max-agent-turn policy, READY protocol, Fast Answer

## Execution And Workspace

**Execution Environment**:
The adapter behind exactly three modes: `disabled`, host-trusted `trust`, and `sandbox`. Sandbox is a seam with explicit unavailable failure unless trusted host code supplies a backend.
_Avoid_: implicit downgrade, permission catalog, approval prompt

**Agent Skill**:
A progressively disclosed `SKILL.md` package discovered globally or in the Agent Workspace. Metadata is projected first; contained references are read only through `load_skill` and Skill code is never executed.
_Avoid_: owner Profile Memory, marketplace plugin, arbitrary extension

**Outbound MCP Tool**:
A deployment-declared and allowlisted remote tool invoked through a foreground stdio or streamable-HTTP MCP session.
_Avoid_: MCP registry, marketplace, OAuth platform

**Agent Workspace**:
The model-visible filesystem rooted at the active Workspace Epoch's workspace directory.
_Avoid_: Corpus Workspace, working_dir, workspace when it could mean a corpus scope

**Workspace Epoch**:
One filesystem generation of an Agent Workspace; recovery copies a stable observation of the recorded epoch and never executes in the old one.
_Avoid_: Durable Progress, Fencing Epoch, checkpoint

**Workspace Inventory**:
The current Workspace Epoch's path, type, size, and digest observation of an Agent Workspace.
_Avoid_: Journal Entry, checkpoint, historical epoch listing

## Operations

**Fencing Epoch**:
The monotonically increasing write-generation of one Answer Run lease; every durable write is predicated on the current epoch and a live lease.
_Avoid_: Workspace Epoch, Durable Progress

**Journal Schema Reset**:
The pre-release replacement of all development data with the journal, progress, resource, evidence, and chunked-Blob schema.
_Avoid_: Migration, compatibility cutover