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
The product capability that owns the complete Answer Run lifecycle, including acceptance, following, cancellation, and terminal projection.
_Avoid_: Answer manager, runtime service

**Retrieval**:
A caller-awaited corpus-evidence use case implemented with asynchronous I/O that returns one result without creating an Answer Run.
_Avoid_: Fast Answer

**Fast Answer**:
A durable Answer Run that plans, retrieves, and generates without creating an Agent Session or research workspace.
_Avoid_: Retrieval, non-durable fast path

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

**Prepared Input**:
The immutable, bounded execution description accepted for a new Answer Run after authorization, capability resolution, and profile pinning.
_Avoid_: Request payload, checkpoint

**Public Request Fingerprint**:
The digest of the normalized caller request used for owner-scoped idempotency before Prepared Input is built.
_Avoid_: Prepared-input hash

## Journal And Effects

**Journal Entry**:
An immutable source fact in an Agent Session whose ordered fold reconstructs active model context.
_Avoid_: Message when the fact may be an intent, result, compaction, or terminal outcome

**Context Projection**:
The bounded continuation state that selects a journal suffix and, when needed, summarizes a contiguous older prefix.
_Avoid_: Checkpoint, transcript snapshot

**Compaction Summary**:
Typed continuation memory for one contiguous journal prefix, validated and rendered by the framework but never treated as citable Evidence.
_Avoid_: Free-form checkpoint, evidence summary

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
The optional published Markdown document taken from `artifacts/report.md` after citation finalization; its bytes live in the Blob store and the result names it by Resource Handle.
_Avoid_: answer body, Spill, Compaction Summary, required report

**Publication**:
The fenced terminal transaction that makes staged Agent Workspace files owner-visible as Published Artifacts.
_Avoid_: Staging, Spill settlement, a second model call

**AgentLoop**:
The unlimited research turn loop that stops when the model emits no tool call, or when cancel, provider error, or an all-terminate batch ends the attempt.
_Avoid_: max_agent_turns, READY protocol, Fast Answer

## Execution And Workspace

**Execution Environment**:
The optional file-and-process host that makes path tools, Spill, and workspace staging possible.
_Avoid_: Sandbox, container runtime

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