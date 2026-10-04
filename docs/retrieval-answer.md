# Retrieval And Answer

This document owns how queries become contexts, answers, sources, and citations.
Payloads live in [Interfaces](interfaces.md), fields in
[Configuration](configuration.md), runtime ownership in
[Architecture](architecture.md), and Run lifecycle and recovery in
[RunRuntime](run-runtime.md).

DlightRAG always uses LightRAG `mix` as its graph/vector base. It adds metadata
filtering, optional direct image retrieval, PostgreSQL BM25, RRF fusion,
provenance hydration, reranking, answer packing, and citation validation.
[Architecture](architecture.md#ingestion) owns how ingestion builds the chunks,
graph, fused visual vectors, and BM25 rows that retrieval reads.

- Top-level `/retrieve` is a durable, owner-scoped Run. It is
  knowledge-base-only and may take `query_images`.
- `/answer` creates a durable Answer Run from a query plus optional attachments,
  then resolves `auto | fast | research`.
- Retrieval inside an Answer is an internal Retrieval Stage under the Answer
  Run; it never creates another Run.
- `auto` considers the valid mode set and conversation context. When both paths
  are legal, routing defaults to Research unless the turn is corpus-grounded.

Every accepted Retrieval pins normalized query/options, authorized search scope,
the required `extract` and optional `vlm` model profiles, capability facts, and
policy revisions. Every accepted Answer additionally pins bounded history and
Resources. The Web conversation layer wraps the Answer pipeline; it does not
define another path.

## Query Pipeline

For a top-level request, the Application resolves authorization and accepts a
`run_kind=retrieval` Query-lane Run before this pipeline executes. The same raw
pipeline begins directly inside an Answer without nested Run acceptance.

```text
Retrieval Stage
  -> use accepted authorized concrete workspaces and warm them
  -> plan lexical terms and optional metadata filters once
  -> per workspace:
       LightRAG mix, PostgreSQL BM25, optional direct image-vector search
         (each leg capped before fusion)
       RRF fusion + dedup
       provenance hydration
       final rerank, capped at chunk_top_k
  -> federated round-robin merge
  -> reference canonicalization
  -> optional answer packing/generation
```

`RetrievalPlanner` sees the query, schema, bounded prior turns when appropriate,
and current-image descriptions. It never sees answer attachment
bytes/text/manifests. Explicit BM25 terms and metadata filters remain
authoritative. A Research KB tool's chosen semantic query is preserved while the
planner derives only its supporting lexical and filter context. Only the
provider SDK retries the planner's model request: a transient failure up to the
`extract` model's `max_retries`, an authentication, request, or context-window
rejection never. A request that still fails plans without the model
(`fallback_provider_error`), searching the query as given.

The `LightRAG mix` and BM25 lanes degrade independently. If one fails, the other
may return results and trace records `lightrag_error_type` or `bm25_error_type`.
If both fail, retrieval raises the LightRAG error with BM25 chained. Trace
`lightrag_mix_chunk_count` records the LightRAG count before fusion;
`contexts.chunks` is the final fused/reranked set.

A workspace that publishes no document runs none of these lanes: its result is
empty and its trace records `workspace_empty`. Querying LightRAG there would
spend a keyword-extraction model call only to report LightRAG's no-result status,
`failure`. The publication check reads corpus storage, so a workspace whose
storage fails still fails; multi-workspace retrieval lists it in
`failed_workspaces` beside the workspaces that answered.

Top-level Retrieval records a `planning` phase while it describes query images
and a `searching` phase for retrieval planning and search. A transient
corpus-storage or model-provider failure defers the Run under the
[RunRuntime](run-runtime.md#lifecycle) backoff and deferral cap. The configured
retrieval timeout bounds claimed planning and search, not queue residence, and
expiry fails the Run as `retrieval_timeout`. Unexpected execution failures
settle as sanitized `retrieval_failed` errors.

### BM25

BM25 queries the same `LIGHTRAG_DOC_CHUNKS` rows. Ingestion labels each chunk's
language. Partial pg_textsearch indexes cover the configured languages, plus a
full-table `simple` fallback. Supported query languages select their profile;
unknown/ambiguous languages use `simple`.

At startup the writer recreates any BM25 index whose text configuration,
`bm25_k1`, `bm25_b`, or language bucket differs from the configuration,
and a reader refuses to start until it does. Existing chunks take a changed
language set only through the offline
[BM25 rebuild](operations.md#workspace-bm25-rebuild), which relabels them.
Disabling BM25 removes only this PostgreSQL lane; Resource lexical search
remains run-scoped and in memory.

## Product Document Visibility And Metadata In-Filtering

Product Document visibility is always-on: directly attributable chunk evidence
is admitted only when its document metadata row exists and
`_dlightrag_finalization_complete` is exactly true. Missing, NULL, false, and
out-of-band LightRAG documents are unpublished. PostgreSQL applies `IS TRUE` in
metadata, graph-chunk, BM25, and vector queries. Unscoped ANN/BM25 first rank a
bounded over-fetch window and then correlate those candidates to metadata;
non-pushdown vector stores similarly post-filter only bounded hit IDs. A short
result is preferable to leakage.

Named filters map to typed columns: `filename`, `file_extension`, `title`,
`author`, and creation-date bounds. Arbitrary keys use `filters.custom` against
one JSONB column, with case-insensitive comparison.

Filename matching is corpus-aware: exact stored name/stem first, then literal
substring. The caller/planner supplies one `filename` value rather than guessing
which operator will match unseen data.

- Explicit caller filters are strict; zero candidates means zero results.
- An inferred filter applies only when the planner states `high` confidence; its
  confidence and evidence reach only the planner log. If it resolves to no
  candidates or retrieves no chunks, DlightRAG retries unfiltered.
- Non-empty inferred candidates constrain semantic and BM25 legs.

Every chunk-producing leg enforces visibility, with any user filter as an
additional scope:

- `FilteredVectorStorage` enforces it on every query. It returns immediately
  for empty filtered candidates, uses exact scoring for small filtered sets,
  HNSW iterative scan for larger sets, and a bounded visibility-only path
  without a user filter.
- Graph entity/relation legs resolve source chunks by ID, so
  `FilteredChunkStore` visibility-checks that bounded lookup inside a retrieval
  scope; LightRAG's own ingest, rollback, and delete reads pass through
  unchanged.

The filter controls quotable chunk evidence. It does not rewrite LightRAG's
corpus-level entity/relationship summaries, which may merge descriptions from
multiple documents and have no separable per-document share. Trace
`metadata_kg_chunks_dropped` counts graph-referenced chunk IDs that returned no
row: unpublished, outside the user filter, or missing. Each workspace trace also
reports `visibility_strategy` (`pushdown`, `bounded_pushdown`, or `postfilter`,
from the last leg that ran), `visibility_shortfall` when a vector leg returns
fewer rows than requested, and `visibility_dropped`, which only the post-filter
path counts; these are not snapshot isolation claims.

## Multimodal Retrieval

Query images, from `/retrieve`'s `query_images` or an Answer's current-turn
images, add two transient paths:

```text
query image
  |-- VLM description -> planner context (BM25 terms, inferred filters)
  `-- native image embedding -> direct visual search (when active)
```

A description changes the LightRAG query only where the planner may rewrite it,
which is a Fast Answer with history; top-level `/retrieve` and Research keep the
query as given, and a planner fallback ignores descriptions. The image-only
query vector searches the whole chunk vector store, where text vectors and
fused VLM-description+image document vectors share one space, under the same
visibility and filter scope. Provider adapters apply official query/document
task semantics and split requests in order at the provider's input-count and
per-request image-byte limits. RRF and dedup resolve overlap between semantic
and visual hits.

A definitive probe failure downgrades `auto` to text, which skips direct image
embedding while descriptions still reach the planner, and aborts startup in
explicit `multimodal` mode. A transient probe failure changes neither mode;
startup continues degraded.

## Fusion And Reranking

DlightRAG disables LightRAG's query reranker. It reranks the fused set after
provenance hydration so LightRAG, BM25, and direct-image candidates compete in
one list. Every fused candidate carries its page provenance; image bytes are
attached before rerank only when the reranker reads images, and a text
reranker's survivors receive theirs afterwards.

Ranker classes are:

- chat-model listwise reranking, which sends chunk images unless
  `rerank.input_modality` is `text` or the scoring model is known to lack image
  input;
- multimodal or text HTTP `/rerank` adapters;
- Voyage, Cohere, and Azure Cohere text rerankers.

A configured score threshold is hard: candidates below it disappear, even if a
workspace then contributes none. Runtime reranker failure falls back to the
pre-rerank fused order. Configuration failures (for example, a selected provider
without credentials) fail startup instead of changing strategy.

Reranking has its own image budget. Oversized visual candidates fall back to
text where available; unbounded data URIs are never sent.

## Multi-Workspace Retrieval

The planner runs once, selected workspaces execute concurrently, and each runs
the complete filtering/fusion/rerank pipeline. The federation layer tags chunks
with `_workspace`, round-robin interleaves the per-workspace lists, drops
repeated workspace/chunk pairs, applies the configured output budget plus its
per-workspace fairness floor, and then canonicalizes references.

By default, round-robin preserves representation without pretending scores from
different workspace/model calls are calibrated. If `federated_rerank` is true
and the configured reranker is available, DlightRAG reranks the merged candidate
pool once across Workspaces. An unavailable/build-failed reranker uses the capped
interleave; a runtime rerank failure does the same and records its error type in
trace rather than failing Retrieval.

## Answer Orchestration

An Answer resolves to Fast or Research. Both produce the same canonical result,
whose fields [Interfaces](interfaces.md#common-answer-terms) defines.

### Fast

Fast performs planning, KB retrieval, and one tool-free generation call on the
`query` model. It uses shared Context Contribution, Evidence, citation,
model-call, usage, Agent Session infrastructure, Profile Memory recall, and,
when execution is enabled, an inert Workspace that carries the
[Session notes](#session-notes). It creates no Agent Operation, tools, skills,
or publication.

### Research

Research drives `AgentSessionRuntime` over one selected Lane. Its closed
run-local registry may include:

- knowledge-base, resource, and optional provider-neutral public Web tools;
- the `browser` tool, when the deployment configures an [Agent Browser](#agent-browser);
- rooted file/Bash tools when execution is enabled, among them
  [`materialize`](resource-reading.md#materialize), which copies a Resource into the
  Agent Workspace;
- Profile Memory tools for the parent (children recall only);
- progressive `load_skill`, plus `publish_skill`/`delete_skill` for the parent;
- the tools of every enabled
  [Personal MCP Connection](personal-mcp-connections.md) its owner holds; and
- bounded asynchronous Child Sessions with explicit snapshots and Evidence
  return.

`spawn_agent` admits up to eight children per call and returns durable handles
immediately. A child runs with its parent's tools except the ones that spend the
Run's authority: the roster controls, `remember`/`forget`, and the publication
tools. It holds `ask_parent` instead. The parent's `tools` list narrows that set
for one child — one without a shell, say — and can never restore what the
Run withholds or remove `ask_parent`; a listed name the Run does not offer is left
out, not an error. Children cannot spawn grandchildren. A child started with
`context="parent"` inherits the conversation the parent has settled and the evidence
gathered so far; the turn that is calling `spawn_agent` is left out, because its calls
have no outputs yet and a provider refuses a request that holds one.
Same-Session continuation creates a new Operation on the existing Child Session.
`wait_subagent` returns early, with its child still running, when a sibling settles
or a child asks a question, and then reports every question awaiting an answer. A
settled result the parent has read through `subagent_status`, `wait_subagent` or
`cancel_subagent`, or been sent in a notification it accepted, is not sent to it
again when it ends its turn; unread results are. The built-in `council` Skill is a parent recipe for independent first-pass
investigations and at most one curated cross-examination; it adds no tools, asks
for no narrowing of its children, which hold the default set, and is not a
permission gate. A user's explicit `/skill:` request is made to the Run's own
agent, so a child gets the Skill catalog, while it holds `load_skill`, but not that
request.

Child `model_role` selects a configured model, not a task category or permission.
The objective remains arbitrary free text; omitting the selector chooses `query`.

| Selector | Recommended use |
| --- | --- |
| `query` | Preferred strongest reasoning tier: hardest research, complex planning, evidence adjudication, final review |
| `default` | General-purpose tier below query: ordinary analysis, synthesis, drafting, routine review |
| `extract` | Routine extraction, normalization, structured work |
| `keyword` | Lightweight keywords, labels, query rewriting |
| `vlm` | Visual evidence, images, charts, document pages; not inherently cheap or fast |

`default` resolves `models.chat.default` itself, independently of the `query`
override. Other selectors use complete authenticated role overrides or the
existing default fallback. These recommendations do not guarantee relative model
strength. The spawn tool describes the accepted effective model/profile and
agentic reasoning request (including ordinary-reasoning inheritance); `max` is a
configured request level, not a universal capability guarantee. Image support
comes from the effective profile, never the selector name. Incompatible image
inputs and unsupported provider tool calling fail explicitly, without switching
models or silently dropping images.

Acceptance pins, for all five selectors, the invocation fingerprint, the model
profile, and the ordinary and agentic reasoning request levels. Each Child
Session pins its selected identity/profile and tools through continuation and
same-version restart recovery. Incompatible endpoint or reasoning drift is
rejected before child provider effects.

Tool errors return to the model for correction; they do not terminate research.
A no-tool assistant turn completes the current Operation, unless a steer is
already waiting, which the same Operation then answers. The Run continues while a
follow-up, control command, or Child result is pending or a Child Session is
still running; otherwise it ends, and that last turn's text is the answer. The parent Research Session authorizes a root file under the
Workspace's `artifacts/` directory for publication only through
`attach_artifact`; Fast and Child Sessions do not receive that product tool. A
successful attachment binds the root's normalized relative path, label, media
capability, byte size, and raw-content digest in the same settlement as the tool
result. Reattaching the same path replaces its intent and moves it to the latest
settlement position.

The Answer remains the default deliverable; Workspace and publication-tool
availability do not imply that Research should create an Artifact. The Agent uses
a separate Artifact when the user requests one, when a complete deliverable is
too long or structurally rich for one practical Answer, or when a visual,
interactive, or downloadable surface materially improves use. If the Artifact
contains the complete deliverable, the Answer is a concise orientation and
handoff rather than a substantial copy. Explicit requests for both inline and
file versions are the exception. Independent citation validation governs support
on each surface, not duplicated prose.

At the terminal boundary, the Host verifies every attachment against current
bytes and publishes each valid root plus the safe transitive dependency closure
reachable through Markdown/HTML `artifact:` links. Those links are placement
syntax, not publication authority: an answer link fails validation unless its
target is an attached root or a validated dependency of one, and an attached
root omitted by the answer receives a trailing link in attachment settlement
order (placed before the answer when its ending would swallow a trailing link).
Dependencies are published but are not auto-placed. Failed or stale attachments
receive the single bounded correction pass. There is no reserved filename,
privileged Artifact role, or hidden finalizer call.

Markdown references use the same grammar for validation, result projection, and
browser placement. Publication leaves `artifact:` targets as written and settles
a document-local `artifact_bindings` map. Unresolvable links enter correction
and, if unresolved, display an unavailable resource in their original position.
HTML dependencies use an HTML parser.

The parent Research Session streams native tool-turn text deltas optimistically
when the provider supports them. They are transient presentation: the Host
resets them when the same turn contains tool calls or commits text that differs
from what streamed, a provider attempt fails or is cancelled after emitting
text, the Run defers on a dependency, the next provider turn begins after a
completed one (a steer, follow-up, control command, Child result, or Artifact
correction continued the Session), interrupted generation is recovered, or
citation/Artifact finalization changes the terminal text. Persisted Request
Snapshots, Assistant Turns, tool settlements, and the canonical result remain
the recovery authorities.

### Web Search

When Exa or Tavily is configured, Research can search Web passages as peer
evidence through one provider-neutral tool, `search_web`. Search and Extract use
independently ordered failover chains. Failover occurs only for provider
failures, never to seek a subjectively better result; malformed individual
results are dropped and reported. An empty Search result stops the Search
chain, while an Extract that yields no usable text counts as a provider failure.
Result URLs become inert resource handles that only an explicit `read` or
`view` fetches, under the
[Resource acquisition](resource-reading.md#registration-and-acquisition) rules.
A configured Agent Browser joins the Extract chain at its end unless
`extract_providers` names it elsewhere
([Public Web Sources](configuration.md#public-web-sources)): when the direct fetch
failed or held no text, the chain's steps run in order and the first usable text wins,
so a browser at the end renders the page in the Run's browser only when no hosted
provider supplied any. `read(..., rendered=true)` asks for that rendering directly
([Rendered reads](resource-reading.md#rendered-reads)).

### Agent Browser

A deployment that configures an [Agent Browser](architecture.md#agent-browser) gives
Research four tiers of reading ([ADR 0032](adr/0032-the-agent-browser.md)): `read(url)`
over direct HTTP, always first; the configured hosted Extract chain; a Rendered Read,
for a page whose content a script builds ([Rendered reads](resource-reading.md#rendered-reads));
and the `browser` tool, for a task that needs interaction. The tool's description
teaches the tiers: read pages with `read`, pass `rendered=true` only when a read
returned a JavaScript shell, and use `browser` only for a search form, a filter,
pagination behind a button, or a file behind a download control. Fast has no tools, so
it has no browser.

- **An Agent Page for each Agent Session.** An Agent Session's first `navigate` leases the
  Run's browser, if the Run holds none, and opens that Session's Agent Page. Any other
  first action answers that no page is open and leases nothing. A Child Session holds the
  tool by default ([ADR 0025](adr/0025-a-child-inherits-capability-not-authority.md)) and
  has an Agent Page of its own in the same browser, so two Children never see each other's
  cookies or pages. When a page closes, and how long the Run holds its browser, is in
  [Architecture](architecture.md#agent-browser).
- **Actions.** `navigate`, `snapshot`, `find`, `back`, and `wait` (for text, for text to
  go, or for seconds); `click`, `type` (optionally pressing Enter), `select`, `press`,
  and `scroll`; `screenshot` and `capture`; `upload`, which is offered only where the
  Run has a workspace (`trust`); and `register`, `login`, and `inbox`, which belong to
  [Agent Accounts and the Agent Mailbox](#agent-accounts-and-the-agent-mailbox). Only a
  configured capability's actions are offered, and
  no configured value appears in the description or the schema, so changing a timeout
  never changes a pinned plan. The tool is not read-only and never replays: each call runs
  alone, and a call pending at a crash settles its outcome as unknown.
- **Snapshots and refs.** An action that changes the page returns the page's frame
  (`[browser: <action> | page: <url> | title: <title>]`, with `| HTTP <status>` when the
  page answered 400 or above, which is reported and never a failure) and a depth-limited
  accessibility snapshot whose `[ref=eN]` markers name the elements the next action
  uses. A ref comes from the latest snapshot or `find` of its page and acts only on that
  page. `find` returns the snapshot lines that contain a query, at most 30, with their
  refs, and reaches elements below the snapshot's depth. A snapshot beyond the result
  bounds (51,200 UTF-8 bytes or 2,000 lines) is kept whole in the workspace and its head
  shown, with the `read` call that returns the rest; without a workspace the call says the
  full snapshot is unavailable and does not fail, because its action completed and a
  failure would invite a repeat.
- **Popups, dialogs, downloads.** A popup or a new tab becomes the active page once the
  call that opened it has acted and the page it acted on has settled, and the result says
  so. One that opens later becomes the active page when the next call that names no ref
  begins, and a call that names a ref acts on the page its ref came from. A page that
  closes itself returns the previous page, or leaves none until the next `navigate`. `alert`,
  `confirm`, and `beforeunload` dialogs are accepted, a `prompt` is dismissed, and each
  is reported with its message, because refusing one would undo what the call set in
  motion. A file a page downloads is admitted and named in the result
  ([Browser captures and downloads](resource-reading.md#browser-captures-and-downloads)).
- **Captures and screenshots.** `capture` admits the current page as a new citable Web
  Resource and returns its first window as `read` would. A screenshot is the page's
  pixels attached to the result against the Run's image budget
  (`answer.generation.max_images` and its byte and pixel limits, which `view` shares), and
  it is context, never evidence. Everything else a page yields, its text, snapshots, and
  screenshots, is untrusted model context; only a capture or a download can be cited.
- **Failures.** A failed call is an error result with one fixed sentence per reason: no
  page open, the page closed, the page lost when the browser disconnected, no element
  with the ref, an element not clickable within the action timeout, a key the browser does
  not know, no earlier page, a target that is not a file input, a wait that timed out, a
  page that did not load, and a pool that is busy or unreachable. Driver error text
  enters only the first line of a failed action.
- **A CAPTCHA stops the path.** On a CAPTCHA or any other human-verification check the
  model stops that path and reports it; the tool's description says so
  ([Security](security.md#agent-browser-boundary)).
- **Recovery.** A recovered Run starts with no page and leases a fresh browser at its next
  `navigate`, so the next call that is not a `navigate` says that a Run that resumed after
  an interruption starts with no open page. Captures and downloads that settled are
  restored without a browser.

### Agent Accounts And The Agent Mailbox

A Run with an Agent Browser offers `login`, and `inbox` when the deployment also configures
an Agent Mailbox; it offers `register` only if it may register
([ADR 0034](adr/0034-agent-accounts-and-the-agent-mailbox.md); what keeps a password from the
model is in [Security](security.md#agent-accounts)). An Agent Session acts as an identity of
its own: the tool's description says never to type the owner's details into a form, and that
DlightRAG makes every password and fills it by ref.

A Run may register when the deployment allows it
([`account_registration`](configuration.md#agent-browser), on by default) and its owner's
switch for new sign-ups, which is on until the owner turns it off in Settings, was on when the
Run was accepted. Acceptance reads the switch once, as it reads the Profile Memory capability,
pins the answer in the Run's prepared input, and plans the browser tool with it, so the
accepted plan and the tools the Run executes agree, and what the owner switches afterwards
changes no Run already accepted. The allowance stays the ceiling and is read again at
execution: a Run pinned to register under one since withdrawn is composed without `register`
and refused as incompatible, like any Run whose tools changed. A Child's tools follow its
parent's, and a Child's registration still lasts for the Run alone. A Run that cannot register
is told of no `register`: the tool's description, its action lines and argument descriptions,
and the sentence of a refusal name only the actions the Run has, and an `inbox` window opens
at a `login`. Its description also says that the Run does not open new accounts, so the Agent
must not create one on any site by filling a sign-up form itself; that is an instruction and
not a control ([Security](security.md#agent-accounts)).

- **`register`** acts on the page `navigate` opened and leases nothing. It takes
  `password_refs` (one or two, such as a password and its confirmation), and optionally
  `email_ref` and `username_ref`; the model types a username into its field first. It checks
  every ref (the element exists, is of the right kind, and is in a frame of the page's own
  site) before it fills anything, sizes the password to the smallest `maxlength` the
  password fields state, and refuses below 12 characters. The address is the one the account
  already has, which is filled again; else, with a mailbox, the owner's mailbox alias for the
  site (a Child gets a random one of its own), which is filled; else the address the Agent typed,
  which must hold one `@` with text on each side of it and no whitespace. The username is the
  one typed, 1 to 128 characters with no control character. A new account needs at least one
  of them. It then generates the password, fills it, and records the account before the site
  has accepted the form, because DlightRAG cannot see the site's verdict. A refused sign-up
  leaves the record that `login` will fail with, and so does a password the site rejected.
- **A password reset is a registration.** `register` on a form of a site whose account
  exists gives the account a new password, keeps its account id and address, and replaces its
  envelope. That is also how an account recovers when no key opens its envelope: `login` with
  `email_ref` alone, on the site's reset request form, fills the account's address and opens
  the `inbox` window, the reset mail's link opens with `navigate`, and `register` on the reset
  form seals a new password.
- **`login`** takes any of `email_ref`, `username_ref`, and `password_refs`, and fills the
  account this site has for the Session: a Child's own Run-scoped account first, else the
  owner's. It opens the password's envelope only when a password field is named, and
  refuses an account that lacks the field named, or whose envelope no key opens, with the
  site's reset path where the Run may register. Once it has filled the password of an owner's
  account it records the time as the account's last use, which Settings shows, whichever
  Session logged in: a Child's login with the owner's account records it, and a Child's own
  account, which lives in the worker's memory, has none. A login that fills no password, the
  reset request's, records nothing. A store that cannot record it is logged and does not fail
  the login.
- **What a result says.** Both end like any action that changes the page: the frame, its
  notes, then a sentence (the account recorded or reset, with `for this owner's later Runs`
  or `for this Run only (a Child Session's account)`, and for a mailbox alias a pointer to `inbox`;
  or the fields filled), then the bounded snapshot, in which a password is only
  `********`. They carry no Evidence. A refusal fills nothing and stores nothing. The
  reasons: no key ring; a page with no `https` registrable domain; a ref that is stale,
  in a frame of another site, or not a password, or not a text or email, field; a
  `maxlength` below 12; an address or username that cannot be recorded; a new account with
  neither; a fill that failed or that the page changed, which clears what it filled; an
  account that could not be stored, which does the same and tells the model not to submit;
  no account for the site, or one without the field named; and an envelope no key opens.
- **`inbox`** needs no page and no lease. It shows mail that the mailbox aliases of this
  Agent Session's accounts received since its latest `register` or `login` in the Run (`login`
  alone where the Run cannot register), from two minutes before it, because the time is the
  bucket's own clock; the Session's other mailbox aliases in the Run stay in the window.
  Before either action it says so, and so it does for accounts with no mailbox alias, an
  address the Agent typed being none. It shows the newest five messages of its mailbox
  aliases, each with its time, mailbox alias, sender, subject, up to four links, and up to
  five codes, says how many more there were, and says when a mailbox alias holds more than a
  listing reads ([the bucket's contract](configuration.md#agent-mailbox)). A link over 2,048
  characters or beyond the first four is only counted. A code is a token of four to nine
  characters that looks like one. A message over 1 MiB is listed and not read, and one that
  cannot be parsed is listed as such. An empty window is not an error: the result says mail
  can take a minute and to call `inbox` again after a `wait`. A bucket that cannot be read
  is reported by its error code alone ([Operations](operations.md#agent-mailbox)). Mail is
  untrusted context, and a link in it is followed with `navigate`
  ([Security](security.md#agent-accounts)).
- **Subjects and recovery.** `register` and `login` name the page they act on, as the other
  actions do, and `inbox` names its mailbox aliases. A recovered Run has no Run-scoped
  account, no window, and no page, as it has no browser; the owner's accounts are intact, and
  `inbox` needs a new `register` or `login` first.

## Context And Model Budgets

Each call uses an immutable model profile pinned by normalized provider, model,
and endpoint. It supplies context, input, output, image, and reasoning facts. An
uncatalogued endpoint resolves to a generous fixed fallback profile with
best-effort reasoning controls; when the real endpoint is smaller, the
provider's own rejection reports it.
[API Family](configuration.md#api-family) selects the wire without changing this
capacity profile or the Context Policy. Replay and history boundaries are owned
by [Agent Session Recovery](run-runtime.md#agent-session-recovery).

The Context Policy independently reserves output, dynamic context, retained
tail, episodic continuation, and minimum input. It carries no estimator-safety
margin: a provider rejection followed by compaction and a retry of the same turn
covers that boundary. Each Tool result, including the Evidence text frozen into
it, is fitted to one absolute, model-aware observation capacity; the proactive
compaction trigger is what bounds the request as a whole. Provider output is
limited by both model output capacity and remaining physical context. Full
attachment bytes never enter model context: only bounded text windows, capped
observations, and budgeted images do.

Acceptance fits the caller's history to every model call the Run can still reach,
each measured as it will be sent and against the model that serves it: Fast's
planning (`extract`) and generation (`query`) calls, which must keep the full
dynamic-context reserve with no history at all; Research's planning call
(`extract`) and first Agent request (`query`); and, when `auto` has both modes
valid, the routing call on the `keyword` model. An explicit Fast request that
cannot keep its reserve is refused; `auto` resolves without Fast instead, and a
routing call that cannot fit resolves `auto` to Research, which needs no routing
(a Research request that cannot fit either is refused as too long). A Fast Run
measures its durable Session history against the same Fast calls before it
compacts. Acceptance and the Fast Run build these calls from one definition, so
they agree on which calls exist and how each is measured.

### Conversation History

Fast and routing continue the conversation, not the tool work in it: an earlier
Research turn's tool calls, tool results and provider state stay out of their
history. The images that turn's tools viewed are what its answer saw, so Fast
keeps them as attachments of the latest user message before them (the question,
or the steer the turn was answering), and a follow-up sees them. Routing and
retrieval planning read the history's words alone.

A Follow-Up continues on the Lane it came from, at its tip. A Fork does not: it
opens a new Lane at the state its parent Run settled at, so a branch from an
earlier turn sees that turn's summary and retained tail rather than everything
the conversation has since become, and it starts from that state's projection
rather than the source Lane's current one. Either kind derives its history
from its Session branch point and injects none: every accepted Run records an
Agent Session, so the fold at that branch point is the context.

### Prefix Cache

A Research request is the previous request plus new material, so a provider
prefix cache can reuse it: the Session fold only appends, and admitted Evidence
text is frozen into the Tool result that produced it. What a Run composes
follows the transcript, which already states the question as the Run's own User
Entry, in this order: the Session-notes statement, the question's Resource
manifest and attached images, memory, and the Skill catalog, all byte-stable
for the Run (a Tool's usage text travels in its description, with the Tool
definitions); last comes the lane of admitted evidence images, which
re-renders every request because evidence pixels are not durable. Nothing else
is composed per turn. A replayed Tool call's arguments are serialized with
sorted keys, the same bytes whether the Session was held in memory or read back
from PostgreSQL. The first request of a follow-up Run is therefore the previous
Run's last one plus new material too. No system message states the current time
— a prefix that moves with wall time forfeits the whole cache — so a Research
agent reads the clock from its environment and a Fast answer's own request
states it
([ADR 0015](adr/0015-prompt-prefix-stability-and-cache-anchored-accounting.md)).

Research's compaction trigger is measured from the character estimator,
corrected by the prompt size the provider billed for the last request that
carried no pixels; a Fast Run's history check uses the estimator alone. Each
Research turn's billed prompt and cache hits are aggregated in the Run trace
(`prompt_cache`). Response `input_tokens` and
`input_tokens_details.cached_tokens` feed these same accounting owners; selecting
Response does not introduce a second estimator or permission to silently
truncate the provider request.

### Compaction

A compaction the projection cannot advance is declined rather than failed. Once
a retained tail is a whole oversized exchange, the uncovered prefix holds no
complete exchange for a summary to state, and no smaller tail changes that, so
the run assembles its request against the projection it already has; a request
that genuinely cannot fit still fails on the hard input limit, by name. The
decline is recorded on the operation state for that turn, which is what keeps an
over-trigger turn from asking for the same impossible compaction again instead
of reaching the provider.

A committed compaction keeps the run's re-readable identities beside its typed
summary: the Evidence ledger supplies its citation handles, and the run's
committed spill rows supply the newest spilled Tool outputs, whose bytes and rows
survive the covered prefix while their receipts do not. Spills and published
Artifacts each claim a reserved share of the summary's handle list and Evidence
fills the remainder, so no class can crowd another out; the spill read is
bounded and ordered newest-first by the producing effect intent. A spill handle
states the `read(resource_id=…)` call it authorizes, and reading one back admits
no Evidence and mints no citation handle — a spilled output is continuation
memory, never a source.

A published Artifact joins the same re-readable family: publication registers it
as a Resource of its Agent Session
([ADR 0023](adr/0023-a-published-product-is-a-resource.md)), and the summary
keeps naming its `read(resource_id='artifact-…')` handle after the receipt that
taught it is compacted away. A later turn of that conversation therefore reads
the version it published, edits it in its own working copy, and publishes a new
version — each version keeping its authoring Run, digest, and Evidence — while
another conversation cannot reach those bytes at all
([adoption](resource-reading.md#earlier-runs)).

### Session Notes

Memory belongs to the Agent Session. When the execution environment is enabled,
every Run, Fast included, binds the Session's notes, and a Follow-Up or Fork
never copies its parent Run's. The notes are laid down into the Run's own
Workspace Epoch and recorded in its Inventory before the first request, so a
conversation, its forks, and every later turn of the Session read the same set.
Every request of a Run or Child Session that holds the `write` tool carries one
static list of the notes the Run started with ([Prefix Cache](#prefix-cache));
Fast never states it, because it has no tool that could read a note — the files
are there for the turn after it. A Session whose plane cannot be read, a note
the plane refuses for budget, and a promotion that fails are degradations
recorded on the Run's trace: a Run always proceeds, because a Run that cannot
read its Session's memory still has its transcript, its Evidence, and its
Products. A recovered attempt states what its own epoch holds rather than
re-materializing, because it may have written a note of its own before it was
interrupted. [Run retention](run-runtime.md#retention) never touches the
Session's memory.

The compaction summary also names the Session's notes this Run holds: the files
under `notes/` in its Agent Workspace, bounded to a small list and named by the
`read(path=…)` call that reads each one again. They come from the Run's own
Workspace Inventory — the framework's observation of the working copy, carried
forward across a verified Workspace Epoch handoff — so the summary names what
this Run can actually open, and reading one back is an ordinary workspace read
that admits no Evidence either.

## Answer Input And Packing

Fast's generation call receives structured messages, not raw `contexts` JSON:

```text
system policy
episodic summary (when present)
bounded prior history, with the images earlier tools viewed (when supplied)
current user message:
  User-attached images
  Knowledge graph evidence, naming its documents as [N] (when retrieved)
  Knowledge-base evidence, each excerpt labeled [N-M]
  Current time
  Question
standing Profile Memory recall (when present)
```

Each citation marker labels one excerpt; a sent image and its chunk's text share
that excerpt's `[N-M]` label. Fast renders its evidence through the same
Evidence ledger as Research: documents are numbered by first appearance, and the
answer's citations resolve against the ledger's rows, so a marker always names
the excerpt the model read under it. Retrieved document images are preceded by
their text label and sent only when they fit.

`top_k` controls KG breadth. `chunk_top_k` caps the reranked text/visual chunks
retrieval returns, and packing fits that list to the query model's remaining
input capacity and image budget; a removed chunk is not replaced.

- Pure visual chunks whose image cannot fit are removed.
- Mixed text+image chunks keep text when the image is skipped.
- Final exact serialization removes whole chunks from the reranked tail and
  rebuilds prompt/citation indexes until it fits; it never truncates a chunk,
  reorders survivors, or retrieves again.
- Returned contexts/sources use the final admitted chunks. Use `/retrieve` for
  the broader pre-answer set.

A JPEG, PNG, or WebP image already within the byte and edge limits, with no EXIF
orientation to apply, passes through unchanged; any other image is re-encoded as
JPEG. Recompression honors configured quality and geometry floors; images that
still do not fit are skipped rather than degraded further.

DlightRAG uses LightRAG `aquery_data()` as the context/reference seed rather
than `aquery_llm()`, because final evidence may include BM25, direct visual,
federated, and reranked results.

## Citation And Presentation Finalization

After generation, DlightRAG validates inline markers against the final packed
context:

- unknown markers are removed;
- a chunk marker pointing only to Markdown headings degrades to its document
  marker because the chunk supports no factual claim;
- cited sources/references are derived from the surviving inline markers.

The shared Answer Markdown grammar identifies citation tokens once for cleanup
and source selection. Highlight extraction, browser badges and portable Markdown
links use that same syntax. Real Markdown links (including numeric reference
labels) take priority; code, math, escapes, link destinations/labels, reference
definitions and image alt text are not citations. Cleanup and public-link
projection edit exact source spans and preserve every other source character,
including line endings and table escapes. A `References` heading never authorizes
deleting prose. Prompts ask the Model to omit duplicate bibliographies; source
authority continues to come only from validated inline citations.

Finalization also derives `evidence_images`, and each reader projection derives
the ordered Markdown/Artifact/image `parts` from the stored Markdown. Image
placements and `evidence_images` come only from server-derived evidence and
Artifact identities, never from a model-written URL. Every published Markdown
Artifact is citation-finalized against the same admitted context and stores its
cited sources under that Artifact's resource identity. Artifact presentations
therefore resolve their own citation indexes without borrowing sources from the
chat Answer or another Artifact. Streaming may expose tokens immediately, but
the final `done` result contains normalized text and authoritative metadata.

Semantic highlights run only after citation validation. They mark up to three
verbatim phrases in each cited chunk that support the finalized answer's citing
sentence. Web always requests them; REST, MCP, and Application callers opt in,
and `answer.citations.highlights.enabled` gates every caller. `/retrieve` never
emits them. Timeout or failure leaves original sources unchanged.

## Answer Attachments And Resources

Attachments, caller links, Web Search results, and URLs the Agent chooses become
Resources of one Answer Run, which Research reads as bounded text, views as pixels,
and, with execution enabled, copies into its Agent Workspace; they never become corpus
data. [Answer Resource Reading, Viewing, and Copying](resource-reading.md) owns that
contract, and their blobs follow [Run retention](run-runtime.md#retention).
