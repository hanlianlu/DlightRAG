# Observability

DlightRAG's traces are a product surface: they are how a Run is reviewed, how
cost is attributed, and what evaluators and dashboards target. This document is
the contract. It is enforced by `tests/unit/test_observability.py`, which fails
when a name leaves the vocabulary, when a name is unused, when a name carries a
dynamic value, or when core code restates an observation kind, and by
`tests/unit/test_observability_wire.py`, which reads what the OTLP exporter
renders — a trace's tree, its attribution, its usage buckets, and the masking in
front of them — off a recording span exporter.

## One unit of work, one trace

A trace is one self-contained unit of work. The root observation is opened
**where that work is executed**, not in the HTTP request that admitted it:

| Unit of work | Root | Opened by |
| --- | --- | --- |
| Answer Run (Fast or Research, root or Child) | `run-answer` (`agent`) | the claimed Run's executor, inside the worker that owns the lease; a Child Run is driven inside its parent's trace and opens no second root |
| Document ingestion | `ingest-documents` (`chain`) | the ingestion engine for the accepted document batch |
| Retrieval Run | `run-retrieval` (`chain`) | the claimed Run's retrieval executor, inside the worker that owns the lease |

Everything the unit of work orchestrates nests inside its root: agent Turns,
Tool calls, provider calls, embeddings, reranks, and retrieval. A Run that
executes in a background worker therefore produces one trace, and none of its
own calls can surface as a root: the root span is open for the whole operation,
so the OTel context carries it into every nested task and thread it starts.
Work a library performs outside any root is the one exception, described
below. A Corpus Mutation Run that ingests is traced by its `ingest-documents`
batch. One that only deletes or resets opens no observation of its own; when a
delete makes LightRAG rebuild entities and relations that other documents share,
those model calls can surface as traces of their own.

Capability checks are not a unit of work. The probes a process runs at start
(whether the chat model accepts images, whether the embedding endpoint accepts an
image query and a fused document), the same image probe when a published model
catalogue makes it run again, and the sweep that resumes interrupted ingestion open
no observation, so starting a container adds nothing to the trace list.

Traces are grouped by **Agent Session** (`session_id`) and **owner**
(`user_id`). A conversation is a session; each of its Runs is its own trace,
which keeps traces small and makes the session view a replay. Attribution is
applied once, at the Run boundary, through `Telemetry.trace(...)`; no call site
threads ids, and no observation restates them.

### Known boundary: model work outside any root

Ingestion nests: an `ingest-documents` trace holds its embeddings and completions
in one tree under the batch root, with no separate completion or embedding root.

What can still surface as a trace of its own is model work LightRAG performs
outside any root: the entity and relation rebuild after a delete that other
documents share, and whatever the startup sweep finds to resume. Those calls have
no ambient parent, and they are exported rather than dropped, because each is real
spend a reviewer would otherwise never see.

The exporter owns an isolated OpenTelemetry TracerProvider, never the
process-global one
([ADR 0036](adr/0036-traces-hold-only-product-work-and-usage-counts-each-token-once.md)).
The spans other libraries open (FastAPI's request span, the MCP SDK's, HTTP
clients, database drivers) are therefore neither exported nor ancestors of an
observation, and every observation DlightRAG opens is a child of one DlightRAG
opened or a root. The root is opened in the worker that owns the work, not in the
request that admitted it, so a request's own headers do not reach it. Sharing the
global provider is what once produced traces with no root and no name: a framework
span became the recording parent of an observation, and it was filtered out of the
export, leaving the observation a child of a span the collector never received.

## The span vocabulary

Names are an API. They are verb-first, stable, and closed: a new observation
requires a new registry entry, not a new string.

| Name | Type | One observation covers | Metadata |
| --- | --- | --- | --- |
| `run-answer` | `agent` | One claimed Answer Run, start to settlement | `run_id`, `parent_run_id`, `workspaces` |
| `run-retrieval` | `chain` | One claimed Retrieval Run, start to settlement | `run_id`, `workspaces` |
| `generate-answer` | `chain` | The Run's generation phase, after mode resolution | `resolved_mode`, `history_turns`, `query_image_count`, `semantic_highlights` |
| `generate-agent-turn` | `generation` | One model invocation in the agent loop | `model`, `provider`, `endpoint_fingerprint`, `api_family`, `tool_names`, `tool_choice` |
| `compact-session` | `generation` | The compaction summary invocation over a rich tool transcript | `model`, `provider`, `endpoint_fingerprint`, `api_family` |
| `execute-agent-tool` | `tool` | One Tool call, including its arguments | `call_id`, `intent_id` |
| `generate-completion` | `generation` | One non-agent provider completion (Fast answer, planner, highlights, ingest extraction) | `provider`, `endpoint_fingerprint`, `api_family`, request parameters |
| `embed-text` | `embedding` | One embedding request | `context`, `modality`, `provider`, `endpoint_fingerprint`, `request_count`, `inline_image_bytes` |
| `rerank-passages` | `span` | One rerank stage of a retrieval | `chunk_count`, `top_k` |
| `call-rerank-model` | `span` | One rerank provider call inside that stage | `document_count`, `top_n` |
| `retrieve-context` | `retriever` | One knowledge-base/web retrieval | `workspaces`, `top_k`, `chunk_top_k`, `has_filters`, `federated_rerank` |
| `plan-retrieval` | `chain` | One query plan | `workspaces`, `history_messages` |
| `highlight-sources` | `chain` | One semantic-highlight enrichment | `source_count`, `text_chunk_count` |
| `ingest-documents` | `chain` | One accepted document batch | `document_count`, `doc_ids` |

**Rules**

- **Verb first, no dynamic values.** A name identifies the operation, never one
  execution of it: the model, workspace, Run, document, or strategy belongs in
  `model` or `metadata`, so a completion is `generate-completion` whatever model
  serves it.
- **A model invocation is a `generation`.** One `generation` per model call in
  an agent loop, interleaved with the `tool` calls it requested. Never one
  `generation` wrapping a whole loop: per-step reasoning, tokens, and cost must
  stay visible, and the step that blows up the context window must be
  identifiable.
- **A Tool call is a `tool`, nested under the agent observation** that
  orchestrated it, as a sibling of the `generation` that requested it. Its input
  is the arguments the model chose, bounded; its output is a summary, because
  the full result reaches the model as transcript text.
- **Kinds come from the registry**, never from a call site (`SPAN_TYPES` in
  `engine/ai/telemetry.py`). The registry's kinds are `agent`, `chain`,
  `embedding`, `generation`, `retriever`, `span`, and `tool`; how each one is
  rendered is the section below.
- **Root input/output answer a reviewer's question.** The root's input is the
  user's question and its output is the answer (or the terminal outcome for a
  deferred or failed Run). Raw payloads belong in `metadata`.
- **Model, usage, and cost ride on the observation.** `model` is rendered as
  `gen_ai.request.model`, and `provider` — the configured provider that the call is
  against — as `gen_ai.provider.name`. The provider is stated in the app's own
  vocabulary and rendered onto the registry's enum on the way out: the app's
  `gemini` is the registry's `gcp.gemini`, its `azure_cohere` is
  `azure.ai.inference`, and the providers the registry does not name (`jina`,
  `openai_compatible`, `voyage`) pass through as the app spells them rather than
  being dropped or renamed to something they are not. The provider's counters are
  mapped once, in
  the provider helpers of `engine/ai/providers/base.py`, onto the buckets the GenAI
  registry describes, which **nest** rather than partition: `input` is the whole
  prompt, `cache_read_input_tokens` is the subset of it a provider-managed cache
  served (omitted when the provider reports no caching at all, which is a different
  fact from a reported zero), `output` is the completion with its reasoning, and
  `total` is the total the provider states or, when it states none, `input +
  output`. The registry says `gen_ai.usage.input_tokens` *"SHOULD include all types
  of input tokens, including cached tokens"* and that
  `gen_ai.usage.cache_read.input_tokens` *"SHOULD be included in
  `gen_ai.usage.input_tokens`"*, so a consumer that sums `input` across spans gets
  the prompt total and one that wants the **fresh** prompt tokens — what a price
  table prices at the uncached rate — derives `input - cache_read_input_tokens`. A
  subset is never added to the whole it sits inside, or a cache hit is counted
  twice. OpenAI, DeepSeek, Gemini and the Responses wire count a cache hit inside
  the prompt already; Anthropic states cache reads and writes *beside*
  `input_tokens`, and the helpers add both back, so `input` is the prompt either
  way. A dialect the helpers do not recognise contributes no usage attribute, and
  because a price table keys on the buckets a span carries, an unrecognised dialect
  is a visible gap rather than a silently free call. An embedding observation
  reports the provider's billable `total_tokens` as `input` and `total`. Cost is
  never computed here.

### Rendering on the wire

One exporter renders every observation
([ADR 0040](adr/0040-observability-lands-in-clickstack.md)), and it states no span
name of its own: the composition root hands it the registry (`SPAN_TYPES`) and the
provider-dialect mapper as constructor data.

**Where an attribute name comes from.** Every `gen_ai.*`, `code.*`, `session.*`,
and `error.*` key above is the
[OpenTelemetry GenAI registry](https://github.com/open-telemetry/semantic-conventions-genai)
and the main semantic conventions — checked against *those* files, never against
this document or the exporter, which once agreed with each other while both were
stale (`gen_ai.usage.input_cached_tokens` existed in neither registry, and the
deprecated `code.filepath`/`code.lineno` had shipped in place of the stable
`code.file.path`/`code.line.number`). Only the `dlightrag.*` keys and
`gen_ai.usage.cost` are the product's or HyperDX's own.

- Every span carries `dlightrag.observation.kind`, the registry's kind — the
  aggregation key a dashboard, a saved filter, or an evaluator targets. Its span
  **kind** follows the convention: a `generation` (a call to a model) and an
  `embedding` are `CLIENT`, because the work happens in another process, and a
  `tool` is `INTERNAL`, because this process performs it. An agent, chain,
  retriever, or plain span is DlightRAG's own orchestration and is `INTERNAL`.
  Span *names* stay as they are: the convention states them as SHOULD, and these
  names are a deliberate API.
- The kinds the GenAI convention names also carry `gen_ai.operation.name`:
  `generation` is `chat`, `tool` is `execute_tool`, `embedding` is `embeddings`.
  An agent, chain, retriever, or plain span keeps its kind and carries no
  operation.
- The usage buckets become `gen_ai.usage.input_tokens`,
  `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, and
  `dlightrag.usage.total`; a charge a provider reported becomes
  `gen_ai.usage.cost`. Every one of those names is the registry's except
  `dlightrag.usage.total` and `gen_ai.usage.cost`, which are DlightRAG's and
  HyperDX's own dialect: the registry states usage and cost as *metrics*, and the
  total is kept because a provider's stated total is a fact no consumer can derive
  when it bills differently from the buckets.
- The two attributes the convention calls **Required** and the seam cannot infer
  are stated by the call site that knows them: `gen_ai.provider.name` (the
  configured provider the call ran against, in the app's vocabulary, which the
  exporter renders onto the registry's enum as the bullet above describes) and
  `gen_ai.tool.name` (the tool an
  `execute_tool` span ran). A call site that does not know one leaves it out — the
  seam states what it was told and never guesses. A provider neither the app nor the
  registry names is passed through as it was stated, because the convention defines
  the attribute as the instrumentation's best knowledge. `gen_ai.response.model` is
  emitted when the wire states the model that answered, which is a different fact
  from the alias that was asked for, and `error.type` is emitted on a failed span:
  the application's own classified kind where it has one, and the exception's class
  name where it does not.
- `input`, `output`, and `metadata` become the JSON string attributes
  `dlightrag.input`, `dlightrag.output`, and `dlightrag.metadata`, each of
  them already the classifier's output and each bounded at 64 KiB. A model's
  request parameters are one JSON attribute, `dlightrag.model_parameters`.
  A Tool may state bounded facts about its own call, which the Runtime puts in its
  span's `dlightrag.output` beside `is_error` and `cached` — those two are the
  Runtime's and a Tool cannot overrule them. The Web search tool is the one that
  does: `web_search_provider` (empty when no provider answered),
  `web_search_hits`, `web_search_degraded`, and `web_search_cost_dollars`, which two
  dashboard tiles group by — calls per provider with failovers marked, and cost with
  passages per interval. Provider failures are also log records with `provider`,
  `operation`, `reason`, and `status` beside the message, so an outage is countable
  from the logs without a tile of its own.
- A failure whose kind the application already classified carries it as
  `dlightrag.error.kind` (`dependency_unavailable`, `run_abandoned`,
  `workspace_integrity_error`, …), beside the status message rather than inside
  it: an alert groups on the attribute, and the message stays for the spans
  recorded before the kind existed.
- Attribution is `session.id` and `user.id` on every span the `trace` scope
  covers; the resource carries `service.name`, `service.version` (the release),
  and `deployment.environment.name`.
- An exception that escapes the body sets the span's status to `ERROR` with the
  message as its description; a caller that named its own `level` first keeps it,
  and a cancellation is not relabelled a failure.

### Log records

The same seam carries the process's own log records, so a trace and the lines it
logged meet in ClickHouse rather than only in a terminal. stdout does not change:
the export is one additional handler on the root logger, and the process keeps
writing to its console exactly as before.

- A record is rendered where it is logged — masked by the classifier, bounded,
  and stamped with the **trace id and span id of the span that logged it**, or
  zeros when no observation is open. That pair is what ClickStack's trace view
  joins on.
- Its severity is the Python level mapped to the OpenTelemetry names (`WARNING` is
  `WARN`, `CRITICAL` is `FATAL`); its message is the formatted record; its
  attributes are everything a caller attached with `extra` — which is where a
  secret-named field is masked — plus the code location the record came from, and
  `exception.type`/`message`/`stacktrace` when it carries one. The location is the
  registry's stable `code.file.path` / `code.line.number`, with
  `code.function.name` qualified the way the runtime names it
  (`module.function`), not the bare `funcName`.
- The handler does no I/O: it hands the record to a processor whose queue is
  bounded and drops the oldest record when it is full, so a logging call never
  waits on the network, and a destination that is down costs records rather than
  latency. A record the export cannot shape is dropped, never exported unmasked.
- The export does not export its own voice: this package's logger records — the
  breaker's warning above all — stay on stdout, because a per-batch line about a
  failing export would itself be a record that fails to leave.

## Redaction

One classifier decides what leaves the process, and monitoring is never switched
off ([ADR 0039](adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md)).
Every value that would be exported — an observation's `input` or `output`, a log
record's message, a browser replay field — is **private** or **exportable**. A
private value is masked to its presence, length, and digest
(`[private: 1,234 chars, sha256:ab12…]`), never carried in the clear; a *credential*
is the exception that keeps neither: it becomes `[redacted]`, because a digest of a
low-entropy secret is a brute-force oracle. An exportable value is exported, still
through the secret masker below.

- **Private** is credentials and tokens, inline image bytes, the browser's DOM
  text and input values, request and response bodies, console output, memory
  record bodies — which were already exported as counts and character totals
  rather than as records — and **a path that names a home directory**, because
  that path names the operator: the account name and the layout of a personal
  machine. It is private as *content*, not as a credential, so it is rendered by
  the ADR's content rule rather than redacted: the directory goes and the basename
  survives, `[private path: toolchain.py: 88 chars, sha256:ab12…]`, which keeps a
  masked log line followable (the basename is what a reader follows) while the
  digest is the correlation handle for a file whose name is not distinctive. The
  rule reads the key *and* the value, so it fires for any path-shaped name the
  application logs (`search_toolchain.fd.path`, `cwd`, `workspace_directory`) while
  an endpoint path such as `/v1/chat/completions`, a container path such as
  `/app/.venv/…`, and a sentence that merely mentions `/Users/` stay exportable.
- **Exportable** is structural metadata — identifiers, counts, model and
  provider identity, endpoint fingerprint, durations, status, usage, error kind
  and error text, span and workspace names, and a path that names nobody — and the
  product's bounded review content: the Run root's question (at most 2000
  characters), its answer, a message list whose every text is bounded (4000
  characters), and Tool summaries. A review starts at one tree per Run and the
  trace table shows the question and the answer, which is why those two stay.
- The classifier is applied when an observation opens and when it updates, so a
  span cannot leak a query, an answer, or a Tool argument by updating later.
- An observation that names its own level keeps it: the adapter maps an
  exception that escapes the body to `ERROR`, but a Run that reported a
  cancelled or fenced outcome first is not re-labelled a failure.
- `mask` replaces secrets and image bytes in every exported payload. A secret is
  named by the same fragments that hide it from a settings dump (`api_key`,
  `api-key`, `authorization`, `token`, …); every non-empty value under such a
  name is replaced except a flag, and a count under a name that merely contains
  `token` (such as `max_tokens`); inline image bytes are removed from message
  telemetry before it leaves the process.
- The browser channel states the same rule in its own terms: replay stays on,
  `maskAllText` and `maskAllInputs` are set, and request and response bodies,
  console output, and canvas recording are not captured. A replay keeps the
  interactions — clicks, navigation, timing, errors, network status — and not
  the text they happened over. Replay is the one signal a deployment may sample
  (`observability.browser_replay_sample_rate`): the page flips one coin per
  session and an unsampled session initialises with replay off, while its errors
  and its RUM are reported exactly as a sampled session's are. Sampling is a
  volume knob; it is not an off state.
- A reported browser stack names minified, hashed assets. The production build emits a source map
  for each, hidden (no bundle points at one), and moves the maps out of the served directory into
  `frontend/.sourcemaps/<build>/assets/` (gitignored; `<build>` is the application entry's hash), so
  a stack can be symbolicated and the maps are never served. The container image does not carry
  them: build the same commit with `make frontend-build` and match the hash in the asset's name.
- Masking fails closed and never raises: a classification or mask that fails
  yields the placeholder, never the original value, and its exception never
  reaches the request path. Monitoring failure never fails the product either:
  an initialization that fails installs no-op telemetry, an unreachable
  destination drops telemetry from a bounded queue and nothing else, and the
  browser SDK is loaded off the render path, where a failure to load is ignored.
  **Turning the observing stack off is quiet**: an outage is reported once — one
  warning when the destination stops answering, one line when it answers again —
  and the failures in between are counted, not logged. A logging call never waits
  on the export, and a shutdown flushes under a timeout rather than waiting on a
  dead collector.
- The document-conversion stack's ONNX runtime is told to report nothing: its 1DS telemetry is
  on by default, and every DlightRAG process sets `ORT_DISABLE_TELEMETRY=1` before that runtime
  initialises and calls the binding's own switch too, so the process creates no uploader, event,
  or persistent device identifier. Sidecars that ship their own runtime are not covered here.
- Only DlightRAG's own observations are exported: the exporter runs on a tracer
  provider of its own, so no span another library opens in this process reaches the
  collector, and the HTTP client and database spans of those libraries cannot
  pollute the tree. Every payload the collector receives has therefore passed the
  masking above.

**Not provided.** Cost attribution: it needs a price table over `otel_traces` that is the
operator's to build.

## Cost

DlightRAG reports usage and never computes cost. The buckets above are the keys a
price table needs, and `gen_ai.request.model` is the DlightRAG model alias (such
as `deepseek-flash` or `voyage-multimodal-3.5`). That table lives in ClickHouse,
beside the traces it prices, and it is the operator's: this repository holds no
prices and ships no table, so until one exists a model's cost is simply not
computed. ClickStack's own LLM cost panel reads a built-in static catalogue and is
not the deployment's money account. A charge a provider reports itself
(OpenRouter's `usage.include`, in [Operations](operations.md#cost-and-recovery)) is
passed through as `gen_ai.usage.cost` and outranks any derived price.


## Deployment

The `observability` fields are in
[Configuration](configuration.md#observability). Startup does not call a
reachability check: tracing is enabled by configuration, and a failure to
initialize degrades to no tracing rather than to a failed boot. Local stack
startup, the ingestion key, and the first-run administrator are in
[Operations](operations.md#local-clickstack-observability).
