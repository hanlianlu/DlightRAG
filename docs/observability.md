# Observability

DlightRAG's traces are a product surface: they are how a Run is reviewed, how
cost is attributed, and what evaluators and dashboards target. This document is
the contract. It is enforced by `tests/unit/test_observability.py`, which fails
when a name leaves the vocabulary, when a name is unused, when a name carries a
dynamic value, or when core code restates an observation type, and by
`tests/unit/test_observability_wire.py`, which reads what a real Langfuse client
exports: a trace's roots, its attribution, and its usage keys.

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

Ingestion nests. The `ingest-documents` trace of release 2.0.40 (2026-10-03) holds
all 3,057 of its observations in one tree, the batch root with 421 embeddings and
2,635 completions beneath it, and no orphan completion or embedding root has
appeared since. Release 2.0.23 did detach them (86 image-embedding roots on
2026-09-30); that boundary has closed.

What can still surface as a trace of its own is model work LightRAG performs
outside any root: the entity and relation rebuild after a delete that other
documents share, and whatever the startup sweep finds to resume. Those calls have
no ambient parent, and they are exported rather than dropped, because each is real
spend a reviewer would otherwise never see. Neither has occurred in the measured
data, so this is what can happen, not what has.

Langfuse owns an isolated OpenTelemetry TracerProvider, never the process-global
one ([ADR 0036](adr/0036-traces-hold-only-product-work-and-usage-counts-each-token-once.md)).
The spans other libraries open (FastAPI's request span, the MCP SDK's, HTTP
clients, database drivers) are therefore neither exported nor ancestors of an
observation, and every observation DlightRAG opens is a child of one DlightRAG
opened or a root. The root is opened in the worker that owns the work, not in the
request that admitted it, so a request's own headers do not reach it. Sharing the
global provider is what once produced traces
with no root and no name: a framework span became the recording parent of an
observation, and it was filtered out of the export, leaving the observation a
child of a span Langfuse never received.

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
- **Types come from the registry**, never from a call site (`SPAN_TYPES` in
  `engine/ai/telemetry.py`). The registry uses these Langfuse observation types:
  `agent`, `chain`, `embedding`, `generation`, `retriever`, `span`, `tool`.
- **Root input/output answer a reviewer's question.** The root's input is the
  user's question and its output is the answer (or the terminal outcome for a
  deferred or failed Run). Raw payloads belong in `metadata`.
- **Model, usage, and cost ride on the observation.** `model` is passed to the
  adapter, which maps each provider dialect's usage counters onto Langfuse's usage
  keys. Langfuse prices every key on its own and takes `total` as their sum, so the
  keys are mutually exclusive buckets and a token is counted in exactly one:
  `input` is the prompt without the tokens the provider's prefix cache served,
  `input_cached_tokens` is that cache hit (omitted when zero), `output` is the
  completion with its reasoning, and `total` is the total the provider states or,
  when it states none, the sum of those keys. OpenAI, DeepSeek, Gemini and the
  Responses wire count a cache hit inside the prompt; Anthropic counts cache reads
  and writes beside `input_tokens`, not inside it. Either way `input` ends as the
  prompt minus the reads, so Anthropic's cache writes stay inside `input`. The
  dialect arithmetic lives once, in the provider helpers of
  `engine/ai/providers/base.py`. A dialect the adapter does not recognise
  contributes no usage keys. An embedding observation reports the provider's
  billable `total_tokens` as `input` and `total`. Cost is never computed here.

## Redaction

`langfuse_trace_sensitive_data` decides content capture at the seam: the adapter
drops an observation's `input` and `output` whether they arrive when it opens or
when it updates, so no call site can leak after the fact. Call sites that would
have to *build* a payload consult `capture_sensitive_data` first, so nothing
user-authored is even shaped while capture is off:

- Observation `input` is dropped at open time and again at update time;
  `output` is dropped at update time. A span cannot leak a query, an answer,
  Tool arguments, or an error detail by updating it later.
- An observation that names its own level keeps it: the adapter maps an
  exception that escapes the body to `ERROR`, but a Run that reported a
  cancelled or fenced outcome first is not re-labelled a failure.
- Error observations carry `level=ERROR` and, only when capture is enabled, the
  provider's message; otherwise the exception's type name (the literal `error`
  when the adapter maps an escaping exception).
- Structural metadata is not content and stays: counts, workspaces, model and
  provider identity, Run and Tool identifiers. A redacted trace must still be
  diagnosable, and none of those values is user text.
- `mask` replaces secrets and image bytes in every exported payload. A secret is
  named by the same fragments that hide it from a settings dump (`api_key`,
  `api-key`, `authorization`, `token`, …); every non-empty value under such a
  name is replaced except a flag, and a count under a name that merely contains
  `token` (such as `max_tokens`); inline image bytes are removed from message
  telemetry before it leaves the process.
- Only DlightRAG's own observations are exported: Langfuse runs on a tracer
  provider of its own, so no span another library opens in this process reaches it,
  and the HTTP client and database spans of those libraries cannot pollute the
  tree. Every payload Langfuse receives has therefore passed the masking above.

## Cost

DlightRAG reports usage and never computes cost. Langfuse infers it from a Model
definition in the project: the definition's match pattern is tested against the
observation's `model` attribute, which is the DlightRAG model alias (such as
`deepseek-flash` or `voyage-multimodal-3.5`), and its prices are keyed by the usage
keys above (`input`, `output`, `input_cached_tokens`). A charge a provider reports
itself (OpenRouter's `usage.include`, in [Operations](operations.md#cost-and-recovery))
is passed through as `cost_details`, and Langfuse prefers an ingested cost to an
inferred one. Until the project has a definition that matches an alias, that
alias's cost shows 0. The prices are the operator's to enter; the repository holds
none.

## Deployment

The `observability` fields are in
[Configuration](configuration.md#observability). Startup does not call
`auth_check`: tracing is enabled by configuration, and a failure to initialize
degrades to no tracing rather than to a failed boot. Local stack startup, keys,
and project bootstrap are in
[Operations](operations.md#local-langfuse-observability).
