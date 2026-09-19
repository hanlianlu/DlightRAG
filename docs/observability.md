# Observability

DlightRAG's traces are a product surface: they are how a Run is reviewed, how cost is attributed, and what evaluators and dashboards target. This document is the contract. It is enforced by `tests/unit/test_observability.py`, which fails when a name leaves the vocabulary, when a name is unused, when a name carries a dynamic value, or when core code restates an observation type.

## One unit of work, one trace

A trace is one self-contained unit of work. The root observation is opened **where that work is executed**, not in the HTTP request that admitted it:

| Unit of work | Root | Opened by |
| --- | --- | --- |
| Answer Run (Fast or Research, root or Child) | `run-answer` (`agent`) | the claimed Run's executor, inside the worker that owns the lease; a Child Run is driven inside its parent's trace and opens no second root |
| Document ingestion | `ingest-documents` (`chain`) | the ingestion engine for the accepted document batch |
| Startup ingestion recovery | `recover-ingestion` (`chain`) | the workspace RAG startup path |
| Retrieval Run | `run-retrieval` (`chain`) | the claimed Run's retrieval executor, inside the worker that owns the lease |

Everything the unit of work orchestrates nests inside its root: agent Turns, Tool calls, provider calls, embeddings, reranks, and retrieval. A Run that executes in a background worker therefore produces one trace, and none of its own calls can surface as a root: the root span is open for the whole operation, so the OTel context carries it into every nested task and thread it starts. Work a library detaches from that context is the one exception, described below. A Corpus
Mutation Run that only deletes or resets traces nothing, because it calls no model and opens
no observation of its own; one that ingests is traced by its `ingest-documents` batch.

Traces are grouped by **Agent Session** (`session_id`) and **owner** (`user_id`). A conversation is a session; each of its Runs is its own trace, which keeps traces small and makes the session view a replay. Attribution is applied once, at the Run boundary, through `Telemetry.trace(...)`; no call site threads ids, and no observation restates them.

### Known boundary: work a library detaches

A provider call made from a library's own executor — LightRAG's ingestion and extraction workers, which do not propagate the trace context into their tasks — carries no ambient parent observation and therefore surfaces as its own trace. `ingest-documents` still records the accepted batch, and the deployment's startup recovery embeddings behave the same way. This is measured, not assumed: in the pre-contract deployment 470 of 476 batch spans had no children, and this deployment's startup embeddings arrive as single-span traces. Closing it requires the library to run its callbacks inside the caller's context; until then, ingestion's per-chunk cost is visible as individual `embed-text` and `generate-completion` traces rather than as children of the batch.

## The span vocabulary

Names are an API. They are verb-first, stable, and closed: a new observation requires a new registry entry, not a new string.

| Name | Type | One observation covers | Metadata |
| --- | --- | --- | --- |
| `run-answer` | `agent` | One claimed Answer Run, start to settlement | `run_id`, `parent_run_id`, `workspaces` |
| `run-retrieval` | `chain` | One claimed Retrieval Run, start to settlement | `run_id`, `workspaces` |
| `generate-answer` | `chain` | The Run's generation phase, after mode resolution | `resolved_mode`, `history_turns`, `query_image_count`, `semantic_highlights` |
| `generate-agent-turn` | `generation` | One model invocation in the agent loop | `model`, `provider`, `endpoint_fingerprint`, `tool_names`, `tool_choice` |
| `compact-session` | `generation` | The compaction summary invocation over a rich tool transcript | as above |
| `execute-agent-tool` | `tool` | One Tool call, including its arguments | `call_id`, `intent_id` |
| `generate-completion` | `generation` | One non-agent provider completion (planner, highlights, probe, ingest extraction) | `provider`, `endpoint_fingerprint`, request parameters |
| `embed-text` | `embedding` | One embedding request | `context`, `modality`, `provider`, `endpoint_fingerprint`, `request_count`, `inline_image_bytes` |
| `rerank-passages` | `span` | One rerank stage of a retrieval | `chunk_count`, `top_k` |
| `call-rerank-model` | `span` | One rerank provider call inside that stage | `document_count`, `top_n` |
| `retrieve-context` | `retriever` | One knowledge-base/web retrieval | `workspaces`, `top_k`, `chunk_top_k`, `has_filters`, `federated_rerank` |
| `plan-retrieval` | `chain` | One query plan | `workspaces`, `history_messages` |
| `highlight-sources` | `chain` | One semantic-highlight enrichment | `source_count`, `text_chunk_count` |
| `ingest-documents` | `chain` | One accepted document batch | `document_count`, `doc_ids` |
| `recover-ingestion` | `chain` | Startup promotion of interrupted ingestion | `trigger` |
| `probe-image-capability` | `generation` | One image-capability probe against a provider | `provider` |

**Rules**

- **Verb first, no dynamic values.** A name identifies the operation, never one execution of it: the model, workspace, Run, document, or strategy belongs in `model` or `metadata`. `generate-completion`, never `llm_deepseek-v4-flash`.
- **A model invocation is a `generation`.** One `generation` per model call in an agent loop, interleaved with the `tool` calls it requested. Never one `generation` wrapping a whole loop: per-step reasoning, tokens, and cost must stay visible, and the step that blows up the context window must be identifiable.
- **A Tool call is a `tool`, nested under the agent observation** that orchestrated it, as a sibling of the `generation` that requested it. Its input is the arguments the model chose, bounded; its output is a summary, because the full result reaches the model as transcript text.
- **Types come from the registry**, never from a call site (`SPAN_TYPES` in `engine/ai/telemetry.py`). The accepted values are Langfuse's observation types: `agent`, `chain`, `embedding`, `generation`, `retriever`, `span`, `tool`.
- **Root input/output answer a reviewer's question.** The root's input is the user's question and its output is the answer (or the terminal outcome for a deferred or failed Run). Raw payloads belong in `metadata`.
- **Model, usage, and cost ride on the observation.** `model` is passed to the adapter, which maps each provider dialect's usage fields onto Langfuse's `input`/`output`/`total`/`input_cached_tokens`. Cost is never computed here.

## Redaction

`langfuse_trace_sensitive_data` decides content capture at the seam: the adapter drops an
observation's `input` and `output` whether they arrive when it opens or when it updates, so
no call site can leak after the fact. Call sites that would have to *build* a payload consult
`capture_sensitive_data` first, so nothing user-authored is even shaped while capture is off:

- Observation `input` is dropped at open time and again at update time; `output` is dropped at update time. A span cannot leak a query, an answer, Tool arguments, or an error detail by updating it later.
- An observation that names its own level keeps it: the adapter maps an exception that escapes the body to `ERROR`, but a Run that reported a cancelled or fenced outcome first is not re-labelled a failure.
- Error observations carry `level=ERROR` and, only when capture is enabled, the provider's message; otherwise the literal `error`.
- Structural metadata is not content and stays: counts, workspaces, model and provider identity, Run and Tool identifiers. A redacted trace must still be diagnosable, and none of those values is user text.
- `mask` replaces secrets and image bytes in every exported payload, and inline image bytes are removed from message telemetry before it leaves the process.
- Only DlightRAG's own observations are exported (`langfuse_export_external_spans` is off by default), so HTTP client and database spans do not pollute the tree.

## Deployment

| Key | Default | Meaning |
| --- | --- | --- |
| `langfuse_environment` | unset (Langfuse default `default`) | Deployment label: `local`, `staging`, `production`. Set it so one deployment's traces never share a bucket with another. |
| `langfuse_release` | the running package version | Release label. |
| `langfuse_sample_rate` | `1.0` | Exported fraction. |
| `langfuse_export_external_spans` | `false` | Export third-party OTel spans. |
| `langfuse_trace_sensitive_data` | `true` | Capture content. |

Startup does not call `auth_check`: tracing is enabled by configuration, and a failure to initialize degrades to no tracing rather than to a failed boot. Local stack startup, keys, and project bootstrap are in [Operations](operations.md#local-langfuse-observability).
