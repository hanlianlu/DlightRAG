# Observability span contract

## Status

Accepted and implemented. No configuration key beyond the existing Langfuse settings: the vocabulary is code, the contract is one registry plus one test.

## Context

Traces are a product surface, not a by-product: Langfuse groups observations into traces by the OTel context, and every dashboard, evaluator, and saved filter targets observation *names*. DlightRAG's names had drifted into three separate problems, all measurable in the deployment's own Langfuse project (78,730 observations, 58,419 traces, 23 names):

- **Names carried model ids** (`llm_deepseek-v4-flash`, 44,790 lonely traces; `embed_voyage-multimodal-3.5`; `rerank/voyage_reranker`). Swapping a model renamed the operation and silently broke every aggregate. Langfuse already stores the model as its own attribute.
- **Retired names lived on** with no version marker (`answer_pipeline`, `answer_stream_pipeline`, `query_planning`, `answer_generation`, `agent_tool`), in a project where every row was `environment=default`, `version=NULL`, so old and new naming shared one bucket.
- **The tree did not hold.** 22,333 observations were roots, 36,097 pointed at a parent that was never exported, and 57,019 of 58,419 traces held a single span. The Run's root span was opened *before* the agent branch — the loop, its Turns, and their provider calls ran outside it — so an agent Run produced a trace of dangling provider calls, and ingestion's per-chunk calls did not nest under its batch. No production call site passed `session_id` at all.

## Decision

- **A closed, typed span vocabulary.** `engine/ai/telemetry.py` owns `SpanName`, `SpanType`, and `SPAN_TYPES`; the adapter derives every observation's type from it, so a call site cannot disagree with the registry and cannot restate a type. Names are verb-first and free of dynamic values (no model, workspace, Run, document, or strategy), matching Langfuse's low-cardinality and "treat names like an API" guidance.
- **One unit of work owns one root, opened where the work is executed.** The Answer Run's root is the claimed Run's executor — inside the worker that owns the lease, wrapping deferral, recovery, and settlement — not the HTTP request that admitted it. Ingestion opens its batch span in the ingestion engine. Tools are `tool` observations nested under the agent observation that requested them, one `generation` per model call in the loop, so per-step reasoning and cost stay visible.
- **Attribution is applied once, at the boundary, by the seam.** `Telemetry.trace(session_id, user_id)` carries the Agent Session and the owner to every observation opened inside its scope; no call site threads ids and no span restates them.
- **Redaction is a property of the seam.** The privacy switch decides whether `input` and `output` exist, both when an observation opens and when it updates; before, `update(input=...)` bypassed the switch, so a span could leak a query or an answer after the fact.
- **Deployment identity is explicit.** Every client carries a `release` (defaulting to the running package version) and takes `environment` from configuration, so two deployments and two releases never share a bucket.
- **The vocabulary is enforced, not documented.** A unit test fails when an observed name is not registered, when a registered name is unused, when an observed name is an f-string, or when code outside the observability adapter mentions `as_type`.

## Considered options

- **Keep model ids in names and filter by model elsewhere.** Rejected: the model is already an attribute on `generation`, and a name that changes with configuration cannot be an aggregate key or an evaluator target.
- **Keep the retired names for continuity.** Rejected: they describe phases that no longer exist, and no version marker separated them from current names, so continuity was already fiction. The deployment's legacy history was deleted rather than migrated.
- **Name spans after the Tool, workspace, or Run (`search_knowledge_base`, `run-abc123`).** Rejected: cardinality explodes and every dashboard fragments. Tool identity is metadata on `execute-agent-tool`.
- **Open the Run root in the HTTP request so the trace starts at admission.** Rejected: the Run outlives the request by design (durable worker, leases, deferral), so a request-scoped root closes before the work it claims to describe — exactly the dangling-parent failure observed above.
- **Type the Run root per mode (`agent` for Research, `span` for Fast).** Rejected: the registry maps a name to one type, the root opens before mode resolution, and a name that changes type with configuration is the same defect as a name that changes with the model. The root types the Run; the agentic loop and its Tool calls are visible as `generate-agent-turn` and `execute-agent-tool` inside it.
- **Let a generic exception mapping own the trace level.** Rejected: `RunCancellationObserved` and `LeaseLostError` are terminal outcomes a caller asked for, so an observation that reports its own `level` first keeps it and is not re-labelled `ERROR`.
- **Forward an unrecognized provider usage dialect verbatim.** Rejected: Langfuse derives cost only from `input`/`output`/`total`, so arbitrary keys produce silently empty usage instead of a diagnosable gap; unrecognized dialects now contribute no usage keys and log at debug.
- **Accept any `langfuse_environment` string.** Rejected: Langfuse drops values outside its alphabet and length silently, which is the drift this contract exists to prevent, so configuration rejects them at startup.
- **Let each call site decide its own observation type.** Rejected: it is how one operation ended up as both `llm_*` and `agent_model_turn` with different types, and rerank as a `span` that carried a model.
- **Gate every metadata field under the privacy switch too.** Rejected: counts, workspaces, model identity, and Run or Tool identifiers are not user content, and dropping them would leave a redacted trace undiagnosable while adding nothing to privacy. The switch suppresses content — observation inputs, outputs, and error text — and the configuration and observability docs now say exactly that; `configuration.md` previously promised to suppress "content/IDs", which the code never did.

## Consequences

- A new observation costs one registry entry, and the contract test refuses a name that is not registered, a registered name that is unused, a computed name, a type that disagrees with the table `docs/observability.md` publishes, an `observe` reached through an alias, and a Run root opened without its attribution scope. Each of those five cases was injected into the tree and observed to fail the suite before this ADR was written.
- Reviews start at one tree per Run: the trace table shows the question and the answer because the root carries them, and a session view replays a conversation in order.
- Cost and latency keep their model dimension without keeping it in a name, so a model swap changes an attribute rather than an aggregate key.
- Deployments are separable by `environment` and `release`, which the previous project could not express at all: every historical row was `default`/`NULL`.
- Ingestion's per-chunk provider calls still surface as their own traces: LightRAG's workers do not run inside the caller's context. They remain correctly named and typed, and closing the gap is a library-side change, not a naming one.
- The pre-contract history was deleted rather than migrated, so the project contains only traces that follow this contract.
