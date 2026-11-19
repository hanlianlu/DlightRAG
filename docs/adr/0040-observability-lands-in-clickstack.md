# Observability lands in ClickStack

Traces are exported over OTLP/HTTP into a self-hosted ClickStack by the `dlightrag-observability` workspace package, which also owns the privacy classifier and the stack itself. The Langfuse client, its nine `langfuse_*` settings, its local stack scripts, and its dependency are deleted, not disabled.

## Status

Accepted and implemented. Revises the deployment clause of [ADR 0017](0017-observability-span-contract.md) — the vocabulary, the seam, and the one root per unit of work stay; the destination does not — and completes the migration clause of [ADR 0039](0039-observability-is-always-on-and-the-seam-masks-private-data.md), whose classifier now masks what this exporter sends.

## Context

DlightRAG exported its traces to Langfuse through a client the product constructed at startup, with ten `langfuse_*` settings (nine once [ADR 0039](0039-observability-is-always-on-and-the-seam-masks-private-data.md) deleted the privacy key) and a local stack prepared from Langfuse's official Compose file: web and worker containers with their own ClickHouse, Redis, MinIO, and PostgreSQL. The deployment used exactly two of that product's surfaces: traces and the usage on them. A grep of the tree found no prompt management, dataset, experiment, LLM-as-judge, score, or playground call — the adapter used the client's observation context, its attribute propagation, and its constructor, and nothing else. The review surface ADR 0017 built is the trace tree and the attributes on it, and cost was the one thing Langfuse inferred rather than received.

Three things changed at once. ClickHouse acquired Langfuse and positions ClickStack for infrastructure and operations while Langfuse is positioned for the LLM quality loop, so the two planes now have different owners. The destination speaks OpenTelemetry: the GenAI conventions carry a model, an operation, usage counters, and `session.id`/`user.id`, which is everything the seam already puts on an observation. And a workspace package can own the export, the privacy classifier, and the stack together, so the export has a home that states no product vocabulary.

What the migration must not cost is the property ADR 0039 wrote down: monitoring is always on, it never fails the product, and an outage is reported once rather than per batch.

## Decision

**The export moves to OTLP/HTTP into the self-hosted ClickStack.** A new `dlightrag_observability.otlp` module owns the TracerProvider, the span rendering, the masking in front of it, the bounded queue, and the circuit breaker. It states no span name and no provider dialect: the composition root hands it the registry (`SPAN_TYPES`) and the dialect mapper (`provider_usage_buckets`) as constructor data, and an observation renders as a span carrying `dlightrag.observation.kind`, the GenAI operation when the kind has one, `gen_ai.request.model`, `gen_ai.provider.name`, the usage buckets, `session.id`, and `user.id`. Its **span kind** follows the convention rather than being uniform: a `generation` and an `embedding` are `CLIENT`, a `tool` is `INTERNAL`, and the spans of DlightRAG's own orchestration are `INTERNAL`.

**The attribute names are the registries', and they are checked there.** The GenAI conventions live in their own repository, and a self-consistent set of names can be stale in the same direction as the exporter and the documents that describe it. `gen_ai.usage.input_tokens` is the *inclusive* prompt (the registry says it "SHOULD include all types of input tokens, including cached tokens") and `gen_ai.usage.cache_read.input_tokens` is a subset of it; the product's exclusive "fresh" count is derived (`input - cache_read`) rather than a redefinition of the standard key. The Required attributes the seam cannot infer — `gen_ai.provider.name`, `gen_ai.tool.name` — are stated by the call site that knows them, and `error.type` rides on every failed span. Where the product cannot know a value, the attribute is absent rather than guessed.

**Configuration loses the plane's nine keys and gains only operational ones.** `otlp_endpoint`, `otlp_headers` (the ingestion key, hidden from a settings dump), `service_name`, `environment`, `release`, `sample_rate`, and `timeout` replace them. Settings stay `extra="forbid"`, so a deployment that still sets `langfuse_host` fails validation exactly as ADR 0039 made the privacy switch fail — there is no compatibility shim.

**Langfuse is deleted, not disabled.** The adapter, the CLI scripts, the `langfuse-*` Makefile targets, the dependency, and the tests that read its wire are removed. The retired stack's data directory is the operator's to delete.

**The process's log records travel beside the traces.** The same package attaches one handler to the root logger; a record is masked by the classifier, carries the trace and span id of the observation that logged it, and reaches ClickHouse from a bounded queue a logging call never waits on. stdout does not change.

**The quality loop is out of scope.** Prompt management, datasets, experiments, LLM-as-judge, scores, and the playground are not reimplemented, and this repository does not gain a product that provides them. Cost stops being inferred by the backend: the app keeps reporting usage and never computing cost, and a price table over `otel_traces` is the operator's to build (ADR 0017's boundary, unchanged).

## Considered options

- **Dual-write both planes for a bake-in period.** Rejected. Two providers would keep the Langfuse dependency, its containers, and its settings alive to answer a question the wire test already answers: the export is one seam, and what it renders is asserted directly against a recording span exporter.
- **Keep the Langfuse client behind a flag as a fallback.** Rejected. It is the switch ADR 0039 deleted under another name, and a fallback destination that is never exercised is a fallback that does not work.
- **Keep the exporter in `dlightrag/adapters/observability/`.** Rejected. The package already owns the classifier and the stack, and a thin app-side adapter would have restated the vocabulary the composition root can pass as data. The directory is deleted, and the import-linter contracts that named it now name `dlightrag_observability`.
- **Render each payload field as its own attribute.** Rejected. An OTLP attribute is a flat scalar or array, and a message list is a tree; one JSON attribute per payload is what keeps the classifier's output intact, bounded at 64 KiB, and masked where it is shaped.
- **Let the exporter own the span names and their kinds.** Rejected. The application owns what it observes (ADR 0017), so the registry stays in `engine/ai/telemetry.py` and reaches the exporter as constructor data; the package cannot import the product anyway.
- **Port Langfuse's Model definitions into the app so cost keeps being inferred.** Rejected. It would move prices into the product, which ADR 0017 forbade, and put them in source rather than where the traces are.
- **Send the payloads to the collector unmasked and redact there.** Rejected by ADR 0039: a private value must not cross the network to be redacted.
- **Wait for a managed plane or a Kubernetes deployment.** Rejected. The deployment is a single machine, the compose stack is four containers, and the failure mode this decision has to answer — the collector is down — is the one the circuit breaker already answers.

## Consequences

- Review, dashboards, and evaluators retarget ClickHouse SQL over `otel_traces`. The vocabulary, attribution, and usage buckets are unchanged in meaning, so a query changes its column names; `dlightrag.observation.kind` is what an aggregate groups by when it wants the product's kind.
- The deployment has one telemetry ClickHouse and a four-container stack instead of Langfuse's.
- The quality loop the deployment could have had is gone. If prompt management, datasets, experiments, LLM-as-judge, or scores are wanted, a product that owns them has to be introduced deliberately — that is the price of this decision, and it is recorded rather than discovered.
- Cost is no longer inferred for free. Until the operator's price table exists, `gen_ai.request.model` and the usage buckets are stored and no cost is shown.
- The exporter's failure modes are the ones ADR 0039 requires, and its tests hold them: no endpoint is a no-op with one info line, a failed initialization installs no-op telemetry, an unreachable collector is one `WARNING` when it starts failing and one `INFO` line when it recovers with nothing in between, and a shutdown flushes under a timeout.
- The ClickStack alerts and dashboard tiles are part of the contract, not merely a consumer of it, so a query is checked against the same primary sources the exporter is: an alert that reads a key no registry defines, or that cannot fire because the signal it names is never emitted, is a defect of this decision's surface and is fixed or deleted rather than left reporting `OK`.
- The alert and dashboard files name the ClickStack connection and source ids of the stack they were written against, and ClickStack mints those per stack; a new stack's ids have to replace them before `alerts push` (see [Operations](../operations.md#alerts-and-the-dashboard)).
- The default ports of the bundled stack are the same ones another ClickStack on the same machine would use, so two stacks side by side need their ports set in `stack/.env`, and the application's `otlp_endpoint` has to follow.
- **Rollback** is a restore, not a migration. The trace history recorded before the cutover stays in the Langfuse project, untouched; ClickStack starts empty for the product's spans, and nothing was copied. To go back, restore the `langfuse` dependency and the `langfuse_*` settings, point them at that project, and re-add the Makefile targets — the seam is one exporter, so no call site moves.
