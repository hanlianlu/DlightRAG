# API Family selects the provider wire, not the model or its authority

One configured model endpoint may be called through Chat Completions or the
Responses API. DlightRAG selects that wire explicitly per model, keeps the model
catalogue and Agent Session authoritative, and preserves Chat as a supported
family instead of treating every OpenAI-compatible endpoint as one uniform
Responses implementation.

## Status

Accepted and implemented. See the [qualification record](../response-api-qualification.md)
for verification and remaining limits, and [Configuration](../configuration.md#api-family)
for current role selection.

This decision extends the [API Family](../domain-language.md#configuration-and-deployment)
term. It does not revise [ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md):
a provider response or conversation identity is never a second history authority.
It also does not revise [ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md):
provider-reported input and cache usage continue to anchor context accounting,
and a measured mismatch is a reason to revisit accounting rather than silently
change context composition.

## Context

Before this decision, all five `provider=openai` model entrypoints called
`client.chat.completions`, covering direct DeepSeek, OpenRouter, OpenAI and
arbitrary compatible base URLs. OpenAI recommends its Responses API for new
agentic work; OpenRouter offers a stateless Responses endpoint; and DeepSeek
documents native Responses support for its Flash model at `POST /responses`.

The useful difference is structural. Chat Completions puts visible text, tool
calls, and provider-specific reasoning fields into one assistant message.
Responses emits ordered typed items for reasoning, function calls, function
outputs, and assistant text, with typed stream events and terminal states. That
shape fits a durable tool loop better, but the name does not make implementations
identical: direct DeepSeek requires complete plaintext reasoning replay when a
later request carries tools, does not support remote conversation continuation,
and silently ignores some unsupported parameters; OpenRouter is stateless and
rejects `previous_response_id`; OpenAI supports encrypted reasoning and optional
remote state.

Three existing facts constrain the change:

1. `ModelProfile` and its catalogue describe one model endpoint's capacity,
   image support, and reasoning ladder. Unknown endpoints deliberately receive a
   permissive fallback and let the provider reject unsupported capabilities.
   Selecting another wire for the same endpoint must not duplicate those facts.
2. The Agent Session Entry Tree and its Context Projection are the durable
   transcript. A provider-native item is replay material attached to one complete
   Assistant Entry, not another transcript.
3. Tools are composed, authorized, executed, and settled locally. A hosted tool
   that happens to share a Responses item type is not the same operation and may
   not bypass the local intent/settlement boundary.

Zero Data Retention is not an API-family property. `store=false` disables ordinary
remote response state where the endpoint honours it; it does not prove the
absence of provider logs, caches, legal retention, or a contractual ZDR control.
Direct DeepSeek publicly documents disk-backed context caching and no OpenAI-like
contractual ZDR switch. This decision therefore minimizes remote state without
introducing a `zdr` product setting or claim.

## Decision

**API Family is one explicit per-model setting.** Its closed values are
`chat_completion | response`, and omission means `chat_completion`. The provider,
model, endpoint, and API Family are separate facts: `provider=openai` continues
to select the OpenAI-compatible implementation, while API Family selects either
`client.chat.completions` or `client.responses`. There is no separate provider
name such as `openai_responses`.

**One provider owns two transports.** Chat keeps its current wire projection.
Response has one common item/request/stream implementation; DlightRAG does not
introduce user-visible `responses_dialect` configuration or one provider class per
vendor. Existing `ReasoningProfile.format` plus API Family decides how a resolved
reasoning level is written. A vendor-specific exception enters only after a
falsifying contract test proves the common subset cannot express the endpoint.
It stays at the provider edge and never reaches Session, Tool, or compaction code.

**The model catalogue remains a model catalogue.** `ModelProfile`, runtime
overlays, and `FALLBACK_MODEL_PROFILE` keep their current meanings and precedence.
A known or explicitly configured endpoint may use Response; an uncatalogued
endpoint uses the same permissive model fallback and the common Responses subset,
logs that its model facts are unverified, and fails explicitly if the provider
rejects the request. DlightRAG never retries by deleting tools, images, or
reasoning, and never changes API Family after an error.

**Endpoint identity and invocation identity are distinct.** The model endpoint
identity (`provider`, `model`, safe endpoint fingerprint) resolves catalogue
facts. The invocation identity adds API Family and pins a Run's wire, opaque
replay, and transport capability caches. Chat-native state can never satisfy a
Response invocation identity, or the reverse. There is no production deployment
to migrate: implementation may invalidate and reset existing development
Run/Session state rather than add permanent missing-field or old-envelope
compatibility branches.

**The common Response contract is deliberately stateless.** Every request is
rebuilt from the selected local Context Projection. It sets `store=false`, keeps
remote conversation state disabled, and does not use `previous_response_id`,
Conversations, background execution, WebSocket continuation, hosted Web Search,
File Search, remote MCP, Code Interpreter, hosted files, or remote compaction.
`truncation` stays disabled so the local context policy, not a provider, chooses
what the model sees.

**The first Response surface is parity, not new authority.** It supports the
same five provider entrypoints, local function tools, `tool_choice=auto`, parallel
Tool batches, streaming and non-streaming text, structured output, user images,
and images carried by local Tool results. A Tool result image is projected into
its own `function_call_output`; Chat retains its existing post-batch user-image
projection. The canonical Tool and attachment contracts do not change.

**Complete native output is replay material, never a partial settlement.** One
complete Assistant Entry may retain the finalized ordered provider items needed
by the same invocation identity. On the next request those native items replace,
not accompany, a synthesized copy of the same assistant output. Cross-family or
cross-model continuation drops opaque replay and reconstructs from canonical
text and Tool calls. A same-identity envelope that claims replay state but is
malformed fails rather than silently discarding reasoning. Text and argument
stream deltas remain ephemeral; no Tool executes until the whole provider
response reaches a valid completed terminal state and the existing validation
accepts its calls.

**Existing domain outcomes do not expand for provider transport states.** A
completed response with function calls maps to `tool_use`; completed text maps to
`stop`; a token-cap incomplete response maps to `length`. Refusal, filtering,
failed, cancelled, or missing terminal events are typed provider failures, not an
empty successful Assistant turn. Structured JSON Schema may use the existing
narrow retry to JSON Object only after an explicit unsupported-format rejection,
before any streamed output, with the rejection cache keyed by invocation
identity.

**Remote-state minimization is fixed behavior, not a ZDR claim.** Response sends
`store=false` and excludes persistent hosted features. No first slice adds
`retention_requirement`, `retention_attestation`, or `zdr` configuration. Provider
logs, caching, agreements, and routing policies remain deployment facts. An
OpenRouter ZDR routing control or contractual provider qualification requires a
separate decision; success here must never be reported as ZDR.

## Considered options

- **Replace Chat Completions everywhere.** Rejected: arbitrary configured
  endpoints may implement only Chat, and Chat remains useful for existing models.
  A qualified endpoint may prefer Response without imposing that choice globally.
- **Add an `openai_responses` provider.** Rejected: provider and wire would become
  one overloaded name, duplicate model configuration, and invite the inference
  that OpenAI-compatible means OpenAI-hosted or ZDR-qualified.
- **Add `responses_dialect=openai|deepseek|openrouter|generic`.** Rejected before
  implementation: the common stateless subset plus the existing reasoning format
  owns every difference established so far. A new public noun without a distinct
  owner would duplicate `ModelProfile` and base-endpoint facts.
- **Put API Family into `ModelProfile`.** Rejected: capacity and modality belong
  to the model endpoint; the same endpoint can support both wires. Invocation
  identity, not capacity identity, owns the choice.
- **Require every Response endpoint to appear in the catalogue.** Rejected: it
  contradicts the deliberate permissive fallback used by custom endpoints. The
  generic path remains explicit and non-adaptive.
- **Probe, auto-detect, or fall back between API families.** Rejected: a 400/404
  can mean model, policy, schema, or routing failure, and a compatible endpoint may
  silently ignore a parameter. Retrying another family adds cost and changes the
  request without proving capability.
- **Use provider conversation state for cheaper continuation.** Rejected: it
  would make restart, fork, and compaction depend on remote state and create a
  second history authority.
- **Treat `store=false` as ZDR.** Rejected: application response storage, abuse
  logs, prompt caches, legal retention, and contractual controls are separate
  facts.

## Consequences

Implementation lands as destination-shaped slices, each with a falsifier written
before the change:

1. Add API Family and split endpoint from invocation identity without changing
   model catalogue semantics. Provider replay and structured-output caches are
   family-bound. Existing development Run/Session data may be reset.
2. Add the non-streaming Response request/output projection for ordinary and Tool
   turns, including structured output, exact `call_id`, status handling, and
   finalized native replay.
3. Add the typed streaming state machine and cancellation/close behavior; partial
   items remain ephemeral and no incomplete call executes.
4. Project user and Tool-result images without changing their canonical ownership,
   then cover all five provider entrypoints with SDK-level mock HTTP/SSE contracts.
5. Verify compaction and history boundaries preserve or remove complete
   assistant/tool exchanges. Keep current context accounting and compare its
   estimates with provider input usage; a systematic undercount or pathological
   early compaction stops this work for a separate accounting decision.
6. Live-qualify direct `deepseek-flash` first with bounded synthetic requests,
   then OpenRouter independently. Implement the official OpenAI contract as an
   explicitly experimental profile using mocks; live OpenAI qualification waits
   for a customer-owned key and is not a gate for DeepSeek/OpenRouter.

Code supports every provider entrypoint, while rollout changes one role at a time:
Query tool turns first, then ordinary structured roles, VLM/image, and any reranker
that explicitly selects the same model. Chat remains available throughout. A
provider failure never authorizes a silent family, Tool, image, reasoning, or
privacy downgrade.

Live canaries may use the deployment's already configured DeepSeek and OpenRouter
credentials with synthetic non-sensitive inputs and bounded request counts. They
must not read, print, or persist credentials or prompt/reasoning payloads in
ordinary telemetry.

Stop and revisit this decision if the common Response transport requires a change
to Tool definitions, local Tool execution/settlement, Session ancestry, Fork Point,
Context Projection, or compaction authority; if replay requires a remote response
identity; if an endpoint can succeed only after silently dropping a requested
capability; or if measured native replay makes the existing context estimator or
compaction trigger materially unsafe. Hosted tools, remote state, WebSocket
continuation, server-side compaction, and a product ZDR policy remain separate
future decisions.
