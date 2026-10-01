# Gemini uses the stateless Interactions API

DlightRAG calls Gemini only through Google's Interactions API, never
`generateContent`. Every request rebuilds the conversation from the local Context
Projection and sets `store=false`. A Gemini invocation's API Family is
`interactions`, a value no other provider takes and no configuration chooses.

## Status

Accepted and implemented. It extends
[ADR 0027](0027-api-family-selects-the-provider-wire.md) with the `interactions` API
Family; that ADR's stateless contract, invocation identity, replay and failure rules
apply to it unchanged. [ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md)'s
Session authority is untouched: no provider conversation becomes a second history.

## Context

The Gemini adapter called `generateContent`, which google-genai still serves but no
longer extends. The Interactions API became generally available in June 2026 and is
Google's recommendation for new work. Its shape is the one ADR 0027 chose Responses
for: typed, ordered steps for thoughts, model output, function calls and function
results, a terminal status, and typed stream events.

Three facts shape the move:

1. Thoughts carry signatures, and in stateless mode every thought must come back
   exactly as received. `generateContent` attached a signature to a function-call
   part, which DlightRAG kept as `ToolCall.thought_signature`. Interactions puts it
   on a thought step, between the text and calls that thought led to.
2. The API keeps remote state when asked: `store=true` retains requests and
   responses (55 days by default on paid tiers), and `previous_interaction_id`
   continues from them. DlightRAG's Sessions already own history, forks, compaction,
   crash replay and switching providers.
3. Interactions take no `temperature` or `top_p`, and name thinking by level
   (`minimal`, `low`, `medium`, `high`), none of which turns it off.

## Decision

**One wire, not a choice.** `provider: gemini` always resolves API Family
`interactions`. An explicit `chat_completion` or `response` on Gemini is a
configuration error, and so is `interactions` on another provider. There is no
`generateContent` fallback; replay state recorded under another family is stripped
like any other invocation's.

**Stateless, so Google keeps no interaction.** Every request carries the full local
context with `store=false`, and never `previous_interaction_id`, background
execution, webhooks, environments or agents. As in ADR 0027, `store=false` is not a
Zero Data Retention claim.

**A turn that thought keeps its native steps.** Its output steps (thoughts with
signature and summary, model output, function calls) are stored as JSON in the
Assistant Entry's `provider_state`, bound to the invocation identity. The same
invocation resends them verbatim and in their original order, after checking them
against the entry's canonical text and calls; a malformed or contradicting replay
fails. Another invocation sees only the canonical text and calls.
`ToolCall.thought_signature` is gone.

**Reasoning levels are thinking levels.** The catalogue's `gemini` reasoning format
maps each DlightRAG level to a `thinking_level`, holds no `off`, and is validated to
the four API values. A configured level also asks for `thinking_summaries: auto`,
whose text becomes the turn's reasoning. An uncatalogued Gemini model gets the four
levels, and a higher request clamps to `high`.

**No sampling parameters.** A configured `temperature` on a Gemini model or a Gemini
chat reranker fails settings validation. Internal defaults written for other
providers, such as the image probe's temperature 0, are never sent.

**Narrow options, parity outputs.** `model_kwargs` accept `safety_settings` and
`service_tier`, plus the raw thinking controls where no typed level owns them;
anything else fails before a request. Structured output is a JSON text
`response_format` with the strict schema. A Tool result's images ride in its own
`function_result` as image subcontent. `completed` and `requires_action` map to `stop`
and `tool_use`; `incomplete` maps to `length` and executes no call; `failed`,
`cancelled` and stream errors are typed provider failures. HTTP statuses keep their
retry budget and classification, and usage keeps the counter names the rest of
DlightRAG reads: `prompt_tokens`, `cached_content_tokens`, `candidates_tokens`,
`thoughts_tokens` and `total_tokens`.

## Considered options

- **Keep `generateContent` as a second Gemini family.** Rejected: two wires to
  qualify for no product gain, and new Gemini features land only in Interactions.
- **Continue with `previous_interaction_id`.** Rejected: a second history authority,
  and Google would retain users' evidence.
- **Ignore a configured temperature.** Rejected: a setting that silently does
  nothing.

## Consequences

Development Sessions recorded through `generateContent` replay as canonical text and
calls; that data is reset, not migrated. google-genai's Interactions bridge reads
`HttpRetryOptions.attempts` as a retry count and cannot express zero, so the adapter
sets the retry budget on the resource, and a request-count test guards the next SDK
upgrade.

The offline tests pin the REST contract. A live key must still confirm what they
cannot: the shape of real HTTP error bodies, whether a completed stream carries its
steps, which status ends a function-calling turn, whether Gemini accepts canonical
history without thought steps, which JSON Schema keywords the structured output
honors, and whether the catalogued Flash model also takes `minimal`, which its
profile does not yet claim.
