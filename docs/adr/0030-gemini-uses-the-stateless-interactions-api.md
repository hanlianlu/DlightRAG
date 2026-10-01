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
3. google-genai 2.25's `GenerationConfig` for Interactions has no `temperature` or
   `top_p` (Google's guides still mention `temperature`; the SDK the wire is built
   from has no field for it), and names thinking by level (`minimal`, `low`, `medium`,
   `high`), none of which turns it off.

## Decision

**One wire, not a choice.** `provider: gemini` always resolves API Family
`interactions`. An explicit `chat_completion` or `response` on Gemini is a
configuration error, and so is `interactions` on another provider. There is no
`generateContent` fallback; replay state recorded under another family is stripped
like any other invocation's.

**Stateless.** Every request carries the full local context with `store=false`, so
the API stores no interaction to resume or retrieve. DlightRAG never sends
`previous_interaction_id`, background execution, webhooks, environments or agents.
As in ADR 0027, `store=false` is not Zero Data Retention.

**A turn that thought keeps its native steps.** Its output steps (thoughts with
signature and summary, model output, function calls) are stored as JSON in the
Assistant Entry's `provider_state`, bound to the invocation identity. The same
invocation resends them verbatim and in their original order, after checking them
against the entry's canonical text and calls; a malformed or contradicting replay
fails. Another invocation sees only the canonical text and calls.
`ToolCall.thought_signature` is gone.

**A stream is assembled from its deltas.** The completed event carries status and
usage, not steps, so a streamed turn is built from its step starts and deltas: a
thought starts with an empty signature and its first summary, and the signature and
call arguments arrive in deltas. Steps a completed event does carry win. A terminal
status update ends the turn as the completed event does; only a stream that ends with
neither was cut short.

**Reasoning levels are thinking levels.** The catalogue's `gemini` reasoning format
maps each DlightRAG level to a `thinking_level`, holds no `off`, and is validated to
the four API values; `gemini-3.8-flash` takes `low`, `medium` and `high`. A configured
level also asks for `thinking_summaries: auto`, whose text becomes the turn's
reasoning. An uncatalogued Gemini model gets the four levels, and a higher request
clamps to `high`.

**No sampling parameters.** A configured `temperature` on a Gemini model or a Gemini
chat reranker fails settings validation. Internal defaults written for other
providers, such as the image probe's temperature 0, are never sent.

**Narrow options, parity outputs.** `model_kwargs` accept `safety_settings` and
`service_tier`, plus the raw thinking controls where no typed level owns them;
anything else fails before a request. Google still lists custom safety settings as
not yet supported by Interactions, so the API decides what it makes of them.
Structured output is a JSON text `response_format` with the strict schema. A Tool
result's images ride in its own `function_result` as image subcontent. `completed`
and `requires_action` map to `stop` and `tool_use`; `incomplete` maps to `length` and
executes no call.

**Failures classify like their HTTP twins.** `failed`, `cancelled` and stream errors
are typed provider failures. One whose error code names an outage (`unavailable`,
`resource_exhausted`, `deadline_exceeded`, `internal`, `aborted`, recognized in any
segment of the code URI) and a stream cut short are transient, so a Run retries and
defers them as it does an HTTP 503. HTTP statuses keep their classification, without
the SDK's schema-mismatch parse failure chained behind them.

**Usage counts thinking as output.** Interactions report thought tokens apart from
the output; DlightRAG reports `completion_tokens` as output plus thoughts, as every
other provider reports reasoning and as Google bills it, beside `prompt_tokens`,
`cached_content_tokens`, `thoughts_tokens` and `total_tokens`.

## Considered options

- **Keep `generateContent` as a second Gemini family.** Rejected: two wires to
  qualify for no product gain, and new Gemini features land only in Interactions.
- **Continue with `previous_interaction_id`.** Rejected: a second history authority,
  and Google would retain users' evidence.
- **Ignore a configured temperature.** Rejected: a setting that silently does
  nothing.

## Consequences

Development Sessions recorded through `generateContent` replay as canonical text and
calls; that data is reset, not migrated.

google-genai's Interactions client reads `HttpRetryOptions.attempts` as its retry
count, though google-genai documents attempts as including the first request, and
google-genai raises an `attempts` of 0 to 1 before that client reads it. The adapter
passes `max_retries` through the public option, so a Gemini request retries at least
once; a request-count test pins this for the next SDK upgrade.

A completion or stream without tools returns only text, so a Fast turn on Gemini keeps
no thought steps, and its next request repeats only the canonical answer.

The offline tests pin the REST contract against Google's documented shapes. A live key
must still confirm what they cannot: the shape of real HTTP error bodies and of in-band
error codes, whether Gemini accepts a same-model history whose Fast turns lack thought
steps, and which JSON Schema keywords the structured output honors.
