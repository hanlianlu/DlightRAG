# A step records the model, parameters and prompt digests it ran on

Every step a model call writes into a Session says, in its own entry, which model produced it, with which parameters, over which prompt, and how long the call waited for a slot and took. It does that with a `call` object in the entry's payload, committed in the same transaction as the entry. The prompt is recorded only as three SHA-256 digests, and the parameters only from an allowlist, so the record holds no prompt text, no raw provider option, no header, no credential and no URL.

## Status

Accepted and implemented. There is no migration and no new table: an entry written before this has no `call`, and every reader takes it as absent. What anything later builds from those records, such as an evaluation set, is separate work this decision does not describe.

## Context

The database could not answer "which model, with which parameters, over which prompt produced this step?". Of the entries a model call writes:

- An Assistant Entry held content, reasoning, tool calls, usage, cost and the provider's replay state.
- A Compaction Entry held the summary and its coverage, with no model and no usage.
- A Fast answer's Assistant Entry held its usage and cost.
- Temperature was stored nowhere, and `max_tokens` lived only in the transient Request Snapshot, which the Runtime deletes when the response commits.
- The system prompt had neither text nor digest, and `prompt_revision` on the Agent Run Plan is a constant nobody passes.
- The model name the server returned was dropped by every adapter, so an alias that a gateway routes elsewhere left no trace.

The operator's traces ([ADR 0040](0040-observability-lands-in-clickstack.md)) name a generation's model and temperature, but a trace is a masked, bounded view held by another store with its own retention. It is keyed by trace and not by Session step, and it holds no digest of the request and no served name. Comparing steps over time, and telling a regression from a configuration change, need the record to live and die with the step.

Two objects know different halves of it. The wrapper that calls the provider (`ToolModel`, `CompletionModel`) knows the model, the sampling, the reasoning level resolved against the model's profile, the merged options and which attempt produced the result, but it is cached per role and shared by Runs and knows neither the Run nor the entry. The code that commits the entry knows the exact request, the Plan and the role, and none of the rest.

## Decision

**A `call` on the three entries a model call writes.** An Assistant Entry of the main agent or of a Child, a Compaction Entry of Research or of Fast, and a Fast answer's Assistant Entry each carry an optional `call`, written in the entry's own transaction. It describes the attempt that produced the result and is absent when no wrapper stated how the call was made.

```json
{
  "model":  {"role": "query", "provider": "openai", "model": "deepseek-flash",
             "endpoint_fingerprint": "<sha256>", "api_family": "response",
             "served_model": "deepseek-v4-flash"},
  "params": {"temperature": 1.0, "max_tokens": 65536, "tool_choice": "auto",
             "response_format": null, "reasoning": {"requested": "high", "effective": "high"},
             "options_digest": "<sha256>"},
  "queued_ms": 0, "duration_ms": 8412,
  "prompt": {"system": "<sha256>", "tools": "<sha256>", "request": "<sha256>"},
  "plan_digest": "<sha256>"
}
```

**The model.** `role` is the model role the call was made for, the Request Snapshot's `model_role` (a Child's own, so a Child on another model says so), or `query` for Fast. `provider`, `model`, `endpoint_fingerprint` and `api_family` are the invocation fingerprint the Run's recovery validation already uses: the endpoint is the SHA-256 of its scheme, host, port and path, never the URL and never its query. `served_model` is the model name the server stated for the response (`response.model` on Chat Completions and Responses, the `message_start` event on Anthropic, the interaction's `model` on Gemini). It arrives from outside the process, so it is kept only when it is a plain name of at most 200 printable characters, and is null otherwise.

**The parameters are an allowlist.** The temperature, the output limit and the tool choice the wrapper sent; the *type* of the response format and nothing of its schema; the reasoning level requested and the one it resolved to against the model's profile; and `options_digest`. The digest covers the provider options the wrapper merged, after two rules. Options whose names mark a request container are dropped whole, whatever is inside: a name containing `header`, `auth`, `cookie`, `credential`, `query`, `url` or `endpoint`. This is why `extra_headers: {"X-Custom": …}` cannot leak, although `X-Custom` looks like no secret. A value under any other secret-looking name, by the same list the configuration dump and the telemetry masker use, is replaced by `[redacted]` before hashing. A credential is never digested, because a digest of a low-entropy secret is a brute-force oracle, so rotating a key moves nothing. The record states what the attempt was sent, not what was planned: an agentic call that fell back to ordinary reasoning, a strict schema that fell back to a JSON object, and a compaction that retried each state their final attempt.

**Timing.** `queued_ms` is how long the request waited for a slot in the model scheduler before it was sent, and 0 when it did not wait. `duration_ms` is how long the attempt that produced the result took, from sending its request to the end of its response; for a stream that is the last chunk read, so a consumer that is slow to take the chunks adds to it. Both are whole milliseconds, read by the wrapper off the monotonic clock and never from a difference of wall-clock times. The wrapper sees the wait without a hook in the scheduler: the scheduler starts a tool turn, and a stream, the moment it admits the request, so the time since the caller asked is the wait. A stream counts from the moment it was requested, because the scheduler binds its owner then, and every stream in the product is read at once. A refused first attempt, such as a strict schema, and an agentic attempt that said nothing are in neither number, as they are in none of the parameters. The two numbers are in the allowlist, in no digest and in no check a Run's recovery makes, and they are not counts of tokens.

**The prompt is three digests, never text.** They are lowercase hex SHA-256, computed by the code that built the request: for an Agent turn the Runtime, from the canonical strings the Request Snapshot already holds. `system` covers the text of the system messages alone, joined by a blank line. `tools` covers the canonical tool definitions alone. `request` covers the canonical JSON object `{"messages": [...], "tools": [...]}`, so a later reader who rebuilds a request from the Session can prove it is byte for byte the one that was sent. A Fast answer and a compaction have no Snapshot: their digests are taken over the messages they sent and no tools, so `tools` is the digest of `[]`. A Research step also carries `plan_digest`, the immutable Plan it ran under. The system prompt takes capability flags and never a clock, so `system` stays equal across Runs and across time while the flags and tools stay the same; turning Memory on changes it, and a Fast question states the time, so only its `request` moves.

**How the halves meet.** A wrapper returns its half, the model, parameters and timings, in its result: `AssistantTurn.call` for a tool turn, with the adapter's `served_model` on the turn, and a `call_holder` for the streams Fast and compaction use, beside `usage_holder` on `CompletionModel`. The adapters write the served model into a holder of the wrapper's own, never the caller's usage, so a stream's caller sees only counters in `usage_holder`. The code that built the request adds the prompt half (the Runtime for an Agent turn, the compaction coordinator for a summary, the synthesizer for a Fast answer), and whoever commits the entry adds the role and the Plan. Nothing travels through telemetry or a context variable.

**What is not recorded.** The calls that write no step: the retrieval planner's extraction, keyword highlights, reranking, image captions and embeddings. Failed attempts. Cost and tokens: the record holds no counter, so a token is counted once, in `usage`. Wire-level digests, which would tie the record to a vendor SDK's conversion of the request. A code version: the package version is the same on every build of this tree and could not tell two builds apart, so the `system`, `tools` and `request` digests are what tells prompt changes apart.

**Recovery never reads it.** `call` is in no usage or cost record, no Plan or acceptance digest, no `routing.model_fingerprints` and no check that decides whether an active Run may resume, and `prompt_revision` is unchanged. A Run in flight when a deployment adds this resumes as before, its earlier steps without a `call` and its later ones with it. A step committed again by recovery records the attempt that committed it: a Research turn whose provider call was interrupted is made again and states the new attempt, at whatever temperature the model is configured with then. A Fast answer settled from a stored result carries no `call`, because the process that made the call is gone and the call is kept nowhere else.

**Why in the entry.** A step's lifetime is its entry's. Stored in its payload, the record is deleted with the Session, cascades with a conversation's deletion, and is kept as long as Runs are, with no rule of its own to state or test. A table of calls would need its own cascade, retention and writer, and would be a second write path beside the entry's.

**What a client sees.** Nothing. The transcript routes (REST, Web, MCP and the Child views) project an entry's content and tool calls by whitelist, and the identity prompt requires that the agent not reveal the model underneath. The record is for the operator who reads the database, and it is not routed through the `Telemetry` seam, so a trace is unchanged.

## Considered options

- **A `model_calls` table keyed by entry.** Rejected, as above: a second lifetime, a join to read a step, and a second write that could miss the entry's transaction.
- **The prompt text beside the digests.** Rejected. The Session already holds the conversation, and the request is rebuilt from it; a copy would double storage, put user content under a retention of its own, and drag inline image bytes into every row.
- **Carry the record through telemetry or a context variable.** Rejected. Wrappers are shared across Runs, so ambient state would attribute one Run's call to another's, and a trace is masked and lossy where the record must be exact.
- **Put the lineage in `usage`, `cost` or the Plan digest.** Rejected. Usage and cost are summed and priced, so a field among them is counted or priced by accident; a Plan digest is compared when a Run resumes, and a new field in it would strand every Run that was running at the deploy.
- **Store the raw options, or hash everything under a secret-looking name.** Rejected. Raw options hold gateway headers and keys. Hashing secrets is a brute-force oracle. Matching header *names* misses the one called `X-Custom`, which is why the container is dropped by what it is.
- **Digest what is on the wire.** Rejected. The wire form is each SDK's conversion of the request and moves with its version; the request the product built is the stable thing, and `request` proves a rebuild against it.
- **Record every model call, auxiliary ones included.** Rejected for now. They write no step, so there is nothing to attach the record to, and a table for them is the option rejected above.
- **A hook in the scheduler to report the wait.** Rejected. The scheduler starts the work the moment it admits it, so the wrapper reads the wait off a clock without one.
- **Bump `prompt_revision`.** Rejected. It belongs to the Plan, which the resume check compares.

## Consequences

- An entry grows by about 0.8 KB of JSON. Entries written before this have no `call`, and every reader uses `.get`; the PostgreSQL round trip writes a turn with a `call` and one without and reads both back as written.
- Credentials and locators cannot reach a row by construction, because the record has no field that takes them: the allowlist is the shape, `served_model` is bounded, and the options are only hashed. Tests configure a key, a signed URL and an `X-Custom` header, and search every stored entry for them.
- The model's identity can be known from `model` and `served_model` together: two steps with one alias and two served models are visible as such.
- The tests hold the rule where it is met, with the digests recomputed from their definition and not from the product's helpers: a Research Run against a recording provider stores a `call` equal to what the provider was sent, and the request digest equals the SHA-256 of the exact messages and tools it received; a compaction, a Fast compaction and a Fast answer state the call that wrote them; each adapter hands up the model its server named; a request held by a full scheduler states the wait apart from its own time; the Run's usage is the provider's alone and the `call` holds no counter.
