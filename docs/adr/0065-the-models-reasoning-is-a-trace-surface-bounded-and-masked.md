# The model's reasoning is a trace surface, bounded and masked

A model call may return the model's own reasoning beside the answer it produces. DlightRAG
already keeps that text: every provider adapter fills `AssistantTurn.reasoning`, the Session
writes it into the assistant entry, and provider state carries whatever the wire needs to
continue a tool-use turn. What it never did was show it. The transcript routes project an entry
by whitelist, and reasoning was deliberately left out; the public Child observation names it as
one of the fields it strips. A trace viewer that hides the agent's own reasoning can describe
what a Run did but not why it took a step, so the Inspector — the product's only trace surface —
now shows it, and a Run still thinking streams it to the reader who is watching. The two
surfaces are the same text at two moments: the live draft passes the reader content gate every
other projection passes, and the trace page keeps it bounded.

## Status

Accepted and implemented for the live `reasoning` event and the Inspector activity page's
`reasoning` field. The Run timeline's page field is retired by
[ADR 0070](0070-the-inspectors-run-timeline-is-a-bounded-index-followed-by-delta.md); reasoning
stays readable through the activity page and the timeline's on-demand step read.

## Context

[ADR 0061](0061-a-protected-skills-text-is-masked-at-the-readers-edge.md) established that a
reader-facing projection is a display action: the record keeps the text a Run reasoned with, and
the projection a reader is handed is bounded and masked. The transcript routes already project
messages by whitelist, so the record holds more than any reader sees. Reasoning was one of those
withheld fields, and `docs/security.md` states the reason it was withheld: a public Child
observation is status, lineage, and evidence handles, not provider-private reasoning.

Two things changed the decision. The Inspector became the product's review surface — a Run's
steps, Tool calls, facts, and decisions are read there, and its purpose is to answer why the Run
did what it did. The owner also asked for the model's reasoning to be visible the way a modern
agent product shows it: while the model is thinking, the reader watching the answer sees the
thinking appear, and after the Run settles the trace keeps it readable.

The reasoning text is the most model-private content the product holds. It is not the answer:
it may quote a protected Skill document, restate a system prompt's rule, or show a provider's
half-formed plan. It is also the most direct evidence of what the Run was doing. The product's
answer to that tension is the one ADR 0061 already applies to every other reader surface: keep
the record whole, project through the reader's content gate, bound what a reader pane carries,
and never let the projection reach the model.

## Decision

**Two reader surfaces, one text.** Reasoning is shown in exactly two places, both belonging to
the owner of the Run and both read from the same captured text:

- **While the Run is non-terminal**, the browser conversation receives `reasoning` durable run
  events, batches of the root agent's reasoning as it streams. They are masked at the edge with
  the same safe-prefix carry the answer's `token` stream uses, so a protected passage crossing a
  batch boundary is withheld rather than released early. A Tool event closes the block: the live
  carry is discarded there, so no block's held tail can bleed into the next one, and the trace
  page is what completes the block's last characters. The chat draws them as collapsible folds
  inside the turn's process group — the same rolling, capped list that holds the Tool rows,
  interleaved in the order the work happened — and clears them when the Run settles: the reading
  surface keeps its rule that a finished answer shows the answer only, and the Inspector is where
  the process stays readable.
- **When the trace is read back**, the Inspector carries the assistant message's `reasoning` on
  both of its surfaces, bounded to 4000 characters and masked through the same reader content gate
  as the message's words. The activity page draws it on the step that message produced — the words
  it said, the Tool calls it chose, or the answer it concluded — and a message that said nothing
  but chose Tools stands as a Thinking step before them, so the "why this tool" case is read exactly
  where the Tool is. The Run timeline states a `thinking` step ahead of the words or calls the same
  message produced, so a reader following the Run's own order sees the reasoning in place rather
  than having to open another pane. This is the durable, bounded record; the live blocks exist only
  to make the wait legible and are discarded when the Run settles.

**The record is not masked.** The Session entry, the run events as committed, the Session the
model continues from, and the training export keep the reasoning whole. The observability plane
keeps the facts it already states — model, provider, usage, the step's shape — and gains no
reasoning text, because a trace's telemetry is about what a step cost, not what it thought.
Masking is a projection applied where a reader is handed text, exactly as ADR 0061 decided for
the transcript; a masked span must never reach the model.

**The canonical result and transcript surfaces do not gain it.** REST and MCP's transcript tail,
the transcript search, and the public Child observation keep the field out.
The owner's review pane is where a private model artifact belongs; an integration surface and a
status observation are not. The canonical event stream is not one of those withheld surfaces: it
is the Run's own event log, so both transports serve the same masked `reasoning` frames live,
and a reader of the canonical stream sees exactly what the browser projection hands its own
reader. The security document's statement about the public Child observation stays true and is
the contract those surfaces are held to.

**One run's root agent streams it live; every agent's turns carry it in the trace.** Only the
root Run publishes provider text (`publish_provider_text`), so only its reasoning streams while
the Run runs. Children do not publish provider text, and their reasoning arrives when their
turns are read back from the transcript. Adding live child reasoning would mean publishing
another agent's provider stream into the parent's conversation, which this decision does not do.

**Empty is ordinary, not a failure.** Provider support varies: a dialect may return encrypted
reasoning, no text, or only a summary (Gemini's `thinking_summaries`, for one). A model call that
returns none simply has no block to show, and the trace says nothing about it rather than
pretending the model did not think.

## Considered options

- **Keep it private (the status quo).** Rejected: the review surface could not answer why a Run
  took a step, and the product's own trace is the place a reader goes for that answer. The
  question "what did the agent think before it chose that tool" is the one an owner asks when a
  Run goes wrong.
- **Put reasoning in the observability plane only.** Rejected: ClickStack is the operators'
  surface, not the owner's. A decision is not visible to the person who pays for the Run if it
  can only be seen in a telemetry backend.
- **Show it on every reader surface, including REST, MCP and the public Child observation.**
  Rejected: the canonical transcript feeds integrations, and the public Child observation is
  deliberately status-and-evidence shaped. Reasoning is the owner's private model artifact; it
  belongs in the owner's review pane, not in a machine contract.
- **Persist masked reasoning.** Rejected for the reason ADR 0061 gives: masking is a display
  action, and a record that lost the text would change what the next turn of the Run can do and
  what the training export holds.
- **Summarize the reasoning with another model call before showing it.** Rejected: it adds a
  model call to the read path, and a summary is the product's words about the thinking rather
  than the thinking the model wrote. The trace states evidence, not paraphrase.
- **Stream child reasoning through the parent's conversation.** Rejected: it would publish one
  agent's provider stream into another agent's reading surface, and a Child's turn is already
  reviewed in its own pane. The retrospective transcript covers it.

## Consequences

- A reader watching a live answer sees a "Thinking" block appear before the answer's words. The
  block is one more stream the browser projection masks, so it costs one additional
  `TokenFrameMasker` carry per subscription and the same holdback as the answer: the block's last
  few dozen visible characters wait on the next batch, and a block that ends at a Tool call is
  completed by the trace page rather than the live draft.
- The retrospective surfaces carry at most 4000 characters of reasoning per assistant message; a
  timeline row further cuts its excerpt to the one line every spine step shows and opens the full
  bounded text. A longer turn is truncated silently; the full text stays in the Session entry and
  the run events, and the trace's job is to show the shape of a step, not to reproduce megabytes.
- A thinking step joins the Run's totals and composition, so the spine's summary counts the model's
  thinking beside its calls and messages. It is not a decision point: the four mechanical marks are
  stated about the step that made the decision (a Tool call), a thought states none, and the
  decision count is the number of marked steps rather than the number of rows that share their
  identity. A step that names a reasoning message shares that message's Session sequence with the
  words or calls it precedes, and the page cursor names the step's kind as well so a page boundary
  cannot lose half of one message.
- Run events grow by the reasoning text a model writes on each turn — one more stream at the
  same batching as tokens. The record was already keeping this text in the Session; the event
  log is now a second copy, and retention prunes it with the Run.
- A provider that returns no readable reasoning shows nothing. The UI has no empty-state label:
  there is no fact to state, and a "the model did not think" message would be false for the
  providers that keep it encrypted.
- The live block clears when the Run settles, so a reader who looks after completion sees the
  answer only and must open the Inspector for the reasoning. That is the existing product taste
  about a finished reading surface, applied to one more stream rather than changed.
- `reasoning` joins the edge-masking list ADR 0061 names. A new live text stream must state its
  masker and its carry, or it is a reader surface outside the gate.
