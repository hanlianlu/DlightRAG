# Run Notes are re-readable handles, and the compaction summary stays the only continuation narrative

A long Run's work product can only survive compaction if the model may write it
somewhere durable, and only one thing in this system is allowed to say what
happened. This decision adds the first without weakening the second: an Agent
Workspace file under a reserved notes path is registered as a re-readable,
non-Evidence handle, carried by the Context Projection as a continuation
identity, and never treated as a record that can contradict the Session.

## Status

Accepted, not implemented. The glossary terms land with this decision; the
behaviour, its tests, and the live-spec revisions land in the slices under
[Consequences](#consequences).

## Context

Continuation memory today is exactly two things, both of them the Context
Projection: the typed compaction summary (goal, constraints, progress,
decisions, next steps, critical context, `paths`, `durable_handles`) and the
retained tail kept verbatim. What the next turn may re-read is `durable_handles`,
and its only producer is `run.evidence.citation_handles()` — Evidence identities.

Two gaps follow. First, the Run already has a second class of re-readable
identity and the summary ignores it: a committed spill is durable on the volume,
durable as a row, digest-recorded, carried across a Workspace Epoch handoff, and
explicitly taught to the model (`read(resource_id=…, cursor=…)`) — and it becomes
unreachable by identity the moment the prefix carrying its handle is covered.
The bytes survive; the name does not. Second, the agent's own work product has no
home at all: the glossary already makes `Compaction Summary`, `Resource Handle`,
`Artifact Attachment`, and `Memory Record` mean four different things, and none
of them is "what this Run worked out and must not lose".

Read-only context injection is not the fix. `CompactionSummary.paths` has been
declared since the schema was written, is rendered when present, and is passed
`None` at its only call site — and that is correct, because a path list describes
a mutable plane with no staleness check, while every other continuation identity
in this system is a digest-addressed identity that a read re-authorizes. A
Workspace Inventory listing would also have to decide which paths matter, which
is the semantic judgement the framework cannot make and the model does not need.

External review of two memory systems informed the shape and is not a template
for it. A coding-agent memory plugin with agent-authored Markdown memory
explicitly **forbids** persisting current-task plans (its own Skill: route them
to the current task only), keeps agent-written knowledge for durable
repository-scoped reuse maintained by a separate policy job, and gates its memory
writes behind a verification bar rather than a nag. A memory-OS service never lets
the model write its store at all: extraction happens post-turn, inside the
framework. Neither injects a workspace manifest; one injects bounded
preferences/procedures only at turn start and after compaction. Both publish
reliability claims that the repositories themselves do not substantiate — one
hard-codes `confidence=0.99` on every extracted record, the other's README scores
contradict its own paper. The lesson taken here is the gate, not the machinery,
and the requirement to measure before claiming.

One asymmetry in the summary schema is load-bearing for this decision: **a field
may be added and may never be removed.** `CompactionSummary.from_canonical_json`
rejects unknown fields, so deleting `paths` would make every Projection already
committed in a deployment undecodable, and recovery of any Run that has ever
compacted would fail closed. Adding a field with a default is backward compatible
in both directions.

## Decision

**One continuation narrative.** The Session projection remains the only statement
of what happened. A Run Note is a re-readable artifact, never a competing
narrative: there is no arbitration rule between a note and the Entries, so the
note must never be able to assert something the Entries do not.

**Run Notes are declared by path, not by a tool argument.** A write inside the
Agent Workspace's reserved notes directory, outside `artifacts/` (whose meaning is
publication authorization for user-facing deliverables), registers that path at
settlement with its raw digest and byte size. Identity is the path with its
current digest: rewriting a note replaces its handle rather than adding one, which
is the same rule an Artifact Attachment already uses for a reattached path. No
Tool schema changes, so no pinned Tool Plan changes.

**Notes render in their own typed field, and non-Evidence handles join the existing
list.** The summary gains a notes field with a small cap, so the deliberately
written continuation files cannot be crowded out by a retrieval-heavy Run's
Evidence handles. `durable_handles` keeps its existing role and cap and starts
receiving every re-readable non-Evidence handle the Run holds — committed spills
first, because their omission is a defect rather than a design choice.

**Registration is non-Evidence by construction.** Reading a Run Note admits no
Evidence row and mints no citation handle. The committed-spill read path already
has exactly this property (a read produces Evidence effects only when the
resource reports evidence available), and Run Notes reuse it: a note is
continuation memory, never a source.

**The habit is stated once, not repeated.** The system prompt states that a long
task records conclusions under the notes path. There is no per-turn or
every-N-turns framework reminder: a framework-authored reminder is not an accepted
Steer and would need a new Session Entry variant, and per-turn prose breaks the
prompt prefix the request shape exists to preserve. The framework's half of the
bargain is deterministic instead — a written note is registered, so the next
compaction carries its handle whether or not the model remembers writing it.

**Admission has a floor.** The prompt's rule is conclusions, decisions, paths, and
numbers the next turn cannot cheaply re-derive — not a narrative of the work. The
render cap bounds it mechanically, and the experiment below is what decides
whether the floor holds.

## Considered options

- **Populate `paths` with a Workspace Inventory listing.** Rejected: a locator
  list over a mutable plane with no staleness check, next to identity-bearing
  handles that are re-authorized on every read; and the "which files matter"
  judgement is semantic, so the framework would either list junk or guess.
- **A parallel agent-authored record of the Run, alongside the summary.** Rejected:
  two narratives with no arbitration rule. The Session Entry tree is the only
  record of what happened, and a note that can contradict it is worse than no note.
- **A per-turn or cadence framework reminder to write notes.** Rejected: it needs a
  new Entry variant (`ControlMessageEntry` is an accepted Steer), it puts moving
  prose after the transcript, and the external evidence favours a gate over a nag.
- **Register every `write`.** Rejected: deliverables, generated reports, and scratch
  files would flood the cap and the notes would become the second transcript that
  compaction exists to prevent.
- **Delete the inert `paths` field instead of leaving it.** Rejected: it would
  strand every already-compacted Run at recovery (see the decode asymmetry above).
- **Adopt the external cadence-reminder-plus-gate design wholesale.** Rejected: its
  gate governs durable repository-scoped knowledge maintained by a separate job,
  not current-task progress, and importing it would bring a second agent-authored
  memory plane this design refuses.

## Consequences

Landing order, one sequence shared with
[ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md):

1. Compose `durable_handles` from Evidence handles plus every committed spill.
2. Register a Run Note under the reserved notes path at write settlement.
3. Carry the notes field in the summary, with the experiment below as its gate.

The shipped claim is narrow and testable, and it is the only claim this change
makes: after forced compactions, (a) the summary carries the note handle, (b) the
model can and does read the note back at a later turn, and (c) the note stays
bounded rather than growing into a second transcript, measured against total input
tokens and prompt-cache hits. Answer quality, summarizer quality, and anything
across Runs are out of scope for that experiment and may not be argued from it.

The decode asymmetry becomes a rule rather than an observation: summary fields may
be added with defaults and are never removed, so a future schema change is
additive or it is a migration with recovery consequences.

Residual risks, to revisit rather than extend silently: models may under-use the
habit, and the deterministic reward only guarantees that a note that exists
survives — the experiment measures adoption, not enthusiasm; the notes directory
consumes Workspace quota and its bytes are not free, which is why Workspace
reclamation travels with the continuation carry in
[ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md); and a note whose
digest changes between compaction and re-read must produce an explicit stale
answer rather than silently serving different bytes.

Stop conditions: Run Notes being used for user-facing deliverables (that is what
Artifacts are for), the notes field displacing Evidence handles or the reverse, or
notes degenerating into a per-turn log.

Live documents revised with the implementation: `docs/retrieval-answer.md` (the
context and budget section), `docs/durable-answer-runs.md` (Agent Session
recovery), and `docs/domain-language.md` where the terms are already recorded.
