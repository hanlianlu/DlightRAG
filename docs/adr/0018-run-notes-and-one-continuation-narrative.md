# Run Notes are re-readable handles, and the compaction summary stays the only continuation narrative

A long Run's work product can only survive compaction if the model may write it
somewhere durable, and only one thing in this system is allowed to say what
happened. This decision adds the first without weakening the second: an Agent
Workspace file under a reserved notes path is registered as a re-readable,
non-Evidence handle, carried by the Context Projection as a continuation
identity, and never treated as a record that can contradict the Session.

## Status

Accepted and implemented. The three slices under [Consequences](#consequences)
have landed: committed spills join `durable_handles`, a write under the reserved
notes path registers a Run Note, and the compaction summary carries the notes
field. Live specs name the terms.

The ownership clause is revised by [ADR 0022](0022-session-owned-memory-and-run-owned-products.md):
a write under the reserved notes path still declares a note, and the note set is
still a filter over one observation, but the authority on what a Session
remembers is the Session's own notes plane rather than a Run's Workspace
Inventory. The path rule, the notes field, the render cap, and the one-narrative
rule are untouched by that revision.

The experiment that gates the notes field has now been run at the seam and against
a live provider (DeepSeek Flash, the low agent level, ten Runs, about $0.72).
Measured: a registered note is carried by every later compaction and rendered as
the call that reads it again (2/2 compactions in the one Run that wrote a note
before compacting); the note stays bounded (227–3065 bytes against 0.5M–1.8M
prompt tokens); and a later Run reads it, unprompted — the one continuation this
experiment ran opened both notes it had inherited and appended its own
recomputation to one. That last observation was taken through the continuation
carry, which [ADR 0022](0022-session-owned-memory-and-run-owned-products.md) has
since replaced with a Session-owned plane; the property it measured, a later Run
reading the note instead of re-deriving it, is the same one the plane serves
without a copy.

The write side did not hold, and the reason was where the habit landed rather
than whether it was stated. Across four unsteered Runs and five
compactions, no Run wrote a note *before* a compaction; two wrote one only as a
closing summary. `notes/` was the only one of the workspace's
three planes the Run had to create for itself, so a Run's working state — a stage
table, in the measured case — went to `tmp/`, which nothing carries, and the
guidance's trigger, "when a task runs long enough that earlier steps stop being
visible", is a condition the model cannot observe: it knows neither its own token
count nor the compaction trigger. A Run told by an accepted Steer to write its
values before the next search did so on its third turn, and every later
compaction carried them.

Three follow-ups landed in the change that measured them: the workspace
pre-creates `notes/` beside `artifacts/` and `tmp/`; the guidance names two
triggers a model can check for itself and states that `tmp/` is scratch; and a
note handle names the moment as well as the call while the re-readable identities
render before the plan they serve. The guidance's promise now matches ADR 0022's
ownership: `notes/` is the conversation's memory, every Run of it starts with what
was written there, and no wording promises a carry that no longer exists. The admission floor was measured
next, with a fixture that asks for corpus facts rather than arithmetic: an
unsteered Run over five definitions in a textbook wrote `notes/findings.md` on its
sixth turn, *before* the first of its four compactions, and all four named it
(444 bytes against 1.09M prompt tokens). The four earlier fixtures that never saw
a note were arithmetically re-derivable, which is the category this decision says a
note is not for, so the write side holds once the floor applies — with the caveat
that one Run writing one note is one data point and not a rate.

The read side was then measured where reading is the only cheap path, and it holds.
The fixture asked a Run to generate 1200 random IDs into `notes/ids.txt` without
pasting them into the chat, so the bytes existed in the file and nowhere else within
reach: the transcript held the generating command rather than its output, a summary
can say what an eight-hex-digit list is but cannot state one, and nothing re-derives
a CSPRNG. A compaction then named the note, and on its own initiative the Run went
back to the file, read the three lines it was asked for, recomputed the SHA-256 it
reported, and checked the file's shape; every value it answered with matches the
bytes on disk exactly. It also flagged, unprompted, that a later edit of the note
would change both the lines and the hash — the live-file property this decision
records as a residual, understood by the Run it applies to.

Two properties of that observation are worth keeping rather than smoothing over. It
read through `bash` (`awk`, `sha256sum`) rather than the `read(path=…)` call its
handle names, which is the freedom ADR 0022 kept when it refused to route `notes/`
through framework Tools: a note is a file, and the call in the handle is a
suggestion rather than the only door. And the read was conditional on need rather
than habitual — the same guidance produced no read at all in the four Runs whose
context already carried the answer, which is the behaviour the decision asks for.
Memory is where a value lives, not a ritual performed on every compaction, and the
earlier negative results are a statement about those fixtures rather than about the
habit.

### Re-measured under Session-owned memory and a confined shell

The experiment was run again after [ADR 0022](0022-session-owned-memory-and-run-owned-products.md)
replaced the carry with a Session-owned plane and
[ADR 0024](0024-the-agent-sees-only-its-workspace.md) confined the Agent's processes, on
a staged task that establishes page numbers early and needs them two compactions later
(DeepSeek Flash at the low agent level, three Runs, about $0.35):

- **The mechanism holds end to end.** A note written mid-Run (`notes/page_map.md`) is
  named by every later compaction (2/2), promoted to the plane under the writing Run's
  own lease (revision 2, 5298 bytes), and a **later Run of the same Session starts with
  it materialized** — the continuation's working copy and the plane agree byte for byte
  (5777 bytes, digest `2bbf85c1…`), and that Run answered, edited the note, and did not
  search the corpus once.
- **The model does read it back**, and the read is conditional on need rather than
  habitual: after a compaction it opened the note it had written, and in the fixtures
  whose context already carried the answer it did not.
- **Bounded**: 227–5777 bytes against 0.5M–1.8M prompt tokens across every run of both
  experiments.
- **Unprompted adoption is fixture-dependent, and that is the residual this decision
  already records.** On this staged task the first Run crossed **four compactions without
  writing a note at all**, re-deriving page numbers by searching again; the steered twin
  of the same task wrote the note on its twenty-second turn and kept it current. The
  earlier fixture, whose values had to be gathered sequentially and quoted later, saw an
  unsteered Run write its note on turn six. So the habit fires on the shape the guidance
  names — a value a later step depends on — and not on a Run that never stops searching.
  **The shortfall is accepted as measured, and the write side will not be strengthened.**
  A framework sentence at compaction time was considered and refused: it would arrive
  after the values were already lost, and it cannot make a retrieval sweep cheaper,
  because what such a Run re-searches for is the source's own citable text. Reading a
  note costs a Tool call while the summary is already in the request, and an Evidence
  handle for retrieved corpus text carries no `read(resource_id=…)` identity, so a note
  is the model's own paraphrase rather than the authority. The deterministic half is what
  this decision owes, and it holds on every run of both experiments.
- **The confinement changed the model's behaviour as intended**: that Run tried four
  direct routes into the corpus (a bare filename, a `local://` uri, an absolute path, a
  parsed-element id). All four were refused by the workspace-rooted tools, and it went
  back to retrieval.

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
publication authorization for user-facing deliverables), makes that path a Run
Note. There is no second registry: the Workspace Inventory the framework already
observes is the one authority on what the Run holds, and the note set is a filter
over it. That makes the Inventory's own completeness a precondition rather than a
detail — a verified Workspace Epoch handoff therefore records the copied epoch's
observation instead of emptying the table, because a filter over an authority that
forgets on recovery is not an authority. No Tool schema changes, so no pinned Tool
Plan changes. A note's identity is its path with the size the Inventory states —
never a digest it may not have, because a `bash` call re-observes the whole
workspace without digests and only the paths this framework wrote itself carry one.
Rewriting a note keeps that identity and replaces its size, which is the same
replace-by-path rule an Artifact Attachment already uses for a reattached path.

**Notes render in their own typed field, and non-Evidence handles join the existing
list.** The summary gains a notes field with a small cap, ordered by path so a note
the Run keeps updating holds its place, and a note is named by the call that reads
it again — a path, not a resource handle, because the file is the Run's own working
state. The deliberately written continuation files therefore cannot be crowded out
by a retrieval-heavy Run's Evidence handles. `durable_handles` keeps its existing
role and cap and starts receiving every re-readable non-Evidence handle the Run
holds — committed spills first, because their omission is a defect rather than a
design choice.

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
[ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md); and a note is a live
file, so a re-read serves the bytes it holds now rather than bytes frozen at the
compaction that named it. That last property is deliberate rather than overlooked:
the writer is the same Run, and the newest content is what the next steps want. If
a note ever misleads a turn by having moved on, revisiting it means recording the
projection's note digests and refusing a changed note explicitly — a new decision,
not a quiet tightening of this one.

Stop conditions: Run Notes being used for user-facing deliverables (that is what
Artifacts are for), the notes field displacing Evidence handles or the reverse, or
notes degenerating into a per-turn log.

Live documents revised with the implementation: `docs/retrieval-answer.md` (the
context and budget section), `docs/durable-answer-runs.md` (Agent Session
recovery), and `docs/domain-language.md` where the terms are already recorded.
