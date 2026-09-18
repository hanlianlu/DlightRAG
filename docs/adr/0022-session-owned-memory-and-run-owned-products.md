# Memory belongs to the Session, products belong to the Run

The agent's file plane splits by kind. The reserved `notes/` set becomes
Session-owned memory that every Run of that Session materializes and promotes
back under its own lease; artifacts, scratch, and spills stay Run-owned. The
continuation carry is deleted, so memory stops depending on how a Run arrived.

## Status

Accepted. Implementation lands in the slices under [Consequences](#consequences).

It supersedes four clauses, and each of those ADRs points here:

- [ADR 0018](0018-run-notes-and-one-continuation-narrative.md), that a Run's
  Workspace Inventory is the one authority on which paths are notes. The path rule
  stands; the authority moves to the Session.
- [ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md), the carry from
  the parent Run's own Agent Workspace, and with it the typed refusals a reclaimed
  or disagreeing parent Workspace produced.
- [ADR 0020](0020-uniform-environment-fast-inert-workspace.md), that Fast's inert
  Agent Workspace exists to receive carried Run Notes and carry them onward.
- [ADR 0021](0021-web-fork-per-turn-and-composer-continuation.md), its residual
  that a Web turn submitted through the composer carries no Run Notes because it
  records no `parent_run_id`, and its decision to keep the browser follow-up route
  for a client that names a Run explicitly.

## Context

Three kinds of state share one plane today, and their lifetimes differ. A
*published product* is byte-compared against the Run's Workspace at publication
and needs that Run's provenance. *Scratch and spills* are working state that
should die with the Run. *Memory* is what a later turn must still be able to
read, and it is the only one of the three that has to outlive its Run.

Because memory was Run-owned, the only way to keep it was to copy it: ADR 0019's
carry materialized the parent Run's registered notes into the continuation's own
Workspace Epoch, all-or-nothing and digest-verified, refusing with
`run_notes_unavailable` when the parent's tree, its Inventory, or its byte
comparison disagreed. Two consequences followed.

Memory availability depended on how a Run arrived. Only a continuation that named
a parent Run could receive the previous turn's notes, so the Web's per-turn
Follow-Up control was the one path that carried them, a Fork carried them into a
new conversation, and a composer turn carried nothing. That asymmetry was
observable in the product before it was visible in the design.

And the carry forced structure on Runs that have no files of their own. Fast
composes no tools, yet it binds an inert Agent Workspace because that Workspace
is what receives and forwards the notes — a Run that exists partly to transport
memory.

Reference harnesses do not have this problem, and the reason is ownership rather
than failure handling. Pi keeps one working directory per Session, memory is
files at stable paths beside the project's own instructions, and compaction and
branch summaries are derived from the Session log; a fork seeds a new session file
over the same tree. DeepSeek's harness is an append-only event log whose surface
fold is the only source of model messages, with compaction replacing a span by a
single summary node and a fork seeding a child from an existing log — its session
and compaction subsystem documents contain no Workspace, filesystem, or carry
concept at all. Both design around the Session for context and let files be files.

What Run-owned trees buy is real, and none of it is about memory: per-Run
provenance for publication, safe deletion at retention, per-lane fencing for
writers, and a Workspace quota. Those belong to products and scratch.

## Decision

**Durable memory is Session-owned.** One authoritative note set per Agent
Session, keyed by the reserved `notes/` path, holding each note's bytes, size,
digest, revision, writing Run, and update time. The bytes ride in the row, bounded
by the plane's own budget, so a note never depends on a Run reference for its
bytes to survive. The rows live in the operational store beside the Session's
entries and registers, and cascade with the Session, which is deleted only when no
routing row references it. A Session is the unit a Web conversation, its forks,
and a stateless caller's continuation already share.

**Products and scratch stay Run-owned.** Publication and Artifact authority keep
reading the Run's own Workspace and keep their byte comparison; spills keep dying
with the Run; Workspace Epochs, fencing, quota, and per-Run reclamation are
unchanged for them — and reclamation no longer touches memory.

**The write rule does not change; the authority does.** A write under the
reserved `notes/` path still declares a note, so no Tool schema and no pinned
Tool Plan changes, and `bash`, `grep`, and a path in the model's hand keep
agreeing with each other. The Run's Workspace holds a materialized working copy:
the Session's notes are laid down when the Run binds, and the Run's changes are
promoted at Tool settlement under its own lease, driven by the Workspace
observation the framework already performs. A promoted deletion removes the row,
because deleting a note is a correction.

**Memory never fails a Run.** Materialization, promotion, and integrity failures
degrade with one typed, observable event carrying its reason, and the Run
proceeds; a note that cannot be promoted lives and dies with that Run's working
copy. Fork Point resolution keeps its fail-closed refusals, because those bound
the transcript and the branch's products rather than memory. This dissolves the
carry-strictness question instead of answering it: nothing is copied, so nothing
is required.

**A fork shares its Session's memory.** Branch-accurate state is the transcript's
and the products'; memory is live and shared, so a fork from an earlier turn
reads memory as it stands now rather than as it stood at its Fork Point.

**Bounds belong to the plane, and refusal is not eviction.** The render cap is
unchanged, so a summary still names few notes. The plane bounds count and bytes
per Session — 64 notes and 256 KiB as the initial defaults, configurable — and
refuses an over-budget promotion for that note with the same event rather than
truncating or evicting memory. Notes stay non-Evidence, never citable, and never a
second statement of what happened; ADR 0018's one-narrative rule is untouched.

**The Web records lineage, not memory transport.** A Web turn submitted through
the composer names the conversation's tip Run as its `parent_run_id`, so a turn's
lineage is stated once instead of implied by turn order. The browser's
`POST /web/api/answer/{run_id}/follow-up` is deleted rather than left without a
client, following ADR 0016's rule that a replaced browser path gets no
compatibility alias. The REST and MCP endpoints keep their own contracts; their
callers name a Run explicitly and their copy already states what the call appends
to.

**Migration is one last carry.** When a Run with registered notes binds into a
Session whose plane is empty, its registered notes are promoted into the plane
once under the rules above, and the Session owns them from then on. A note whose
Run row is already pruned is not recoverable, and the migration says so rather
than pretending otherwise.

## Considered options

- **Keep the carry and choose its strictness per kind** — required for a Fork,
  best-effort for a lane append. Rejected: the defect is ownership, and both
  reference harnesses show it. Memory would still depend on a parent link and on
  a tree that retention reclaims.
- **A Session-scoped working tree** — the option ADR 0019 rejected. Still
  rejected, and this decision is not it: two lanes writing one directory without a
  lock is a fencing problem, while a notes table whose writes settle under each
  Run's own lease and whose rows are per path has no shared tree to race in.
- **Route reads and writes of `notes/` through framework Tools only**, as a
  virtual directory. Rejected: the model's `bash` and `grep` would stop seeing
  what its `write` produced, and the working copy keeps files being files for no
  more machinery than it costs.
- **A filesystem plane per Session instead of rows.** Rejected: it reintroduces
  the fencing, recovery, and reclamation questions the rows answer for free, for
  bytes that are small and bounded.
- **Notes in the Run blob plane**, addressed by digest like other bytes. Rejected:
  that plane's reclamation follows Run references and orphan sweeps, and a note
  must outlive the Run that wrote it — its bytes may not be collected because no
  Run still names them. The plane's own byte budget is what bounds them instead.
- **Copy memory at Fork to keep as-of memory.** Rejected: memory is not a
  checkpoint. A copy would need digests, staleness rules, and a second narrative
  to explain disagreements, and a fork in either reference harness shares its tree.
- **Notes as Session Entries**, reusing the transcript's own machinery. Rejected:
  they would join compaction and the narrative, and a note is live memory.
- **A `remember`-style Tool for notes.** Rejected: a Tool schema change breaks the
  pinned Tool Plan and the request prefix, and the path rule needs no new API.

## Consequences

The vocabulary moves with the ownership: the durable unit is a **Session Note**,
and *Run Note* retires to mean the working copy a Run holds under `notes/`. Live
documents to revise with the implementation: `docs/domain-language.md`,
`docs/retrieval-answer.md`, `docs/durable-answer-runs.md`, and
`docs/interfaces.md` where the carry and the browser route are described.

Landing order, one sequence:

1. The plane and materialization at bind, with the degradation event.
2. Promotion at Tool settlement, including deletion and the plane's bounds.
3. Delete the carry, its Inventory filter, and its refusals; keep the migration
   rule and pin it with a test that a Session's notes outlive the Run that wrote
   them.
4. The composer's lineage field, and the browser follow-up route's removal with
   its route tests.
5. Revisit ADR 0020's inert Workspace: a Fast Run now carries nothing forward,
   and whether it keeps an inert Workspace at all is that decision's to make.

Residual risks, recorded rather than solved. Two lanes writing one path is
last-settled-wins, which suits prose memory and is observable through the row's
writer and revision — but it is a lost update, not a merge. A fork reads memory
that may have moved after its Fork Point. The plane's rows now outlive individual
Runs, so its bounds and the degradation event are the only guards against memory
becoming a second transcript. A stateless REST or MCP caller keeps memory across
calls only by continuing the Run, because naming a Session is not part of
`/answer`. And a Run Note written before this decision whose Run row is already
pruned is gone.
