# Fast shares Research's environment; only tools, skills, and publication are disabled

Fast was the cheaper Answer path, and the cheaper path had been given a thinner
world: no Agent Workspace, no Profile Memory recall, and a compaction that
cleared the Lane's continuation handles whenever it ran after Research. The
owner's product intent is the opposite of that thinning: Fast keeps its
definition — one durable Answer Run, planning plus KB retrieval plus one
generation call, no Agent Operation — and shares the environment Research already
has. What Fast does not get are the capabilities that would make it Research:
tools, skills, and publication.

## Status

Accepted.

The carry rationale this Decision records for Fast's inert Agent Workspace is
superseded by [ADR 0022](0022-session-owned-memory-and-run-owned-products.md):
memory is Session-owned, so a Fast Run carries nothing forward and has nothing to
receive. Whether Fast keeps an inert Workspace at all is that decision's
remaining question; the uniformity conclusion here — Fast and Research follow one
environment contract, differing only in capability — stands either way.

## Context

[ADR 0018](0018-run-notes-and-one-continuation-narrative.md) made a Run Note a
re-readable handle named in the compaction summary. [ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md)
made a continuation copy those notes into its own Workspace Epoch. Both decisions
were implemented on the Research path only: `bind_run_workspace` sat inside the
research-only branch of `AnswerExecutor._execute`, Fast compaction called
`CompactionCoordinator.prepare` with empty `run_notes` and `durable_handles`, and
Profile Memory recall was gated on `resolved_mode == "research"`.

The result was a mode change that dropped the mutable plane. A Research turn that
wrote `notes/plan.md`, followed by a Fast follow-up, followed by Research again,
arrived at the third Run with an empty workspace. Fast compaction on a Lane whose
previous summary named notes and spill handles recomposed both fields from Fast's
own Evidence and Workspace — of which it had neither — and cleared them for every
later reader. And a Fast generation never saw the owner's Profile Memory, even
though recall is context rather than a write capability.

ADR 0019 recorded the framework-field question as a residual: whether those
fields should travel across a mode change was left open because the planes
differ, a note path being honourable by a continuation whose Workspace holds the
file, a spill handle being unreadable by any later Run. That residual is closed
here by making the planes uniform, not by copying another Run's summary.

Two operational facts bound the change. Reclamation was composed only when
`execution_environment != "disabled"`, so a deployment that turned execution off
stopped deleting the trees earlier enabled runs had left. And Fast acceptance
refused to pin Profile Memory at all, so even an executor that wanted to recall
would have found `profile_memory_enabled=False` on the prepared input.

## Decision

**The environment is uniform; capabilities are removed.** Fast binds its own
Agent Workspace epoch, including a continuation carry, and recalls Profile
Memory under the same owner, auth, capability, and epoch gates Research uses.
It still creates no Agent Operation, composes no tools (it never calls
`compose_research_tools`), loads no skills as tools, and authorizes no
publication. The workspace is therefore inert for Fast: it exists to receive
carried Run Notes and to carry them onward. `remember` and `forget` stay
Research-only; recall is context, not a write.

**Framework summary fields are recomposed per Run, and that is correct under
uniform planes.** Each compaction names the compacting Run's own Evidence and
Workspace. Fast's compaction therefore passes `run_notes` from its Inventory —
which, after the carry, holds what it inherited — and passes no
`durable_handles`. Fast has no `EvidenceLedger` at compaction time (compaction
is before retrieval) and no committed spills (it has no tools). Inventing a
citation ordinal the model never saw is worse than naming none. A spill handle
cannot cross Runs; a note path can, because the file travels with the
continuation. This supersedes the "who owns the fields" residual ADR 0019 left
open.

**Reclamation follows the root, not the execution mode.**
`agent_workspace_reclaimer` is built when a workspace root is configured, even
if execution is `disabled`. Deletion creates nothing. Carry stays gated on the
Run actually having a workspace: `disabled` still means no root, no bind, no
notes, no carry, for every mode.

**Acceptance pins recall for Fast.** Fast's prepared input records
`profile_memory_enabled` and the capability epoch the same way Research does, and
Fast's generation capacity measure reserves the standing memory block, because
execution will inject it.

## Considered options

- **Keep Fast workspace-less and copy the previous summary's notes field into
  Fast compaction.** Rejected: it makes the summary a second narrative of a plane
  Fast does not hold, and a later Research turn would be told to `read(path=…)` a
  file that is not there.
- **Give Fast the `read` tool so it can use the notes it carries.** Rejected: that
  is Research. The owner's cut is tools, skills, and publication, not "almost
  Research."
- **Pass Fast's retrieval contexts as `durable_handles`.** Rejected: Fast
  compaction runs before retrieval, and even after it the citation ordinals Fast
  would mint are not the identities a previous Research turn taught. A handle
  whose `[n]` the model never saw is worse than none.
- **Leave reclamation gated on execution mode.** Rejected: turning execution off
  would strand every earlier Run's tree.

## Consequences

A continuation chain that passes through Fast keeps the note: Research writes it,
Fast binds an epoch that names it, and the next Research turn copies the file
with a matching digest. Fast's own compaction names the notes its Inventory
holds. Fast's synthesizer receives recalled Profile Memory, and a disabled memory
capability still suppresses it. Fast's tool set stays empty, which is what makes
the workspace inert.

Residual risks, to revisit rather than extend silently: a spill handle cannot
cross Runs, so a Fast compaction after a Research one drops the Lane's spill
list for the next reader — correct under uniform planes, and the bytes remain on
the volume until reclamation, but the name does not travel; the same compaction
drops the Lane's Evidence handles, including the `[resource: res-…]` aliases a
later Run could otherwise have adopted by name (ADR 0013), because a handle
bundles the minting Run's own citation ordinal with that alias and the ordinal
must not travel — ADR 0013 already treats compaction as the shrink of the
adoptable set, and re-surfacing the bare identities would be a new summary field
rather than a fix here; Fast acceptance now
reserves standing memory, so an `auto` request whose Fast window cannot hold the
worst-case recall block loses Fast rather than overflowing at generation; and a
disabled execution environment with no configured root still builds no reclaimer,
so trees left under the default home path by an earlier trust deployment are not
swept unless the operator names that path.

Live documents revised with the implementation: `docs/architecture.md` (the Fast
overview line), `docs/durable-answer-runs.md` (reclamation follows the root, and the
disabled contract), `docs/retrieval-answer.md` (the
Fast section), `docs/configuration.md` (the disabled contract), and
`docs/domain-language.md` where Fast Answer and Run Note are already recorded.
