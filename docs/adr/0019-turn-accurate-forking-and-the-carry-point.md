# Turn-accurate forking, and the carry point for the mutable plane

Fork and Follow-Up are the same continuation machinery today: both branch from
the source Lane's *current* head, so forking an earlier turn branches from the
conversation as it now stands, and the fork's Agent Workspace starts empty while
its transcript is fully shared. This decision makes the two operations mean what
their own contracts already say — a Follow-Up appends to a line, a Fork opens a
branch at one Run's exact state — and makes the mutable plane travel with it.

## Status

Accepted, not implemented. The glossary terms land with this decision; the
recording, seeding, carry, reclamation, and their tests land in the slices under
[Consequences](#consequences).

## Context

Three of this system's own surfaces described Fork and the implementation did not
match any of them. The MCP tool says "sibling branch from one terminal run's
accepted context"; the browser dialog says "Start a new conversation from the same
context. **The previous answer is not carried over**"; and the continuation
builder says a fork "branches from the same starting point without carrying the
answer it is meant to redo". In the implementation, `ensure_session_lane` seeds a
new Lane at `source.head_entry_id` — the source Lane's *current* head, which for a
terminal Run is the state *after* its answer — and the fork's fold is rebuilt from
that ancestry with no projection to hide anything. So a fork does carry the
answer; the only channel the flag steers is the injected history, where the same
content is then repeated. A Follow-Up on a session-backed Lane repeats it too,
because the Lane fold already holds the parent's question and answer, while the
Web path that continues the same conversation passes no history at all and does
not. Two code paths implement "continue here" and only one of them avoids the
duplicate.

Fork also carries nothing it can edit. Under the mapping the product already uses
— Lane as branch, Entry tree as history, Agent Workspace as working tree — a Fork
is a fully shared history with an empty working tree, which is not `switch -c` and
not `switch --orphan`; it is a combination no branch operation has.

The fix is small because the durable model already holds almost everything it
needs. Entries are immutable and parent-linked, so branching is pointing a new
Lane head at an existing Entry; `ensure_session_lane` already builds a new Lane's
head from an existing Entry and only ever uses the source Lane's current one.
`CompactionEntry` records a complete
projection (`projection_id`, `summary`, coverage sequences, both boundary Entry
identities, source digest), so the projection active at a given head is
reconstructible from the tree. And an Agent Workspace is per Run
(`workspace_root/<owner shard>/<run_id>`), so in a Web Conversation — where each
turn is its own Run — the turn-granular snapshot of the mutable plane already
exists on disk. Nothing needs a new snapshot mechanism.

One durable fact is missing. `dlightrag_answer_run_routing` records the Session,
the Lane, and the source Lane, and no Entry identity: nothing records the Lane
head a Run settled at, so "the state this Run ended in" is currently
unaddressable, and deriving it from timestamps or Entry ordering would be
reasoning from clocks that this system refuses elsewhere.

Two operational facts bound the change. Epoch retirement works — a successful
handoff deletes the previous epoch — but a Run root is never reclaimed: nothing in
Answer retention or pruning touches `agent_workspaces/<shard>/<run_id>`, and the
only thing that clears the root is the development reset script. And carrying
bytes multiplies that debt by the number of continuations.

## Decision

**Fork is a tree operation; Follow-Up is a line operation.** A Follow-Up appends
to the selected Lane's tip and nothing can be inserted into the middle of a Lane.
A Fork opens a new Lane and a new Web Conversation at its Fork Point. The
per-turn Fork control is therefore correct as a product affordance and the
per-turn Follow-Up control is not: only the tip offers Follow-Up.

**The Fork Point is recorded, not inferred.** At terminal settlement — success,
failure, or cancellation alike — the Run's routing row records the Lane head Entry
its state ended at and the projection active there. The state at a Run's end is a
real state whether the Run succeeded or died, and a Fork from a failed Run is
meaningful: everything the Run established before it failed.

**A Fork inherits the as-of projection, not the raw ancestry.** The new Lane is
opened with the Fork Point as its head and the projection reconstructed from the
newest `CompactionEntry` at or before that head, instead of re-folding the entire
transcript and forcing an immediate recompaction. That is what "from the same
context" means, and it keeps the first request bounded by whatever the parent was
already carrying. It buys no prompt-cache reuse: the new question sits before the
Session fold in the composed request.

**The carry point is the Fork Point's own Agent Workspace, and the carried set is
its registered Run Notes.** Materialization happens under the consuming Run's own
fence into its own epoch, digest-verified, with Inventory rows written as for any
other file. The rule is uniform across both continuation kinds, so "your notes
follow you" needs no case analysis. There is no working-tree history and no
per-turn file snapshot: carrying the parent's whole Workspace is rejected (it
propagates generated junk and multiplies the epoch-copy headroom every Run needs),
and an empty Workspace is rejected because a branch with no tree is not a branch.

**History is derived from the branch point.** A continuation whose branch point
already contains the conversation injects no history at all; only a caller that
supplies history with no Agent Session branch point — a stateless REST or MCP call
— has it injected. `include_answer` retires to a history-only flag, and because
turn-accurate forking already covers "redo this answer" (fork at the preceding
turn and ask again), no separate retry operation is introduced.

**Degradation is explicit.** A Workspace already reclaimed, a Run outside this
Session, a missing Fork Point, or a head no longer present fails the Fork with a
typed refusal and a remedy. Silently falling back to the Lane tip is exactly the
divergence between contract and behaviour that this decision removes.

**Reclamation lands with the carry, not after it.** Per-Run Workspace deletion
joins the existing per-Run retention prune, an orphan sweep removes Run roots with
no Run row under a row-and-lease guard rather than by directory age, and the
acceptance tests below cover both plus the ordering between carrying and pruning.

## Considered options

- **Keep the empty Workspace and treat a Fork as a fresh start.** Rejected: it
  discards the state the branch is supposed to branch from, and it is what makes
  the product's own copy false.
- **Define Fork as redoing the answer, seeding the Lane before the terminal
  assistant Entry.** Rejected as a product action, not as a mechanism: it is a
  retry, turn-accurate forking already expresses it by forking at the preceding
  turn, and one operation with two meanings is what produced this divergence.
- **Make the Agent Workspace Session-scoped so continuations share files.**
  Rejected: it introduces a second fencing domain (the Workspace Epoch CAS is
  anchored to the Run row), two Lanes of one Session would write one directory
  with no lock, and it breaks the epoch-copy headroom invariant by making the
  copied state unbounded across Runs.
- **Extend lineage adoption to filesystem-backed kinds instead of carrying.**
  Rejected: adoption is lazy, handle-triggered, blob-plane, and read-only, against
  bytes another Run holds; a branch seeds a mutable plane the new Run will edit
  under its own fence. Adoption also stays available for what it is for.
- **Offer Fork only on the newest turn.** Rejected: choosing the divergence point
  *is* Fork's main scenario, so restricting it to the tip reduces the operation to
  "duplicate the conversation as it stands".
- **Let the caller name the branch Entry.** Rejected: identifiers embedded in
  message text are not references, clients would have to know Entry identities,
  and a Run's terminal state is a server-side fact.

## Consequences

Landing order continues [ADR 0018](0018-run-notes-and-one-continuation-narrative.md)'s
sequence:

4. Record the Fork Point on the routing row at terminal settlement.
5. Seed a Fork from that Fork Point with its as-of projection and the typed
   refusals, aligning the MCP and dialog copy in the same change.
6. Reclaim per-Run Agent Workspaces in the retention prune, with the orphan sweep.
   This slice is a prerequisite of the carry slice, not a follow-up to it.
7. Carry Run Notes from the Fork Point's own Agent Workspace, with the experiment
   below as its gate.
8. Derive history injection from the branch point, retire `include_answer` to a
   history-only flag, and align `docs/interfaces.md` with the landed behaviour.

The experiment that gates the carry is separate from the Run Note experiment and
may not borrow from it: interrupt a Run after a compaction, then start a
continuation, and assert that the note file is present in the new Workspace with a
matching digest, that the new Run's first request states the carry once, and that
the task continues without re-deriving. Whether the model *would have needed* the
note is not claimed here.

Out of scope, deliberately, and not to be imported later by analogy: any mid-Run
rewind (a Run's internal turns are not product checkpoints), merge, diff, compare,
or linearize operations, working-tree history, and any branch/commit/ref vocabulary
in the product surface. Lane and Web Conversation remain the names.

Residual risks, to revisit rather than extend silently: existing Runs have no
recorded Fork Point, so the new column is populated from the point of deployment
and older Runs refuse a Fork with the remedy instead of branching from the tip
(acceptable, since the dialog copy they shipped with was already untrue); a Fork's
projection must remain valid against a *shorter* ancestry than the parent's, which
the existing digest check already enforces and the tests must pin; carried bytes
raise Workspace quota consumption, which is why the reclaim slice is a
prerequisite of the carry slice and not a follow-up to it; and the Web copy, the
two MCP descriptions, and the continuation builder's comment must move in the same
change as the behaviour, since three surfaces stating three things is how this
happened.

Live documents revised with the implementation: `docs/interfaces.md` (the fork
endpoint and continuation fields), `docs/durable-answer-runs.md` (Agent Session
recovery and run retention), `docs/retrieval-answer.md` where continuation context
is described, and `docs/domain-language.md` where the terms are already recorded.
