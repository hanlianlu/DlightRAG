# Turn-accurate forking, and the carry point for the mutable plane

Fork and Follow-Up are the same continuation machinery today: both branch from
the source Lane's *current* head, so forking an earlier turn branches from the
conversation as it now stands, and the fork's Agent Workspace starts empty while
its transcript is fully shared. This decision makes the two operations mean what
their own contracts already say — a Follow-Up appends to a line, a Fork opens a
branch at one Run's exact state — and makes the mutable plane travel with it.

## Status

Accepted and implemented. The slices under [Consequences](#consequences) have
landed: Fork Point recording, turn-accurate seeding, workspace reclamation, Run
Note carry, and history derived from the branch point. Live specs name the terms.

Evidence for the compaction and carry experiments is unit- and integration-level
at the product seams (the write tool, Inventory settlement, `bind_run_workspace`,
the request assembler, coordinator PG tests), not a live-provider run. "The model
actually re-reads a carried note" is therefore structurally demonstrated — a later
turn can `read` the file, the summary names it, the continuation copies it — and
not empirically demonstrated against a provider.

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

**The Fork Point is recorded under the claim that ends the Run.** Immediately
before its owner writes the terminal row — success, failure, or cancellation alike
— the Run's routing row records the Lane head Entry its state ended at and the
projection active there. A Fast Run commits its own terminal row, so it records
the point first; every other path records on the way out, before the coordinator's
terminal write. An attempt that defers or waits for repair records nothing, because
it has not settled and a head it merely passed through would outlive a later
attempt whose own write failed. A Run whose worker died before it could record one
is not forkable, and a Fork from it refuses with the remedy rather than branching
from wherever the Lane has since moved.

**A Fork inherits the as-of projection, not the raw ancestry.** The new Lane is
opened with the Fork Point as its head and the projection reconstructed from the
newest `CompactionEntry` at or before that head, instead of re-folding the entire
transcript and forcing an immediate recompaction. That is what "from the same
context" means, and it keeps the first request bounded by whatever the parent was
already carrying. It buys no prompt-cache reuse: the new question sits before the
Session fold in the composed request.

**Recording it reuses the Session seam's own read.** A research Run passes the view
it just drove, so the point costs nothing extra; a Fast Run refreshes the Host's
cached view, or reads once on a failure path that never held one. A narrower
"read two registers" primitive was rejected: the Session repository's seam is
deliberately snapshot-or-transaction — an architecture test pins exactly that
method set — and widening it to save one delta refresh would trade a checked
invariant for a measurably cheap read.

**A Run that has not settled carries no Fork Point.** The two columns are cleared
with the requeue that follows a graceful shutdown and with a deferral, so an
attempt that never ended cannot leave a head behind for a later attempt whose own
write fails to inherit.

**The carry point is the Fork Point's own Agent Workspace, and the carried set is
its registered Run Notes.** Materialization happens under the consuming Run's own
fence into its own epoch, digest-verified, with Inventory rows written as for any
other file. The rule is uniform across both continuation kinds, so "your notes
follow you" needs no case analysis. There is no working-tree history and no
per-turn file snapshot: carrying the parent's whole Workspace is rejected (it
propagates generated junk and multiplies the epoch-copy headroom every Run needs),
and an empty Workspace is rejected because a branch with no tree is not a branch.

**History is derived from the branch point.** A continuation whose branch point
already contains the conversation injects no history at all; only a caller with no
Agent Session branch point — a stateless REST or MCP call — has its parent's
accepted history injected. Where the branch point holds a turn that never got an
answer, the Fold carries that turn, because it is the one turn the continuation is
continuing. `include_answer` retires to a history-only flag, and because
turn-accurate forking already covers "redo this answer" (fork at the preceding
turn and ask again), no separate retry operation is introduced.

**Degradation is explicit.** A Workspace already reclaimed, a Run outside this
Session, a missing Fork Point, or a head no longer present fails the Fork with a
typed refusal and a remedy. Silently falling back to the Lane tip is exactly the
divergence between contract and behaviour that this decision removes.

**Reclamation lands with the carry, not after it.** Per-Run Workspace deletion
joins the existing per-Run retention prune, and an orphan sweep removes Run roots
whose Run row is gone. The guard is the row, not directory age: retention selects
only terminal Runs, and a lease is renewed through its own row, so a Run with no
row can hold nothing. A root whose row still exists is never touched. Residual risk,
noted rather than solved: deleting a conversation removes the rows of its Runs,
including a turn that is still executing, so a dying worker can see filesystem
errors for up to one heartbeat while the sweep removes the tree it was writing to.
The acceptance tests below cover both reclamation paths, the disabled-environment
gate, and the ordering between carrying and pruning.

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

4. Record the Fork Point on the routing row under the claim that ends the Run,
   before its terminal write.
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

Residual risks, to revisit rather than extend silently: a summary is rendered to
the Run that reads it rather than rewritten for it, so a Run without a `read` tool
sees the summary's content and not the re-read calls it cannot make — but the
framework fields themselves are recomposed by each compaction from the compacting
Run's own Evidence and Workspace. [ADR 0020](0020-uniform-environment-fast-inert-workspace.md)
makes those planes uniform — Fast binds an inert workspace and names the notes it
carries — and records the residual that a spill handle still cannot cross Runs; a continuation's admission
fingerprint describes the caller's submission rather than the identities this
process draws: a Fork mints a Lane and a stateless continuation mints a Session, and
both are excluded from that hash, because leaving them in made every retry of an
identical submission look like changed input. Deriving the two identities from the
submission instead was tried and withdrawn — a Lane outlives the idempotency row
that named it, so reusing a key after retention would have reopened the branch a
previous Run left behind; existing Runs have no
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
