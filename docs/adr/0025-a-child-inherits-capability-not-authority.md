# A Child Session inherits its parent's capability, not its authority

`spawn_agent` admits a Child Session that shares its parent Run's working copy. Its
tool set is pinned per child, and it defaults to the parent's composed set minus the
tools that spend the Run's authority. This decision makes that default match what a
child actually does — compute, fetch, build — and states the line it may never
cross.

## Status

Accepted and implemented. Every slice under [Consequences](#consequences) has
landed and passed a two-axis review (Standards and Spec) before the next one began. It refines the Child Session contract in
[domain language](../domain-language.md#current-execution-and-workspace-concepts)
and complements [ADR 0024](0024-the-agent-sees-only-its-workspace.md): a child's
processes are confined by the same policy as its parent's, because the working
copy they run in is the same one.

Amended 2026-10-03: the built-in `council` Skill no longer asks for a narrower child.
Its children are ordinary children and hold the default set, with no read-only
restriction and no list of tool names. A read-only first pass had been the recipe's
intent, and a nine-name `tools` list had stood in for the old read-only default
after this decision flipped it. It bought nothing the rule above does not already
give, since the line is authority, not side effects, and it failed any Run that did
not offer one of the nine tools, whole `spawn_agent` call included. `tools` itself
stays, as a way to narrow one child, and never fails a spawn. A user's explicit
`/skill:` request goes to the Run's own agent and is no longer sent to its
children, which keep the Skill catalog while they hold `load_skill`; a child
narrowed away from it is not shown Skills it cannot load.

Amended 2026-10-05: `set_skill_enabled`, which turns one of an owner's own Skills off
or on without deleting it, joins `publish_skill` and `delete_skill` in the table
below. A Skill that is off is not served to any later Run of its owner, so a child
that turned one off would change what its parent and every Run after it find, which is
the authority of the Run that owns the answer. The skills bundle still offers a child
only `load_skill`, and the table keeps a `tools` list from naming the tool back.

The objective prefix is no longer a prefix: [ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md)
moved the child's instructions, the scratch convention among them, into its system
prompt, so the objective is stated once, by the child's own User Entry, and a steer
that redirects the child still comes after it. The convention itself is unchanged.

## Context

A Child Session is one parent Run's worker, not a second Run. It has its own
Session, plan, model, budget, and lease, and it shares the parent's working copy:
`run_child_session` states that a Child "shares its parent Run's working copy", so
a note it writes is promoted from the child's own Tool settlement under the same
lease rather than waiting for the parent to settle another batch.

Three authorities are already out of a child's reach, each by a different
mechanism:

- **Splitting.** The spawn controls (`spawn_agent`, and the subagent roster's
  `status`/`wait`/`cancel`/`steer`/`continue`/`reply`) are removed from every
  child's composed set, so a parent cannot grant them either.
- **Durable owner memory.** `remember` and `forget` are removed the same way.
  Session Notes are a different plane and stay writable: the plane is owned by
  the Session, and a child's contribution reaching it is the point.
- **Publication.** `attach_artifact` is not composed at all when the run is a
  child, and the skills bundle offers a child only `load_skill` — never
  `publish_skill`, `delete_skill` or `set_skill_enabled`. A product belongs to the Run
  that owns the answer.

What the default withholds is capability rather than authority. The default set
is `search_knowledge_base`, `search_web`, `read`, `view`, `grep`, `find`, `ls`,
`recall_memory`, `load_skill`, and `ask_parent`, so a child asked to count pages,
parse a fetched document, compute a statistic, or assemble a file has no way to
run code and must be granted `bash`. The tool description says the caller may
"explicitly list a narrower host-permitted set when side effects are required",
which reads as narrowing only and hides that `tools` is how a child is given the
capability its objective requires.

## Decision

**A tool is either forbidden to a child or granted by default. There is no state
in which a child could be given it but is not.** A middle state — present in the
parent's composition, withheld from the child, restorable by name — makes the
default encode a guess about child work and makes every caller negotiate
capability one sentence at a time. The rule partitions the tool surface once:

*Forbidden, because it spends the Run's authority:*

| Authority | Tools |
|---|---|
| Splitting and the roster | `spawn_agent`, `subagent_status`, `wait_subagent`, `cancel_subagent`, `steer_subagent`, `continue_subagent`, `reply_subagent` |
| Durable owner memory | `remember`, `forget` |
| Publication | `attach_artifact`, `publish_skill`, `delete_skill`, `set_skill_enabled` |

*Granted by default, because it is capability:* every other tool the parent's
composition offers — the path tools (`bash`, `write`, `edit`, `read`, `ls`,
`find`, `grep`), retrieval and reading (`search_knowledge_base`, `search_web`,
`view`), `recall_memory`, `load_skill`, the child's own `ask_parent`, and every
tool a Connection contributes (`mcp__<connection>__<tool>`), which the owner's
enabling of that Connection authorizes for that owner's Runs.

**The default is computed, not listed.** A child's set is the parent's composed
set minus the forbidden set, computed in one place, instead of a hardcoded
read-only mask. That is what makes the rule hold for a tool nobody has written
yet: a new capability is a child's capability the moment the parent has it, and
withholding it becomes the explicit act of adding a row to the table above rather
than the accident of not adding it to a list.

**The line is authority, not side effects.** A child writing files in the shared
working copy is the work it was spawned to do; a child publishing a product,
steering its siblings, or writing durable owner memory is spending authority that
belongs to the Run that owns the answer. That is the rule a reader should carry,
because it decides the next case without another decision.

**`tools` narrows, and that is its only use.** A caller that wants a strictly
read-only investigator or a child without a shell passes that narrower list. It
can never restore what the table above withholds, and where a tool is structurally
absent (publication is not composed for a child at all, the skills bundle offers
a child only `load_skill`) no list can reach it either. Two mechanisms therefore
express one rule, and the test asserts the composition rather than the table: a
name may be added to the table for a tool composition already withholds, and the
two must agree.

The list never fails a spawn. The child holds the tools its Run composed, minus the
table, restricted to the names listed; a name the Run does not offer (no web
provider, no execution environment) or one the table withholds is left out rather
than refused. A refusal cost more than it protected: the table already withholds
what must not be held, a bad name failed the whole `spawn_agent` call, siblings
named correctly included, and a recipe that listed tools by name failed in every
deployment that did not offer one of them. Every child of a call is prepared
before the first is persisted, so a child that cannot be prepared leaves no
sibling behind.

**Parallel children share one tree, so scratch is per child.** The child objective
prefix tells a child to keep intermediate and scratch files under
`tmp/children/<its own child session id>/`, which is the one convention that keeps
simultaneous children from overwriting each other's work. The id needs no new
identity: `spawn_agent` already derives a child's session id deterministically
before the child runs, and the prefix is composed where that id is in hand. Nothing enforces it:
two children that write one path are last-writer-wins, the same rule Session
Notes already have, and it is recorded here rather than discovered later.

**The contract is stated where the caller reads it.** The `spawn_agent`
description says that a child runs with its parent's tools except the three
authority groups, and that `tools` narrows a specific child — instead of today's
"explicitly list a *narrower* host-permitted set when side effects are required",
which reads as narrowing only and hides that the default is where the capability
lives.

**The set is pinned, as it already is.** A child's granted tools are part of its
pinned plan, so what a child could call is answerable from its own durable plan,
the same way a Run's tool plan is ([ADR 0018](0018-run-notes-and-one-continuation-narrative.md)'s
narrative rule already assumes the plan is the record).

## Considered options

- **Keep the read-only default.** Rejected: it encodes a shape of child work that
  does not exist. Every child whose objective needs a computation, a parse, or a
  generated file must spend a sentence of the parent's prompt asking for `bash`,
  and a caller that forgets gets a child that reports it cannot do the task.
- **Give a child everything the parent has, including authority.** Rejected: a
  child that can publish can spend the parent's product authority, and a child
  that can write owner memory can leave durable facts the parent never reviewed.
  Neither is recoverable from the child's own record.
- **Isolate each child in its own working copy.** Rejected: children are workers
  of one Run, and their work product has to land where the Run's memory
  promotion, quota, and observation already look. A per-child copy would need its
  own carry, its own promotion, and its own answer to who merges the trees —
  structure this product does not need.
- **Leave the default read-only and treat `tools` as the normal path.** Rejected
  as the *default*, not as a mechanism: per-spawn improvisation makes the common
  case verbose and the rare case no safer, since the danger was never the tool.
- **Keep the council recipe read-only with a list of tool names, or give `spawn_agent` an
  `exclude` list for the same purpose.** Rejected (2026-10-03): the list encoded a
  restriction the rule above says protects nothing (a child's `bash` is confined like
  its parent's), failed in every Run that did not offer one of its names, and left out
  every Connection tool and every tool added later. An `exclude` parameter would be a
  second way to state what the withheld table already states, would cost a schema,
  digest and contract change, and would have no first user once the recipe stopped
  restricting children.
- **Keep a grantable-but-withheld middle state for the workspace tools.** Rejected
  by the rule above: it leaves the default wrong for every child that needs to
  compute something, and it makes "what can a child do" a per-spawn answer that
  nothing can audit against a single table.
- **Withhold a Connection's tools from a child until it names them.** Rejected:
  the owner's enabling of a Connection authorizes its discovered tools for that
  owner's Runs ([ADR 0012](0012-personal-connections-and-hot-plug.md)), and a
  child is part of its Run; withholding them would put a child's ability back
  behind per-spawn negotiation, which is the state this decision removes.
- **Let a child publish into the parent's Run.** Rejected: the Run that owns the
  answer owns its products; a child contributes findings, and the parent decides
  what becomes a deliverable.

## Consequences

Landing order, one sequence:

1. The child's tool set is computed as the parent's composed set minus the
   authority table, replacing the hardcoded read-only mask, in the one place that
   computes it.
2. The child objective prefix states the scratch convention, naming the child's
   own directory from the id the spawn already derived, and the `spawn_agent`
   description states the capability-minus-authority rule in both of its
   branches.
3. Tests: the forbidden set and the default set are disjoint and together cover
   the parent's composition, so a new tool is classified by that test rather than
   by inspection; an explicit `tools` list cannot restore a forbidden name and cannot fail the spawn; a
   child runs a command and writes a file with the default set, and a note it
   writes reaches the Session's note plane; the artifact and skill-publication
   tools are absent for a child.

Live documents to revise with the implementation:
[domain language](../domain-language.md) (the Child Session term gains the
capability-minus-authority rule) and the `spawn_agent` contract wherever the
tool's description is quoted.

Residual risks, recorded rather than solved: parallel children writing one path
are last-writer-wins, and the convention is guidance rather than enforcement; a
child's `bash` consumes the Run's workspace quota and its output can spill like
any other command; and a child's capability set is now a broader default, so a
parent that wants a strictly read-only investigator must ask for it — which is
the one case where `tools` narrows, and the description now says so. The `council`
Skill does not ask for it.
