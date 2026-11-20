# Runtime Policies are read at the door and pinned into the Run

A Runtime Policy decides an Answer once, when it is accepted, and the Answer
carries what the policy said. A change to a policy reaches the next submission
of every process and no Run already accepted. Two policies decide an Answer
this way: `agent_effort_ceiling`, the highest effort any of its agents may run
at, and `active_answer_runs`, how many Answers one owner may have in progress.

## Status

Accepted and implemented. It revises
[ADR 0014](0014-caller-chosen-agent-effort.md) in one place: a Run's child
agents now run under the same ceiling as its own agent. Everything else that
decision says stands, including that a caller's choice reaches the Run's own
agent only.

## Context

A Runtime Policy is a setting an administrator changes while the deployment
runs. It is one JSON document per key in PostgreSQL, replaced whole, and read by
every request that depends on it, so a change holds on the next request of every
API process with no restart and no cache to invalidate. A policy is *targeted*:
a default and ordered rules, each naming a verified JWT claim and giving the
callers it matches another value; the first match decides. A deployment that
verifies no JWT cannot tell its callers apart, so only the default applies
there. Reading and replacing policies is one deployment-wide action,
`runtime_policy.manage`, because a policy lists the people it targets. The
application registers each policy by its key, with the shape its document must
have and the value that holds until one is written, and a stored document that
no longer fits is never read as that default: a policy can switch something off
for the people it names, and a default would switch it back on. The first policy,
`answer_feedback`, is read when a reader rates, so a change takes effect on
their next request. That is the right moment for it.

An Answer is different, and two things an administrator asks for show how. They
want to lower how much effort a group's Answers spend, since a high reasoning
level is the cost the product most easily lets one person raise
([ADR 0014](0014-caller-chosen-agent-effort.md) lets a caller choose `max`
and names a per-owner cap on high-effort Runs as the thing to add if misuse
shows). And they want to bound how many Answers one person has in progress at
once.

A Run is durable. It is accepted once, may wait in the queue, may be deferred for
a dependency and claimed again, and is recovered after a crash, and none of
that reads current configuration: its input is the audit surface, and recovery
runs what was accepted
([ADR 0007](0007-unified-durable-runs-and-execution-lanes.md),
[ADR 0014](0014-caller-chosen-agent-effort.md)). A Run that read a policy each
time it was claimed would run at one effort before a crash and another after.
And a submission is keyed so that a retry returns its Run: if the policy a Run
was accepted under were part of what the key is compared by, an administrator's
edit between a submission and its retry would turn the retry into a conflict.

## Decision

**A policy is read once for a submission, at its door, and the Run is pinned to
what it said.** `AnswerService.accept` reads both policies for the submitting
caller after it has answered a replay and before it prepares the input. It is
the one pipeline Web, REST, MCP, follow-up and fork all pass through, and
it takes the caller as a required argument beside the owner, so no transport
can omit it. The caller is whoever submits, with the claims their token
verified; it is the identity a policy names people by, and the owner id it
projects to is not. A replay is answered before any policy is read, so a retried
key returns its Run whatever the policies have become, and no unreadable
document can refuse it.

**The effort ceiling is pinned into the Run's input.** `agent_effort_ceiling`
is `low`, `high` or `max`; `max` is no ceiling at all and is what holds until an
administrator writes one. The accepted Run's input carries `effort_ceiling`
beside `effort`, which still holds exactly what the caller asked for. The ceiling
is not part of the idempotency fingerprint, which is taken from the request, so
a policy changed between a submission and its retry leaves the retry a replay.
Replays, retries, deferrals and recovery run the pinned ceiling and never read
the policy.

**Every agent of the Run runs at the level it would have, no higher than the
ceiling.** The level is what the caller chose, else what the role was configured
with, lowered to the ceiling on the model's own reasoning ladder: the highest
level the model offers that is at or below the ceiling. The ladder is walked
downward and not rounded, which `resolve_reasoning` does for a level the model
lacks, because a rounding that crossed the ceiling would be no ceiling. `off` is
never a target of the clamp. A model that offers nothing at or below the ceiling
runs at its lowest level, and the trace says it is above the ceiling. A model
left on its provider's own level, with nothing configured and nothing chosen,
cannot be bounded by leaving it alone, so under a ceiling below `max` it is
asked for the ceiling's level. This applies to the Run's own agent and, by the
same function, to each child agent, which runs at the lower of the level its
role was configured with and the ceiling. A caller's choice still reaches the
Run's own agent only. A role that owns its reasoning through raw
`agentic_model_kwargs` takes no typed level, so neither the choice nor the
ceiling reaches it: the trace records `ignored: raw_kwargs`, as it does for a
choice. The Run's trace states `agent_effort: {requested, ceiling, effective,
ignored?, above_ceiling?}`.

**The offer follows the ceiling.** What a caller is offered, in the bootstrap
`agent_effort` and in MCP `get_capabilities`, is computed for that caller: the
levels above their ceiling are removed, and `default` is the level their Answers
run at when they choose nothing, so the control marks what is true. The
frontend needs no change: a stored choice the offer no longer lists is read as no
choice, and the server lowers one that arrives anyway, which is what a page left
open across an edit sends.

**The active-run limit is pinned at admission and counted under the lane's
lock.** `active_answer_runs` is a whole number of at least one, or none, which is
no limit and holds until an administrator writes one. At acceptance the store
counts the owner's Answer Runs that are `queued` or `running` while it holds the
lane's acceptance lock, which also admits the Run, and after a replay has been
answered. At the limit it refuses the new Run and stores nothing. The count and
the insert cannot be separated by another submission of that owner, so racing
submissions admit exactly the limit. The count includes Runs deferred for a
dependency, which are queued still. A limit lowered below what an owner holds
refuses more and cancels nothing. The limit is a condition of one acceptance and
is not stored with the Run. The refusal is HTTP 429 with `error_kind:
active_run_limit` on REST and the Web command envelope, and `Error
[active_run_limit]: ...` on MCP. It stays the Web's retryable command kind, so
the browser keeps the draft and offers Retry. It is a different thing from the
503 a lane at its deployment-wide limit answers: the caller is over their own
limit, and the deployment has room. A partial index,
`idx_dlightrag_runs_owner_active` on the owner's active Answers, serves the
count (see Consequences).

**An unreadable document fails a submission closed and never the page.**
Acceptance raises `PolicyUnreadableError`, which every transport maps as it maps
any outage: 503, with the policy named and the document never repeated. The
bootstrap and `get_capabilities` degrade as `answer_feedback_enabled` does: the
page loads and offers no effort at all, since the document that would have
lowered it for someone is the one that cannot be read, and an administrator
replaces it to restore both.

**`none` and `simple` authentication have one owner.** Every caller they admit
is the deployment's owner, so the limit there is the deployment's: how many
Answers it may have in progress. Only the default of a targeted policy applies.

**What is deliberately not a policy.** The deployment-wide cap on a lane's
nonterminal Runs, worker and model concurrency, the number of children one
Run runs at once, and the bound on how often a Run may be deferred stay
configuration and code
([RunRuntime](../run-runtime.md#admission-limits)). They bound the machine and
are decided with the deployment; a policy bounds a person and is edited while
people use it. A per-owner policy is not a second spelling of the lane cap: in a
deployment with one owner it bounds that owner, which is the whole deployment's
Answers, and the lane cap still bounds Retrieval and the rest.

## Considered options

- **Read the ceiling when a Run executes.** Rejected. A Run could run at different
  efforts across a recovery, and a policy edit would change a Run already
  paid for. The pin is the Run's own record of what it was allowed.
- **Put the ceiling in the idempotency fingerprint.** Rejected. An edit between a
  submission and its retry would make the retry a 409.
- **Refuse a request above the ceiling.** Rejected. A chosen effort never fails a
  Run ([ADR 0014](0014-caller-chosen-agent-effort.md)), and a page left open across an
  edit would otherwise fail submissions the server can serve.
- **`resolve_reasoning` to round a level down.** Rejected. It rounds a tie up, so
  a level the model lacks can land above the ceiling.
- **Treat `off` as the floor of the clamp.** Rejected. `off` is an explicit,
  hard request everywhere else, and a ceiling that switched reasoning off for a
  model without a low level would be a different feature.
- **Re-level every role's children by the caller's choice.** Rejected, as in
  ADR 0014: a caller's pick for the Run's own agent says nothing about a child.
  Only the ceiling reaches a child.
- **Count the owner's Answers in the application, before the store.** Rejected.
  Two submissions would both read the room that is left. The count belongs under
  the lock that admits the Run.
- **Count every Run of the lane, or Retrieval too.** Rejected. The limit is on
  Answers, the Runs a person waits on and pays for.
- **Cancel the newest Runs when a limit drops.** Rejected. An accepted Run is
  the caller's, and the limit is a condition of admission.
- **No index.** Measured and rejected. On a database of 558,000 Runs with
  28,000 queued or running (near the default lane cap of 30,000) and one owner
  holding 250,000 finished Answers, as a deployment with no tell-apart owners
  would, the count walked the active rows of the whole lane: 28,236 buffers,
  6.8 ms, under the lane lock that every Answer acceptance waits on. The
  partial index reads the owner's active rows only: 0.07 ms and no heap. The
  lane-wide count that already ran on each acceptance reads 272 buffers in
  2.4 ms, so without the index the limit would have read a hundred times as many
  buffers as the cap it sits beside. This is a vacuumed table, which an index-only
  scan is costed against. On one never vacuumed, and for an owner that holds most
  of the table, the planner estimates the owner's active Runs from the table's
  statistics, finds them many, and keeps to the lane's claim index: the cost
  the limit would have had without an index, never more.
- **A policy for the deployment-wide cap.** Not built. It is a capacity
  decision made with the deployment, and in a one-owner deployment this policy
  already is it for Answers.
- **Fail the bootstrap on an unreadable document.** Rejected. The page must load
  so an administrator can sign in to replace the document.
- **Read an unreadable document as its default.** Rejected, for the reason the
  mechanism has always given: a default would admit what the document restricts.

## Consequences

- A policy edit holds on the next submission of every process and never on a
  Run already accepted. A Run's agents all stay within the ceiling it was accepted
  under through its deferrals and recoveries, and the trace says what ran.
- Children run lower under a ceiling than they were configured to. A Run
  accepted before ceilings existed carries none, and runs as it always did. The
  spawn tool's model guidance, which the Run's Agent Plan pins as text, keeps
  listing each role's configured reasoning request. Under a ceiling below `max` it
  adds one sentence that names the ceiling as the most a Child runs at, and with
  none it is the text it always was. Acceptance and execution render it with one
  function from the same pins and ceiling, so the plan comparison cannot drift.
- A page open across an edit may offer a level the server then lowers. The trace
  states `requested`, `ceiling` and `effective`, so it is seen.
- A role whose reasoning is raw model kwargs is not bounded by the ceiling, and
  says so in the trace. The deployment owns that role's reasoning; the ceiling
  cannot reach what it cannot name.
- `PreparedRunEnvelope` carries `active_run_limit`, a condition of one
  acceptance that is neither stored nor fingerprinted, and `AnswerService` takes
  a `caller`. The store refuses the new `ActiveRunLimitExceededError`, which
  admission names `ActiveRunLimitError`, an unavailable-family error with its own
  429 mapping.
- The schema gains `idx_dlightrag_runs_owner_active` on
  `dlightrag_runs (owner_id) WHERE run_kind = 'answer' AND status IN
  ('queued', 'running')`, in the baseline and as one more migration, so a
  database made before it gains the index at the next writer start.
- `error_kind: active_run_limit` joins the answer error vocabulary; the Web
  localizes it and the vocabulary test holds its words to the server's.
- Tests hold the rule at the product's boundaries: a provider that records the
  request is sent no level above the ceiling for the Run's own agent or any
  child, and the Run's own agent, a child and a role that owns its reasoning each
  get what they should; the Run's input carries the ceiling and the request keeps
  the choice; a policy edit between a submission and its retry leaves a replay; a
  replay at the limit is answered and an unreadable document never refuses one;
  several connections racing for an owner's last places admit exactly the limit
  and another owner is unaffected; a limit lowered cancels nothing; an edit made
  on another connection holds on the next request; the 429 and the 503 reach a
  browser as retryable commands that keep the draft; the bootstrap and MCP
  offers differ by caller and survive an unreadable document; and the count is
  served by the index.
