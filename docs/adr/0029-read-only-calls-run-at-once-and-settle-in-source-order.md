# Read-only calls of one Tool batch run at once and settle in source order

The durable Agent runtime executed a model turn's Tool calls strictly one at a time.
Adjacent calls whose Tool declares them read-only now run at once, a bounded group at
a time. Their results still settle in the batch's source order, and so does what they
do to the Run's evidence, image budget, trace, and settlements. Citations and the
transcript are therefore exactly what one-at-a-time execution produces from the same
answers. Every other call still runs alone and is a barrier between groups.

## Status

Accepted and implemented. It adds a contract fact to `ToolDeclaration` and lets one
pending record of the Agent operation state name several calls. The Effect Intent,
Effect Settlement, and replay rules it works within are unchanged:
[ADR 0012](0012-personal-connections-and-hot-plug.md)'s never-replay Connection
calls, and [ADR 0015](0015-prompt-prefix-stability-and-cache-anchored-accounting.md)'s
evidence frozen into the Tool result that admitted it.

## Context

`AgentSessionRuntime` records a call's intent (`ToolEffectPending`) before running it
and its result after. Recovery after a crash or lease reclaim therefore knows which
call was in flight: it replays that call if the call is replayable, and otherwise
settles its outcome as unknown. Because only one call could be pending, a batch ran
serially.

On the development deployment none of 446 Tool calls overlapped. Turns averaged 1.6
to 1.7 calls, but a web-heavy turn issues four to eight `search_web` and
`search_knowledge_base` calls, and each waited for the one before it. The Research
prompt already tells the model that independent tools may run in the same turn.

Those calls only read, so their own work can overlap. What they do next cannot,
because it touches state the whole batch shares:

- evidence admission, where the order of admission assigns citation numbers;
- the ADR 0015 freeze, which renders the rows admitted since the previous freeze into
  this call's result;
- the image budget;
- the Run trace;
- the host update that persists the evidence ledger with the call's settlement.

Letting that work run in completion order would make citations, frozen text, and
persisted evidence depend on network timing. A crash could also persist one call's
evidence with another call's settlement.

## Decision

**Read-only is a declared contract fact, and the default is sequential.**
`ToolDeclaration.read_only` means a call changes nothing outside its own Run's record
of what it read. The Agent Run Plan and every Tool batch item pin it next to the
replay policy, so a changed flag settles `tool_contract_changed` like any other
contract change. Nothing infers the flag: not a Tool name, and not an MCP server's
`readOnlyHint`. Connection tools are declared sequential, and the Connection effect
gate refuses a pending Connection call marked read-only.

| Read-only Tool | Why |
|---|---|
| `search_knowledge_base` | Reads the selected corpus. It admits what it found into this Run's evidence, in source order. |
| `search_web` | Reads a search provider. It registers result links as inert Resources and admits passages, in source order. |
| `read` | Reads a workspace path, a Run Resource, or a public URL. A URL fetch lands in a Resource under its own intent's settlement. Adopting an earlier Run's Resource spends this Run's attachment allowance, so it waits for source order. |
| `view` | Reads pixels from the same sources. It spends the shared image budget only in source order. |
| `ls`, `grep`, `find` | Read the workspace under the access scheduler. Any output they spill goes to a uniquely named Run-owned file. |

Every other Tool stays sequential. That covers `bash`, `write`, `edit`,
`attach_artifact`, `remember`, `forget`, the Skills tools, the subagent tools, and
every Connection tool, whose effects belong to the remote account. `recall_memory`
also stays sequential, though it only reads, because nothing has measured a need for
it to run in a group.

**Adjacent read-only calls begin together, at most eight at a time.** The pure
interpreter starts a group from a read-only executable call and the read-only
executable calls directly after it, up to `MAX_CONCURRENT_TOOL_CALLS = 8`. Eight
covers a web-heavy turn in one round while bounding what one batch asks of a
provider at once. The bound is per Agent Session, so a parent and each of its Child
Sessions run their own groups. Any other position ends the group: a call with side
effects, or a call that never executes (unknown, denied, invalid, or truncated). That
call runs alone, so a barrier stays exactly where the model put it.

**One intent record per group, one settlement per call, in source order.** Before
anything runs, the Runtime commits one `ToolEffectPending`. It names the first call
of the group and its attempt, plus one attempt for each read-only call beside it
(`concurrent_attempt_ids`). The calls then run at once. Each result commits in its
own transaction once the results before it have, and that commit moves the pending
record past it. The pending record always names exactly the calls that may be in
flight, and every call before them is settled. A single call writes the same record
as before, with no concurrent attempts.

**Shared state is touched in source order.** The Runtime gives each effect an
`in_source_order` awaitable, which a Tool reaches through
`ToolRuntime.in_source_order()`. It returns once every earlier call of the group has
returned, and a call returns only after reaching it. A call does its own work beside
its neighbours: a retrieval, a search, a fetch, a file read, a subprocess. It awaits
`in_source_order` before it admits evidence, adopts an earlier Run's Resource, spends
the image budget, or writes the trace, and the Research host awaits it before the
freeze and the host update. Evidence, citation numbers, the image budget, the trace,
and every settlement are therefore exactly what running the calls alone in source
order would produce.

Acquiring a Resource is not ordered. The Resource registry already shares one fetch
and one conversion among the calls that read the same Resource, and a group relies on
that. Only rare races there depend on timing:

- Reading a URL and the page it redirects to in one group can print either of two
  handles for that page.
- A presentation header can meet a snapshot the other call has already admitted.
- `grep` and `find` create the shell's scratch home under `tmp/` the first time
  either runs.

**Recovery is per call, in source order.** After a crash, recovery closes the pending
calls one by one under the existing rule. A replayable call runs again under a fresh
attempt. A call that never replays settles `outcome_unknown` under the attempt that
was pending. A call settled before the crash keeps its result. A call that finished
its work but had not settled is simply still pending, and recovery treats it like the
others. Read-only and replayable remain separate facts. The two searches are read-only
but never replay, so a crash in their group settles them as unknown outcomes.

**Cancellation closes the whole group.** Cancelling the drive cancels every call of
the group. `Cancelling` carries the pending attempts, so each pending call closes as
`outcome_unknown` under its own attempt. Positions after the group close as
`interrupted`.

## Considered options

- **Infer read-only from names or from MCP annotations.** Rejected. A name is not a
  contract, and a remote server's hint describes an account this product cannot see.
- **One process-wide semaphore that dispatches calls as slots free.** Rejected. A
  call waiting for a slot would be recorded as pending before it was dispatched. Slots
  held by calls waiting for their turn in source order could also deadlock. Bounded
  groups keep "pending" meaning "dispatched" and need no slot bookkeeping. The cost is
  that a slow call holds back the next group.
- **Settle in completion order.** Rejected. Transcript order and citation numbers
  would follow network timing, and a crash could persist one call's evidence with
  another call's settlement.
- **Guard shared state with a lock instead of an order.** Rejected. A lock makes the
  calls exclusive but not deterministic: whichever finished first would still number
  the citations.
- **One pending register per call.** Rejected. The operation state register is the
  Runtime's one program counter, and a group needs no more than an extra list of
  attempts in it.
- **Make read-only imply replayable.** Rejected. Replay safety after a reclaim is its
  own question. The searches are safe to overlap long before they are safe to replay.

## Consequences

A web-heavy turn's four to eight searches run in one round instead of in sequence.
Committing results takes as many transactions as before, and a group writes one
intent record instead of one per call.

A crash in the middle of a group can leave up to eight calls unknown instead of one.
Each is closed by its own contract, and the model can ask again.

A group's `tool_start` events arrive together, so the live trace shows several
running rows at once. The trace keeps its five newest rows, so a group of more than
five shows only its last five while they run. A call's measured duration includes any
time it spent waiting for an earlier call. Concurrent searches share their provider's
rate limit, and a refusal reaches the model as an ordinary Tool error.

The contract binds the Tool author. A read-only Tool that touches shared state before
`in_source_order` makes the batch nondeterministic. Tests pin the order for search
admission, resource-backed evidence admission, adoption, both of `view`'s image paths,
and the Research host's settlement.

Revisit this decision if any of these happens:

- A read-only Tool needs to hold shared state across its own work.
- Grouped searches regularly meet provider rate limits.
- A deployment needs a different bound.
