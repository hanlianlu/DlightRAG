# ADR 0071 — A goal round is a same-lane continuation

## Status

Accepted by the owner on 2026-10-24 as Phase 4 of the goal-continuity work. **4a** is implemented
(`c1375a6f`): goal rounds are delivered through the one admission/next-turn path, so there is no
second producer. **4b** (continuation inside the live agent loop) was retired on 2026-10-25 by the
[amendment below](#amendment-2026-10-25--phase-4b-is-retired-a-run-is-the-durable-answer-boundary):
in this product a Run is the durable answer boundary, so carrying rounds inside the loop would
re-introduce the complexity Phases 0–2 removed rather than converge further.

**Read the continuation-permission parts through
[ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md).** The round
boundary and the one admission path decided here stand. The `armed`/disarmed continuation language
in the Consequences is superseded: continuation is decided by the durable goal phase plus CAS, an
active goal continues automatically (after a restart, the deployment's recovery scan takes it up),
and pause is immediate: it stops the goal now, cancels the goal round already running, and starts no
next round.

## Context

ADR 0060 made a goal round a **new Run per round**, submitted by a goal-round driver when the
previous Run settled. That was enough to continue a goal, and it produced the product's most
expensive defect: the round and a human message are two producers contending for one agent lane, and
the loser used to fail as a generic answer (ADR 0060 §Not in v1).

Phases 0–3 of the goal-continuity work removed the consequences of that shape without yet removing
the shape itself:

- **S0.1–S0.3**: a send while a Run is live is a durable conversation queue entry;
  settlement coalesces it into one next-turn Run. The queue is the product's next-turn channel.
- **S2.1**: every conversation turn is admitted through the Session work head, in the same
  transaction that links the turn. A second producer can no longer be admitted concurrently; it is
  refused with the typed `session_busy`.
- **S2.2**: queue delivery is fenced by the rows it read and the work-head claim.
- **S1.1–S1.3, S3.1–S3.2**: the Session owns its branch workspace head; a round already continues
  the previous round's files, notes and published products.

The reference implementation (`DeepSeek Harness`, and the Pi harness it derives from) takes the
shape this product now needs: **one live agent per session**, with an inbox that accepts a
**`next-step` (steer)** or a **`next-turn` (queue)** message. A goal round there is a `next-turn`
continuation, not a second producer. Run-per-round is a budget choice, not an architecture.

### Two decisions this work rests on

**A conversation owns its Agentspace.** A goal round is a separate Run, and a new Run used to bind a
fresh working tree that carried only the Session's `notes/`. A generated model and its scripts ended
up in `notes/` (85% of the 256 KiB memory plane), the agent reported its work "lost", and the goal
prompt promised a continuity that did not exist. The goal changed the unit of continuity from the Run
to the conversation, so the conversation's Session now owns the durable file state and each Run works
in a private copy:

- **A Run never writes another Run's copy.** Its working directory is its own epoch; the Session's
  committed head is written only by the settlement commit, never by a live Run.
- **Seed on bind.** A Run that continues the conversation (a follow-up, a goal round) copies the
  committed head into its private epoch before it starts, everything except `tmp/`; `notes/` still
  comes from the memory plane, which stays authoritative.
- **Commit on success only.** A succeeded Run's files become the new head by compare-and-set in the
  terminal transaction; a failed or cancelled Run never advances it. Each settled phase also leaves a
  restorable snapshot, and a fork seeds from its fork-point's snapshot (ADR 0019).

A single mutable directory shared by every Run was rejected for the reason ADR 0019 and ADR 0022
gave: a failed Run would leave half-written files the next Run trusts, two live Runs would write the
same paths without a lock, and "which Run wrote this" would be lost. A Session-owned head with
private Run copies and one commit point keeps the fencing those ADRs required.

**One admission, two deliveries.** A reader's message sent while a Run is live used to become a second
Run on the same lane; it lost the lane and failed as a generic answer. Whether a send is a new turn
needs no new state, only the fact the platform already has: is a Run live on this conversation's
Session?

- **Idle:** a send is an ordinary new turn.
- **A Run is live:** the send goes to a durable conversation queue by default (coalesced, in arrival
  order, into one next-turn prompt when the Run settles), or steers the live Run when the reader asks
  for it. A steer that lost its window degrades to the queue; no path ends in a failed answer.
- **One admission authority:** every producer is admitted through the Session work head's
  compare-and-set, in the transaction that links the turn, so a second producer cannot run on a live
  lane and a refusal is the typed `session_busy`. Queue delivery is fenced by the rows it read.

The reference implementation (below) has exactly this inbox shape; DlightRAG keeps run-per-turn
durability and adopts its admission property and the Session-owned working directory.

## Decision

**A goal round is a continuation delivered through the same lane and admission path as a human
turn.**

1. **One lane, one admission authority.** A goal round never becomes a second producer. The goal
   driver stops submitting Runs directly (`AnswerService.create`/web `accept_goal_round`); it
   *requests* a continuation. Delivery and admission belong to the Session work head introduced by
   S2.1, in the transaction that links the turn. `_session_busy` stays deleted; no scan decides
   precedence.
2. **The continuation channel is the next-turn channel.** Human queued messages and a goal
   continuation share one delivery mechanism (the conversation queue and its settlement trigger).
   **Human work has priority:** a queued human turn is delivered first, and the goal continuation is
   offered only after that turn settles. The driver must not starve behind an endlessly growing
   human queue, so the continuation is offered on every settlement and re-offered on the next one.
3. **Phase 4a — same admission path, Run per round still allowed.** The goal continuation is
   enqueued as an internal delivery (source `goal_round`) in the same durable queue; the fenced
   delivery transaction admits the round as it admits any next-turn at rest. This removes the second
   producer and the lane race even while a round remains a Run. The round cap counts continuations
   *delivered*, not requests refused.
4. **Phase 4b — continuation inside the live lane.** *(Superseded by the 2026-10-25 amendment: no
   in-loop continuation is adopted; the Run boundary is the durable answer boundary.)* Where the
   executor has a live agent lane, the continuation was proposed to be delivered as a next-turn
   inbox message to that lane, and the run-per-round boundary would become a budget policy rather
   than the only way to continue. The turn boundary keeps the durable settlement and workspace
   commit of Phases 1–3; a continuation that ends the Run still settles and commits exactly as a
   round does today.
5. **One producer is deleted, not deprecated.** The slice that lands the continuation deletes the
   goal driver's direct acceptance path and its tests in the same commit (plan §7 decision 0). There
   is no feature flag and no dual path.
6. **Failure semantics stay the ones the goal domain already has.** A continuation that is not
   admitted (cap reached, lane busy in a way the work head refuses, quota refusal) returns to the
   goal driver, which keeps the durable goal active and retries on the next settlement or recovery
   scan; only the existing three-consecutive-round condition blocks it. A failed round records
   `blocked` and a cancelled round records `paused` — [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md)
   supersedes the arming/disarming wording this decision first used.
7. **The prompts state the truth of the shape they run in.** A round continues the Session's
   workspace, its notes and its published products (S1.3); a run-per-round boundary is a budget, not
   a reset.

## Consequences

- The lane contention recorded in ADR 0060 §Not in v1 disappears by construction: there is one
  producer and one admission path. The front-door `session_busy` refusal becomes a rare
  misconfiguration guard rather than a race between producers.
- The goal driver shrinks to: decide whether a continuation is due, and hand it to the admission
  path. Its newest-first scan and direct `AnswerService.create` call are deleted with 4a.
- A human turn can still displace a goal continuation by arriving first; that is the wanted
  precedence, and the continuation follows on the next settlement.
- Phase 4b changes the executor's turn loop and the prompt/persistence boundary around a
  continuation. It is gated on the 4a slice and on the workspace head being the only continuity
  mechanism (already true after Phase 1).
- Not in v1: no concurrent rounds in one Session, no second live agent, and no driver-initiated
  steer of a running goal round. Humans keep the queue/steer switch (S0.2/S0.4); the driver only
  offers the next turn.

## Amendment 2026-10-25 — Phase 4b is retired: a Run is the durable answer boundary

**Status: retired, not deferred.** 4a is the whole of Phase 4.

An Answer Run is one unit in five senses at once: it is the web turn's durable answer, the stream
the reader watches, the state that queue delivery is fenced against, the failure unit that blocks or
pauses a goal ([ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md)),
and the commit that produces one session-workspace snapshot. Those ties are the Phases 0–3
result: the fenced queue delivery deletes rows in the same commit that links a turn, and the
terminal transaction commits the workspace head and the snapshot. Carrying a goal round inside the
live loop would break at least one of them in either direction:

- **One answer at the end.** The Run's visible answer would cover several goal rounds, dropping the
  per-round deliverable that the goal prompt, the round cap and the Volvo-shaped four-phase build
  rely on;
- **Several answers per Run.** The product would need a multi-answer turn model, a new turn row for
  each in-loop answer, and a second continuation path the queue fence has to reason about — the
  same two-producer shape S0.1–S2.3 and ADR 0060's `session_busy` work removed. That is a new
  architecture, not a Phase-4 tail.

**Decision.** Same-lane continuation is delivered at the admission/delivery layer, where it is
indistinguishable from a human next turn (4a): one producer, one next-turn channel, one admission
authority, the same Session lane and branch head. **Run-per-round is the budget policy** — the
`max_rounds` cap counts delivered continuations — not a second lane or a second producer. The
one-answer-per-Run invariant stands.

If the product ever wants rounds inside one live loop, that needs its own ADR and a durable
multi-answer turn model; it is not Phase 4. The plan's Phase 4 is therefore complete with 4a, and the
tests pin the shape: the goal driver requests a continuation, never accepts a turn directly.

## References

- [ADR 0060](0060-goal-mode-is-a-harness-feature-skills-declare-checkpoints.md) — goal mode and its
  round driver; §Not in v1 records the producer race.
- [ADR 0019](0019-turn-accurate-forking-and-the-carry-point.md) — the workspace carry point this
  continuation relies on.
- [ADR 0022](0022-session-owned-memory-and-run-owned-products.md) — Session-owned memory, whose
  fencing objection the commit rules below answer.
