# ADR 0060 — Goal mode is a harness feature, and a Skill plugs in by declaration

## Status

Accepted; implemented through the reference's Web surface. The goal domain, the durable session row
and its compare-and-set store, the three tools, the settlement-driven round driver, rounds that are
Web conversation turns, the goal bar and the `/goal` commands are in place. The Round's evidence is
an in-run pre-delivery checkpoint, not a registered provider: the `CheckpointProvider` seam in
Decision 4 was retired — see the
[2026-10-24 Amendment](#amendment-2026-10-24--the-provider-seam-is-retired-the-checkpoint-runs-inside-the-turn).
Machine-enforced completion predicates remain open. The round-driver crash recovery, the
token-limit handling and the process-local arming named under Not in v1 below are superseded by
[ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md): recovery scans
durable state at startup and periodically, a token-limit round records `blocked`, and no arming
bounds the claim.

**Superseded in part by [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md).**
The process-local `armed`/`disarmed` permission is retired: the durable phase is the continuation
authority, an active goal continues automatically (after a restart, the recovery scan takes it up),
and pause is immediate: it stops the goal now and cancels the goal round already running. The goal
domain, the durable row and its CAS store, the three tools and the
settlement-driven round driver remain as decided here; the arming language in Decision 3, the
2026-10-20 Web-surface amendment and the 2026-10-25 amendment is historical.

## Context

A live Run of a built-in Skill (`01a155ca…`, 2026-10-19) settled after 5.4 minutes and 32 turns
with an Answer that said the sealed model was unfinished and could be built "next round". The
Skill's delivery contract was prose, and a Run is one message: the platform had no durable objective
to continue, and no point at which the evidence the reader would see could be handed back to the
Agent while the task still had a next step.

DeepSeek Harness's goal mode is the reference shape for that missing capability: one persisted goal
per session, event-sourced in the session log, a compare-and-set `revision`, `roundsStarted` derived
only from admitted goal-round user messages, a driver that starts the next round when the agent goes
idle, completion decided by the model's own tool call with no evaluator, blocking only after the same
condition persists for three consecutive rounds, a new process starting `disarmed`, and goals created
by the model or the user but never forced. This product keeps the threshold of started rounds that
gates an autonomous block; it does not persist a condition streak, and the prompt and ADR say so. Its budget is the round cap; a max-tokens turn end stops
the continuation.

At the same time the product has two development tracks that must stay independent. A Skill is a
bundle of text, scripts and references that must gain monitoring, insertion and checkpoints without a
harness release; the harness must provide those general points without knowing which Skill is loaded.
The existing Skill-facing surface is already declarative and generic — `required_reads` and
`attestation` in frontmatter, read by `engine/agent/skill_reads.py` and
`engine/agent/skill_delivery.py`, rendered for the reader by `application/answer/provenance.py`
(ADR 0057, ADR 0058). The goal must consume that surface, not replace it, and no goal code may name a
Skill.

## Decision

**The harness owns the goal; Skills contribute declarative checkpoints; neither imports the other's
specifics.**

1. **Goal state and its transitions are engine domain objects with no I/O.** One goal per answer
   session: `{id, revision, objective, phase, max_rounds, rounds_started, blocked_reason}`, phases
   `active | paused | blocked | complete`. Every mutation carries the current `{id, revision}` and is
   refused as stale once superseded; only `edit` changes the objective or the round cap, and it may
   not fall below the rounds already started; a cap is any positive integer, with no product
   ceiling (the reference's deployment default is 256, while this product defaults to 8 because a
   round is a whole Run); an autonomous `blocked` is accepted only after the
   threshold of started rounds (the reference shape's gate; the prompt asks for the same condition
   across those rounds, and no condition streak is persisted); `complete` is the model's call and
   is never machine-enforced. Arming is process-local and is not part of the row (see Not in v1).
2. **The engine owns the goal service; an adapter implements its one store port.** Authority,
   activation and the compare-and-set live in `engine/answer/goal_service.py`, beside the Session
   Memory service the answer runtime already owns, and it is typed by a narrow `GoalStore` port. The
   PostgreSQL adapter implements that port; the application composes it. A goal is one current row
   per session, written with a compare-and-set on `revision`; the reference shape replays a session
   event log, while this product already stores mutable session-owned rows, so one row gives the same
   no-lost-update guarantee without a replay fold. `create_goal`, `get_goal` and `update_goal` follow
   the reference authority rules: creation, edit, pause and resume happen in a direct human turn;
   complete and blocked are also accepted from a goal round.
3. **The round driver is application orchestration.** When a Run settles and the session's goal is
   active and armed with rounds left, the driver starts the next Run in the same session with the
   objective and the round number. A human message wins over a queued round; at the round cap the
   driver writes `blocked` with the round-limit reason
   and disarms, and `resume` refuses at the cap until a human edits it; a goal round that fails or is
   cancelled disarms, while a human turn's own failure does not; a new process starts disarmed, and
   a human must resume. A round is always research, because that is the mode whose tool set contains
   the goal tools.
4. **The extension point was a `CheckpointProvider` (retired; see Amendment 2026-10-24).** The round
   prompt is composed from registered providers; each provider reads durable Run evidence and
   returns findings as plain lines, or nothing. The Skill-derived provider (a declared read the Run
   did not make, and a declared verifier it never named or that failed) is one implementation; page
   provenance stays a v2 item. The goal domain, the driver and the prompt contain no Skill name and
   no Skill-specific branch: with every Skill removed, goal mode still works; adding a monitor is
   registering a provider, never editing the goal. The provider is the only code that may name a
   Skill, because the record it derives names one; no harness branch keys on it.
5. **Skills declare; they do not wire.** The Skill-facing surface remains `required_reads` and
   `attestation`, plus at most a future declarative `checkpoints:` block read by the in-run delivery
   check. A Skill's own checks live in its scripts and references and are invoked by the Agent; the
   harness never executes Skill content in order to monitor it. No harness module branches on a
   Skill's name.
6. **v1 is evidence, not enforcement.** The in-run checkpoint injects the same record the reader
   will see. The Agent may mark a goal complete with items outstanding, and the reader's record stays
   honest. Goals are opt-in — created by the Agent for a long-running direct request, or by the user
   — and a one-Round answer without a goal is unchanged.

## Amendment 2026-10-20 — the reference's Web surface is implemented

- **A goal round is an ordinary conversation turn.** The driver accepts a round through the Web
  conversation's acceptor when the settled Run has a conversation, so the reader sees the round where
  the turns are; a Run with no conversation (REST or MCP) falls back to plain acceptance.
- **The goal bar is the human's surface.** It shows the phase, the round, the objective and a blocked
  condition, and offers pause, resume, edit and clear. Resume is offered for a paused goal, a blocked
  goal, and an armed-less active goal (the state a restart leaves); a complete goal is not resumable.
  When the round cap is exhausted, Resume asks for a higher cap and raises it before resuming.
  The bar reads again when the conversation changes, after a command, and when a turn settles.
- **`/goal` is the same surface without the bar.** `/goal` reads the goal, `/goal <objective>` creates
  one, `/goal edit <objective>` edits it, `/goal cap <n>` raises the round cap, and
  `/goal pause|resume|clear` sends the human control. It never reaches the model as a user turn.
  It is one entry in the composer's command catalog: the slash menu and the command result line are
  generic, so another command registers its words and handler rather than branching the composer.
  The server executes and keeps each command event in the conversation, with its input and result,
  so a reload draws the command where it happened and it never becomes a model turn.
- **A terminal round says how to close.** When `complete` or `blocked` is recorded, the tool result
  asks the model to write the closing message and stop calling tools, as the reference's terminal
  rounds do.
- **The prompt states the reference rules.** A goal section tells the model when a goal is for, that
  complete means achieved, that blocked needs the same condition across the round threshold, and that
  pause and resume are the human's. A loaded Skill's documents may inform how the objective is
  scoped; they are context, not a contract the platform enforces.
- **The round cap ends the way the reference does.** The driver writes `blocked` with the round-limit
  reason and disarms; resume refuses at the cap until a human edits it.
- **A conversation materializes its Agent Session identity.** A Web conversation is its Session from
  creation, so the Session row is written with the conversation (and a goal command writes it for an
  older empty conversation); the first Run's claim replaces the placeholder lease. Session-owned
  state can then exist before the first Run, which is where a goal lives.
- **Resume re-arms an active goal a restart left disarmed.** Activation moves; the durable row's
  revision does not, because arming is process state and the objective did not change.
- **A Web round's submission id is a UUID derived from the goal and the round**, because a
  conversation stores submission ids as UUIDs; a retry of the same round therefore replays instead of
  opening a second turn.
- **No new constraints.** No delivery contract drives the objective, no condition identity is
  tracked, no no-progress gate exists, and no turn, time or token limit is invented beyond the
  user's round cap. max-tokens disarm stays out (the user's decision); a crash between a terminal
  commit and the detached round loses that round, and a single coordinator process is the deployment
  the claim is made for.

**Superseded in part by [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md).**
The arming/disarming language in this amendment (the "Resume re-arms" and "disarms" bullets, and the
max-tokens-disarm paragraph above) is historical. Continuation is decided by the durable phase, a
token-limit round records `blocked` with a retry reason, and pause is immediate: it stops the goal
now and cancels the goal round already running (a human turn already running is not cancelled).

## Amendment 2026-10-24 — the provider seam is retired; the checkpoint runs inside the Turn

Decision 4 no longer describes the code, and the Status above now says so. Commits `85d70cff` and
`fb456692` (2026-10-22) deleted `application/answer/checkpoints.py` (`CheckpointProvider`) and
`application/answer/skill_evidence.py` (`SkillEvidenceProvider`), removed the provider registry and
the findings injection from the goal-round prompt, and added the **pre-delivery checkpoint** to the
Agent Session Runtime. The round driver now carries the objective and the round number only; the
evidence is injected inside the Run that is still writing the answer.

The retirement is deliberate, not a missing restoration:

- **The settled-Run boundary was too late.** A provider read a settled Run's records to advise the
  *next* Round, after the reader's record was already delivered. The in-run check reads the same
  delivery ledger (`AnswerOrchestrator.delivery_findings()`) while the model can still correct the
  answer; its findings become a `DeliveryWarningEntry` the model sees before completing. One
  mechanism serves a goal round and an ordinary one-Round answer.
- **There was exactly one consumer.** The Skill-derived evidence was the only implementation. ADR
  0012 rejects a reusable seam until a second real consumer proves it; a registry for one
  implementation is the speculative abstraction that decision forbids. The capability survives as
  the typed `delivery_check` callback on `AgentSessionRuntime`, the one point that can inject a
  warning before completion.
- **The invariants survive.** The goal domain, the driver and the prompt still name no Skill. The
  check still derives its findings from the Run's own records, remains advisory ("advice, not a
  gate"), is bounded to two warning attempts per Operation, and never blocks completion.

`CONTEXT.md` now records the same thing: it defines **pre-delivery checkpoint** and no longer
defines a checkpoint provider; **goal round** carries the objective and its number only. A
provider-style extension point is out of v1; the trigger to propose one is a second real consumer
(page provenance, an operator monitor) with at least two implementations, recorded in a new ADR.

## Amendment 2026-10-24 — a human resume continues an idle goal

A resume used to re-arm the goal and wait for a settlement to follow it. That left a dead end:
after a restart, a goal round that was in flight settles disarmed and starts no next round, and a
resume typed after that settlement has no Run left to wait for. The reference drives on the goal
change itself, and so does this driver now:

- **Resume is the driver's second trigger.** A successful human `resume` asks the driver to start
  the next round from the conversation's newest turn when no Run is advancing that Session. The
  driver admits the round with the same compare-and-set, request projection and idempotency key as
  a settlement-driven round, so a resumed round is an ordinary goal round.
- **A Run in flight keeps its priority.** If anything is still advancing the Session, the resume
  only arms and the settlement path starts the round when that Run finishes; the two triggers
  cannot both start the same round, because the second admission is refused as stale.
- **The Web re-reads its turns.** The goal bar tells the chat a resume happened, and the chat
  re-reads the conversation, so a round started by the resume appears where the turns are instead
  of waiting for a reload.

## Amendment 2026-10-25 — a restart takes up the active goals

Decision 3's "a new process starts disarmed, and a human must resume" is superseded. The product
owner's decision: an active goal is live work, not a proposal, and a deployment restart must not
leave it behind a Resume button that says nothing about why.

On startup, once the coordinator and the Web conversations exist, the process reads every goal the
database records as `active` and re-arms it. The round driver then continues each goal whose Session
is idle from the Run that last carried it: a terminal Run's settlement-equivalent starts the next
round, a capped goal is blocked the way settlement would block it, and a goal whose Session already
has a Run in flight is only armed — that Run's settlement is the trigger, so no second round races
the one already working. A failure for one goal never costs the others, and a store that cannot
answer leaves the process otherwise healthy.

The safety tradeoff is taken knowingly: a restart after a crash now also continues the work, so a
crash loop can keep spending rounds. This is acceptable because the round cap still bounds the loop,
the goal is visible in the reader's conversation, and the owner prefers the continuation to a
silently inactive goal. Disarming still happens on pause, complete, block, and on a failed or
cancelled goal round. Arming remains process-local (the one-process caveat under Not in v1
stands); this amendment only adds the startup recovery that fills the set.

**Superseded by [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md).**
The startup scan this amendment added is kept and generalized; the activation set it filled is
removed, so there is no arming and no human re-arm. Pause is now immediate: it stops the goal now
and cancels the goal round already running, while a human turn already running is not cancelled;
failed/cancelled rounds record durable `blocked`/`paused` phases.

## Not in v1

- **One process.** Arming was process-local by the reference shape. A multi-process deployment could
  claim a goal round in a process that had not armed the goal, so continuation stopped there; v1 was
  documented for one API coordinator process. A shared arming epoch was the additive v2 step.
  *(Superseded by [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md):
  continuation follows the durable phase and a CAS revision from any process, and the recovery
  process takes active goals up.)*
- **No crash sweep.** A crash between a terminal commit and the detached driver task loses that one
  round. *(Superseded by [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md):
  the recovery process scans at startup and periodically, and the Web pending-continuation row is
  the durable record recovery retries from.)*
- **No max-tokens disarm.** The provider stop reason is not in the terminal result, so a `length`
  turn end does not yet disarm; the round cap still bounds the loop.
  *(Superseded by [ADR 0077](0077-goal-mode-continues-automatically-and-pause-stops-the-running-round.md):
  the provider stop reason reaches settlement, and a token-limit round records `blocked` with a
  retry reason instead of disarming; the round cap still bounds the loop.)*
- **No machine completion predicates.**
- **A human turn is not an admission-time cancel.** A human message accepted after a round was
  admitted does not cancel that round; the session lane settles at most one of them, and the
  queued-turn delivery order gives the human turn precedence only at the next-turn boundary.
- **Rounds are admitted under default policies, not the original caller's** (`caller=None`), so a
  per-caller effort ceiling or `active_answer_runs` limit is not replayed. A deployment that needs
  those on rounds keeps goal mode off until v2 carries the caller.

## Consequences

- The harness gains a general insertion, callback and monitoring point for long Runs, while a Skill
  opts in by declaration and keeps working in normal one-Round answers.
- The goal stack is testable without any Skill installed, and a Skill is testable without a goal:
  the engine goal domain may not import the Skill modules (import-linter contract), and the
  Skill-derived delivery evidence imports only the generic declarations and the Run's own records.
- The evidence in a goal round is the same derivation as the reader's record, so the Agent and the
  reader cannot be shown different facts; the reader's record remains evidence, never a gate.
- Execution of Skill-provided checker code is out of v1; when it comes it is one more in-run check or
  a declared tool, not a harness branch.
