# ADR 0077 — Goal mode continues automatically, and pause stops the running round

**Summary / 摘要**

ADR 0060 made a goal's permission to continue process-local: an `armed` flag in memory, disarmed by
every session-start edge and re-armed by a human resume. That mirrored the reference at the cost of
a product that told the reader and the model a story the work did not need: a restart parked live
work behind a Resume button, a second process could disagree about the same goal, and the
model-visible text described a permission it could neither see nor change. This ADR removes that
permission. A goal's durable phase is the whole continuation authority: `active` means the platform
continues it, `paused`/`blocked`/`complete` stop. Continuation is driven from durable state and a
compare-and-set revision: a settlement can continue it from any process, and after a restart the
process that serves the Web/API surface scans the durable record and starts the next round
(`deployment.goal_recovery`, default true, off in the MCP-only process, so one process scans).
**Pause is immediate**: it stops the goal now, cancels the goal round already running, and
starts no next round, while a human turn already running is not cancelled. Resume is a human
decision. A round stays one Answer Run (ADR 0071) and continues the same Session; it is a budget
boundary, not a context reset.

ADR 0060 把目标继续的许可放在进程内存里：`armed` 标记，每次会话开始即解除，由人 resume 后重新
授予。这照搬了参考实现的代价是：重启把仍在进行的工作停在一个 Resume 按钮后面，第二个进程可能对同一
目标有不同判断，而模型看到的提示在描述一个它既看不见也改不了的许可。本 ADR 移除这个许可：目标的持久
phase 就是全部继续权——`active` 表示平台会继续，`paused`/`blocked`/`complete` 表示停止。继续由持久
状态与 CAS revision 驱动：settlement 在任意进程都能继续；重启后由承载 Web/API 的进程扫描持久记录并开始
下一轮（`deployment.goal_recovery`，默认 true，纯 MCP 进程关闭，因此只有一个进程扫描）。**Pause 立即
生效**：立即停止目标、取消正在运行的 goal round，并且不再开始下一轮；已在运行的人类 turn 不会被取消。
resume 由人决定。round 仍是一个 Answer Run（ADR 0071），延续同一 Session——它是预算边界，不是上下文重置。

## Status

Accepted 2026-10-25. The domain, service, round driver, Web/API surface, model-visible prompt and
tool text, and docs land together in this rewrite. This ADR supersedes ADR 0060's process-local
arming and the "disarmed after restart" semantics everywhere, including the 2026-10-25 amendment's
activation vocabulary (its behavior — a restart continues active goals — is kept and generalized).
Amended 2026-10-25 (owner decision): pause is immediate — it cancels the goal round already running
and starts no next round, while a human turn already running is not cancelled. The rest of this ADR
is unchanged.

## Context

ADR 0060 chose the reference shape for goal mode: one persisted goal per Session, a compare-and-set
`revision`, a driver that starts the next round when the agent goes idle, the model's own completion
call, and a process-local `armed` permission disarmed by every session-start edge and re-armed by a
human. The permission was there to answer a real worry — a resumed or forked conversation should not
silently spend rounds, and two processes should not race one objective.

The product learned what the permission cost:

- **A restart parked live work.** An active goal came back "Inactive", with a Resume button and no
  explanation of why the platform had stopped. The 2026-10-25 amendment added a startup scan to
  soften this, but the activation set, its disarm rules and the reader's "Inactive goal" state
  stayed.
- **The model was told a false story.** The system prompt and all three tool descriptions said a
  goal "starts disarmed after a restart and a human re-arms it", a permission the model could not
  observe or change, while the actual continuation was decided by settlement outside its turn.
- **Two processes could disagree.** Arming was process-local by design, so a round admitted by a
  process that had not armed the goal simply stopped; the documented deployment was one API process.
- **A round claimed a fresh context.** The round prompt told the model every round began a new
  context and to reload its Skills, contradicting the same-Session continuation the round actually
  is; the model re-read documents it already had and re-derived state it had already committed.

**Pause was a rounding-off control, and the owner changed that.** The first version let a running
goal round finish and only stopped the next one, so a human asking to stop a long round kept paying
for it until it ended. A pause now means stop now: the running goal round is cancelled, no next
round starts, and a human turn that is not a goal round is left alone. The trade is stated under
Known limits: the cancelled round's uncommitted work is not carried into the resumed round.

The durable goal row, its phase graph, the CAS revision and the settlement-driven driver were the
right shape. The permission was the part that did not pay for itself, and pause was the part that
did not stop.

## Decision

1. **The durable phase is the only continuation permission.** One goal per Agent Session remains,
   with phases `active | paused | blocked | complete`. `active` means the platform may continue the
   goal. There is no `armed`/`disarmed` state, no activation cache and no process-local permission;
   the phase and the round count are the complete continuation state.
2. **Continuation is automatic, from durable state plus CAS.** When a successful Run settles and the
   Session is idle, an `active` goal with rounds remaining is continued: the next round is admitted
   under a CAS on the goal's revision and the pending-round row, so a settlement may run in any
   process and a retry replays instead of duplicating. (A failed or cancelled human Run changes
   nothing; the recovery scan retries the active goal.) A goal created by `/goal` outside a turn
   starts round 1 at once, including in a conversation with no prior Run.
3. **Recovery is a deployment scan, not a permission.** The process that serves the Web/API
   surface scans durable `active` goals when it starts and about once a minute afterwards; for each
   idle goal it delivers a continuation a previous process left pending, starts the next round, or
   records the cap. `deployment.goal_recovery` (default `true`) keeps exactly one process scanning —
   the MCP-only service sets it false — so two processes do not scan the same goals.
   The scan also takes up an active goal with no answer Run to continue from — a `/goal` command
   whose first round never started — through the conversation that owns its Session; a Session with
   no conversation is not a Web goal and waits for the next human trigger. CAS and the pending-round
   row fence concurrent attempts; no arming is involved.
4. **Pause is immediate.** A human pause stops the goal now: it records `paused`, cancels the goal
   round already running — that Run is cancelled rather than allowed to finish — and starts no next
   round. A human turn already running is not a goal round and is not cancelled by a goal pause.
   Cancellation is best-effort: the phase is already `paused`, so a round the runtime cannot
   interrupt finishes without starting another. Resume re-enters `active` and, when the Session is
   idle, starts the next round immediately. **Resume is the human's decision**: resuming a paused
   goal is a human-control action (the goal bar or `/goal resume`), and the model's tool path is
   refused on a paused goal. Resuming a `blocked` or `active` goal from a direct human turn remains
   allowed.
5. **Failures are durable phases, never silent retries.** A failed goal round records `blocked` with
   the concrete reason and a resume instruction; a goal round cancelled by a human's pause settles
   into the already-paused goal without changing it, while a goal round cancelled for any other
   reason records `paused` when the goal is still active; a round that ends on the model token limit
   records `blocked` when the stop reason is known. A harness stop applies only to the goal revision
   the round was admitted for — a newer human decision (pause, resume, edit, clear) wins. A human
   turn's failure or cancellation changes nothing about the goal; the idle Session continues it, and
   an active goal left behind by a failed human Run is taken up by the next recovery scan in a
   deployment that runs recovery.
6. **The cap is a round budget, default 16.** `rounds_started` counts admitted rounds; at the cap
   the driver records `blocked` with the round-limit reason and a human raises the cap and resumes.
   The model is told to name `max_goal_rounds` when the objective needs more than the default.
7. **A round continues the Session.** ADR 0071's Run boundary stands: one round is one Answer Run,
   and the boundary is a budget, not a context reset. The round continues the Session's committed
   Agentspace, its Session Notes, its published products and its conversation history. The
   model-visible text says so, asks for a short `notes/goal.md` handoff when work remains, and keeps
   a previous round's `load_skill` as context while making clear that only a load in the delivery
   round counts for delivery attestation.
8. **The model owns completion; humans own pause, resume and edit.** `complete` and `blocked` are
   the model's calls, with an autonomous block still gated on the started-round threshold; `edit`,
   `pause` and `resume` act on a direct human turn only. **2026-10-25 amendment:** a Run that loaded
   a Skill declaring a sealed artifact may not complete while that delivery is unattested — a
   skipped required read, a verifier that did not pass, or a refused seal makes `update_goal
   complete` a refusal. The goal stays active for the next round; `blocked` with the concrete
   condition remains the honest early exit. This is a delivery-evidence gate, not a machine
   evaluator of the analysis.
9. **A completed goal may be replaced.** `create` on a completed goal replaces the row atomically
   with a new goal; an unfinished goal must be paused, blocked or cleared before another takes its
   place.
10. **The text states the shape it runs in.** Model-visible guidance and the three tool descriptions
    state automatic continuation, with process-restart recovery named for the serving process rather
    than claimed as an absolute, and no re-arm; same-Session rounds with the handoff note; `get_goal`
    before `update_goal`; complete-only-when-achieved; the blocked threshold; an immediate pause
    that cancels the running goal round; and cap sizing. The round prompt no longer claims a new
    context or a
    Skill reload. The terminal round's result is a full closing brief: outcome, evidence and
    verification, concrete results, and what remains or what the reader must do.

## Consequences

- In the shipped deployment a restart continues work: the serving process's recovery scan takes the
  durable goal up again, so the reader sees "Ongoing goal", not an inactive goal behind a Resume
  button, and the model is no longer told about a permission it cannot observe.
- The reader's surface states the same rules: the bar's Pause control says pause stops the goal now
  and cancels the round in progress, a complete or blocked goal shows one neutral terminal
  line (no further round will start; resume to continue, or raise the cap when the cap stopped it),
  and a refused goal command shows the server's specific reason instead of a generic failure. Goal
  rounds stay silent per round; reaching a terminal phase sends one notice.
- Multi-process continuation is grounded in durable state and CAS; the only deployment switch is
  whether a process runs the recovery scan (startup and periodic), which is not a permission to
  continue.
- Pause stops spending at once: the in-flight goal round is cancelled rather than allowed to
  finish, so a human can stop a long round immediately; a human turn already running is untouched.
  The cost — that round's uncommitted work — is recorded under Known limits.
- The prompt, tool text and round prompt agree with the state machine: no arming vocabulary,
  same-Session continuity, `get_goal` before mutation, and closing briefs instead of status lines.
- A deferred prompt read of "the same blocking condition" still describes the model's judgment; the
  runtime enforces the started-round count, as ADR 0060 recorded.
- Known limits: no in-loop continuation (ADR 0071's Phase 4b stays retired), no machine completion
  evaluator beyond the sealed-delivery evidence of the completing Run, no resource budget beyond the
  round cap, and no abnormal-failure auto-retry — a failed or token-limited round records a durable
  stop and waits for a human resume. The completion gate reads this Run's attempt ledger; a final
  round that loads no sealed Skill is not gated, and the durable reader record still marks an
  unattested delivery. Pause is immediate,
  so it cancels the in-flight goal round: that round's uncommitted Agentspace changes are not carried
  into the resumed round (only a succeeded Run commits), and a continuation still waiting to become
  the next turn is dropped with the pause. The crash window is
  narrower but not closed: on the Web path a crash after the continuation is queued is replayed from
  the pending row, and a lost settlement observer is retried by the next periodic recovery scan
  (about once a minute, and only in the process that runs recovery). A non-Web goal can still
  lose one admitted round when the process crashes between admission and Run creation. The Web
  `/goal` create path has the same class of window: a crash between `admit_round` and
  `accept_goal_round_now` burns that admitted first round — recovery admits the next round, and the
  burned one counts against the cap. A deployment with `goal_recovery` off (the MCP-only service)
  does not run the scan, so a goal whose settlement was lost there waits for the recovery process or
  a human trigger.
- Supersedes ADR 0060 Decision 3's arming language and its 2026-10-25 amendment (behavior kept:
  a restart continues active goals; the activation vocabulary is removed), and ADR 0071's
  arming/disarming wording in its Consequences. The goal domain, durable row, tools and
  settlement-driven driver remain as ADR 0060 decided.
