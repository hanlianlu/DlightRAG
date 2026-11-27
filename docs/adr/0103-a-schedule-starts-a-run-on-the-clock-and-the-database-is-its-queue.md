# ADR 0103 — A Schedule starts a Run on the clock, and the database is its queue

**Summary / 摘要**

Every Run today starts because a human sent a turn or because a Run that just settled continues
(a goal round, a queued turn, a Child's report). Nothing starts a Run because a time arrived, so
recurring work — a weekly review, a watch on a filing, a digest of a corpus — is driven from
outside by a script or by hand. This ADR adds the **Schedule**: an owner's durable record that
starts an Answer Run at times it names, as that owner, through the same admission and conversation
path a human turn takes. The database is the queue: a process claims a due Schedule with
`SKIP LOCKED`, writes one **Fire** per slot under a unique key, and starts the Run with an
idempotency key built from that slot, so each slot starts at most one Run in any number of
processes, with no leader and no cron. Missed slots collapse into at most one late fire; a fire
whose previous Run is still live is skipped, never queued; authority and policy are read at each
fire, not frozen at creation; a spend ceiling, a failure streak or lost access pauses the Schedule
with a stated reason. A fire can start a goal, and the goal's own rounds continue it (ADR 0077).
The owner manages Schedules in the Schedules dialog the sidebar's Scheduled row opens, and the Agent
manages them from a chat on a turn a person started, under the same rules and never with a goal or a
spend ceiling of its own.

今天每个 Run 的起点要么是人发来的一轮，要么是刚结束的 Run 接着做（goal round、排队消息、Child 回报）。
没有任何东西因为"到时间了"而启动 Run，所以周期性的工作——每周复盘、盯一份公告、汇总一个 corpus——
只能靠外部脚本或手动。本 ADR 增加 **Schedule**：owner 的持久记录，在它指定的时刻、以该 owner 的身份、
走与人类 turn 相同的准入和对话路径启动一个 Answer Run。数据库就是队列：进程用 `SKIP LOCKED` 认领到期的
Schedule，按 slot 写入唯一的 **Fire**，并用由该 slot 构成的 idempotency key 启动 Run，因此无论多少进程，
每个 slot 至多启动一个 Run，不需要 leader，也不需要 cron。错过的 slot 合并为至多一次补跑；上一次的 Run
还在跑时本次直接跳过，不排队；权限和 policy 在每次触发时读取，不在创建时冻结；花费上限、连续失败或失去
访问权会暂停 Schedule 并写明原因。一次触发可以启动一个 goal，由 goal 自己的轮次继续（ADR 0077）。
owner 在 Scheduled 区管理 Schedule；Agent 也能在人发起的 turn 里从对话中管理它们，规则与 owner 相同，
但不能设置 goal，也没有自己的花费上限。

## Status

Accepted and implemented 2026-11-27: the domain, store, look, settlement, REST and Web routes, and the
Scheduled list. Amended 2026-11-27: the Agent manages Schedules from a chat through four tools (Decision 12). This is a harness decision: it names no vertical domain and ships to DlightRAG with the
rest of the runtime.

## Context

The runtime already holds every part a clock-started Run needs except the clock:

- **Durable Runs and admission** (ADR 0007): a Run is admitted under per-owner and per-process
  limits, survives a disconnect, and accepts an idempotency key, so a retried request replays the
  Run it already created instead of creating another.
- **Turns that nobody typed.** A goal round (ADR 0071, ADR 0077) and a queued turn become the next
  turn of a Web conversation without a human at the keyboard. The goal recovery scan already
  proves the pattern this ADR needs — durable state, a periodic scan in the serving process, and a
  pending row that fences concurrent attempts.
- **The database as a queue.** Run claims, event-log trimming and the Session Memory Pass's look
  for lost passes all claim work with `SKIP LOCKED` batches that need no leader. A due Schedule is
  the same shape: a row whose `next_fire_at` has passed.
- **Authority at the door** (ADR 0042): Runtime Policies are read when a Run is admitted and pinned
  into it. A Run started by a clock must take the owner's authority as it stands at that moment, not
  as it stood when the Schedule was written.

What is missing is a durable record of *when*, a way to turn a due time into exactly one Run, and
the rules for the cases a human turn never meets: the process was down at the time, the previous
fire is still running, the owner lost access in between, or the work costs more than the owner
meant to spend while nobody is watching.

Recurring work is also where an unattended agent can spend most. One research Run with goal rounds
can consume millions of tokens; a Schedule multiplies that by its cadence with no human in the loop.
The spend rules are therefore part of the decision, not an afterthought.

## Decision

1. **A Schedule is the owner's durable record of recurring or one-time work.** It holds:
   - the **turn**: the prompt text that becomes the user message of each fired Run, the mode
     (`research` or `fast`), the workspace selection as the owner made it, and optionally a
     **goal round cap**, which makes it a goal Schedule whose prompt is the goal's objective;
   - the **target**: `new_conversation` (each fire opens a new Web conversation, titled from the
     Schedule and the slot — the default) or `same_conversation` (each fire is the next turn of one
     conversation, so the Session's notes, memory and Agentspace carry from fire to fire);
   - the **cadence**: either a recurrence — a five-field cron expression evaluated in an IANA time
     zone the owner chooses — or a single `at` instant;
   - the **state**: `active | paused | completed` (a one-time Schedule whose Fire started a Run; a
     one-time Fire that started nothing pauses it instead, so the owner can move `at` and resume),
     the pause reason, `next_fire_at`, and the spend ceiling.

   A Schedule belongs to one owner, as a conversation does. It is not shared and not a workspace
   object.
2. **Wall-clock time in the owner's zone.** The recurrence is evaluated in the Schedule's zone, so
   "Monday 08:00 Europe/Stockholm" stays 08:00 across daylight-saving changes. A local time that
   does not exist (the spring gap) fires at the first valid instant after it; a local time that
   occurs twice (the autumn overlap) fires once, at its first occurrence. A recurrence whose slots
   are closer than `schedules.min_interval_minutes` (default 60) is refused at creation and at
   edit — the minimum is a spend guard, not a technical limit.
3. **The database is the queue, and a slot fires at most once.** Each process that runs Schedules
   (`deployment.schedules`, default `true`) looks about every 30 seconds for active Schedules with
   `next_fire_at <= now`, claiming them `FOR UPDATE SKIP LOCKED`. In the same transaction it inserts
   the slot's **Fire** — unique on `(schedule_id, slot)`, state `pending` — and advances
   `next_fire_at` to the next slot after now. It then starts the Run with the idempotency key
   `schedule:<schedule_id>:<slot>` and records the Run id on the Fire. A crash after the Fire is
   written and before the Run exists leaves a `pending` Fire; the next look retries it with the same
   key, and it first looks for the Run that key already created, so a retry records that Run rather
   than judging the rules again; a `pending` Fire
   that could not be started while its slot was still within its grace becomes `missed`, so a
   retry never starts stale work. Any number of
   processes may run the look; none of them is a leader. Time is the database's: every "now" the
   look and the Schedule service decide by is read from the database, not from the process, so
   replicas whose clocks differ still agree on what is due.

   Only a process that hosts Web conversations runs it — the MCP-only service sets
   `DLIGHTRAG_DEPLOYMENT__SCHEDULES=false`, as it does for goal recovery — because a fire becomes a
   conversation turn.
4. **Missed slots collapse into at most one late fire.** When the look finds a Schedule whose
   `next_fire_at` is in the past by more than one slot (the deployment was down, the Schedule was
   paused and resumed), it fires the **latest** missed slot once, if that slot is within the grace
   window — half the Schedule's interval, at most six hours; a one-time Schedule's grace is one
   hour. The other missed slots start nothing; one `missed` Fire per gap records how many slots it
   covered and the first and last of them. Resuming a paused Schedule never replays the slots it
   slept through beyond that one.
5. **A fire whose previous work is still live is skipped, not queued.** If the Schedule's previous
   Fire has a Run that has not settled — including the goal rounds that Run started — or the target
   Session already has a Run in flight, the new slot is recorded as `skipped_busy` and starts
   nothing. Queuing would let a slow Schedule build a backlog that runs back to back after an
   outage; the next slot is the next chance.
6. **A fire runs as the owner, with the owner's authority at that moment.** The fire builds the
   same Answer request a human turn in the target conversation would, and admission reads the
   owner's Runtime Policies, workspace access and Connections at the fire (ADR 0042). The owner's
   identity is the one place the fire cannot read fresh: with no request there is no token, so the
   Schedule keeps the principal — user id, auth mode and claims — the owner last presented when
   creating, editing or resuming it, and the access gate evaluates the current access rules and
   policies against that principal at every fire. A principal that no longer maps to the
   Schedule's owner (the deployment's auth mode or issuer changed) is refused. Claims the identity
   provider has since revoked would otherwise outlive the revocation, so a token-issued (`jwt`)
   principal kept longer than `schedules.principal_max_age_days` (default 30) is refused too, and
   the Schedule pauses until the owner resumes it, which renews the principal. A workspace
   the owner can no longer read is dropped from the selection; if none remains, or the owner is
   disabled, or a Connection the turn names needs re-authorization, the Fire records `refused` with
   the reason and the Schedule pauses. A Schedule is not a service identity and never holds
   authority the owner lacks. Its Run and every Child it starts inherit exactly what a human turn's
   would (ADR 0025).
7. **Admission limits apply unchanged.** A fire refused by the owner's active-Run limit or the
   process admission limit records `skipped_limit` and starts nothing; the Schedule stays active and
   the next slot tries again.
8. **Spend is bounded per Schedule, read at admission.** A Schedule may carry
   `max_tokens_per_day` (a rolling 24 hours, counted from the usage its Runs and their Children
   recorded, ADR 0036); `schedules.default_max_tokens_per_day` sets the value a new Schedule starts
   with. A fire whose Schedule has already reached its ceiling records `refused` with the spend
   reason and the Schedule pauses. Every count of a goal Schedule includes the later rounds of the
   goals its fires started. The ceiling is checked before a Run starts; it does not cut a
   running Run short — a goal fire is bounded inside the Run by its round cap. An owner may hold at
   most `schedules.max_active_per_owner` (default 20) active Schedules.
9. **Failure pauses; nothing retries on its own.** A Fire whose Run fails records `failed` with the
   Run's error. A refusal only the owner can clear — lost identity or access, the spend ceiling, a
   turn admission rejects as invalid — pauses the Schedule at once. Otherwise three consecutive
   Fires that end `failed` or `refused` (`schedules.failure_streak`) pause the Schedule with the
   last reason; a Fire that could not be started before its grace ran out counts too, and carries
   the last error as its reason. The streak counts only Fires after the last resume. A Fire still
   `pending` when its Schedule is paused is cancelled, not started. A failed fire is not
   retried before its next slot, and a paused Schedule fires again only when the owner resumes it.
10. **A goal fire is a goal start.** When the Schedule carries a goal round cap, the fire creates the
    goal — its objective is the Schedule's prompt — in the target Session, and the fire's Run is the
    goal's round 1: a real goal round, admitted through the goal's own revision and pending-round
    fence, so ADR 0077's driver settles and continues it exactly as it does any goal (a failed round
    blocks the goal, a cancelled one pauses it). The Fire tracks round 1 only; the later rounds are
    the goal's, shown in its conversation. With
    `same_conversation`, a fire finds the previous goal either finished — and replaces it, as ADR
    0077 Decision 9 allows — or still unfinished, and never overwrites it. An `active` goal is live
    work, so the fire records `skipped_busy` (Decision 5). A `paused` or `blocked` goal is waiting
    for the owner, so the fire records `refused` with that reason; it counts toward the failure
    streak (Decision 9), and the Schedule pauses where the owner can see why instead of skipping
    in silence forever.
11. **The turn says it was scheduled.** The fired Run's user message is the Schedule's prompt, the
    owner's own words, after one harness line that names the Schedule and the slot's local time, so
    "this week" in the prompt has a date. The Fire records the Run it started, so the conversation,
    the Run panel and the trace can show which Schedule and which slot started a Run; a goal fire's
    first round keeps the goal-round continuation kind and is linked through its Fire the same way.
    The Session Memory Pass reads the turn like any other owner turn.
12. **Owners and the Agent manage Schedules.** The Web sidebar and the REST API create, edit, pause,
    resume, run now and delete a Schedule, and list its Fires. "Run now" is a manual Fire with its own
    slot — the request time — under every rule above; it is refused while the Schedule has a Fire
    pending or started, and it is allowed on a paused Schedule, because the owner is present, without
    resuming it. Editing the cadence or zone recomputes `next_fire_at` from now. Deleting a Schedule
    stops future fires and deletes its Fire history; the conversations its Runs wrote stay, as any
    conversation does.

    The Agent has four tools over the same service — `list_schedules`, `create_schedule`,
    `update_schedule` and `delete_schedule`, where pausing and resuming are `update_schedule(paused)` —
    and every rule above applies to them unchanged: the minimum interval, the cap on active Schedules,
    cadence validation, and no edit of a completed Schedule. A refusal comes back as a tool error the
    model can act on. The limits are these:

    - **A turn a person started.** Only the continuation kinds a person starts — a new turn, the next
      turn of a conversation and a fork — are offered the tools, an allowlist: a Schedule's fire
      (`scheduled`), a goal round (`goal_round`) and any kind added later have none declared, so a
      Schedule never creates a Schedule and an unattended goal loop never does either, whoever the
      fire acts as. A REST or MCP answer submission by an authenticated client is a turn its own
      client started and is offered them. A Child Session never holds them (ADR 0025). Acceptance
      pins whether a Run is offered them in its input, as it pins its plan, so whichever process
      claims the Run composes the tools it was accepted with: every process holds the Agent's way to
      the Schedule service, and `deployment.schedules` only says which processes look for due
      Schedules.
    - **No goal and no ceiling.** The Agent's tools have no round cap and no `max_tokens_per_day`. A
      Schedule the Agent creates starts with `schedules.default_max_tokens_per_day`, and an edit by the
      Agent never touches the ceiling or the goal; only the owner changes either, or makes a goal
      Schedule, in the Schedules dialog. A Schedule that starts a goal is the owner's to change and
      resume: the Agent may rename, pause or delete it, nothing more.
    - **The owner's latest human turn.** The Schedule acts under the identity of the owner's latest
      human turn — the caller acceptance keeps, one record per owner, replaced by each such turn and
      never written into a Run's input — which is what an owner's own request would store (Decision 6).
      A record older than a day is no one's latest turn, so with none under a day old the tools answer
      `schedule_unavailable`, and a turn that could not keep it still runs. Nothing the model passes
      names an owner, and a workspace named for the Schedule is one of the workspaces the Run already
      reads.
    - **Bounded against text that asks for work.** What a Run read — a page, a file, a tool result —
      can ask for a recurring instruction. A turn makes at most one Schedule, a second waiting for the
      user to confirm; the minimum interval, the cap on active Schedules and the Schedule's ceiling
      bind it; and every Schedule records whether its owner or the Agent last made, edited or resumed
      it (`changed_by`; pausing, deleting and running now leave it).
    - **Said back and visible.** The Agent creates one only when the user asks for recurring or deferred
      work, takes the user's time zone from the user rather than assuming one, and tells the user the
      name, the cadence in words with its zone, and the next run. When an edit is saved and the pause
      or resume that followed is refused, the result says which was which. Every Schedule the Agent
      made is in the owner's Schedules dialog, whose list (and the Scheduled row's count) is read
      again when a Run settles, where the owner can pause or delete it.

13. **Fire history is bounded.** A Schedule keeps its latest `schedules.fires_kept` (default 200)
    Fires; older ones are trimmed in the claim transaction that writes a new one. A Fire's state is one of `pending`, `started`,
    `succeeded`, `failed`, `cancelled`, `refused`, `skipped_busy`, `skipped_limit` or `missed`, with
    the reason for every state that started nothing.

## Consequences

- Recurring work moves inside the product: it runs under the owner's authority, appears as
  ordinary conversations, and stops on its own when it fails, loses access or overspends, instead
  of depending on an outside script that holds a token and knows none of these rules.
- Each slot starts at most one Run with no leader process: the unique Fire and the slot's
  idempotency key fence any number of processes and any crash between writing the Fire and
  creating the Run.
- A deployment that was down does not wake up to a burst of runs: at most one late fire per
  Schedule, and none for slots beyond the grace window.
- `same_conversation` gives a Schedule a memory of its own past fires through the Session it
  shares — notes, Agentspace and recalled memory — at the cost of a Session that grows with every
  fire and is compacted as any long Session is. `new_conversation` keeps each fire independent.
- The spend ceiling is checked between Runs, not inside them, so one fire can overshoot it by up to
  that fire's own cost. The minimum interval and the round cap keep that overshoot bounded.
- New durable state: the Schedule and Fire tables and the one-row-per-owner identity record the
  Agent's Schedules act under (`dlightrag_owner_principals`, kept a day), owned by the Run store's
  Adapter family; a new deployment switch, `deployment.schedules`; and the `schedules.*` settings named above.
- Known limits: a slot fires up to about 30 seconds late, plus admission time; there is no
  notification beyond the conversation appearing unread — email, push and webhooks are later
  decisions; and a Schedule cannot depend on another Schedule or on an external event, only on the
  clock.
- A user can set up recurring work by asking for it in a chat, and it costs what the owner's own would:
  the same minimum interval, cap and default ceiling bind it, and it is listed where the owner can
  pause or delete it. What the Agent cannot do is spend unattended beyond that default or start a
  goal; those stay the owner's decision.
