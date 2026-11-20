# The Web reads Run activity from the Runs it keeps

The page shows what is running and what finished by reading the durable Runs, not a second
notification record.

## Status

Accepted and implemented.

## Context

A Web Run is already durable: the page that accepted it can close, and the Run finishes in a worker
under its own lease ([ADR 0007](0007-unified-durable-runs-and-execution-lanes.md),
[`docs/run-runtime.md`](../run-runtime.md)). What the page lacked was the reader's side of that
fact. It followed one Run per tab, so it knew the open conversation's state and no other
conversation's; the conversation list carried no running state; and the only completion
notification fired for the one followed chat Run while the page was away
(`frontend/ui/notifications.ts`). A Run that finished in another conversation, or after the reader
left the followed one, produced no visible trace until they opened that conversation again.

Two shapes were available. The first is a server notification record: a table of per-run delivery
and read state, written at settlement. It duplicates lifecycle facts the Run already owns, and it
would have to answer the same questions the Run's row answers — status, start, finish, which
conversation it belongs to. The second is a bounded read of the existing records: the Runs table
for lifecycle, the conversation turn map for the conversation link, and the conversation row for
the title. The second keeps one truth and makes the page's poll a projection.

## Decision

- **One bounded read.** `GET /web/api/run-activity?limit=` answers the authenticated owner's recent
  conversation-linked Answer Runs, newest first: `{"items": [{"run_id", "conversation_id",
  "conversation_title", "status", "phase", "created_at", "started_at", "finished_at",
  "goal_round"}]}`. `limit` is 1–100, 30 unless given; the page is a hard-bounded list rather than a
  cursor traversal, because a reader's recent activity is small and the page refreshes it.
- **No second state.** The projection creates nothing and stores nothing. It contains only
  `run_kind = 'answer'` Runs that a `web_conversation_turns` row links to a conversation: a Run
  accepted through REST or MCP has no conversation, and the Web has no surface for it. The read
  touches no lease, admission, event, or Run state.
- **Running and unread are the page's own.** A conversation is running while the projection holds a
  nonterminal Run for it. A terminal Run is unread until the page marks its conversation seen; the
  page keeps those terminal Run ids in its own bounded storage, and opening the conversation or
  clicking a completion notification marks them. P0 has no server read state, so a second browser or
  device does not inherit "read".
- **The page watches only while there is work.** `start()` performs one discovery read; the page
  polls the projection (five seconds visible, fifteen seconds hidden) only while the last read held
  a nonterminal Run, makes one tail read after the last Run settles — the goal-round driver starts
  its continuation within milliseconds — and then holds no timer and asks for nothing. `wake()`
  resumes a dormant page for one read (and, if it finds nothing, the same single tail look: the
  page hears a submission start before the accepted Run is guaranteed to be in a read), and the
  shell wakes it when a submission starts running and when the page becomes visible again. A hidden
  page with a live Run keeps the hidden cadence so a completion can be delivered; a hidden page
  with nothing running makes no request at all.
- **One notification owner, and goal rounds are not announcements.** `DlRunNotifications` is the
  only constructor of a completion `Notification`, for every conversation including the open one;
  `notifications.ts` keeps only the permission offer. A settlement is announced only while the page
  is away and permission is already granted, and a goal round is never announced per round: it is
  activity in the projection (the conversation runs across rounds and the last round leaves the
  unread marker), but announcing each round would claim the conversation finished while the goal is
  still active and duplicate the goal bar. A goal-level completion notice is a later,
  goal-state-aware step (P0.5). A closed tab, a service worker, Web Push, email, and webhooks are
  out of this decision: each is separate delivery infrastructure, and the projection does not depend
  on one.

## Considered options

- **A server notification table with per-run read receipts:** rejected for P0. It duplicates the Run
  row's lifecycle and makes "read" a cross-device feature with a schema and an API; the page can
  derive both from the Runs it already has, and a device-local unread state is honest about what it
  is.
- **A server-sent event stream of all of the owner's Runs:** the more natural long-run transport,
  and it would remove the poll. It needs one durable owner-wide subscription surface plus fan-out
  and reconnect semantics; P0 is a bounded read and a timer, and the projection it reads is the same
  one such a stream would push. Adding the stream later changes the transport, not this projection.
- **A conversation-list field (`running`, `last_finished_at`) instead of a Run read:** rejected
  because a conversation summary would then carry another aggregate's state, and a run that
  outlives the conversation's newest page would be invisible. The Run is the unit the reader is
  watching.
- **Service worker and Web Push in P0:** rejected as new infrastructure with its own registration,
  VAPID, and browser-permission surface. Best-effort while the page is open is the honest first
  slice; the projection is what a push would read.

## Consequences

- No migration, no new table, no Run state change; one bounded join per poll, and no poll at all
  while the reader's Runs are all settled.
- Hidden tabs are timer-throttled by the browser, so a completion notification may arrive late or,
  when the browser suspends the page, not until the reader returns; the conversation list's unread
  marker is then what tells them.
- A goal round's settlement is silent by design; the goal bar and the conversation's unread marker
  are what show it, and a goal-level notice would need the goal state the projection does not
  carry (P0.5).
- The open conversation's notice now also comes from the poll, so it can arrive up to one interval
  after the Run settles rather than from its SSE: one owner and one timer are the simplicity the
  product gains in exchange.
- A second device does not inherit unread. Moving unread to the server later is additive: the
  projection stays, and a read receipt is a new record rather than a new truth.
- The page's poll can become an owner-wide stream without changing the wire projection or the
  client's derivation of running and unread.
- A Runs page or popover is additive: it reads the same projection and adds no backend state.
