# P0 design — Run activity: completion notification + run visibility (Web)

Status: implemented and verified (P0 + P0.1: goal-round silence, one notification owner, an idle-sleeping watchdog). Verification record: [p0-verification.md](p0-verification.md). The drift fix (ADR-0060 / CONTEXT.md) is separate and was committed as 2d8eaea9.
Scope: a read-only projection + browser polling. No new Run state, no server notification table, no schema migration.

## Why this slice

The durable Run already survives a disconnect and finishes without the page. What is missing is the
reader's ability: the Web does not show which conversation is running, does not show a finished run
the reader has not opened, and the only completion notification fires for the one followed chat run
while the page is away. P0 closes that gap:

- one owner-scoped read of the reader's recent conversation-linked Answer runs;
- a browser store that polls it and derives `running` / `unread` per conversation;
- sidebar row markers: running dot, finished-unread dot;
- a completion notification for a run in a conversation other than the open one, while the page is
  away. The current conversation keeps its existing notification path.

Explicitly out of P0: service worker/Web Push (closed tab), email/webhook, a server-owned "read"
state, a Runs page/popover, cross-device unread, and any change to RunRuntime or the Run state machine.

## Wire contract (frozen)

`GET /web/api/run-activity?limit=30`

- Auth: the existing Web cookie/bearer principal (`UserContext` via the route's `_user`). Owner-scoped:
  only `principal_id == owner` runs.
- `limit`: int, 1..100, default 30.
- Response 200 `application/json`:

```json
{
  "items": [
    {
      "run_id": "018f…",
      "conversation_id": "018f…",
      "conversation_title": "Volvo quarterly results",
      "status": "running",
      "phase": "researching",
      "created_at": "2026-10-24T10:11:12.123456Z",
      "started_at": "2026-10-24T10:11:13.000000Z",
      "finished_at": null
    }
  ]
}
```

- `status` is one of `queued | running | succeeded | failed | cancelled`; `phase` is the
  executor-owned string or null; `finished_at` is null until terminal; `conversation_title` is null
  for an untitled conversation.
- Order: `created_at DESC, run_id DESC`. The list is one hard-bounded page; no cursor in P0.
- Only `run_kind = 'answer'` runs that a `web_conversation_turns` row links to a conversation are
  returned (a run accepted through REST/MCP has no conversation and is not Web run activity).
- Errors follow the existing Web application rules: unavailable store → the same 503 mapping other
  Web routes use; malformed `limit` → 422 before the service runs.
- Response model name: `WebRunActivityPage`; item: `WebRunActivity`.

## Backend shape (frozen names)

- `application/web_conversations/models.py`: `WebRunActivity` frozen dataclass (fields above as
  `str / str | None / datetime`), exported from `application/web_conversations/__init__.py`.
- `application/web_conversations/models.py`: `WebConversationStore.recent_answer_runs(
  principal_id: str, *, limit: int) -> tuple[WebRunActivity, ...]`.
- `application/web_conversations/service.py`: `WebConversationService.run_activity(
  user: UserContext | None, *, limit: int) -> tuple[WebRunActivity, ...]`; normalizes
  `owner_id_from_user(user)` and delegates. Optional `limit` validation (1..100) in the service too,
  raising `ValueError` for an internal caller — the route's `Query` is the public bound.
- `adapters/http/browser/routes/run_activity.py`: new router module, `GET /run-activity`; included in
  `routes/__init__.py` under `api_router`. Uses `get_web_conversation_service` and `_user`-equivalent
  local helper (do not import a route's private helper across modules; define the small `_user`
  function locally as `conversations.py` does).
- `adapters/http/browser/conversation_models.py`: `WebRunActivityPage` / item models.
- `adapters/http/browser/conversations.py`: `project_run_activity(activity) -> WebRunActivity`.
- `adapters/postgres/web/web_conversations.py`: `_RECENT_ANSWER_RUNS` SQL constant and
  `recent_answer_runs`; uses `_ensure_initialized`, `_run_read`, and maps rows explicitly.
  SQL joins `dlightrag_runs` (`owner_id`, `run_id`, `run_kind='answer'`, lifecycle columns) to
  `web_conversation_turns` (`principal_id`, `answer_run_id`) and `LEFT JOIN web_conversations` for the
  title. No new index or migration: the turn indexes and the runs PK cover the read; verify with
  `EXPLAIN` in the PG test notes if the query plan is surprising.

## Frontend shape (frozen names)

- `frontend/api/run-activity.ts`: valibot schema + `listRunActivity(limit = 30, signal?)` using
  `parseWire`; camelCase transform: `runId`, `conversationId`, `conversationTitle`, `status`, `phase`,
  `createdAt`, `startedAt`, `finishedAt`.
- `frontend/stores/run-activity-store.ts`: `RunActivityStore extends Store`.
  - Read-only state: `runs`, `activeConversationIds`, `unreadConversationIds`, `activeCount`,
    `unreadCount`.
  - Lifecycle: `start()`, `stop()`, `refresh()`, `forget()`; poll 5 s visible / 15 s hidden, one
    request in flight (AbortController), hidden/visible interval switch on `visibilitychange`.
  - Seen state: terminal run ids in `localStorage` key `dlightrag.run-activity-seen`, bounded to the
    newest 200 terminal ids; a malformed value behaves as empty. `markConversationSeen(conversationId)`
    and `markAllSeen()` persist and notify subscribers.
  - Settlement callback: `subscribeSettled(handler)` fires only for a transition observed after the
    first successful load (a run that was nonterminal in the previous snapshot, or a terminal run that
    appeared after the first load), and never for a run already marked seen. The first load never
    fires settlement callbacks. This is what the notifier uses; the store itself never constructs a
    `Notification`.
  - The store injects the list call in tests; no module-level singleton.
- `frontend/stores/app-handles.ts`: add `runActivity: RunActivityStore`; `reset()` calls
  `forget()`.
- `frontend/ui/app.ts`: start the store once the shell is ready and signed in; stop/forget on sign-out
  and disconnect; when `handles.conversations.activeConversationId` changes, call
  `markConversationSeen`; render one `<dl-run-notifications>` element.
- `frontend/ui/conversation-list.ts`: subscribe to `runActivity` through the existing
  `StoreController`; a row shows a running dot when its conversation has an active run and a
  finished-unread dot when it has an unread terminal run. Dots are `role="img"` with an `aria-label`
  (message ids below), and the styles live in `frontend/styles/conversations.module.css`.
- `frontend/ui/run-notifications.ts`: non-visual `DlRunNotifications` element. On a settlement
  callback: only if enabled, `document.hidden || !document.hasFocus()`, the run's conversation is not
  the active conversation, and `Notification.permission === 'granted'` → one `Notification` titled
  "Answer ready" with the conversation title (fallback to the existing untitled copy). Click focuses
  the window, navigates to the conversation, and marks it seen. No permission prompting in P0; the
  existing offer component keeps prompting.
- The existing `frontend/ui/notifications.ts` component is unchanged: it keeps the current
  conversation's notification and permission offer. The new notifier excludes the active conversation,
  so the two never double-notify.
- i18n: every new `msg()` id gets a `frontend/i18n/locales/zh.ts` entry. Proposed ids:
  `conversationList.running` ("Running"), `conversationList.unreadFinished` ("Finished, unread"),
  `runNotifications.answerReadyBody` ("Your answer for {title} is ready.") — the title is a `str`
  interpolation; reuse `notifications.answerReadyTitle` for the title.

## Tests (acceptance)

Backend:
- Unit route test (ASGI + `application_double`): projection fields, owner path through the service,
  `limit` bounds (422), empty list, service unavailable mapping. Fake service returns DTOs; assert
  the route does not read the DB itself.
- PG integration test: seed two owners, conversations, turns and runs (queued/running/terminal,
  plus a retrieval run and an unlinked answer run); assert ordering, limit, owner isolation, title
  join, null title, and that a non-answer/unlinked run is absent.
- `uv run ruff check` + `ruff format --check` + `uv run pyright` + `uv run lint-imports` on changed
  files (or the project defaults).

Frontend:
- Node store tests with an injected list function: first load suppresses settlement; a nonterminal →
  terminal transition fires once; a terminal run appearing after the first load fires once; a seen
  run never fires; persistence round-trips; poll interval switches on visibility; overlapping refresh
  does not duplicate.
- Browser test: `DlRunNotifications` notifies for an away page and a non-active conversation, stays
  silent when focused or when the conversation is active, and navigates on click; conversation-list
  rows render the running and unread markers.
- `npm --prefix frontend run test`, `typecheck`, `lint`, `test:browser`, `build` as available.

Docs (Lead):
- `docs/interfaces.md` Web section: the route, the read-only projection, and the P0 boundary
  (client-owned unread, tab-open polling, no server notification state).
- A short ADR: run activity is a bounded read of the durable Runs and the conversation turn map; no
  second queue, no server read state, no service worker in P0.

---

## Amendment 2026-10-24 (P0.1) — watch only while there is work; one notification owner

Two corrections after review. They stay inside the P0 boundary (no new store, no migration, no
RunRuntime change) and replace parts of the text above where they conflict.

### A. A goal round is activity, but it is not a completion notice

A goal round is an ordinary Answer Run with a Web turn, so it belongs in the projection: the
conversation is still running across rounds, and the final round leaves the unread marker. But a
per-round completion notice would fight the goal bar and claim the conversation finished while the
goal is still active. The projection therefore states the Run's goal-continuation fact, and the page
uses it only to stay silent:

- Wire item gains `goal_round: boolean` (additive). Derivation: the Run's accepted input carries
  `parent_run_id` and `continuation_kind="goal_round"`; the projection states
  `continuation_kind == "goal_round"` as this one product fact. No raw request field crosses.
- The notifier skips a settlement whose Run is a goal round. The unread marker is unchanged: a
  reader who returns sees the conversation finished and the goal bar state; a goal-level notice is
  a later, goal-state-aware step (P0.5), not a per-round one.
- Follow-up and fork continuations are human actions and still notify.

### B. The page watches only while there is something to watch

The first P0 text polled for the shell's whole lifetime. That makes "activity" a resident process
even when every Run is finished, which is the opposite of the product's surface: an all-clear page
should hold no timer and ask for nothing, and the row markers themselves render only while a
conversation is running or unread.

The store becomes demand-driven:

- `start()` performs one discovery read (the page may have opened on a live Run) and then polls only
  while the last read held a nonterminal Run: 5 s visible, 15 s hidden.
- When the last active Run settles, the store makes **one** tail read after 5 s — the goal-round
  driver starts its continuation within milliseconds, and that read catches it — and then sleeps:
  no timer, no request, no marker beyond a legitimate unread one.
- `wake()` resumes a dormant store for one read, and keeps polling again if that read finds work.
  An attention read that finds nothing still earns the one tail look: the wake can arrive while the
  accepted Run's projection row is still a moment away (the page hears "submission started" before
  the GET observably contains it), and a wake that slept immediately could lose the Run it woke for.
  The shell wakes it when a submission starts running (`dl-chat-running-change` active), when the
  page becomes visible again, and when the user starts the page on a live Run (the initial read
  covers that). A first discovery read that finds nothing sleeps immediately, without the tail.
- `stop()` still means "the page is gone/signed out"; `wake()` is the in-page attention signal.
- A hidden page with a live Run keeps the hidden cadence so the completion notice can be delivered;
  a hidden page with nothing running makes no request at all.

### C. One notification owner

The old `notifications.ts` both asked for permission and announced the followed Run; the new
notifier announced other conversations. With goal rounds in play, that split cannot suppress the
current conversation's per-round notices without giving the old component a second source of truth.
The ownership is therefore separated:

- `notifications.ts` keeps **only the permission offer**: "Notify you when an answer finishes?" is
  asked from the same missed-answer moment, and the enable/decline state stays where it is.
- `DlRunNotifications` is **the only constructor of a completion Notification**, for every
  conversation (the open one included), gated by enabled + page away + granted permission + not a
  goal round. No active-conversation exclusion; no duplicate by construction.
- The notification is observed by the poll rather than by the open conversation's SSE, so the open
  conversation's notice may arrive up to one poll later (5 s visible / 15 s hidden). That delay is
  the price of one owner and of the page not keeping a second notification path; the notice is a
  background courtesy, not a transcript event.

Acceptance for P0.1: wire carries `goal_round`; a goal round settles unseen and no Notification is
constructed for it in either conversation state; the existing offer still appears and persists its
decline; an idle page (all Runs terminal, nothing unread) schedules no further read after the tail;
a running page resumes and polls; the page wakes on submission start and on becoming visible; FE and
BE suites, the browser suite, and the docs contracts pass.
