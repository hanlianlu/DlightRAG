# P0 verification — Run activity: completion notification + run visibility

> Date: 2026-10-24 · Repository HEAD at verification: 310e9102 (the drift fix landed as 2d8eaea9)
> Contract: [p0-design.md](p0-design.md) · Decision: [ADR-0069](../../adr/0069-the-web-reads-run-activity-from-the-runs-it-keeps.md)
> Scope delivered: P0 only. No migration, no RunRuntime change, no server notification/read state.

## Delivered

Backend (read projection):

- `GET /web/api/run-activity?limit=` — owner-scoped, conversation-linked Answer Runs, newest first,
  `limit` 1–100 (30 default). Fields: `run_id`, `conversation_id`, `conversation_title`, `status`,
  `phase`, `created_at`, `started_at`, `finished_at`.
- Application: `WebRunActivity`, `WebConversationStore.recent_answer_runs`,
  `WebConversationService.run_activity`, `RUN_ACTIVITY_DEFAULT_LIMIT` / `RUN_ACTIVITY_MAX_LIMIT`.
- HTTP: `routes/run_activity.py`, wire models, `project_run_activity`, router registration.
- PG: `_RECENT_ANSWER_RUNS` joins `web_conversation_turns` → `dlightrag_runs`
  (`run_kind='answer'`) with `LEFT JOIN web_conversations` for the title, owner predicate from the
  turn's unique index, `ORDER BY created_at DESC, run_id DESC LIMIT`, `_run_read` 503 mapping.
- Tests: 7 ASGI route unit tests (exact wire JSON, bounds 422, empty page, owner/limit passthrough,
  503 unavailable), new PG integration file with 4 tests (ordering/tie-break, owner isolation, limit,
  title/null title, non-answer and unlinked-run exclusion).

Frontend (page projection and notification):

- `api/run-activity.ts` — valibot wire schema, `listRunActivity`, `runActive` / `runTerminal`.
- `stores/run-activity-store.ts` — 5 s visible / 15 s hidden polling, visibility re-arm, single
  in-flight read, `seen` persistence in `localStorage` (`dlightrag.run-activity-seen`, ≤200 ids,
  malformed = empty), `markConversationSeen` / `markAllSeen`, `subscribeSettled` (first load never
  announces; only a watched run settling or a terminal run appearing after the first load, unseen),
  `forget` on sign-out.
- `ui/run-notifications.ts` — non-visual `DlRunNotifications`: away page + non-active conversation +
  granted permission → one `Notification`, click focuses, navigates, marks seen. Existing
  `notifications.ts` (current conversation + permission offer) is untouched; the new notifier excludes
  the active conversation, so no double notification.
- `ui/conversation-list.ts` + styles — running dot and finished-unread dot per conversation row
  (`role="img"`, aria-labels), observing the new store.
- `ui/app.ts` + `stores/app-handles.ts` — store lifecycle follows the shell; opening a conversation
  marks its settled runs seen; the notifier is mounted by the shell.
- i18n catalog entries for the new copy; shell/structure tests updated for the new shared store.

Docs:

- `docs/interfaces.md` Web route family: the new route and its read-only contract.
- `docs/adr/0069-the-web-reads-run-activity-from-the-runs-it-keeps.md`; ADR index regenerated (68 entries).
- `CONTEXT.md`: **run activity** defined.
- Drift fix (separate, committed as 2d8eaea9): ADR 0060 amendment retires the `CheckpointProvider`
  seam in favour of the in-run pre-delivery checkpoint; `CONTEXT.md` defines the pre-delivery
  checkpoint and no longer a checkpoint provider; the goal round carries the objective and number.

## Verification (Lead re-ran, 2026-10-24)

Backend:

| Command | Result |
|---|---|
| `uv run pytest tests/unit/test_web_conversations.py tests/integration/test_web_run_activity_pg.py -q` | **100 passed** |
| `uv run pytest tests/unit/test_web_conversations.py tests/unit/test_web_routes.py tests/unit/test_api_server.py tests/unit/test_application.py tests/integration/test_web_run_activity_pg.py -q` | **423 passed** in 40.4 s |
| `uv run ruff check <changed Python>` | All checks passed |
| `uv run ruff format --check <changed Python>` | formatted |
| `uv run lint-imports` | 32 kept, 0 broken |
| `uv run pyright <changed backend files>` | 0 errors, 0 warnings |
| `uv run python scripts/adr_index.py --check` | current, 68 entries |
| `uv run pytest tests/unit/test_documentation_contracts.py -q` | 12 passed |
| `uv run python scripts/verify_release_contract.py --scope platform` | passed |

Frontend:

| Command | Result |
|---|---|
| `npm --prefix frontend run test` | 436 pass, 0 fail |
| `npm --prefix frontend run typecheck` | clean |
| `npm --prefix frontend run lint` | exit 0 (3 pre-existing warnings in unrelated scripts) |
| `npm --prefix frontend run test:browser` | 74 files, 1131 passed, 0 failed (Chromium) |
| `npm --prefix frontend run build` | built successfully |

Not run: a full-stack HTTP+PG e2e of the route (route is covered with a fake service against the
exact wire JSON, and the store against real PostgreSQL 18.6; the seam between them is the typed
service call, and OpenAPI confirmed the route shape). The endpoint was not exercised against a live
signed-in browser session in this environment.

## Contract conformance

- Wire shape: exact, asserted byte-for-byte in the route test; OpenAPI shows the 8 required/optional
  fields, the 5-value status enum, and `limit` `1..100` default `30`.
- Ownership: the SQL starts at `web_conversation_turns` for the principal; the PG test proves two
  owners see only their own rows.
- No second truth: no migration, no notification table, no read state on the server. The projection
  reads the durable Run row and the conversation turn map; `frontend/ui/notifications.ts` untouched.
- No RunRuntime/Engine change.

## Judgments made during implementation (to review)

1. **`forget()` clears the unread set on sign-out** (not only the snapshot), because the app-handles
   `reset()` wording is "what the signed-in reader sent, rated, and read". A different reader on the
   same browser must not inherit the first reader's unread markers.
2. **`activeCount` / `unreadCount` count conversations**, not Runs: the sidebar marks conversations,
   and one conversation with two running Runs is one running row.
3. **Disconnect only stops polling, never forgets**; only sign-out forgets. A navigation or HMR
   remount therefore does not lose what the reader has seen.
4. **A settlement in the open conversation is marked seen by the shell**, so it gets no unread dot;
   its browser notification remains the existing `notifications.ts` path. The new notifier excludes
   the open conversation, so the two never double-notify.
5. **No permission prompt from the new notifier.** A reader who has never granted permission gets no
   browser notification (they still get the row markers); the existing offer is the prompt path. If
   the first missed completion happens in another conversation, the offer may not appear until the
   reader grants it elsewhere — a P0.5 candidate (trigger the existing offer from a missed
   non-active Run).
6. **Settlement is transition-based after the first successful read**, so a run that settled while
   the page was closed is unread (marker) but is not announced as a new notification. This is the
   frozen contract and matches "the page opened with it".

## Known limits (P0)

- The page must stay open; hidden-tab timers are browser-throttled. Closed-tab delivery needs a
  service worker / Web Push, explicitly out of P0 (ADR 0069).
- Polling is one bounded join per open page every 5 s visible / 15 s hidden; an owner-wide SSE stream
  is the natural later transport and does not change the wire projection.
- Unread is per-browser, not cross-device.
- The feed is a hard page of the newest 30 Runs: a running Run that falls out of that window before
  it settles is not announced (its conversation can still show the settled turn in history).
- A Runs page/popover is additive (P0.5); this slice delivers the row markers + notification.

## Working tree note


---

## P0.1 amendment — goal-round silence, one notification owner, watchdog sleeps when idle

Design: [p0-design.md](p0-design.md) "Amendment 2026-10-24 (P0.1)"; decision text updated in
[ADR-0069](../../adr/0069-the-web-reads-run-activity-from-the-runs-it-keeps.md); route field list
updated in [interfaces.md](../../interfaces.md).

Delivered:

- **`goal_round` on the wire.** `WebRunActivity.goal_round: bool` is derived inside the PG adapter
  from the Run's accepted envelope (`continuation_kind = 'goal_round'`); the raw continuation field
  never crosses. Wire field is a required boolean, last in the item; OpenAPI verified.
- **One notification owner.** `notifications.ts` keeps only the permission offer (its Notification
  construction was removed); `DlRunNotifications` is the only completion-notification constructor and
  now covers every conversation, including the open one. It stays silent for a goal round, for a
  focused page, and without granted permission.
- **A watchdog that sleeps.** The store performs one discovery read, polls only while a Run is
  nonterminal (5 s visible / 15 s hidden), takes one 5 s tail read after the last Run settles (to
  catch a goal-round continuation), and then holds no timer and makes no request. An all-clear page
  therefore has no resident activity process. `wake()` is the attention signal (submission starts
  running, page becomes visible); an attention read that finds nothing still earns the one tail look
  because the accepted Run's projection row can trail the page's "started" event by a moment.

Re-verification (Lead, 2026-10-24):

| Command | Result |
|---|---|
| `uv run pytest tests/unit/test_web_conversations.py tests/unit/test_web_routes.py tests/unit/test_api_server.py tests/unit/test_application.py tests/integration/test_web_run_activity_pg.py tests/integration/test_web_conversations_pg.py -q` | **432 passed** |
| `uv run pytest tests/unit/test_web_conversations.py tests/integration/test_web_run_activity_pg.py -q` | 102 passed (97 unit + 5 PG) |
| OpenAPI `WebRunActivity` | `goal_round` boolean, required |
| `uv run ruff check` + `uv run lint-imports` + `pyright` (changed files) | clean / 32 kept / 0 errors |
| `npm --prefix frontend run test` | 445 pass, 0 fail |
| `npm --prefix frontend run typecheck` / `lint` / `build` | clean / exit 0 / built |
| `npm --prefix frontend run test:browser` | 74 files, **1135 passed**, 0 failed |
| `scripts/adr_index.py --check` + documentation contracts | current (68) / 12 passed |

Updated behaviour notes:

- A goal round's settlement is silent in both conversation states; the unread marker and the goal bar
  are what state it. A goal-level completion notice needs goal state the projection does not carry
  and remains P0.5.
- A dormant page does not discover a Run started by another tab/device; it wakes on this page's own
  submission, on becoming visible, or on the initial discovery read. That is the price of "no request
  when there is nothing to watch" and is consistent with P0's conversation-linked, page-local scope.
- The open conversation's completion notice now arrives through the poll (up to 5 s visible / 15 s
  hidden after settlement) rather than its SSE: one owner and one timer, deliberately.
