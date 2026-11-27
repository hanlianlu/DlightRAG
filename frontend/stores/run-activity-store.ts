// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Which of the reader's conversations are running, and which have an answer they have not opened.

 * The durable Run outlives the page, so the page reads the reader's recent conversation-linked
 * Answer runs and projects them onto the conversation list: a conversation is running while one of
 * its runs is queued or running, and unread while one of its runs has settled since the reader last
 * looked. The server keeps no read state; what the reader has seen is this browser's, kept in
 * localStorage and bounded so the value stays small. The store itself never draws anything: it
 * notifies subscribers, and the Features above it choose what to show.
 *
 * The page watches only while there is work. `start()` takes one discovery read, and a read that
 * finds no nonterminal Run leaves the page asleep: no timer, no request, while the snapshot and the
 * unread marks stay for what is drawn from them. A read that finds a Run under way keeps polling,
 * closer while the reader is here and calmer while they are away. When the last active Run settles,
 * one tail read five seconds later catches a goal round's continuation (its driver starts the next
 * round within milliseconds); after that the page sleeps again. `wake()` is the page's attention
 * signal, for a submission that started running or a page shown again: read once now, and resume
 * watching if that read finds work. An attention read that finds nothing still earns the one tail
 * read, because a submission's Run can be accepted a moment after the page hears it start.
 *
 * A run is only "settled" to this store when the page watched it settle after its first successful
 * read: the first read is what the page opened with, and the reader has not missed it. A settled run
 * already marked seen is never announced, and a run the page started watching after it had already
 * settled has nothing to announce either. That is what the completion notifier uses, and what tells the
 * Scheduled list that the Agent may have changed it.
 *
 * The visibility of the page chooses the polling interval (closer while the reader is here, calmer
 * while they are away), one request is in flight at a time, and stopping ends the timers and the
 * request.
 */

import {
  listRunActivity,
  runActive,
  runTerminal,
  RUN_ACTIVITY_LIMIT,
  type RunActivity,
  type RunActivityPage,
} from '../api/run-activity.ts';
import {readStored, writeStored} from '../lib/storage.ts';
import {Store} from './base.ts';

/** Where this browser keeps the terminal runs the reader has seen. */
const SEEN_STORAGE_KEY = 'dlightrag.run-activity-seen';

/** How many terminal run ids the stored value keeps: the newest, and only the newest. */
const SEEN_BOUND = 200;

/** How often the activity is read while the reader is looking at the page, and while they are not. */
const VISIBLE_INTERVAL_MS = 5000;
const HIDDEN_INTERVAL_MS = 15000;

/** How long after the last active Run settles the one tail read waits for a goal round's continuation. */
const TAIL_INTERVAL_MS = 5000;

/** Whether the page is being looked at, and when that changes. */
export interface ActivityVisibility {
  visible(): boolean;
  /** Call `handler` whenever the page is shown or hidden; the result stops it. */
  onChange(handler: () => void): () => void;
}

/** Where the seen run ids are kept; injectable so a test needs no browser storage. */
export interface SeenStorage {
  read(): string | null;
  write(value: string | null): void;
}

/** The read that fills the store: the Run activity page, abortable with the store. */
export type RunActivityList = (limit: number, signal: AbortSignal) => Promise<RunActivityPage>;

/** A run the page watched settle; it was not terminal in the snapshot the page read before. */
export type SettledHandler = (run: RunActivity) => void;

/** Why a read was made: the first look, a poll of known work, the one look after the last Run
 *  settled, or the page's attention read (a wake). It decides what the page does when the read finds
 *  no Run under way: the tail and attention reads earn the one tail read, the first look sleeps. */
type ReadReason = 'discovery' | 'poll' | 'tail' | 'wake';

/** What a pending timer will do when it fires: poll known work, or make the one tail read. */
type TimerKind = 'poll' | 'tail';

export interface RunActivityStoreOptions {
  /** The read; defaults to the Wire client. Injected by tests. */
  list?: RunActivityList;
  /** The page's visibility; defaults to the document's. */
  visibility?: ActivityVisibility;
  /** Where this browser keeps what the reader has seen; defaults to localStorage. */
  storage?: SeenStorage;
  /** How many runs one read asks for. */
  limit?: number;
}

function documentVisibility(): ActivityVisibility {
  return {
    visible: () => typeof document !== 'undefined' && !document.hidden,
    onChange: (handler) => {
      if (typeof document === 'undefined') return () => {};
      document.addEventListener('visibilitychange', handler);
      return () => { document.removeEventListener('visibilitychange', handler); };
    },
  };
}

function browserSeenStorage(): SeenStorage {
  return {
    read: () => readStored(SEEN_STORAGE_KEY),
    write: (value) => { writeStored(SEEN_STORAGE_KEY, value); },
  };
}

/** The ids a stored value states, or none when it is unreadable: a stale value must not hide an
 *  answer, so anything but a list of non-empty strings reads as nothing seen yet. */
function parseSeen(raw: string | null): readonly string[] {
  if (raw === null) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];
  if (!parsed.every((id): id is string => typeof id === 'string' && id.length > 0)) return [];
  return parsed.slice(-SEEN_BOUND);
}

/** Whether two snapshots say the same thing about every run, so a redraw is not asked for a poll
 *  that found nothing new. Order is part of it: the server orders by created_at, newest first. */
function sameRuns(left: readonly RunActivity[], right: readonly RunActivity[]): boolean {
  if (left.length !== right.length) return false;
  return left.every((run, index) => {
    const other = right[index];
    return other !== undefined
      && run.runId === other.runId
      && run.status === other.status
      && run.phase === other.phase
      && run.conversationTitle === other.conversationTitle
      && run.finishedAt === other.finishedAt
      && run.goalRound === other.goalRound;
  });
}

function sameIds(left: ReadonlySet<string>, right: ReadonlySet<string>): boolean {
  if (left.size !== right.size) return false;
  for (const id of left) if (!right.has(id)) return false;
  return true;
}

/** The conversation-level projection of one Run snapshot. */
function project(runs: readonly RunActivity[], seen: ReadonlySet<string>) {
  const active = new Set<string>();
  const unread = new Set<string>();
  for (const run of runs) {
    if (runActive(run.status)) active.add(run.conversationId);
    else if (!seen.has(run.runId)) unread.add(run.conversationId);
  }
  return {active, unread};
}

/** Recent Run activity, projected per conversation, read while there is work to watch. */
export class RunActivityStore extends Store {
  readonly #list: RunActivityList;
  readonly #visibility: ActivityVisibility;
  readonly #storage: SeenStorage;
  readonly #limit: number;
  readonly #settled = new Set<SettledHandler>();

  #runs: readonly RunActivity[] = [];
  #activeConversationIds: ReadonlySet<string> = new Set();
  #unreadConversationIds: ReadonlySet<string> = new Set();
  readonly #previous = new Map<string, RunActivity>();
  /** The ids the first successful read stated: a settled run among them is what the page opened with. */
  readonly #firstRead = new Set<string>();
  /** The ids this page ever watched running; a settled one of them is a transition, however long ago. */
  readonly #watched = new Set<string>();
  /** The ids this browser has seen, oldest first, so bounding drops the oldest. */
  readonly #seen: Set<string>;

  /** The first successful read has answered: only what arrives after it can be news. */
  #loaded = false;
  /** The page is watching: started and not stopped. A stopped store reads for nobody. */
  #started = false;
  #inFlight: Promise<void> | null = null;
  #request: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #timerKind: TimerKind | null = null;
  #stopListening: (() => void) | null = null;

  constructor(options: RunActivityStoreOptions = {}) {
    super();
    this.#list = options.list ?? listRunActivity;
    this.#visibility = options.visibility ?? documentVisibility();
    this.#storage = options.storage ?? browserSeenStorage();
    this.#limit = options.limit ?? RUN_ACTIVITY_LIMIT;
    this.#seen = new Set(parseSeen(this.#storage.read()));
  }

  /** The newest runs the last read reported, newest first. */
  get runs(): readonly RunActivity[] {
    return this.#runs;
  }

  /** The conversations with a run still queued or running. */
  get activeConversationIds(): ReadonlySet<string> {
    return this.#activeConversationIds;
  }

  /** The conversations with a settled run the reader has not opened since. */
  get unreadConversationIds(): ReadonlySet<string> {
    return this.#unreadConversationIds;
  }

  /** How many conversations are running. */
  get activeCount(): number {
    return this.#activeConversationIds.size;
  }

  /** How many conversations have a settled run the reader has not opened since. */
  get unreadCount(): number {
    return this.#unreadConversationIds.size;
  }

  /** Whether the page has read the activity once; a settlement is only news after that. */
  get loaded(): boolean {
    return this.#loaded;
  }

  /** Whether the page is watching; false while it sleeps with the snapshot kept, or after `stop`. */
  get watching(): boolean {
    return this.#started && (this.#timer !== null || this.#inFlight !== null);
  }

  /** Hear a run settle that the page watched; returns the unsubscribe. */
  subscribeSettled(handler: SettledHandler): () => void {
    this.#settled.add(handler);
    return () => { this.#settled.delete(handler); };
  }

  /** Read once, then poll only while a Run is under way. Idempotent while already started. */
  start(): void {
    if (this.#started) return;
    this.#started = true;
    this.#stopListening = this.#visibility.onChange(this.#visibilityChanged);
    void this.#refresh('discovery');
  }

  /** Stop watching and end a read in flight; the snapshot and the seen marks stay for the next `start`. */
  stop(): void {
    this.#started = false;
    this.#stopListening?.();
    this.#stopListening = null;
    this.#clearTimer();
    this.#request?.abort();
    this.#request = null;
    this.#inFlight = null;
  }

  /** The page has reason to look now: read at once, and watch again if there is work to watch. */
  wake(): void {
    if (!this.#started) return;
    void this.#refresh('wake');
  }

  /** Read the activity now. A read already in flight is shared, never duplicated. */
  refresh(): Promise<void> {
    return this.#refresh('wake');
  }

  /** Mark every settled run of one conversation seen, which takes away its unread mark. */
  markConversationSeen(conversationId: string): void {
    this.#markSeen((run) => run.conversationId === conversationId);
  }

  /** Mark every settled run the page knows about seen. */
  markAllSeen(): void {
    this.#markSeen(() => true);
  }

  /** Forget the activity and what the reader saw, as signing out must: the next reader starts new. */
  forget(): void {
    this.stop();
    this.#seen.clear();
    this.#writeSeen();
    this.#runs = [];
    this.#activeConversationIds = new Set();
    this.#unreadConversationIds = new Set();
    this.#previous.clear();
    this.#firstRead.clear();
    this.#watched.clear();
    this.#loaded = false;
    this.changed();
  }

  /** Start a read with its reason, unless one is already in flight. A read that interrupts the one
   *  tail timer is that tail look, made early: one tail read, never two. */
  #refresh(reason: ReadReason): Promise<void> {
    if (this.#inFlight !== null) return this.#inFlight;
    if (this.#timerKind === 'tail') reason = 'tail';
    this.#clearTimer();
    const controller = new AbortController();
    this.#request = controller;
    const read = this.#read(controller, reason);
    this.#inFlight = read;
    return read;
  }

  async #read(controller: AbortController, reason: ReadReason): Promise<void> {
    let read = false;
    try {
      const page = await this.#list(this.#limit, controller.signal);
      if (this.#request !== controller) return;
      this.#apply(page.items);
      read = true;
    } catch {
      // Polling is an enhancement: a read that fails keeps the snapshot, and the scheduling below
      // leaves a page with work watching and an idle page asleep.
    } finally {
      if (this.#request === controller) {
        this.#request = null;
        this.#inFlight = null;
        if (this.#started) this.#settle(reason, read);
      }
    }
  }

  /** Choose what the page does after a read: poll while a Run is under way, take the one tail look
   *  after the last one settled or after an attention read that found nothing, and otherwise sleep. */
  #settle(reason: ReadReason, read: boolean): void {
    const active = this.#activeConversationIds.size > 0;
    if (!read) {
      // A failed read keeps a page that had work watching; an idle page is not retried until the
      // page has a reason to look again.
      if (active) this.#schedulePoll();
      return;
    }
    if (active) {
      this.#schedulePoll();
      return;
    }
    if (reason === 'poll' || reason === 'wake') this.#scheduleTail();
  }

  #apply(activity: readonly RunActivity[]): void {
    const settled: RunActivity[] = [];
    if (this.#loaded) {
      for (const run of activity) {
        if (!runTerminal(run.status) || this.#seen.has(run.runId)) continue;
        const before = this.#previous.get(run.runId);
        // A run already read as settled was read then; only a run that was watched running, or that
        // appeared already settled after the first read, is news.
        if (before !== undefined && runTerminal(before.status)) continue;
        if (this.#watched.has(run.runId) || !this.#firstRead.has(run.runId)) settled.push(run);
      }
    } else {
      for (const run of activity) this.#firstRead.add(run.runId);
    }
    for (const run of activity) if (!runTerminal(run.status)) this.#watched.add(run.runId);
    const {active, unread} = project(activity, this.#seen);
    const redraw = !this.#loaded
      || !sameRuns(this.#runs, activity)
      || !sameIds(this.#activeConversationIds, active)
      || !sameIds(this.#unreadConversationIds, unread);
    this.#loaded = true;
    this.#runs = activity;
    this.#activeConversationIds = active;
    this.#unreadConversationIds = unread;
    this.#previous.clear();
    for (const run of activity) this.#previous.set(run.runId, run);
    if (redraw) this.changed();
    // The snapshot the reader sees is in place before the news it carries is announced.
    for (const run of settled) for (const handler of [...this.#settled]) handler(run);
  }

  #markSeen(matches: (run: RunActivity) => boolean): void {
    let marked = false;
    // Oldest first: the newest terminal ids are the ones the bound must keep.
    for (const run of [...this.#runs].reverse()) {
      if (!matches(run) || !runTerminal(run.status) || this.#seen.has(run.runId)) continue;
      this.#seen.add(run.runId);
      marked = true;
    }
    if (!marked) return;
    this.#trimSeen();
    this.#writeSeen();
    const {unread} = project(this.#runs, this.#seen);
    if (!sameIds(this.#unreadConversationIds, unread)) this.#unreadConversationIds = unread;
    this.changed();
  }

  #trimSeen(): void {
    while (this.#seen.size > SEEN_BOUND) {
      const oldest = this.#seen.values().next().value;
      if (oldest === undefined) return;
      this.#seen.delete(oldest);
    }
  }

  #writeSeen(): void {
    this.#storage.write(this.#seen.size === 0 ? null : JSON.stringify([...this.#seen]));
  }

  /** Keep watching at the cadence the page's visibility asks for. */
  #schedulePoll(): void {
    this.#scheduleTimer('poll', this.#visibility.visible() ? VISIBLE_INTERVAL_MS : HIDDEN_INTERVAL_MS);
  }

  /** Make the one look that catches a goal round's continuation, then sleep unless it finds work. */
  #scheduleTail(): void {
    this.#scheduleTimer('tail', TAIL_INTERVAL_MS);
  }

  #scheduleTimer(kind: TimerKind, delay: number): void {
    this.#clearTimer();
    if (!this.#started) return;
    this.#timerKind = kind;
    this.#timer = setTimeout(() => {
      this.#timer = null;
      this.#timerKind = null;
      void this.#refresh(kind);
    }, delay);
  }

  #clearTimer(): void {
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.#timer = null;
    this.#timerKind = null;
  }

  readonly #visibilityChanged = (): void => {
    if (!this.#started || this.#request !== null) return;
    if (this.#visibility.visible()) {
      // The reader is back: read now, then keep the nearer cadence if that read finds work.
      this.wake();
      return;
    }
    // Hidden: a poll waits the calmer interval; the one tail look keeps its short deadline.
    if (this.#timerKind === 'poll') this.#schedulePoll();
  };
}
