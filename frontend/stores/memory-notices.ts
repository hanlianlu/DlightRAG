// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What each answer's memory notice has read of the memory its Run left.
 *
 * Two writers leave Profile Memory in a Run's name: the Agent, with its memory tools while it answers, and
 * the Session Memory Pass, a few seconds after the Run succeeds. The server lists a Run's records that are
 * still active; this store reads that list for the notice under each answer and keeps what it read, because
 * the chat draws only the turns in view and a turn that scrolls away is drawn again from here.
 *
 * It reads three ways. A turn that was already answered when the page opened is read once, when it is first
 * drawn. A Run this page watched being answered is read again after it settles, at a short backoff, until the
 * pass has reported or the last try has passed: the pass runs detached and takes as long as its model does, so
 * nothing in the Run says when it ends. And a live memory event for a running turn says the Agent wrote,
 * which is read as soon as the Run settles. No try is made while the page is hidden; the one that fell due
 * is made when it is shown again.
 *
 * The list the server returns is the records still active, so a record the reader took back is gone from the
 * next read. The store keeps it, marked undone, so the notice can say so instead of the row vanishing under
 * the reader's hand.
 */

import {listRunMemoryChanges, type RunMemoryChange} from '../api/memory.ts';
import {Store} from './base.ts';

/** Milliseconds after a Run is seen to settle at which the notice looks again for what the pass learned. */
const WATCH_DELAYS_MS: readonly number[] = [3000, 8000, 20000];

/** A change the Run left, and whether the reader has since taken it back. */
export interface MemoryNoticeChange extends RunMemoryChange {
  readonly undone: boolean;
}

/** Whether the page is being looked at, and when that changes: the tries of a watch wait for it. */
export interface PageVisibility {
  visible(): boolean;
  /** Call `handler` whenever the page is shown or hidden; the result stops it. */
  onChange(handler: () => void): () => void;
}

interface MemoryNoticesOptions {
  delays?: readonly number[];
  read?: (runId: string, signal: AbortSignal) => Promise<readonly RunMemoryChange[]>;
  visibility?: PageVisibility;
}

const documentVisibility: PageVisibility = {
  visible: () => document.visibilityState === 'visible',
  onChange: (handler) => {
    document.addEventListener('visibilitychange', handler);
    return () => { document.removeEventListener('visibilitychange', handler); };
  },
};

/** A Run's watch: which try is next, its timer, and whether a try fell due while the page was hidden. */
interface Watch {
  attempt: number;
  timer: ReturnType<typeof setTimeout> | null;
  owed: boolean;
}

interface RunNotice {
  changes: readonly MemoryNoticeChange[];
  /** The list has been read at least once. */
  read: boolean;
  /** A read for a turn that was already answered is out. */
  reading: boolean;
  /** This page watched the Run being answered, so what arrives is news to the reader. */
  live: boolean;
  /** The Run's own agent wrote to memory while it ran. */
  agentWrote: boolean;
  announced: boolean;
  watch: Watch | null;
}

const NONE: readonly MemoryNoticeChange[] = Object.freeze([]);

export class MemoryNotices extends Store {
  readonly #delays: readonly number[];
  readonly #read: NonNullable<MemoryNoticesOptions['read']>;
  readonly #visibility: PageVisibility;
  readonly #notices = new Map<string, RunNotice>();
  #abort = new AbortController();
  #stopListening: (() => void) | null = null;

  constructor(options: MemoryNoticesOptions = {}) {
    super();
    this.#delays = options.delays ?? WATCH_DELAYS_MS;
    this.#read = options.read ?? listRunMemoryChanges;
    this.#visibility = options.visibility ?? documentVisibility;
  }

  /** What the Run left that the notice shows, oldest first, including what the reader has taken back. */
  changes(runId: string): readonly MemoryNoticeChange[] {
    return this.#notices.get(runId)?.changes ?? NONE;
  }

  /** Whether this page watched the Run being answered: only then is what turns up worth announcing. */
  announces(runId: string): boolean {
    const notice = this.#notices.get(runId);
    return (notice?.live ?? false) && !notice?.announced;
  }

  /** The announcement for a Run's notice is made; it is made once. */
  announced(runId: string): void {
    const notice = this.#notices.get(runId);
    if (notice) notice.announced = true;
  }

  /** A turn that was already answered is drawn: read what its Run left, once. */
  ensure(runId: string): void {
    const notice = this.#notice(runId);
    if (notice.read || notice.reading || notice.watch !== null) return;
    notice.reading = true;
    void this.#look(runId, notice).finally(() => { notice.reading = false; });
  }

  /** The Run's own agent wrote to memory while it ran: read once it has settled, without waiting for the pass. */
  noteAgentWrote(runId: string): void {
    this.#notice(runId).agentWrote = true;
  }

  /** The Run just settled with an answer on this page: look for what it left, and keep looking for the pass. */
  watch(runId: string): void {
    const notice = this.#notice(runId);
    // An answer already drawn learns that there is news to wait for, so it can make room to announce it.
    if (!notice.live) {
      notice.live = true;
      this.changed();
    }
    if (notice.watch !== null) return;
    notice.watch = {attempt: 0, timer: null, owed: false};
    if (notice.agentWrote) void this.#look(runId, notice);
    this.#arm(runId, notice, notice.watch);
  }

  /** The reader took a change back: it stays in the list, marked, and no longer counts. */
  markUndone(runId: string, changeId: string): void {
    const notice = this.#notices.get(runId);
    if (!notice?.changes.some((change) => change.changeId === changeId && !change.undone)) return;
    notice.changes = notice.changes.map((change) => (
      change.changeId === changeId ? {...change, undone: true} : change
    ));
    this.changed();
  }

  /** A change the reader took back was brought back by an Undo: the record stands again, but as the reader's own
   *  restoring and not as this Run's, so the notice lets it go instead of saying it was taken back. */
  release(runId: string, changeId: string): void {
    const notice = this.#notices.get(runId);
    if (!notice?.changes.some((change) => change.changeId === changeId)) return;
    const kept = notice.changes.filter((change) => change.changeId !== changeId);
    notice.changes = kept.length === 0 ? NONE : kept;
    this.changed();
  }

  /** The owner changed their profile somewhere else (turned Memory off, cleared it, forgot a record): what was
   *  read may no longer be true. Each notice keeps showing what it shows until it is read again, which a turn
   *  in view does at once and one out of view does when it is drawn. */
  refresh(): void {
    for (const notice of this.#notices.values()) notice.read = false;
    this.changed();
  }

  /** Forget everything read, as signing out must, and stop every try still to come. */
  forget(): void {
    this.#abort.abort();
    this.#abort = new AbortController();
    for (const notice of this.#notices.values()) this.#stop(notice);
    this.#notices.clear();
    this.#stopListening?.();
    this.#stopListening = null;
    this.changed();
  }

  #notice(runId: string): RunNotice {
    let notice = this.#notices.get(runId);
    if (!notice) {
      notice = {
        changes: NONE, read: false, reading: false, live: false, agentWrote: false, announced: false, watch: null,
      };
      this.#notices.set(runId, notice);
    }
    return notice;
  }

  #stop(notice: RunNotice): void {
    if (notice.watch?.timer) clearTimeout(notice.watch.timer);
    notice.watch = null;
  }

  /** Schedule the watch's next try, measured from the moment the Run settled. */
  #arm(runId: string, notice: RunNotice, watch: Watch): void {
    const at = this.#delays[watch.attempt];
    if (at === undefined) {
      notice.watch = null;
      return;
    }
    const wait = at - (this.#delays[watch.attempt - 1] ?? 0);
    watch.timer = setTimeout(() => {
      watch.timer = null;
      void this.#try(runId, notice, watch);
    }, wait);
  }

  async #try(runId: string, notice: RunNotice, watch: Watch): Promise<void> {
    if (notice.watch !== watch) return;
    if (!this.#visibility.visible()) {
      // Looking at nothing is not worth a request: the try is made when the page is shown.
      watch.owed = true;
      this.#stopListening ??= this.#visibility.onChange(this.#visibilityChanged);
      return;
    }
    watch.owed = false;
    const found = await this.#look(runId, notice);
    if (notice.watch !== watch) return;
    watch.attempt += 1;
    // The pass writes its records together, so one of its records says it has reported. What the Agent left
    // is known when the Run settles and says nothing of the pass.
    if (found?.some((change) => change.writer === 'conversation')) {
      notice.watch = null;
      return;
    }
    this.#arm(runId, notice, watch);
  }

  readonly #visibilityChanged = (): void => {
    if (!this.#visibility.visible()) return;
    let owing = false;
    for (const [runId, notice] of this.#notices) {
      const watch = notice.watch;
      if (watch?.owed) void this.#try(runId, notice, watch);
      owing ||= watch?.owed ?? false;
    }
    if (!owing) {
      this.#stopListening?.();
      this.#stopListening = null;
    }
  };

  /** Read the Run's list and keep it; the list read, or null when the read failed or was let go. */
  async #look(runId: string, notice: RunNotice): Promise<readonly RunMemoryChange[] | null> {
    const abort = this.#abort;
    try {
      const fetched = await this.#read(runId, abort.signal);
      if (abort.signal.aborted) return null;
      this.#keep(notice, fetched);
      return fetched;
    } catch {
      return null;
    }
  }

  /** Keep a read, in the order the reader has been shown the rows: a change the reader took back stays where
   *  it was, and one that is gone from the server for any other reason goes. */
  #keep(notice: RunNotice, fetched: readonly RunMemoryChange[]): void {
    const fresh = new Map(fetched.map((change) => [change.changeId, change]));
    const kept: MemoryNoticeChange[] = [];
    for (const before of notice.changes) {
      const now = fresh.get(before.changeId);
      if (now) {
        kept.push({...now, undone: false});
        fresh.delete(before.changeId);
      } else if (before.undone) {
        kept.push(before);
      }
    }
    for (const change of fresh.values()) kept.push({...change, undone: false});
    const same = kept.length === notice.changes.length
      && kept.every((change, index) => (
        notice.changes[index]?.changeId === change.changeId && notice.changes[index]?.undone === change.undone
      ));
    notice.read = true;
    if (same) return;
    notice.changes = kept.length === 0 ? NONE : kept;
    this.changed();
  }
}
