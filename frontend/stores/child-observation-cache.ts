// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One Run's loaded child pages, kept for as long as the dock that reads them is open.
 *
 * A child's page is the heaviest read the dock makes, and a reader walks the same children
 * over and over: looking at one, looking at the next, coming back. The page of a child that
 * has settled never changes — its Entries are immutable — so it is worth keeping until the
 * budget says otherwise; a child that is still running is not, so it is never kept and every
 * read of it stays live. The budget bounds memory whatever the reader does, and the child on
 * screen is the last one evicted, so moving between children costs one read each and moving
 * back costs none.
 */

import type {ActivityPage, ChildObservation} from '../api/conversations.ts';
import type {TracesSource} from './traces-source.ts';

/** How much of the reader's memory the kept pages may hold before the oldest one goes. */
const CHILD_OBSERVATION_BUDGET_BYTES = 8 * 1024 * 1024;

/** A TracesSource that keeps what it read, and can read a child ahead of the click. */
interface WarmTracesSource extends TracesSource {
  /** Read a child now, so the click that follows finds it: hovering or focusing its row calls this. */
  warm(childSessionId: string): void;
}

/** Roughly the bytes a kept page holds: the words it would put on screen are the payload. */
function weigh(...texts: (string | null | undefined)[]): number {
  let bytes = 512;
  for (const text of texts) bytes += typeof text === 'string' ? text.length : 0;
  return bytes;
}

/** What one activity page holds, measured the same way the local schema spells it. */
function weighActivity(page: ActivityPage): number {
  return weigh(...page.messages.map((message) => message.content));
}

/** The error a reader that stopped waiting sees, as the dock's own abort checks expect. */
function abortError(): Error {
  const error = new Error('aborted');
  error.name = 'AbortError';
  return error;
}

/** Wrap a source so it keeps what it read.
 *
 * `epoch` says how many times the reader has signed out: a page read under an earlier count is not the
 * current reader's, so the wrapper drops what it holds and what it still had in flight when the count
 * moves.
 */
function withChildObservationCache(
  source: TracesSource,
  budgetBytes: number,
  epoch: () => number,
): WarmTracesSource {
  const kept = new Map<string, {observation: ChildObservation; bytes: number}>();
  const inFlight = new Map<string, Promise<ChildObservation>>();
  const pages = new Map<string, {page: ActivityPage; bytes: number}>();
  const pageInFlight = new Map<string, Promise<ActivityPage>>();
  let held = 0;
  let pageHeld = 0;
  let readUnder = epoch();
  // What a reader is looking at now: eviction takes this last, so the page on screen stays.
  let shownChild: string | null = null;
  let shownPage: string | null = null;

  /** Drop what a reader who has since signed out read, and the reads they still had coming. */
  const forgetIfSignedOut = (): void => {
    if (readUnder === epoch()) return;
    readUnder = epoch();
    kept.clear();
    pages.clear();
    inFlight.clear();
    pageInFlight.clear();
    held = 0;
    pageHeld = 0;
  };

  const forget = (childSessionId: string): void => {
    const entry = kept.get(childSessionId);
    if (!entry) return;
    kept.delete(childSessionId);
    held -= entry.bytes;
  };

  const keep = (childSessionId: string, observation: ChildObservation): void => {
    if (observation.child.status === 'running') return; // still being written: never keep it
    forget(childSessionId);
    const bytes = weigh(observation.child.objective ?? '');
    kept.set(childSessionId, {observation, bytes});
    held += bytes;
    // The one read longest ago goes first, and the one on screen goes only if it is alone.
    while (held > budgetBytes && kept.size > 1) {
      const victim = [...kept.keys()].find((key) => key !== shownChild) ?? null;
      if (victim === null) break;
      forget(victim);
    }
  };

  /** Deliver a shared read to one caller: its own abort stops the waiting, never the read. */
  const deliver = <T>(
    flight: Promise<T>,
    signal: AbortSignal,
  ): Promise<T> => {
    if (signal.aborted) return Promise.reject(abortError());
    return new Promise((resolve, reject) => {
      const stop = (): void => reject(abortError());
      signal.addEventListener('abort', stop, {once: true});
      flight.then(
        (observation) => {
          signal.removeEventListener('abort', stop);
          resolve(observation);
        },
        (error: unknown) => {
          signal.removeEventListener('abort', stop);
          reject(error);
        },
      );
    });
  };

  const load = (childSessionId: string): Promise<ChildObservation> => {
    const bornUnder = readUnder;
    const existing = inFlight.get(childSessionId);
    if (existing) return existing;
    // The read runs on the cache's own signal: a reader that looks away mid-flight stops
    // waiting, but the page it asked for still lands here, so looking back costs nothing.
    // The cleanup rides the promise a reader awaits, so a settled flight is already gone
    // when the next look starts: two looks at a child still running are two reads, and one
    // look that a reader repeats while it is in the air is still one.
    const flight = source
      .observe(childSessionId, new AbortController().signal)
      .then((observation) => {
        if (bornUnder !== epoch()) return observation; // its reader signed out while it flew
        keep(childSessionId, observation);
        return observation;
      })
      .finally(() => {
        if (inFlight.get(childSessionId) === flight) inFlight.delete(childSessionId);
      });
    inFlight.set(childSessionId, flight);
    return flight;
  };

  const pageKey = (agent: string | null, cursor: string | null): string => `${agent ?? ''}|${cursor ?? ''}`;

  const loadPage = (agent: string | null, cursor: string | null, signal: AbortSignal): Promise<ActivityPage> => {
    const bornUnder = readUnder;
    const key = pageKey(agent, cursor);
    const existing = pageInFlight.get(key);
    if (existing) return existing;
    const flight = source
      .activity(agent, cursor, signal)
      .then((page) => {
        if (bornUnder !== epoch()) return page; // as above, for a transcript page
        const bytes = weighActivity(page);
        pages.set(key, {page, bytes});
        pageHeld += bytes;
        while (pageHeld > budgetBytes && pages.size > 1) {
          const victim = [...pages.keys()].find((key) => key !== shownPage) ?? null;
          if (victim === null) break;
          const gone = pages.get(victim);
          pages.delete(victim);
          pageHeld -= gone?.bytes ?? 0;
        }
        return page;
      })
      .finally(() => {
        if (pageInFlight.get(key) === flight) pageInFlight.delete(key);
      });
    pageInFlight.set(key, flight);
    return flight;
  };

  return {
    ...source,
    activity: (agent, cursor, signal) => {
      forgetIfSignedOut();
      const key = pageKey(agent, cursor);
      shownPage = key;
      const entry = pages.get(key);
      if (entry) {
        pages.delete(key);
        pages.set(key, entry);
        return Promise.resolve(entry.page);
      }
      return deliver(loadPage(agent, cursor, signal), signal);
    },
    observe: (childSessionId, signal) => {
      forgetIfSignedOut();
      shownChild = childSessionId;
      const entry = kept.get(childSessionId);
      if (entry) {
        // Re-inserting moves it to the young end, so what a reader returns to stays.
        kept.delete(childSessionId);
        kept.set(childSessionId, entry);
        return Promise.resolve(entry.observation);
      }
      return deliver(load(childSessionId), signal);
    },
    warm: (childSessionId) => {
      forgetIfSignedOut();
      if (kept.has(childSessionId)) return;
      // A prefetch nobody waits for: it fails quietly and the click reads again if it must.
      void load(childSessionId).catch(() => undefined);
    },
  };
}

/** How many Runs keep their kept pages: enough for the one on screen and the one before it. */
const RUNS_KEPT = 3;

/** The sources of the Runs a reader has opened, kept across the times each dock is opened and closed.
 *
 * A cache per opening would forget everything the moment the dock closes, which is exactly when a reader
 * comes back; a Run's pages therefore live beside the Run, and only the last few Runs keep any. The
 * page's bag holds the one of these, so what it keeps goes when the bag is reset on sign-out.
 */
export class KeptTraces {
  readonly #perRun = new Map<string, WarmTracesSource>();
  /** How many times a reader has signed out; a page read before the last one is not ours. */
  #signOuts = 0;

  /** The one source for a Run, built the first time it is asked for. */
  sourceFor(
    runId: string,
    build: () => TracesSource,
    budgetBytes: number = CHILD_OBSERVATION_BUDGET_BYTES,
  ): WarmTracesSource {
    const warm = this.#perRun.get(runId);
    if (warm) {
      this.#perRun.delete(runId);
      this.#perRun.set(runId, warm);
      return warm;
    }
    const created = withChildObservationCache(build(), budgetBytes, () => this.#signOuts);
    this.#perRun.set(runId, created);
    while (this.#perRun.size > RUNS_KEPT) {
      const oldest = this.#perRun.keys().next().value;
      if (oldest === undefined) break;
      this.#perRun.delete(oldest);
    }
    return created;
  }

  /** Forget every kept child page, and stop any source a dock still holds from serving them.
   *
   * Signing out must drop what the last sign-in read, or whoever signs in next on this tab
   * could be shown a child they never opened. */
  forget(): void {
    this.#signOuts += 1;
    this.#perRun.clear();
  }
}
