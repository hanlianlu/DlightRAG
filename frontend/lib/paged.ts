// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {isAbortError} from './errors.ts';

export type PageLoadState = 'idle' | 'loading' | 'error';

/** How the latest next-page load ended; null while none has since the last start. */
export type NextPageOutcome = 'loaded' | 'failed' | null;

/** Any keyset page: what it holds is its owner's, its cursor the pager's. */
export interface KeysetPage {
  readonly nextCursor: string | null;
}

/** Load one page; a null cursor asks for the first page. */
export type KeysetPageLoader<P extends KeysetPage> = (
  cursor: string | null,
  signal: AbortSignal,
) => Promise<P>;

/** What a list shows of its pager: the next-page control reads only this. */
export interface KeysetPagerStatus {
  /** The latest load, the first page's included. */
  readonly state: PageLoadState;
  readonly hasOlder: boolean;
  readonly outcome: NextPageOutcome;
}

/** Single-flight, abort-guarded keyset pager over one forward cursor.
 *  The owner keeps its own collection: start() replaces it with the first
 *  page and loadNext() adds the next one, each through onPage. The pager owns
 *  the cursor, load state, request abort, and stale-flight rejection; onPage
 *  and onError fire only for results that are still current. An owner that
 *  loads its first page itself anchors the pager with reset(nextCursor).
 *  notify reports load progress; cancel and reset are silent, because the
 *  owner that drops work publishes its own change. */
export class KeysetPager<P extends KeysetPage> implements KeysetPagerStatus {
  #cursor: string | null = null;
  #state: PageLoadState = 'idle';
  #outcome: NextPageOutcome = null;
  #flight: Promise<void> | null = null;
  #controller: AbortController | null = null;
  readonly #load: KeysetPageLoader<P>;
  readonly #notify: () => void;

  constructor(load: KeysetPageLoader<P>, notify: () => void = () => {}) {
    this.#load = load;
    this.#notify = notify;
  }

  get state(): PageLoadState {
    return this.#state;
  }

  get hasOlder(): boolean {
    return this.#cursor !== null;
  }

  get outcome(): NextPageOutcome {
    return this.#outcome;
  }

  /** A copy of this status that later loads leave unchanged. */
  snapshot(): KeysetPagerStatus {
    return {state: this.#state, hasOlder: this.hasOlder, outcome: this.#outcome};
  }

  /** Drop in-flight work and keep the cursor. */
  cancel(): void {
    this.#controller?.abort();
    this.#controller = null;
    this.#flight = null;
    this.#state = 'idle';
    this.#outcome = null;
  }

  /** Drop in-flight work and re-anchor the cursor (null clears older pages). */
  reset(cursor: string | null): void {
    this.cancel();
    this.#cursor = cursor;
  }

  /** Load the first page in place of everything before it; the newest start wins. */
  start(onPage: (page: P) => void, onError: (error: unknown) => void = () => {}): Promise<void> {
    this.reset(null);
    return this.#fetch(null, onPage, onError);
  }

  /** Load the next page; resolves with any concurrent flight. */
  loadNext(onPage: (page: P) => void, onError: (error: unknown) => void = () => {}): Promise<void> {
    if (this.#flight !== null) return this.#flight;
    if (this.#cursor === null) return Promise.resolve();
    const flight = this.#fetch(this.#cursor, onPage, onError);
    this.#flight = flight;
    void flight.finally(() => {
      if (this.#flight === flight) this.#flight = null;
    });
    return flight;
  }

  async #fetch(
    cursor: string | null,
    onPage: (page: P) => void,
    onError: (error: unknown) => void,
  ): Promise<void> {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    this.#state = 'loading';
    this.#outcome = null;
    this.#notify();
    const next = cursor !== null;
    try {
      const page = await this.#load(cursor, controller.signal);
      // cancel() and every newer load replace the controller.
      if (controller !== this.#controller) return;
      this.#cursor = page.nextCursor;
      this.#state = 'idle';
      if (next) this.#outcome = 'loaded';
      onPage(page);
    } catch (error) {
      if (controller !== this.#controller) return;
      if (isAbortError(error)) {
        this.#state = 'idle';
      } else {
        this.#state = 'error';
        if (next) this.#outcome = 'failed';
        onError(error);
      }
    } finally {
      if (this.#controller === controller) this.#controller = null;
    }
    this.#notify();
  }
}
