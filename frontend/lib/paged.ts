// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {isAbortError} from './errors.ts';

export type PageLoadState = 'idle' | 'loading' | 'error';

export interface KeysetPage<T> {
  items: T[];
  nextCursor: string | null;
}

/** Single-flight, generation-guarded keyset pager over one forward cursor.
 *  The owner keeps its own collection and merges pages through onPage; the
 *  pager owns cursor movement, load state, request abort, and stale-flight
 *  rejection. */
export class KeysetPager<T> {
  #cursor: string | null = null;
  #state: PageLoadState = 'idle';
  #flight: Promise<void> | null = null;
  #controller: AbortController | null = null;
  #generation = 0;
  readonly #load: (cursor: string, signal: AbortSignal) => Promise<KeysetPage<T>>;
  readonly #notify: () => void;

  constructor(
    load: (cursor: string, signal: AbortSignal) => Promise<KeysetPage<T>>,
    notify: () => void = () => {},
  ) {
    this.#load = load;
    this.#notify = notify;
  }

  get state(): PageLoadState {
    return this.#state;
  }

  get hasOlder(): boolean {
    return this.#cursor !== null;
  }

  /** Drop in-flight work and re-anchor the cursor (null clears older pages). */
  reset(cursor: string | null): void {
    this.#controller?.abort();
    this.#controller = null;
    this.#generation += 1;
    this.#flight = null;
    this.#state = 'idle';
    this.#cursor = cursor;
    this.#notify();
  }

  /** Load the next page; resolves with any concurrent flight. onPage and
   *  onError fire only for results that are still current. */
  loadNext(
    onPage: (page: KeysetPage<T>) => void,
    onError: () => void = () => {},
  ): Promise<void> {
    if (this.#flight !== null) return this.#flight;
    if (this.#cursor === null) return Promise.resolve();
    const flight = this.#loadNextPage(onPage, onError);
    this.#flight = flight;
    void flight.finally(() => {
      if (this.#flight === flight) this.#flight = null;
    });
    return flight;
  }

  async #loadNextPage(
    onPage: (page: KeysetPage<T>) => void,
    onError: () => void,
  ): Promise<void> {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    const generation = this.#generation;
    const cursor = this.#cursor;
    if (cursor === null) return; // guarded by loadNext; kept for the type
    this.#state = 'loading';
    this.#notify();
    try {
      const page = await this.#load(cursor, controller.signal);
      if (
        controller !== this.#controller
        || generation !== this.#generation
        || this.#cursor !== cursor
      ) {
        if (controller === this.#controller) this.#state = 'idle';
        return;
      }
      this.#cursor = page.nextCursor;
      this.#state = 'idle';
      onPage(page);
    } catch (error) {
      if (controller !== this.#controller || generation !== this.#generation) return;
      if (isAbortError(error)) return;
      this.#state = 'error';
      onError();
    } finally {
      if (this.#controller === controller) this.#controller = null;
    }
    this.#notify();
  }
}
