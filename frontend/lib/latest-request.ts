// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/**
 * The one request that matters: the latest one started.
 *
 * A reader who asks again does not want the answer to the question before. Starting a request ends the
 * one under way, and a request that finds, when its answer lands, that it is no longer the latest drops
 * the answer instead of drawing it:
 *
 *     const signal = this.#latest.start();
 *     const page = await fetchPage(signal);
 *     if (!this.#latest.isCurrent(signal)) return;
 *
 * A failure from an aborted request is the same case, so the catch asks `isCurrent` too. A request that
 * ends by itself says so with `finish`, which is how `busy` goes back to false.
 */
export class LatestRequest {
  #controller: AbortController | null = null;
  #signal: AbortSignal | null = null;

  /** Whether a request is under way: started, and neither finished nor cancelled since. */
  get busy(): boolean {
    return this.#signal !== null;
  }

  /** Begin a request and end the one before it. The signal is what the request is made with and what
   *  `isCurrent` is asked about. */
  start(): AbortSignal {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    this.#signal = controller.signal;
    return controller.signal;
  }

  /** Whether the request is still the one that matters: nothing has replaced or cancelled it. */
  isCurrent(signal: AbortSignal): boolean {
    return this.#signal === signal;
  }

  /** The request is over. Whether it was the latest: a request that was replaced has nothing of its own
   *  left to settle. */
  finish(signal: AbortSignal): boolean {
    if (this.#signal !== signal) return false;
    this.#controller = null;
    this.#signal = null;
    return true;
  }

  /** End the request under way, if there is one, and start none. */
  cancel(): void {
    this.#controller?.abort();
    this.#controller = null;
    this.#signal = null;
  }
}
