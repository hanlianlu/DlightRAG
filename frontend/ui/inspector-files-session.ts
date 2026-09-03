// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort, poll, and mutation bookkeeping for one Files panel lifetime.

 *  The Feature still owns snapshot rendering and toasts. This session is the
 *  single in-flight request + poll loop so a third ad-hoc #poll cannot appear.
 */

const POLL_INTERVAL_MS = 2000;

export class InspectorFilesSession {
  #request: AbortController | null = null;
  #pollController: AbortController | null = null;
  #pollTimer: number | null = null;
  #olderController: AbortController | null = null;
  #olderGeneration = 0;
  #mutations = 0;

  get mutating(): boolean {
    return this.#mutations > 0;
  }

  get requestBusy(): boolean {
    return this.#request !== null;
  }

  get olderGeneration(): number {
    return this.#olderGeneration;
  }

  startRequest(): AbortController {
    this.#request?.abort();
    const controller = new AbortController();
    this.#request = controller;
    return controller;
  }

  isCurrent(controller: AbortController, workspace: string, current: string): boolean {
    return this.#request === controller && current === workspace;
  }

  finishRequest(controller: AbortController): boolean {
    if (this.#request !== controller) return false;
    this.#request = null;
    return true;
  }

  beginMutation(): void {
    this.#mutations += 1;
  }

  finishMutation(): void {
    this.#mutations = Math.max(0, this.#mutations - 1);
  }

  startOlder(): AbortController {
    this.#olderController?.abort();
    const controller = new AbortController();
    this.#olderController = controller;
    return controller;
  }

  isOlderCurrent(controller: AbortController, generation: number): boolean {
    return this.#olderController === controller && generation === this.#olderGeneration;
  }

  finishOlder(controller: AbortController): boolean {
    if (this.#olderController !== controller) return false;
    this.#olderController = null;
    return true;
  }

  invalidateOlder(): number {
    this.#olderController?.abort();
    this.#olderController = null;
    this.#olderGeneration += 1;
    return this.#olderGeneration;
  }

  schedulePoll(workspace: string, tick: (workspace: string) => void): void {
    this.stopPolling();
    this.#pollTimer = window.setTimeout(() => {
      this.#pollTimer = null;
      tick(workspace);
    }, POLL_INTERVAL_MS);
  }

  startPollRequest(): AbortController {
    const controller = new AbortController();
    this.#pollController = controller;
    return controller;
  }

  finishPollRequest(controller: AbortController): void {
    if (this.#pollController === controller) this.#pollController = null;
  }

  stopPolling(): void {
    if (this.#pollTimer !== null) window.clearTimeout(this.#pollTimer);
    this.#pollTimer = null;
    this.#pollController?.abort();
    this.#pollController = null;
  }

  pause(): void {
    this.invalidateOlder();
    this.#request?.abort();
    this.#request = null;
    this.stopPolling();
  }
}
