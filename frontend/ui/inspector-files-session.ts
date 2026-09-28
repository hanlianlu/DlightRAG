// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort and mutation bookkeeping for one Files panel lifetime.

 *  The Feature still owns snapshot rendering and toasts, and CorpusRunTracker
 *  follows accepted Runs. This session is the single in-flight request.
 */

export class InspectorFilesSession {
  #request: AbortController | null = null;
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

  pause(): void {
    this.invalidateOlder();
    this.#request?.abort();
    this.#request = null;
  }
}
