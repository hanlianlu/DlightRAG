// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort and mutation bookkeeping for one Files panel lifetime.

 *  The Feature still owns snapshot rendering and toasts; CorpusRunTracker
 *  follows accepted Runs and KeysetPager older pages. This session is the
 *  single in-flight list or mutation request.
 */

export class InspectorFilesSession {
  #request: AbortController | null = null;
  #mutations = 0;

  get mutating(): boolean {
    return this.#mutations > 0;
  }

  get requestBusy(): boolean {
    return this.#request !== null;
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

  pause(): void {
    this.#request?.abort();
    this.#request = null;
  }
}
