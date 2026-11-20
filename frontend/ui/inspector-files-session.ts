// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort and mutation bookkeeping for one Files panel lifetime.

 *  The Feature still owns snapshot rendering and toasts; CorpusRunTracker
 *  follows accepted Runs and KeysetPager older pages. This session is the
 *  single in-flight list or mutation request.
 */

import {LatestRequest} from '../lib/latest-request.ts';

export class InspectorFilesSession {
  readonly #request = new LatestRequest();
  #mutations = 0;

  get mutating(): boolean {
    return this.#mutations > 0;
  }

  get requestBusy(): boolean {
    return this.#request.busy;
  }

  startRequest(): AbortSignal {
    return this.#request.start();
  }

  isCurrent(signal: AbortSignal, workspace: string, current: string): boolean {
    return this.#request.isCurrent(signal) && current === workspace;
  }

  finishRequest(signal: AbortSignal): boolean {
    return this.#request.finish(signal);
  }

  beginMutation(): void {
    this.#mutations += 1;
  }

  finishMutation(): void {
    this.#mutations = Math.max(0, this.#mutations - 1);
  }

  pause(): void {
    this.#request.cancel();
  }
}
