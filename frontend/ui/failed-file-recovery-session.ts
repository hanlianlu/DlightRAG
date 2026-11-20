// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort and generation bookkeeping for failed-document recovery.

 *  The Feature still owns page/recovery rendering and toasts; KeysetPager
 *  loads the failed-document pages and CorpusRunTracker follows the accepted
 *  recovery Run. This session is the retry request and its confirmation.
 */

import {LatestRequest} from '../lib/latest-request.ts';

export class FailedFileRecoverySession {
  readonly #mutation = new LatestRequest();
  readonly #modal = new LatestRequest();
  #contextGeneration = 0;

  get contextGeneration(): number {
    return this.#contextGeneration;
  }

  startMutation(): AbortSignal {
    return this.#mutation.start();
  }

  /** Cancelling the context drops the mutation, so a mutation that is still held belongs to this one. */
  isMutationCurrent(
    signal: AbortSignal,
    workspace: string,
    currentWorkspace: string,
    active: boolean,
  ): boolean {
    return this.#mutation.isCurrent(signal) && workspace === currentWorkspace && active;
  }

  finishMutation(signal: AbortSignal): boolean {
    return this.#mutation.finish(signal);
  }

  startModal(): AbortSignal {
    return this.#modal.start();
  }

  finishModal(signal: AbortSignal): boolean {
    return this.#modal.finish(signal);
  }

  cancelContext(): void {
    this.#contextGeneration += 1;
    this.#mutation.cancel();
    this.#modal.cancel();
  }
}
