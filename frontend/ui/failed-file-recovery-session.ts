// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort and generation bookkeeping for failed-document recovery.

 *  The Feature still owns page/recovery rendering and toasts; KeysetPager
 *  loads the failed-document pages and CorpusRunTracker follows the accepted
 *  recovery Run. This session is the retry request and its confirmation.
 */

export class FailedFileRecoverySession {
  #mutation: AbortController | null = null;
  #modal: AbortController | null = null;
  #contextGeneration = 0;

  get contextGeneration(): number {
    return this.#contextGeneration;
  }

  startMutation(): AbortController {
    this.#mutation?.abort();
    const controller = new AbortController();
    this.#mutation = controller;
    return controller;
  }

  isMutationCurrent(
    controller: AbortController,
    workspace: string,
    currentWorkspace: string,
    generation: number,
    active: boolean,
  ): boolean {
    return this.#mutation === controller
      && workspace === currentWorkspace
      && generation === this.#contextGeneration
      && active;
  }

  finishMutation(controller: AbortController): boolean {
    if (this.#mutation !== controller) return false;
    this.#mutation = null;
    return true;
  }

  startModal(): AbortController {
    this.#modal?.abort();
    const controller = new AbortController();
    this.#modal = controller;
    return controller;
  }

  finishModal(controller: AbortController): boolean {
    if (this.#modal !== controller) return false;
    this.#modal = null;
    return true;
  }

  cancelContext(): void {
    this.#contextGeneration += 1;
    this.#mutation?.abort();
    this.#mutation = null;
    this.#modal?.abort();
    this.#modal = null;
  }
}
