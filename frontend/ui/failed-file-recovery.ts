// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Workspace-scoped failed-document visibility and durable recovery control. */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  FilesApiError,
  getFailedFileRetryStatus,
  getFailedFiles,
  startFailedFileRetry,
  type WebFailedFilesPage,
  type WebFailedRecoveryJob,
} from '../api/files.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement} from '../lib/lit-host.ts';
import {modalResult} from './modal.ts';
import type {ToastRequestDetail} from './toast.ts';

const RECOVERY_POLL_INTERVAL_MS = 2000;
const ACTIVE_RECOVERY_STATES = new Set(['queued', 'running']);

type PageLoadState = 'idle' | 'loading' | 'error';
type ActiveRecoveryJob = WebFailedRecoveryJob & {status: 'queued' | 'running'};

function isRecoveryActive(
  job: WebFailedRecoveryJob | null | undefined,
): job is ActiveRecoveryJob {
  return job != null && ACTIVE_RECOVERY_STATES.has(job.status);
}

function normalizePage(page: WebFailedFilesPage): WebFailedFilesPage {
  return {
    ...page,
    failed: Array.isArray(page.failed) ? page.failed : [],
    next_cursor: page.next_cursor ?? null,
    active_recovery: page.active_recovery ?? null,
  };
}

function failureTime(value: string): string {
  if (!value) return '';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return '';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed);
}

function recoveryRequestError(error: unknown, fallback: string): string {
  if (!(error instanceof FilesApiError)) return fallback;
  if (error.status === 401 || error.status === 403) {
    return msg('You do not have permission to recover documents in this workspace.', {
      id: 'inspectorFiles.recovery.forbidden',
    });
  }
  if (error.status === 409) {
    return msg('This workspace is no longer available.', {
      id: 'inspectorFiles.recovery.workspaceGone',
    });
  }
  return fallback;
}

export class DlFailedFileRecovery extends LightElement {
  static properties = {
    workspace: {attribute: false},
    active: {attribute: false},
    page: {state: true},
    loading: {state: true},
    error: {state: true},
    loadMoreState: {state: true},
    recovery: {state: true},
    recoveryPending: {state: true},
  };

  declare workspace: string;
  declare active: boolean;
  declare page: WebFailedFilesPage | null;
  declare loading: boolean;
  declare error: string | null;
  declare loadMoreState: PageLoadState;
  declare recovery: WebFailedRecoveryJob | null;
  declare recoveryPending: boolean;

  #listController: AbortController | null = null;
  #loadMoreController: AbortController | null = null;
  #mutationController: AbortController | null = null;
  #modalController: AbortController | null = null;
  #pollController: AbortController | null = null;
  #pollTimer: number | null = null;
  #contextGeneration = 0;
  #listGeneration = 0;
  #retryTrigger: HTMLElement | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.workspace = '';
    this.active = false;
    this.page = null;
    this.loading = false;
    this.error = null;
    this.loadMoreState = 'idle';
    this.recovery = null;
    this.recoveryPending = false;
  }

  override disconnectedCallback(): void {
    this.pause();
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('workspace')) {
      this.#cancelContext();
      this.page = null;
      this.recovery = null;
      this.error = null;
      this.loadMoreState = 'idle';
    }
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('workspace') || changed.has('active')) {
      if (this.active && this.workspace) void this.refresh();
      else this.pause();
    }
  }

  async refresh(showLoading = true): Promise<void> {
    if (!this.active || !this.workspace) return;
    const workspace = this.workspace;
    const generation = ++this.#listGeneration;
    const observedRecovery = this.recovery;
    this.#listController?.abort();
    this.#loadMoreController?.abort();
    this.#stopPolling();
    const controller = new AbortController();
    this.#listController = controller;
    if (showLoading) this.loading = true;
    this.error = null;
    this.loadMoreState = 'idle';
    try {
      const response = await getFailedFiles(workspace, null, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      const page = normalizePage(response);
      this.page = page;
      const pageRecovery = page.active_recovery;
      const liveRecovery = this.recovery;
      // A mutation or poll that completed after this request began owns newer
      // state even when the delayed page still reports an older non-null job.
      const recovery = liveRecovery !== observedRecovery
        ? liveRecovery
        : pageRecovery ?? (isRecoveryActive(liveRecovery) ? liveRecovery : null);
      this.recovery = recovery;
      if (isRecoveryActive(recovery)) this.#schedulePoll(workspace, recovery.job_id);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace, generation)) return;
      this.page = null;
      this.error = recoveryRequestError(
        error,
        msg('Document status is temporarily unavailable.', {
          id: 'inspectorFiles.recovery.loadFailed',
        }),
      );
      const liveRecovery = this.recovery;
      if (isRecoveryActive(liveRecovery)) {
        this.#schedulePoll(workspace, liveRecovery.job_id);
      }
    } finally {
      if (this.#listController === controller) {
        this.#listController = null;
        this.loading = false;
      }
    }
  }

  pause(): void {
    this.#cancelContext();
  }

  #cancelContext(): void {
    this.#contextGeneration += 1;
    this.#listGeneration += 1;
    this.#listController?.abort();
    this.#listController = null;
    this.#loadMoreController?.abort();
    this.#loadMoreController = null;
    this.#mutationController?.abort();
    this.#mutationController = null;
    this.#modalController?.abort();
    this.#modalController = null;
    this.#stopPolling();
    this.loading = false;
    this.recoveryPending = false;
    this.#retryTrigger = null;
  }

  protected override render(): TemplateResult | typeof nothing {
    if (!this.active) return nothing;
    if (this.loading && this.page === null) {
      return html`<div class="failed-files-loading" role="status">
        ${msg('Checking document status…', {id: 'inspectorFiles.recovery.loading'})}
      </div>`;
    }
    if (this.error && this.page === null) {
      return html`
        <div class="failed-files-unavailable" role="alert">
          <span>${this.error}</span>
          <button class="failed-files-retry-load" type="button"
                  @click=${() => { void this.refresh(); }}> 
            ${msg('Try again', {id: 'inspectorFiles.recovery.tryAgain'})}
          </button>
        </div>
      `;
    }

    const failed = this.page?.failed ?? [];
    const recoveryActive = isRecoveryActive(this.recovery);
    const recoveryBusy = this.recovery !== null;
    if (failed.length === 0 && !recoveryActive) return nothing;
    const count = `${failed.length}${this.page?.next_cursor ? '+' : ''}`;
    const heading = recoveryActive
      ? msg('Document recovery in progress', {id: 'inspectorFiles.recovery.inProgress'})
      : failed.length === 1 && !this.page?.next_cursor
        ? msg('1 document needs attention', {id: 'inspectorFiles.recovery.oneNeedsAttention'})
        : msg(str`${count} documents need attention`, {
          id: 'inspectorFiles.recovery.nNeedsAttention',
        });

    return html`
      <div class="failed-file-recovery-shell">
        <details class="failed-file-recovery">
          <summary class="failed-file-recovery-summary">
            <span class="failed-file-recovery-mark" aria-hidden="true">!</span>
            <span class="failed-file-recovery-copy">
              <strong>${heading}</strong>
              <span>${recoveryActive
                ? msg('You can close this panel while recovery continues.', {
                  id: 'inspectorFiles.recovery.continues',
                })
                : msg('Review failed documents', {id: 'inspectorFiles.recovery.review'})}</span>
            </span>
          </summary>
          <div class="failed-file-recovery-body">
            <ul class="failed-file-list"
                aria-label=${msg('Documents needing attention', {
                  id: 'inspectorFiles.recovery.listAria',
                })}>
              ${repeat(
                failed,
                (item) => item.document_id,
                (item) => html`
                  <li>
                    <details class="failed-file-row">
                      <summary class="failed-file-row-summary">
                        <span class="failed-file-row-mark" aria-hidden="true">!</span>
                        <span class="failed-file-row-copy">
                          <strong title=${item.file_name}>${item.file_name}</strong>
                          <span>${msg('Processing did not finish.', {
                            id: 'inspectorFiles.recovery.processingFailed',
                          })}</span>
                        </span>
                        <time datetime=${item.updated_at}>${failureTime(item.updated_at)}</time>
                      </summary>
                      <div class="failed-file-technical">
                        <span>${msg('Technical details', {
                          id: 'inspectorFiles.recovery.technicalDetails',
                        })}</span>
                        <code>${item.error || msg('No diagnostic details were provided.', {
                          id: 'inspectorFiles.recovery.noDetails',
                        })}</code>
                      </div>
                    </details>
                  </li>
                `,
              )}
            </ul>
            ${this.page?.next_cursor ? html`
              <div class="failed-file-more">
                <button class="failed-file-more-button" type="button"
                        ?disabled=${this.loadMoreState === 'loading'}
                        aria-busy=${this.loadMoreState === 'loading' ? 'true' : 'false'}
                        @click=${() => { void this.#loadMore(); }}>
                  ${this.loadMoreState === 'error'
                    ? msg('Retry loading more failed documents', {
                      id: 'inspectorFiles.recovery.retryLoadMore',
                    })
                    : msg('Load more failed documents', {
                      id: 'inspectorFiles.recovery.loadMore',
                    })}
                </button>
              </div>
            ` : nothing}
            <div class="failed-file-recovery-note">
              ${msg('Retry uses stored sources. Parsing, embedding, and model usage may apply.', {
                id: 'inspectorFiles.recovery.usageNotice',
              })}
            </div>
          </div>
        </details>
        <button class="dl-btn failed-file-retry" type="button"
                ?disabled=${recoveryBusy || this.recoveryPending || failed.length === 0}
                aria-busy=${this.recoveryPending ? 'true' : 'false'}
                @click=${this.#confirmRetry}>
          ${recoveryActive
            ? msg('Running…', {id: 'inspectorFiles.recovery.running'})
            : msg('Retry all', {id: 'inspectorFiles.recovery.retryAll'})}
        </button>
      </div>
      ${this.#confirmDialog()}
    `;
  }

  async #loadMore(): Promise<void> {
    const page = this.page;
    const cursor = page?.next_cursor;
    const workspace = this.workspace;
    if (!cursor || !workspace || this.loadMoreState === 'loading') return;
    this.#loadMoreController?.abort();
    const controller = new AbortController();
    this.#loadMoreController = controller;
    const generation = this.#contextGeneration;
    const observedRecovery = this.recovery;
    this.loadMoreState = 'loading';
    try {
      const response = await getFailedFiles(workspace, cursor, controller.signal);
      if (
        controller !== this.#loadMoreController
        || generation !== this.#contextGeneration
        || workspace !== this.workspace
        || this.page?.next_cursor !== cursor
      ) return;
      const older = normalizePage(response);
      const seen = new Set(page.failed.map((item) => item.document_id));
      const appended = older.failed.filter((item) => !seen.has(item.document_id));
      const liveRecovery = this.recovery;
      const recovery = liveRecovery !== observedRecovery
        ? liveRecovery
        : older.active_recovery ?? liveRecovery;
      this.page = {
        ...page,
        failed: [...page.failed, ...appended],
        next_cursor: older.next_cursor,
        active_recovery: recovery,
      };
      this.recovery = recovery;
      if (isRecoveryActive(recovery)) {
        this.#schedulePoll(workspace, recovery.job_id);
      }
      this.loadMoreState = 'idle';
    } catch (error) {
      if (isAbortError(error) || controller !== this.#loadMoreController) return;
      if (error instanceof FilesApiError && [401, 403, 409].includes(error.status)) {
        this.#stopPolling();
        this.page = null;
        this.recovery = null;
        this.error = recoveryRequestError(
          error,
          msg('Document status is temporarily unavailable.', {
            id: 'inspectorFiles.recovery.loadFailed',
          }),
        );
        this.loadMoreState = 'idle';
        return;
      }
      this.loadMoreState = 'error';
    } finally {
      if (this.#loadMoreController === controller) this.#loadMoreController = null;
    }
  }

  #confirmRetry = async (event: Event): Promise<void> => {
    const trigger = event.currentTarget as HTMLButtonElement;
    const dialog = this.querySelector<HTMLDialogElement>('#retry-failed-files-dialog');
    if (!dialog || this.recoveryPending || isRecoveryActive(this.recovery)) return;
    this.#retryTrigger = trigger;
    this.#modalController?.abort();
    const controller = new AbortController();
    this.#modalController = controller;
    const result = await modalResult(
      this,
      dialog,
      () => this.#restoreRetryFocus(),
      controller.signal,
    );
    if (this.#modalController === controller) this.#modalController = null;
    if (result !== 'retry') return;
    await this.#startRetry();
  };

  async #startRetry(): Promise<void> {
    const workspace = this.workspace;
    if (!workspace || this.recoveryPending || this.recovery !== null) return;
    this.#mutationController?.abort();
    const controller = new AbortController();
    this.#mutationController = controller;
    const generation = this.#contextGeneration;
    this.recoveryPending = true;
    try {
      const job = await startFailedFileRetry(workspace, controller.signal);
      if (!this.#mutationCurrent(controller, workspace, generation)) return;
      this.recovery = job;
      if (isRecoveryActive(job)) {
        this.#requestToast({
          message: msg('Document recovery started.', {id: 'inspectorFiles.recovery.started'}),
          duration: 3000,
        });
        this.#schedulePoll(workspace, job.job_id);
      } else {
        await this.#settleRecovery(job);
      }
    } catch (error) {
      if (isAbortError(error) || !this.#mutationCurrent(controller, workspace, generation)) return;
      this.#requestToast({
        message: recoveryRequestError(
          error,
          msg('Document recovery could not be started.', {
            id: 'inspectorFiles.recovery.startFailed',
          }),
        ),
        duration: 3000,
      });
    } finally {
      if (this.#mutationController === controller) {
        this.#mutationController = null;
        this.recoveryPending = false;
      }
    }
  }

  async #poll(workspace: string, jobId: string): Promise<void> {
    const controller = new AbortController();
    this.#pollController = controller;
    try {
      const job = await getFailedFileRetryStatus(workspace, jobId, controller.signal);
      if (
        controller !== this.#pollController
        || workspace !== this.workspace
        || !this.active
        || !this.isConnected
      ) return;
      this.recovery = job;
      if (isRecoveryActive(job)) {
        this.#schedulePoll(workspace, jobId);
        return;
      }
      await this.#settleRecovery(job);
    } catch (error) {
      if (isAbortError(error)) return;
      if (
        controller !== this.#pollController
        || workspace !== this.workspace
        || !this.active
        || !this.isConnected
      ) return;
      if (error instanceof FilesApiError && [401, 403, 404, 409].includes(error.status)) {
        this.page = null;
        this.recovery = null;
        this.error = recoveryRequestError(
          error,
          msg('Document recovery status is no longer available.', {
            id: 'inspectorFiles.recovery.statusUnavailable',
          }),
        );
        return;
      }
      this.#schedulePoll(workspace, jobId);
    } finally {
      if (this.#pollController === controller) this.#pollController = null;
    }
  }

  async #settleRecovery(job: WebFailedRecoveryJob): Promise<void> {
    const workspace = this.workspace;
    const generation = this.#contextGeneration;
    this.recovery = job;
    await this.refresh(false);
    if (
      workspace !== this.workspace
      || generation !== this.#contextGeneration
      || !this.active
      || !this.isConnected
    ) return;
    this.dispatchEvent(new CustomEvent('dl-failed-file-recovery-complete', {
      bubbles: true,
      composed: true,
    }));
    if (job.status === 'succeeded' || job.status === 'partial') {
      this.#requestToast({
        message: job.failed > 0
          ? msg(str`Recovery finished: ${job.succeeded} succeeded, ${job.failed} still failed.`, {
            id: 'inspectorFiles.recovery.finishedPartial',
          })
          : msg(str`Recovery finished: ${job.succeeded} succeeded.`, {
            id: 'inspectorFiles.recovery.finished',
          }),
        duration: 5000,
      });
      return;
    }
    this.#requestToast({
      message: msg('Document recovery failed.', {id: 'inspectorFiles.recovery.failed'}),
      duration: 4000,
    });
  }

  #schedulePoll(workspace: string, jobId: string): void {
    this.#stopPolling();
    this.#pollTimer = window.setTimeout(() => {
      this.#pollTimer = null;
      void this.#poll(workspace, jobId);
    }, RECOVERY_POLL_INTERVAL_MS);
  }

  #stopPolling(): void {
    if (this.#pollTimer !== null) window.clearTimeout(this.#pollTimer);
    this.#pollTimer = null;
    this.#pollController?.abort();
    this.#pollController = null;
  }

  #isCurrent(controller: AbortController, workspace: string, generation: number): boolean {
    return this.#listController === controller
      && workspace === this.workspace
      && generation === this.#listGeneration
      && this.active;
  }

  #mutationCurrent(
    controller: AbortController,
    workspace: string,
    generation: number,
  ): boolean {
    return this.#mutationController === controller
      && workspace === this.workspace
      && generation === this.#contextGeneration
      && this.active;
  }

  #restoreRetryFocus(): void {
    const trigger = this.#retryTrigger;
    this.#retryTrigger = null;
    if (trigger?.isConnected) trigger.focus();
  }

  #confirmDialog(): TemplateResult {
    return html`
      <dialog id="retry-failed-files-dialog" class="confirm-dialog"
              aria-labelledby="retry-failed-files-title"
              aria-describedby="retry-failed-files-message">
        <form method="dialog">
          <h2 id="retry-failed-files-title">${msg('Retry failed documents?', {
            id: 'inspectorFiles.recovery.confirmTitle',
          })}</h2>
          <p id="retry-failed-files-message">${msg(
            str`All failed documents in workspace “${this.workspace}” will be processed again from their stored sources. This can take a while and may use parsing, embedding, and model capacity.`,
            {id: 'inspectorFiles.recovery.confirmBody'},
          )}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">
              ${msg('Cancel', {id: 'inspectorFiles.recovery.cancel'})}
            </button>
            <button type="submit" value="retry">
              ${msg('Retry all', {id: 'inspectorFiles.recovery.confirmRetry'})}
            </button>
          </div>
        </form>
      </dialog>
    `;
  }

  #requestToast(detail: ToastRequestDetail): void {
    this.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail,
      bubbles: true,
      composed: true,
    }));
  }
}

customElements.define('dl-failed-file-recovery', DlFailedFileRecovery);

declare global {
  interface HTMLElementTagNameMap {
    'dl-failed-file-recovery': DlFailedFileRecovery;
  }
}
