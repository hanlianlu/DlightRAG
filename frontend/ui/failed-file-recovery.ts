// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Workspace-scoped failed-document visibility and durable recovery control. */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {corpusRunActive} from '../api/corpus-runs.ts';
import {
  getFailedFiles,
  startFailedFileRetry,
  type WebFailedFilesPage,
} from '../api/files.ts';
import {ApiError} from '../api/wire.ts';
import {CorpusRunTracker, type TrackedCorpusRun} from '../lib/corpus-run-tracker.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement} from '../lib/lit-host.ts';
import {corpusRepairNotice, resumeRepairLabel, resumeRepairResult} from './corpus-repair.ts';
import {requestToast} from './toast-request.ts';
import {modalResult} from './modal.ts';
import recoveryStyles from '../styles/failed-file-recovery.module.css';
import {FailedFileRecoverySession} from './failed-file-recovery-session.ts';

type PageLoadState = 'idle' | 'loading' | 'error';

function normalizePage(page: WebFailedFilesPage): WebFailedFilesPage {
  return {
    ...page,
    failed: Array.isArray(page.failed) ? page.failed : [],
    nextCursor: page.nextCursor ?? null,
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

/** HTTP status of a Files or Corpus Run refusal; null for any other failure. */
function refusalStatus(error: unknown): number | null {
  return error instanceof ApiError ? error.status : null;
}

function recoveryRequestError(error: unknown, fallback: string): string {
  const status = refusalStatus(error);
  if (status === null) return fallback;
  if (status === 401 || status === 403) {
    return msg('You do not have permission to recover documents in this workspace.', {
      id: 'inspectorFiles.recovery.forbidden',
    });
  }
  if (status === 409) {
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
    recoveryPending: {state: true},
  };

  declare workspace: string;
  declare active: boolean;
  declare page: WebFailedFilesPage | null;
  declare loading: boolean;
  declare error: string | null;
  declare loadMoreState: PageLoadState;
  declare recoveryPending: boolean;

  readonly #session = new FailedFileRecoverySession();
  readonly #tracker = new CorpusRunTracker({
    onChange: () => { this.requestUpdate(); },
    onSettled: (run) => { void this.#settleRecovery(run); },
    onLost: (error) => {
      this.page = null;
      this.error = recoveryRequestError(
        error,
        msg('Document recovery status is no longer available.', {
          id: 'inspectorFiles.recovery.statusUnavailable',
        }),
      );
    },
  });
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
    this.recoveryPending = false;
  }

  override disconnectedCallback(): void {
    this.pause();
    super.disconnectedCallback();
  }

  /** The accepted recovery Run, kept after it settles. */
  get recovery(): TrackedCorpusRun | null {
    return this.#tracker.run;
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('workspace')) {
      this.#session.cancelContext();
      this.#tracker.clear();
      this.loading = false;
      this.recoveryPending = false;
      this.#retryTrigger = null;
      this.page = null;
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
    const {controller, generation} = this.#session.startList();
    if (showLoading) this.loading = true;
    this.error = null;
    this.loadMoreState = 'idle';
    try {
      const response = await getFailedFiles(workspace, null, controller.signal);
      if (!this.#session.isListCurrent(controller, workspace, this.workspace, generation, this.active)) return;
      this.page = normalizePage(response);
    } catch (error) {
      if (isAbortError(error) || !this.#session.isListCurrent(controller, workspace, this.workspace, generation, this.active)) return;
      this.page = null;
      this.error = recoveryRequestError(
        error,
        msg('Document status is temporarily unavailable.', {
          id: 'inspectorFiles.recovery.loadFailed',
        }),
      );
    } finally {
      if (this.#session.finishList(controller)) {
        this.loading = false;
        if (this.active && this.isConnected) this.#tracker.wake();
      }
    }
  }

  pause(): void {
    this.#session.cancelContext();
    this.#tracker.pause();
    this.loading = false;
    this.recoveryPending = false;
    this.#retryTrigger = null;
  }

  protected override render(): TemplateResult | typeof nothing {
    if (!this.active) return nothing;
    if (this.loading && this.page === null) {
      return html`<div class=${recoveryStyles['failed-files-loading']} role="status">
        ${msg('Checking document status…', {id: 'inspectorFiles.recovery.loading'})}
      </div>`;
    }
    if (this.error && this.page === null) {
      return html`
        <div class=${recoveryStyles['failed-files-unavailable']} role="alert">
          <span>${this.error}</span>
          <button class=${recoveryStyles['failed-files-retry-load']} type="button"
                  @click=${() => { void this.refresh(); }}> 
            ${msg('Try again', {id: 'inspectorFiles.recovery.tryAgain'})}
          </button>
        </div>
      `;
    }

    const failed = this.page?.failed ?? [];
    const repairRun = this.#tracker.waitingForRepair ? this.recovery : null;
    const repairWaiting = repairRun !== null;
    const recoveryActive = this.#tracker.active && !repairWaiting;
    const pending = this.recoveryPending || this.#tracker.resuming;
    if (failed.length === 0 && !recoveryActive && !repairWaiting) return nothing;
    const count = `${failed.length}${this.page?.nextCursor ? '+' : ''}`;
    const heading = repairWaiting
      ? msg('Corpus repair confirmation required', {
        id: 'inspectorFiles.recovery.repairRequired',
      })
      : recoveryActive
        ? msg('Document recovery in progress', {id: 'inspectorFiles.recovery.inProgress'})
      : failed.length === 1 && !this.page?.nextCursor
        ? msg('1 document needs attention', {id: 'inspectorFiles.recovery.oneNeedsAttention'})
        : msg(str`${count} documents need attention`, {
          id: 'inspectorFiles.recovery.nNeedsAttention',
        });

    return html`
      <div class=${recoveryStyles['failed-file-recovery-shell']}>
        <details class=${recoveryStyles['failed-file-recovery']}>
          <summary class=${recoveryStyles['failed-file-recovery-summary']}>
            <span class=${recoveryStyles['failed-file-recovery-mark']} aria-hidden="true">!</span>
            <span class=${recoveryStyles['failed-file-recovery-copy']}>
              <strong>${heading}</strong>
              <span>${recoveryActive
                ? msg('You can close this panel while recovery continues.', {
                  id: 'inspectorFiles.recovery.continues',
                })
                : msg('Review failed documents', {id: 'inspectorFiles.recovery.review'})}</span>
            </span>
          </summary>
          <div class=${recoveryStyles['failed-file-recovery-body']}>
            <ul class=${recoveryStyles['failed-file-list']}
                aria-label=${msg('Documents needing attention', {
                  id: 'inspectorFiles.recovery.listAria',
                })}>
              ${repeat(
                failed,
                (item) => item.documentId,
                (item) => html`
                  <li>
                    <details class=${recoveryStyles['failed-file-row']}>
                      <summary class=${recoveryStyles['failed-file-row-summary']}>
                        <span class=${recoveryStyles['failed-file-row-mark']} aria-hidden="true">!</span>
                        <span class=${recoveryStyles['failed-file-row-copy']}>
                          <strong title=${item.fileName}>${item.fileName}</strong>
                          <span>${msg('Processing did not finish.', {
                            id: 'inspectorFiles.recovery.processingFailed',
                          })}</span>
                        </span>
                        <time datetime=${item.updatedAt}>${failureTime(item.updatedAt)}</time>
                      </summary>
                      <div class=${recoveryStyles['failed-file-technical']}>
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
            ${this.page?.nextCursor ? html`
              <div class=${recoveryStyles['failed-file-more']}>
                <button class=${recoveryStyles['failed-file-more-button']} type="button"
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
            ${repairRun ? html`
              <div class=${recoveryStyles['failed-file-recovery-note']} role="status">
                ${corpusRepairNotice(repairRun)}
              </div>
            ` : html`
              <div class=${recoveryStyles['failed-file-recovery-note']}>
                ${msg('Retry uses stored sources. Parsing, embedding, and model usage may apply.', {
                  id: 'inspectorFiles.recovery.usageNotice',
                })}
              </div>
            `}
          </div>
        </details>
        <button class=${`dl-btn ${recoveryStyles['failed-file-retry']}`} type="button"
                ?disabled=${recoveryActive || pending || (!repairWaiting && failed.length === 0)}
                aria-busy=${pending ? 'true' : 'false'}
                @click=${this.#confirmRetry}>
          ${repairWaiting
            ? resumeRepairLabel()
            : recoveryActive
              ? msg('Running…', {id: 'inspectorFiles.recovery.running'})
              : msg('Retry all', {id: 'inspectorFiles.recovery.retryAll'})}
        </button>
      </div>
      ${this.#confirmDialog()}
    `;
  }

  async #loadMore(): Promise<void> {
    const page = this.page;
    const cursor = page?.nextCursor;
    const workspace = this.workspace;
    if (!cursor || !workspace || this.loadMoreState === 'loading') return;
    const controller = this.#session.startLoadMore();
    const generation = this.#session.contextGeneration;
    this.loadMoreState = 'loading';
    try {
      const response = await getFailedFiles(workspace, cursor, controller.signal);
      if (
        !this.#session.isLoadMoreCurrent(controller, generation)
        || workspace !== this.workspace
        || this.page?.nextCursor !== cursor
      ) return;
      const older = normalizePage(response);
      const seen = new Set(page.failed.map((item) => item.documentId));
      const appended = older.failed.filter((item) => !seen.has(item.documentId));
      this.page = {
        ...page,
        failed: [...page.failed, ...appended],
        nextCursor: older.nextCursor,
      };
      this.loadMoreState = 'idle';
    } catch (error) {
      if (isAbortError(error) || !this.#session.isLoadMoreCurrent(controller, generation)) return;
      const status = refusalStatus(error);
      if (status !== null && [401, 403, 409].includes(status)) {
        this.#tracker.clear();
        this.page = null;
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
      this.#session.finishLoadMore(controller);
    }
  }

  #confirmRetry = async (event: Event): Promise<void> => {
    const trigger = event.currentTarget as HTMLButtonElement;
    if (this.#tracker.waitingForRepair) {
      await this.#resumeRepair();
      return;
    }
    const dialog = this.querySelector<HTMLDialogElement>('#retry-failed-files-dialog');
    if (!dialog || this.recoveryPending || this.#tracker.active) return;
    this.#retryTrigger = trigger;
    const controller = this.#session.startModal();
    const result = await modalResult(
      this,
      dialog,
      () => this.#restoreRetryFocus(),
      controller.signal,
    );
    this.#session.finishModal(controller);
    if (result !== 'retry') return;
    await this.#startRetry();
  };

  async #startRetry(): Promise<void> {
    const workspace = this.workspace;
    if (!workspace || this.recoveryPending || this.#tracker.active) return;
    const controller = this.#session.startMutation();
    const generation = this.#session.contextGeneration;
    this.recoveryPending = true;
    try {
      const run = await startFailedFileRetry(workspace, controller.signal);
      if (!this.#session.isMutationCurrent(controller, workspace, this.workspace, generation, this.active)) return;
      this.#tracker.follow(run);
      if (corpusRunActive(run)) {
        requestToast(this, {
          message: msg('Document recovery started.', {id: 'inspectorFiles.recovery.started'}),
          duration: 3000,
        });
      }
    } catch (error) {
      if (isAbortError(error) || !this.#session.isMutationCurrent(controller, workspace, this.workspace, generation, this.active)) return;
      requestToast(this, {
        message: recoveryRequestError(
          error,
          msg('Document recovery could not be started.', {
            id: 'inspectorFiles.recovery.startFailed',
          }),
        ),
        duration: 3000,
      });
    } finally {
      if (this.#session.finishMutation(controller)) this.recoveryPending = false;
    }
  }

  async #resumeRepair(): Promise<void> {
    const outcome = await this.#tracker.resume();
    if (outcome === 'stale') return;
    requestToast(this, {message: resumeRepairResult(outcome), duration: 3000});
  }

  async #settleRecovery(run: TrackedCorpusRun): Promise<void> {
    const workspace = this.workspace;
    const generation = this.#session.contextGeneration;
    await this.refresh(false);
    if (
      workspace !== this.workspace
      || generation !== this.#session.contextGeneration
      || !this.active
      || !this.isConnected
    ) return;
    this.dispatchEvent(new CustomEvent('dl-failed-file-recovery-complete', {
      bubbles: true,
      composed: true,
    }));
    requestToast(this, {
      message: run.status === 'succeeded'
        ? msg('Document recovery finished.', {id: 'inspectorFiles.recovery.finished'})
        : msg('Document recovery failed.', {id: 'inspectorFiles.recovery.failed'}),
      duration: 3000,
    });
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

}

customElements.define('dl-failed-file-recovery', DlFailedFileRecovery);

declare global {
  interface HTMLElementTagNameMap {
    'dl-failed-file-recovery': DlFailedFileRecovery;
  }
}
