// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Run continuation and child-roster dialogs as first-class Lit components. */

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import type {
  ChildControlReceipt,
  ChildObservation,
} from '../api/conversations.ts';
import {LightElement} from '../lib/lit-host.ts';
import {isAbortError} from '../lib/errors.ts';
import {KeysetPager} from '../lib/paged.ts';
import {publishModalState, showOwnedModal} from './modal.ts';

export type ContinuationKind = 'follow-up' | 'fork';

export interface ContinuationResult {
  kind: ContinuationKind;
  query: string | null;
}

export class DlContinuationDialog extends LightElement {
  static override properties = {kind: {state: true}};
  declare kind: ContinuationKind;

  constructor() {
    super();
    this.kind = 'follow-up';
    updateWhenLocaleChanges(this);
  }

  open(kind: ContinuationKind): void {
    this.kind = kind;
    void this.updateComplete.then(() => {
      const dialog = this.#dialog();
      if (!dialog) return;
      const input = this.#input();
      if (input) {
        input.value = '';
        window.requestAnimationFrame(() => input.focus());
      }
      showOwnedModal(this, dialog);
    });
  }

  #dialog(): HTMLDialogElement | null {
    return this.querySelector<HTMLDialogElement>('dialog');
  }

  #input(): HTMLTextAreaElement | null {
    return this.querySelector<HTMLTextAreaElement>('textarea');
  }

  override render() {
    const forking = this.kind === 'fork';
    const title = forking
      ? msg('Fork this answer', {id: 'runDialogs.forkTitle'})
      : msg('Follow up', {id: 'runDialogs.followUpTitle'});
    const note = forking
      ? msg('Start a new conversation from the same context. The previous answer is not carried over.', {
          id: 'runDialogs.forkNote',
        })
      : msg('Ask a follow-up question; the previous answer is included as context.', {
          id: 'runDialogs.followUpNote',
        });
    return html`
      <dialog class="confirm-dialog" aria-labelledby="dl-continuation-title"
              @close=${() => this.#emitClose()}>
        <form method="dialog">
          <h2 id="dl-continuation-title">${title}</h2>
          <p>${note}</p>
          <textarea class="dl-dialog-input" rows="3"
                    aria-label=${msg('Your question', {id: 'runDialogs.questionLabel'})}
                    placeholder=${msg('Ask a question…', {id: 'runDialogs.askPlaceholder'})}></textarea>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'runDialogs.cancel'})}</button>
            <button type="submit" value="continue" class="dl-btn">${msg('Continue', {id: 'runDialogs.continue'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #emitClose(): void {
    publishModalState(this);
    const dialog = this.#dialog();
    const value = dialog?.returnValue;
    this.dispatchEvent(
      new CustomEvent<ContinuationResult>('dl-continuation-result', {
        detail: {
          kind: this.kind,
          query: value === 'continue' ? (this.#input()?.value.trim() ?? null) : null,
        },
        bubbles: true,
        composed: true,
      }),
    );
  }
}

export interface ChildRosterEntry {
  status: string;
  objective?: string;
  childSessionId?: string;
  operationId?: string | null;
  operationStatus?: string | null;
  cancellationOrigin?: string | null;
  summary?: string | null;
  resultHandles?: readonly string[];
}

export type ChildRosterPageFetcher = (
  cursor: string | null,
  signal?: AbortSignal,
) => Promise<{children: ChildRosterEntry[]; nextCursor: string | null}>;

export interface ChildRosterActions {
  runId?: string;
  observe?: (childSessionId: string, signal?: AbortSignal) => Promise<ChildObservation>;
  control?: (
    childSessionId: string,
    action: 'steer' | 'continue' | 'cancel',
    content: string,
    reauthorizeUserCancelled?: boolean,
    signal?: AbortSignal,
  ) => Promise<ChildControlReceipt>;
  reply?: (
    requestId: string,
    content: string,
    signal?: AbortSignal,
  ) => Promise<ChildControlReceipt>;
}

function commandOutcome(error: unknown): string | null {
  if (error !== null && typeof error === 'object' && 'outcome' in error) {
    const outcome = (error as {outcome?: unknown}).outcome;
    return typeof outcome === 'string' && outcome ? outcome : null;
  }
  return null;
}

function commandStatus(error: unknown): number | null {
  if (error !== null && typeof error === 'object' && 'status' in error) {
    const status = (error as {status?: unknown}).status;
    return typeof status === 'number' ? status : null;
  }
  return null;
}

export class DlChildrenRoster extends LightElement {
  static override properties = {fetcher: {state: true}};
  declare fetcher: (() => Promise<ChildRosterEntry[]>) | null;

  constructor() {
    super();
    this.fetcher = null;
    updateWhenLocaleChanges(this);
  }

  #pageFetcher: ChildRosterPageFetcher | null = null;
  #actions: ChildRosterActions | null = null;
  #entries: ChildRosterEntry[] = [];
  #empty = true;
  #failed = false;
  #announcement = '';
  #controller: AbortController | null = null;
  #observeController: AbortController | null = null;
  #generation = 0;
  #selectedId: string | null = null;
  #observation: ChildObservation | null = null;
  #observeFailed = false;
  #stale = false;
  #outcome = '';
  #busy = false;
  #pager = new KeysetPager<ChildRosterEntry>(
    (cursor, signal) => this.#pageFetcher!(cursor, signal).then((page) => ({items: page.children, nextCursor: page.nextCursor})),
    () => this.requestUpdate(),
  );

  open(
    fetcher: () => Promise<ChildRosterEntry[]>,
    pageFetcher?: ChildRosterPageFetcher,
    actions?: ChildRosterActions,
  ): void {
    this.fetcher = fetcher;
    this.#pageFetcher = pageFetcher ?? null;
    this.#actions = actions ?? null;
    void this.updateComplete.then(() => {
      const dialog = this.querySelector<HTMLDialogElement>('dialog');
      if (dialog) showOwnedModal(this, dialog);
      void this.refresh();
    });
  }

  refreshIfFollowing(runId: string): void {
    const dialog = this.querySelector<HTMLDialogElement>('dialog');
    if (!dialog?.open || this.#actions?.runId !== runId) return;
    void this.refresh();
  }

  async refresh(): Promise<void> {
    const selected = this.#selectedId;
    this.#invalidate();
    this.#entries = [];
    this.#empty = true;
    this.#failed = false;
    this.#selectedId = selected;
    if (this.#pageFetcher) {
      await this.#loadFirstPage();
      await this.#restoreSelection();
      return;
    }
    let children: ChildRosterEntry[] = [];
    if (this.fetcher) {
      try {
        children = await this.fetcher();
      } catch {
        children = [];
      }
    }
    this.#entries = children;
    this.#empty = children.length === 0;
    await this.#restoreSelection();
    this.requestUpdate();
  }

  async #loadFirstPage(): Promise<void> {
    const controller = new AbortController();
    this.#controller = controller;
    const generation = this.#generation;
    try {
      const page = await this.#pageFetcher!(null, controller.signal);
      if (controller !== this.#controller || generation !== this.#generation) return;
      this.#entries = page.children;
      this.#pager.reset(page.nextCursor);
      this.#empty = page.children.length === 0;
      this.#failed = false;
      this.requestUpdate();
    } catch (error) {
      if (controller !== this.#controller || generation !== this.#generation) return;
      if (isAbortError(error)) return;
      this.#empty = false;
      this.#failed = true;
      this.requestUpdate();
    } finally {
      if (this.#controller === controller) this.#controller = null;
    }
  }

  loadOlderChildren(): Promise<void> {
    this.#announcement = msg('Loading older children…', {id: 'runDialogs.loadingOlderChildren'});
    return this.#pager.loadNext((page) => {
      const known = new Set(this.#entries.map((entry) => entry.childSessionId).filter(Boolean));
      const appended = page.items.filter((entry) => {
        if (!entry.childSessionId || known.has(entry.childSessionId)) return false;
        known.add(entry.childSessionId);
        return true;
      });
      this.#entries = [...this.#entries, ...appended];
      this.#announcement = appended.length === 1
        ? msg('Loaded 1 older child.', {id: 'runDialogs.loadedOneChild'})
        : msg(str`Loaded ${appended.length} older children.`, {id: 'runDialogs.loadedOlderChildren'});
    }, () => {
      this.#announcement = msg('Older children could not be loaded.', {
        id: 'runDialogs.olderChildrenFailed',
      });
    });
  }

  #invalidate(): void {
    this.#controller?.abort();
    this.#controller = null;
    this.#observeController?.abort();
    this.#observeController = null;
    this.#generation += 1;
    this.#announcement = '';
    this.#observation = null;
    this.#observeFailed = false;
    this.#stale = false;
    this.#outcome = '';
    this.#busy = false;
    this.#pager.reset(null);
  }

  #loadOlder = (): void => {
    void this.loadOlderChildren();
  };

  async #restoreSelection(): Promise<void> {
    const selected = this.#selectedId;
    if (!selected) return;
    const present = this.#entries.some((entry) => entry.childSessionId === selected);
    if (!present) {
      this.#selectedId = null;
      this.#stale = true;
      this.#observation = null;
      this.#announcement = msg('That child is no longer available.', {
        id: 'runDialogs.staleChild',
      });
      this.requestUpdate();
      return;
    }
    await this.#loadObservation(selected);
  }

  #selectChild = (childSessionId: string): void => {
    this.#selectedId = childSessionId;
    this.#stale = false;
    this.#outcome = '';
    void this.#loadObservation(childSessionId);
  };

  async #loadObservation(childSessionId: string): Promise<void> {
    const observe = this.#actions?.observe;
    if (!observe) return;
    this.#observeController?.abort();
    const controller = new AbortController();
    this.#observeController = controller;
    this.#observeFailed = false;
    this.#observation = null;
    this.requestUpdate();
    try {
      const observation = await observe(childSessionId, controller.signal);
      if (controller !== this.#observeController || this.#selectedId !== childSessionId) return;
      this.#observation = observation;
      this.#stale = false;
      this.requestUpdate();
    } catch (error) {
      if (controller !== this.#observeController || this.#selectedId !== childSessionId) return;
      if (isAbortError(error)) return;
      if (commandStatus(error) === 404) {
        this.#stale = true;
        this.#announcement = msg('That child is no longer available.', {
          id: 'runDialogs.staleChild',
        });
      } else {
        this.#observeFailed = true;
      }
      this.requestUpdate();
    } finally {
      if (this.#observeController === controller) this.#observeController = null;
    }
  }

  #onSteer = (event: Event): void => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const instruction = String(new FormData(form).get('instruction') || '').trim();
    void this.#runControl('steer', instruction, false, form);
  };

  #onContinue = (event: Event): void => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const data = new FormData(form);
    const instruction = String(data.get('instruction') || '').trim();
    const reauthorize = data.get('reauthorize') === 'on';
    void this.#runControl('continue', instruction, reauthorize, form);
  };

  #onCancel = (event: Event): void => {
    event.preventDefault();
    void this.#runControl('cancel', '', false);
  };

  #onReply = (event: Event): void => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const requestId = form.dataset.requestId || '';
    const content = String(new FormData(form).get('reply') || '').trim();
    void this.#runReply(requestId, content, form);
  };

  async #runControl(
    action: 'steer' | 'continue' | 'cancel',
    content: string,
    reauthorize: boolean,
    form?: HTMLFormElement,
  ): Promise<void> {
    const control = this.#actions?.control;
    const childSessionId = this.#selectedId;
    if (!control || !childSessionId || this.#busy) return;
    if (action !== 'cancel' && !content) return;
    await this.#runCommand(
      () => control(childSessionId, action, content, reauthorize),
      form,
    );
  }

  async #runReply(requestId: string, content: string, form: HTMLFormElement): Promise<void> {
    const reply = this.#actions?.reply;
    if (!reply || !requestId || !content || this.#busy) return;
    await this.#runCommand(() => reply(requestId, content), form);
  }

  async #runCommand(
    run: () => Promise<ChildControlReceipt>,
    form?: HTMLFormElement,
  ): Promise<void> {
    this.#busy = true;
    this.#outcome = '';
    this.requestUpdate();
    try {
      const receipt = await run();
      this.#outcome = receipt.outcome;
      form?.reset();
      this.#announcement = this.#outcomeLabel(receipt.outcome);
      if (this.#selectedId) await this.#loadObservation(this.#selectedId);
    } catch (error) {
      if (isAbortError(error)) return;
      const outcome = commandOutcome(error);
      if (outcome) {
        this.#outcome = outcome;
        this.#announcement = this.#outcomeLabel(outcome);
      }
      if (commandStatus(error) === 404) {
        this.#stale = true;
        this.#announcement = msg('That child is no longer available.', {
          id: 'runDialogs.staleChild',
        });
      } else if (!outcome) {
        this.#outcome = 'failed';
        this.#announcement = msg('The child intervention could not be sent.', {
          id: 'runDialogs.interventionFailed',
        });
      }
    } finally {
      this.#busy = false;
      this.requestUpdate();
    }
  }

  #outcomeLabel(outcome: string): string {
    const labels: Record<string, string> = {
      queued: msg('Queued. The child has not necessarily followed it yet.', {
        id: 'runDialogs.outcome.queued',
      }),
      consumed: msg('Consumed at a safe checkpoint. This does not prove the model complied.', {
        id: 'runDialogs.outcome.consumed',
      }),
      accepted: msg('Continuation accepted as a new operation.', {
        id: 'runDialogs.outcome.accepted',
      }),
      cancellation_requested: msg('Cancellation requested.', {
        id: 'runDialogs.outcome.cancellationRequested',
      }),
      replied: msg('Reply sent.', {id: 'runDialogs.outcome.replied'}),
      terminal_child: msg('This child is already terminal and was not revived.', {
        id: 'runDialogs.outcome.terminalChild',
      }),
      run_terminal: msg('The parent run is terminal, so this child cannot continue.', {
        id: 'runDialogs.outcome.runTerminal',
      }),
      child_running: msg('This child is still running.', {
        id: 'runDialogs.outcome.childRunning',
      }),
      reauthorization_required: msg('User-cancelled work needs explicit reauthorization.', {
        id: 'runDialogs.outcome.reauthorizationRequired',
      }),
      queue_full: msg('The pending control queue is full.', {id: 'runDialogs.outcome.queueFull'}),
      idempotency_conflict: msg('This submission id was already used for a different request.', {
        id: 'runDialogs.outcome.idempotencyConflict',
      }),
      unknown_outcome: msg('The child outcome is unknown.', {
        id: 'runDialogs.outcome.unknownOutcome',
      }),
    };
    return labels[outcome] ?? outcome;
  }

  #close(): void {
    publishModalState(this);
    this.#invalidate();
    this.#selectedId = null;
    this.#actions = null;
    this.#entries = [];
    this.#empty = true;
    this.#failed = false;
    this.requestUpdate();
  }

  #statusLabel(status: string): string {
    const labels: Record<string, string> = {
      running: msg('running', {id: 'runDialogs.childStatus.running'}),
      succeeded: msg('succeeded', {id: 'runDialogs.childStatus.succeeded'}),
      failed: msg('failed', {id: 'runDialogs.childStatus.failed'}),
      cancelled: msg('cancelled', {id: 'runDialogs.childStatus.cancelled'}),
    };
    return labels[status] ?? status;
  }

  #cancellationOriginLabel(origin: string): string {
    const labels: Record<string, string> = {
      user: msg('user', {id: 'runDialogs.cancellationOrigin.user'}),
      parent: msg('parent', {id: 'runDialogs.cancellationOrigin.parent'}),
      run: msg('run', {id: 'runDialogs.cancellationOrigin.run'}),
    };
    return labels[origin] ?? origin;
  }

  #roleLabel(role: string): string {
    const labels: Record<string, string> = {
      user: msg('user', {id: 'runDialogs.role.user'}),
      assistant: msg('assistant', {id: 'runDialogs.role.assistant'}),
      tool: msg('tool', {id: 'runDialogs.role.tool'}),
    };
    return labels[role] ?? role;
  }

  #originLabel(origin: string): string {
    const labels: Record<string, string> = {
      user: msg('user', {id: 'runDialogs.origin.user'}),
      parent: msg('parent', {id: 'runDialogs.origin.parent'}),
    };
    return labels[origin] ?? origin;
  }

  #questionStatusLabel(status: string): string {
    const labels: Record<string, string> = {
      pending: msg('pending', {id: 'runDialogs.questionStatus.pending'}),
      replied: msg('replied', {id: 'runDialogs.questionStatus.replied'}),
      expired: msg('expired', {id: 'runDialogs.questionStatus.expired'}),
      cancelled: msg('cancelled', {id: 'runDialogs.questionStatus.cancelled'}),
    };
    return labels[status] ?? status;
  }

  override render() {
    const entries = this.#entries;
    const showEmpty = !this.#failed && this.#empty;
    const interactive = Boolean(this.#actions?.observe);
    return html`
      <dialog class="confirm-dialog roster-dialog" aria-labelledby="dl-roster-title"
              @close=${() => this.#close()}>
        <h2 id="dl-roster-title">${msg('Child agents', {id: 'runDialogs.childAgents'})}</h2>
        <ul class="roster-list" role="list">
          ${this.#failed ? html`
            <li class="roster-error" role="alert">${msg('Child agents could not be loaded.', {id: 'runDialogs.childAgentsFailed'})}</li>
          ` : nothing}
          ${showEmpty ? html`
            <li>${msg('No child agents were started.', {id: 'runDialogs.noChildAgents'})}</li>
          ` : entries.map((child) => {
            const childSessionId = child.childSessionId;
            return html`
            <li role="listitem">
              ${interactive && childSessionId ? html`
                <button type="button" class="roster-child-select"
                        data-child-session=${childSessionId}
                        aria-current=${this.#selectedId === childSessionId ? 'true' : 'false'}
                        @click=${() => this.#selectChild(childSessionId)}>
                  ${this.#statusLabel(child.status)}: ${child.objective || childSessionId}
                </button>
              ` : html`
                ${this.#statusLabel(child.status)}: ${child.objective || child.childSessionId || ''}
              `}
            </li>
            `;
          })}
        </ul>
        ${this.#pager.hasOlder && !showEmpty ? html`
          <div class="roster-page-control">
            <button type="button" data-load-older-children
                    aria-busy=${this.#pager.state === 'loading' ? 'true' : 'false'}
                    ?disabled=${this.#pager.state === 'loading'}
                    @click=${this.#loadOlder}>
              ${this.#pager.state === 'error'
                ? msg('Retry loading older children', {id: 'runDialogs.retryLoadOlderChildren'})
                : msg('Load older children', {id: 'runDialogs.loadOlderChildren'})}
            </button>
          </div>
        ` : nothing}
        ${this.#observationPanel()}
        <span class="sr-only" data-roster-status role="status" aria-live="polite">
          ${this.#announcement}
        </span>
        <form method="dialog">
          <div class="dl-dialog-actions">
            <button type="button" class="dl-btn" @click=${() => void this.refresh()}>${msg('Refresh', {id: 'runDialogs.refresh'})}</button>
            <button type="submit" value="close">${msg('Close', {id: 'runDialogs.close'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #observationPanel(): TemplateResult | typeof nothing {
    if (!this.#actions?.observe) return nothing;
    if (this.#stale) {
      return html`<p role="status">${msg('That child is no longer available.', {id: 'runDialogs.staleChild'})}</p>`;
    }
    if (!this.#selectedId) {
      return html`<p>${msg('Select a child to inspect transcript, questions, and controls.', {
        id: 'runDialogs.selectChild',
      })}</p>`;
    }
    if (this.#observeFailed) {
      return html`<p role="alert">${msg('Child details could not be loaded.', {
        id: 'runDialogs.observationFailed',
      })}</p>`;
    }
    const observation = this.#observation;
    if (!observation) {
      return html`<p>${msg('Loading child details…', {id: 'runDialogs.loadingChild'})}</p>`;
    }
    const child = observation.child;
    const status = child.status;
    const running = status === 'running';
    const terminal = status === 'succeeded' || status === 'failed' || status === 'cancelled';
    const userCancelled = status === 'cancelled' && child.cancellationOrigin === 'user';
    return html`
      <section class="roster-observation" aria-labelledby="dl-child-observation-title">
        <h3 id="dl-child-observation-title">${msg('Selected child', {id: 'runDialogs.selectedChild'})}</h3>
        <p>
          ${this.#statusLabel(status)}
          ${child.operationStatus
            ? html` · ${msg('Operation', {id: 'runDialogs.operation'})}: ${this.#statusLabel(child.operationStatus)}`
            : nothing}
          ${child.cancellationOrigin
            ? html` · ${msg('Cancellation', {id: 'runDialogs.cancellation'})}: ${this.#cancellationOriginLabel(child.cancellationOrigin)}`
            : nothing}
        </p>
        ${child.summary ? html`<p>${child.summary}</p>` : nothing}
        ${observation.result?.handles.length ? html`
          <p>${msg('Evidence handles', {id: 'runDialogs.evidenceHandles'})}:
            ${observation.result.handles.join(', ')}</p>
        ` : nothing}
        ${this.#outcome ? html`<p role="status">${this.#outcomeLabel(this.#outcome)}</p>` : nothing}
        <h4>${msg('Transcript', {id: 'runDialogs.transcript'})}</h4>
        <ol class="roster-lineage">
          ${observation.transcript.length === 0 ? html`
            <li>${msg('No transcript entries yet.', {id: 'runDialogs.noTranscript'})}</li>
          ` : observation.transcript.map((entry) => html`
            <li>${this.#roleLabel(entry.role)}: ${entry.content || entry.name}</li>
          `)}
        </ol>
        <h4>${msg('Controls', {id: 'runDialogs.controls'})}</h4>
        <ol class="roster-lineage">
          ${observation.controls.length === 0 ? html`
            <li>${msg('No control messages yet.', {id: 'runDialogs.noControls'})}</li>
          ` : observation.controls.map((record) => html`
            <li>
              ${record.consumed
                ? msg('Consumed', {id: 'runDialogs.controlConsumed'})
                : msg('Queued', {id: 'runDialogs.controlQueued'})}
              · ${this.#originLabel(record.origin)}: ${record.content}
            </li>
          `)}
        </ol>
        <h4>${msg('Questions', {id: 'runDialogs.questions'})}</h4>
        ${observation.questions.length === 0 ? html`
          <p>${msg('No questions from this child.', {id: 'runDialogs.noQuestions'})}</p>
        ` : observation.questions.map((question) => html`
          <p>${this.#questionStatusLabel(question.status)}: ${question.question}${question.reply ? html` → ${question.reply}` : nothing}</p>
          ${question.status === 'pending' && this.#actions?.reply ? html`
            <form data-request-id=${question.requestId} @submit=${this.#onReply}>
              <fieldset>
                <legend>${msg('Reply to this question', {id: 'runDialogs.replyLegend'})}</legend>
                <label>
                  <span class="sr-only">${msg('Reply', {id: 'runDialogs.replyLabel'})}</span>
                  <textarea class="dl-dialog-input" name="reply" rows="2" required
                            ?disabled=${this.#busy}></textarea>
                </label>
                <div class="dl-dialog-actions">
                  <button type="submit" class="dl-btn" ?disabled=${this.#busy}>
                    ${msg('Reply', {id: 'runDialogs.replySubmit'})}
                  </button>
                </div>
              </fieldset>
            </form>
          ` : nothing}
        `)}
        ${running && this.#actions?.control ? html`
          <form @submit=${this.#onSteer}>
            <fieldset>
              <legend>${msg('Steer this child', {id: 'runDialogs.steerLegend'})}</legend>
              <label>
                <span class="sr-only">${msg('Steering instruction', {id: 'runDialogs.steerLabel'})}</span>
                <textarea class="dl-dialog-input" name="instruction" rows="2" required
                          ?disabled=${this.#busy}></textarea>
              </label>
              <div class="dl-dialog-actions">
                <button type="submit" class="dl-btn" ?disabled=${this.#busy}>
                  ${msg('Steer', {id: 'runDialogs.steerSubmit'})}
                </button>
              </div>
            </fieldset>
          </form>
          <form @submit=${this.#onCancel}>
            <div class="dl-dialog-actions">
              <button type="submit" class="dl-dialog-danger" ?disabled=${this.#busy}>
                ${msg('Cancel child', {id: 'runDialogs.cancelChild'})}
              </button>
            </div>
          </form>
        ` : nothing}
        ${terminal && this.#actions?.control ? html`
          <form @submit=${this.#onContinue}>
            <fieldset>
              <legend>${msg('Continue this child', {id: 'runDialogs.continueLegend'})}</legend>
              <label>
                <span class="sr-only">${msg('Continuation instruction', {id: 'runDialogs.continueLabel'})}</span>
                <textarea class="dl-dialog-input" name="instruction" rows="2" required
                          ?disabled=${this.#busy}></textarea>
              </label>
              ${userCancelled ? html`
                <label class="dl-dialog-checkbox">
                  <input type="checkbox" name="reauthorize" ?disabled=${this.#busy}>
                  ${msg('Reauthorize this user-cancelled work', {id: 'runDialogs.reauthorize'})}
                </label>
              ` : nothing}
              <div class="dl-dialog-actions">
                <button type="submit" class="dl-btn" ?disabled=${this.#busy}>
                  ${msg('Continue child', {id: 'runDialogs.continueChild'})}
                </button>
              </div>
            </fieldset>
          </form>
        ` : nothing}
      </section>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'dl-continuation-dialog': DlContinuationDialog;
    'dl-children-roster': DlChildrenRoster;
  }
}

customElements.define('dl-continuation-dialog', DlContinuationDialog);
customElements.define('dl-children-roster', DlChildrenRoster);
