// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Run continuation and child-roster dialogs as first-class Lit components. */

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {keyed} from 'lit/directives/keyed.js';
import {repeat} from 'lit/directives/repeat.js';
import type {
  ChildControlReceipt,
  ChildObservation,
} from '../api/conversations.ts';
import {LightElement} from '../lib/lit-host.ts';
import {isAbortError} from '../lib/errors.ts';
import {KeysetPager} from '../lib/paged.ts';
import {publishModalState, showOwnedModal} from './modal.ts';

export interface ContinuationResult {
  query: string | null;
}

export class DlContinuationDialog extends LightElement {
  constructor() {
    super();
    updateWhenLocaleChanges(this);
  }

  open(): void {
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
    const title = msg('Fork this answer', {id: 'runDialogs.forkTitle'});
    const note = msg('Start a new conversation from the state this answer settled at, including its answer.', {
      id: 'runDialogs.forkNote',
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
    operationId?: string | null,
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

type ChildEditorKind = 'steer' | 'continue' | 'reply';
type ChildCommandAction = 'steer' | 'continue' | 'cancel' | 'reply';

interface ChildCommandIdentity {
  dialogGeneration: number;
  runId: string | undefined;
  childSessionId: string;
  action: ChildCommandAction;
  requestId: string | null;
  operationId: string | null;
}

interface ChildEditorDraft {
  value: string;
  reauthorize: boolean;
  selectionStart: number;
  selectionEnd: number;
  element: HTMLTextAreaElement | null;
}

function editorKind(form: HTMLFormElement): ChildEditorKind | null {
  const kind = form.dataset.editor;
  return kind === 'steer' || kind === 'continue' || kind === 'reply' ? kind : null;
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
  #dialogGeneration = 0;
  #selectedId: string | null = null;
  #observation: ChildObservation | null = null;
  #observeFailed = false;
  #stale = false;
  #outcome = '';
  #outcomeIdentity: ChildCommandIdentity | null = null;
  #commands = new Set<ChildCommandIdentity>();
  #drafts = new Map<string, ChildEditorDraft>();
  #focusKey: string | null = null;
  #refreshing = false;
  #refreshQueued = false;
  #pager = new KeysetPager<ChildRosterEntry>(
    (cursor, signal) => this.#pageFetcher!(cursor, signal).then((page) => ({items: page.children, nextCursor: page.nextCursor})),
    () => this.requestUpdate(),
  );

  open(
    fetcher: () => Promise<ChildRosterEntry[]>,
    pageFetcher?: ChildRosterPageFetcher,
    actions?: ChildRosterActions,
  ): void {
    this.#dialogGeneration += 1;
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
    this.#refreshQueued = true;
    void this.#flushRefresh();
  }

  async #flushRefresh(): Promise<void> {
    if (this.#refreshing) return;
    this.#refreshing = true;
    try {
      while (this.#refreshQueued) {
        this.#refreshQueued = false;
        await this.refresh();
      }
    } finally {
      this.#refreshing = false;
      if (this.#refreshQueued) void this.#flushRefresh();
    }
  }

  async refresh(): Promise<void> {
    const selected = this.#selectedId;
    this.#controller?.abort();
    this.#controller = null;
    this.#generation += 1;
    this.#announcement = '';
    this.#failed = false;
    this.#pager.reset(null);
    this.#selectedId = selected;
    if (this.#pageFetcher) {
      await this.#loadFirstPage();
      await this.#restoreSelection(true);
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
    await this.#restoreSelection(true);
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
    this.#outcomeIdentity = null;
    this.#pager.reset(null);
  }

  #editorKey(
    kind: ChildEditorKind,
    requestId: string | null,
    childSessionId = this.#selectedId,
    operationId = this.#observation?.child.operationId ?? null,
  ): string | null {
    const runId = this.#actions?.runId;
    if (!runId || !childSessionId) return null;
    return JSON.stringify({
      runId,
      childSessionId,
      kind,
      requestId: requestId ?? '',
      operationId: operationId ?? '',
    });
  }

  #captureEditors(): void {
    const panel = this.querySelector('.roster-observation');
    if (!panel || !this.#selectedId) return;
    const active = document.activeElement;
    this.#focusKey = null;
    for (const textarea of panel.querySelectorAll<HTMLTextAreaElement>('textarea')) {
      const form = textarea.closest('form');
      if (!form) continue;
      const kind = editorKind(form);
      if (!kind) continue;
      const requestId = form.dataset.requestId || null;
      const key = this.#editorKey(kind, requestId);
      if (!key) continue;
      const reauthorize = Boolean(
        form.querySelector<HTMLInputElement>('[name="reauthorize"]')?.checked,
      );
      this.#drafts.set(key, {
        value: textarea.value,
        reauthorize,
        selectionStart: textarea.selectionStart ?? textarea.value.length,
        selectionEnd: textarea.selectionEnd ?? textarea.value.length,
        element: textarea,
      });
      if (active === textarea) this.#focusKey = key;
    }
  }

  #restoreEditors(): void {
    const panel = this.querySelector('.roster-observation');
    if (!panel || !this.#selectedId) return;
    for (const textarea of panel.querySelectorAll<HTMLTextAreaElement>('textarea')) {
      const form = textarea.closest('form');
      if (!form) continue;
      const kind = editorKind(form);
      if (!kind) continue;
      const requestId = form.dataset.requestId || null;
      const key = this.#editorKey(kind, requestId);
      if (!key) continue;
      const draft = this.#drafts.get(key);
      if (!draft) continue;
      if (draft.element === textarea) continue;
      textarea.value = draft.value;
      const box = form.querySelector<HTMLInputElement>('[name="reauthorize"]');
      if (box) box.checked = draft.reauthorize;
      draft.element = textarea;
      if (this.#focusKey === key) {
        textarea.focus();
        try {
          textarea.setSelectionRange(draft.selectionStart, draft.selectionEnd);
        } catch {
          // Native range restore can reject if the control is not text-like.
        }
      }
    }
  }

  #clearEditor(identity: ChildCommandIdentity): void {
    const kind: ChildEditorKind | null = identity.action === 'steer' || identity.action === 'continue'
      || identity.action === 'reply'
      ? identity.action
      : null;
    if (!kind) return;
    const key = this.#editorKey(
      kind,
      identity.requestId,
      identity.childSessionId,
      identity.operationId,
    );
    if (key) this.#drafts.delete(key);
    if (key && this.#focusKey === key) this.#focusKey = null;
    if (!this.#commandMatchesSelection(identity)) return;
    for (const form of this.querySelectorAll<HTMLFormElement>('.roster-observation form')) {
      if (editorKind(form) === kind
        && this.#editorKey(kind, form.dataset.requestId || null) === key) form.reset();
    }
  }

  #commandMatchesSelection(identity: ChildCommandIdentity, replied = false): boolean {
    const dialog = this.querySelector<HTMLDialogElement>('dialog');
    return identity.dialogGeneration === this.#dialogGeneration
      && identity.runId === this.#actions?.runId
      && identity.childSessionId === this.#selectedId
      && identity.childSessionId === this.#observation?.child.childSessionId
      && identity.operationId === (this.#observation?.child.operationId ?? null)
      && (identity.requestId === null || Boolean(this.#observation?.questions.some(
        (question) => question.requestId === identity.requestId
          && (question.status === 'pending' || (replied && question.status === 'replied')),
      )))
      && Boolean(dialog?.open);
  }

  #formBusy(action: ChildCommandAction, requestId: string | null = null): boolean {
    for (const command of this.#commands) {
      if (!this.#commandMatchesSelection(command)) continue;
      if (command.action !== action) continue;
      if (command.requestId !== requestId) continue;
      return true;
    }
    return false;
  }

  #visibleOutcome(): string {
    if (!this.#outcome) return '';
    if (this.#outcomeIdentity
      && !this.#commandMatchesSelection(this.#outcomeIdentity, this.#outcome === 'replied')) return '';
    return this.#outcome;
  }

  #loadOlder = (): void => {
    void this.loadOlderChildren();
  };

  async #restoreSelection(inPlace = false): Promise<void> {
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
    const sameChild = this.#observation?.child.childSessionId === selected;
    await this.#loadObservation(selected, {inPlace: inPlace && sameChild});
  }

  #selectChild = (childSessionId: string): void => {
    if (
      this.#selectedId === childSessionId
      && this.#observation?.child.childSessionId === childSessionId
    ) {
      return;
    }
    this.#captureEditors();
    this.#selectedId = childSessionId;
    this.#stale = false;
    this.#outcome = '';
    this.#outcomeIdentity = null;
    void this.#loadObservation(childSessionId);
  };

  async #loadObservation(
    childSessionId: string,
    options: {inPlace?: boolean} = {},
  ): Promise<void> {
    const observe = this.#actions?.observe;
    if (!observe) return;
    const inPlace = Boolean(options.inPlace);
    const dialogGeneration = this.#dialogGeneration;
    if (!inPlace) {
      this.#observeController?.abort();
      this.#observeFailed = false;
      this.#observation = null;
      this.requestUpdate();
    }
    const controller = new AbortController();
    this.#observeController = controller;
    try {
      const observation = await observe(childSessionId, controller.signal);
      if (controller !== this.#observeController || this.#selectedId !== childSessionId) return;
      if (this.#dialogGeneration !== dialogGeneration) return;
      if (inPlace) this.#captureEditors();
      this.#observation = observation;
      this.#stale = false;
      this.#observeFailed = false;
      this.requestUpdate();
      await this.updateComplete;
      this.#restoreEditors();
    } catch (error) {
      if (controller !== this.#observeController || this.#selectedId !== childSessionId) return;
      if (this.#dialogGeneration !== dialogGeneration) return;
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
    void this.#runControl('steer', instruction, false);
  };

  #onContinue = (event: Event): void => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const data = new FormData(form);
    const instruction = String(data.get('instruction') || '').trim();
    const reauthorize = data.get('reauthorize') === 'on';
    void this.#runControl('continue', instruction, reauthorize);
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
    void this.#runReply(requestId, content);
  };

  async #runControl(
    action: 'steer' | 'continue' | 'cancel',
    content: string,
    reauthorize: boolean,
  ): Promise<void> {
    const control = this.#actions?.control;
    const childSessionId = this.#selectedId;
    if (!control || !childSessionId || this.#formBusy(action)) return;
    if (action !== 'cancel' && !content) return;
    const operationId = this.#observation?.child.operationId ?? null;
    const identity: ChildCommandIdentity = {
      dialogGeneration: this.#dialogGeneration,
      runId: this.#actions?.runId,
      childSessionId,
      action,
      requestId: null,
      operationId,
    };
    await this.#runCommand(
      () => control(childSessionId, action, content, reauthorize, operationId),
      identity,
    );
  }

  async #runReply(requestId: string, content: string): Promise<void> {
    const reply = this.#actions?.reply;
    if (!reply || !requestId || !content || this.#formBusy('reply', requestId)) return;
    const childSessionId = this.#selectedId;
    if (!childSessionId) return;
    const identity: ChildCommandIdentity = {
      dialogGeneration: this.#dialogGeneration,
      runId: this.#actions?.runId,
      childSessionId,
      action: 'reply',
      requestId,
      operationId: this.#observation?.child.operationId ?? null,
    };
    await this.#runCommand(() => reply(requestId, content), identity);
  }

  async #runCommand(
    run: () => Promise<ChildControlReceipt>,
    identity: ChildCommandIdentity,
  ): Promise<void> {
    this.#commands.add(identity);
    if (this.#commandMatchesSelection(identity)) this.#outcome = '';
    this.requestUpdate();
    try {
      const receipt = await run();
      if (!this.#commandMatchesSelection(identity)) return;
      this.#clearEditor(identity);
      this.#outcome = receipt.outcome;
      // An accepted continuation reports the new Operation it created.
      this.#outcomeIdentity = identity.action === 'continue' && receipt.outcome === 'accepted'
        && receipt.operationId ? {...identity, operationId: receipt.operationId} : identity;
      this.#announcement = this.#outcomeLabel(receipt.outcome);
      await this.#loadObservation(identity.childSessionId, {inPlace: true});
    } catch (error) {
      if (!this.#commandMatchesSelection(identity)) return;
      if (isAbortError(error)) return;
      const outcome = commandOutcome(error);
      if (outcome) {
        this.#outcome = outcome;
        this.#outcomeIdentity = identity;
        this.#announcement = this.#outcomeLabel(outcome);
      }
      if (commandStatus(error) === 404) {
        this.#stale = true;
        this.#announcement = msg('That child is no longer available.', {
          id: 'runDialogs.staleChild',
        });
      } else if (!outcome) {
        this.#outcome = 'failed';
        this.#outcomeIdentity = identity;
        this.#announcement = msg('The child intervention could not be sent.', {
          id: 'runDialogs.interventionFailed',
        });
      }
    } finally {
      this.#commands.delete(identity);
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
    this.#dialogGeneration += 1;
    this.#drafts.clear();
    this.#focusKey = null;
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
    const outcome = this.#visibleOutcome();
    const steerBusy = this.#formBusy('steer');
    const continueBusy = this.#formBusy('continue');
    const cancelBusy = this.#formBusy('cancel');
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
        ${outcome ? html`<p role="status">${this.#outcomeLabel(outcome)}</p>` : nothing}
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
        ` : repeat(observation.questions,
          (question) => this.#editorKey('reply', question.requestId),
          (question) => html`
          <p>${this.#questionStatusLabel(question.status)}: ${question.question}${question.reply ? html` → ${question.reply}` : nothing}</p>
          ${question.status === 'pending' && this.#actions?.reply ? html`
            <form data-editor="reply" data-request-id=${question.requestId} @submit=${this.#onReply}>
              <fieldset>
                <legend>${msg('Reply to this question', {id: 'runDialogs.replyLegend'})}</legend>
                <label>
                  <span class="sr-only">${msg('Reply', {id: 'runDialogs.replyLabel'})}</span>
                  <textarea class="dl-dialog-input" name="reply" rows="2" required
                            ?disabled=${this.#formBusy('reply', question.requestId)}></textarea>
                </label>
                <div class="dl-dialog-actions">
                  <button type="submit" class="dl-btn"
                          ?disabled=${this.#formBusy('reply', question.requestId)}>
                    ${msg('Reply', {id: 'runDialogs.replySubmit'})}
                  </button>
                </div>
              </fieldset>
            </form>
          ` : nothing}
        `)}
        ${running && this.#actions?.control ? keyed(this.#editorKey('steer', null), html`
          <form data-editor="steer" @submit=${this.#onSteer}>
            <fieldset>
              <legend>${msg('Steer this child', {id: 'runDialogs.steerLegend'})}</legend>
              <label>
                <span class="sr-only">${msg('Steering instruction', {id: 'runDialogs.steerLabel'})}</span>
                <textarea class="dl-dialog-input" name="instruction" rows="2" required
                          ?disabled=${steerBusy}></textarea>
              </label>
              <div class="dl-dialog-actions">
                <button type="submit" class="dl-btn" ?disabled=${steerBusy}>
                  ${msg('Steer', {id: 'runDialogs.steerSubmit'})}
                </button>
              </div>
            </fieldset>
          </form>
          <form data-editor="cancel" @submit=${this.#onCancel}>
            <div class="dl-dialog-actions">
              <button type="submit" class="dl-dialog-danger" ?disabled=${cancelBusy}>
                ${msg('Cancel child', {id: 'runDialogs.cancelChild'})}
              </button>
            </div>
          </form>
        `) : nothing}
        ${terminal && this.#actions?.control ? keyed(this.#editorKey('continue', null), html`
          <form data-editor="continue" @submit=${this.#onContinue}>
            <fieldset>
              <legend>${msg('Continue this child', {id: 'runDialogs.continueLegend'})}</legend>
              <label>
                <span class="sr-only">${msg('Continuation instruction', {id: 'runDialogs.continueLabel'})}</span>
                <textarea class="dl-dialog-input" name="instruction" rows="2" required
                          ?disabled=${continueBusy}></textarea>
              </label>
              ${userCancelled ? html`
                <label class="dl-dialog-checkbox">
                  <input type="checkbox" name="reauthorize" ?disabled=${continueBusy}>
                  ${msg('Reauthorize this user-cancelled work', {id: 'runDialogs.reauthorize'})}
                </label>
              ` : nothing}
              <div class="dl-dialog-actions">
                <button type="submit" class="dl-btn" ?disabled=${continueBusy}>
                  ${msg('Continue child', {id: 'runDialogs.continueChild'})}
                </button>
              </div>
            </fieldset>
          </form>
        `) : nothing}
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
