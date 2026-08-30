// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Run continuation and child-roster dialogs as first-class Lit components. */

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing} from 'lit';
import {LightElement} from '../lib/lit_host.ts';
import {isAbortError} from '../lib/errors.ts';
import {publishModalState, showOwnedModal} from './modal.ts';

export type ContinuationKind = 'follow-up' | 'fork';

export interface ContinuationResult {
  kind: ContinuationKind;
  query: string | null;
}

export class DlContinuationDialog extends LightElement {
  kind: ContinuationKind = 'follow-up';

  constructor() {
    super();
    updateWhenLocaleChanges(this);
  }

  open(kind: ContinuationKind): void {
    this.kind = kind;
    this.requestUpdate();
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
          <textarea class="ui-dialog-input" rows="3"
                    placeholder=${msg('Ask a question…', {id: 'runDialogs.askPlaceholder'})}></textarea>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'runDialogs.cancel'})}</button>
            <button type="submit" value="continue" class="ui-btn">${msg('Continue', {id: 'runDialogs.continue'})}</button>
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
  child_session_id?: string;
}

export type ChildRosterPageFetcher = (
  cursor: string | null,
  signal?: AbortSignal,
) => Promise<{children: ChildRosterEntry[]; next_cursor: string | null}>;

export class DlChildrenRoster extends LightElement {
  fetcher: (() => Promise<ChildRosterEntry[]>) | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
  }

  #pageFetcher: ChildRosterPageFetcher | null = null;
  #entries: ChildRosterEntry[] = [];
  #nextCursor: string | null = null;
  #loadMoreState: 'idle' | 'loading' | 'error' = 'idle';
  #empty = true;
  #failed = false;
  #announcement = '';
  #controller: AbortController | null = null;
  #generation = 0;
  #flight: Promise<void> | null = null;

  open(
    fetcher: () => Promise<ChildRosterEntry[]>,
    pageFetcher?: ChildRosterPageFetcher,
  ): void {
    this.fetcher = fetcher;
    this.#pageFetcher = pageFetcher ?? null;
    void this.updateComplete.then(() => {
      const dialog = this.querySelector<HTMLDialogElement>('dialog');
      if (dialog) showOwnedModal(this, dialog);
      void this.refresh();
    });
  }

  async refresh(): Promise<void> {
    this.#invalidate();
    this.#entries = [];
    this.#nextCursor = null;
    this.#empty = true;
    this.#failed = false;
    if (this.#pageFetcher) {
      await this.#loadFirstPage();
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
      this.#nextCursor = page.next_cursor;
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
    if (this.#flight !== null) return this.#flight;
    if (!this.#pageFetcher || this.#nextCursor === null) return Promise.resolve();
    const flight = this.#loadOlderPage(this.#nextCursor);
    this.#flight = flight;
    void flight.finally(() => {
      if (this.#flight === flight) this.#flight = null;
    });
    return flight;
  }

  async #loadOlderPage(cursor: string): Promise<void> {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    const generation = this.#generation;
    this.#loadMoreState = 'loading';
    this.#announcement = msg('Loading older children…', {id: 'runDialogs.loadingOlderChildren'});
    this.requestUpdate();
    try {
      const page = await this.#pageFetcher!(cursor, controller.signal);
      if (
        controller !== this.#controller
        || generation !== this.#generation
        || this.#nextCursor !== cursor
      ) {
        if (controller === this.#controller) {
          this.#loadMoreState = 'idle';
          this.#announcement = '';
        }
        return;
      }
      const known = new Set(this.#entries.map((entry) => entry.child_session_id).filter(Boolean));
      const appended = page.children.filter((entry) => {
        if (!entry.child_session_id || known.has(entry.child_session_id)) return false;
        known.add(entry.child_session_id);
        return true;
      });
      this.#entries = [...this.#entries, ...appended];
      this.#nextCursor = page.next_cursor;
      this.#loadMoreState = 'idle';
      this.#announcement = appended.length === 1
        ? msg('Loaded 1 older child.', {id: 'runDialogs.loadedOneChild'})
        : msg(str`Loaded ${appended.length} older children.`, {id: 'runDialogs.loadedOlderChildren'});
      this.requestUpdate();
    } catch (error) {
      if (controller !== this.#controller || generation !== this.#generation) return;
      if (isAbortError(error)) return;
      this.#loadMoreState = 'error';
      this.#announcement = msg('Older children could not be loaded.', {
        id: 'runDialogs.olderChildrenFailed',
      });
      this.requestUpdate();
    } finally {
      if (this.#controller === controller) this.#controller = null;
    }
  }

  #invalidate(): void {
    this.#controller?.abort();
    this.#controller = null;
    this.#generation += 1;
    this.#flight = null;
    this.#loadMoreState = 'idle';
    this.#announcement = '';
  }

  #loadOlder = (): void => {
    void this.loadOlderChildren();
  };

  #close(): void {
    publishModalState(this);
    this.#invalidate();
    this.#entries = [];
    this.#nextCursor = null;
    this.#empty = true;
    this.#failed = false;
    this.requestUpdate();
  }

  override render() {
    const entries = this.#entries;
    const showEmpty = !this.#failed && this.#empty;
    return html`
      <dialog class="confirm-dialog" aria-labelledby="dl-roster-title"
              @close=${() => this.#close()}>
        <form method="dialog">
          <h2 id="dl-roster-title">${msg('Child agents', {id: 'runDialogs.childAgents'})}</h2>
          <ul class="roster-list" role="list">
            ${this.#failed ? html`
              <li class="roster-error" role="alert">${msg('Child agents could not be loaded.', {id: 'runDialogs.childAgentsFailed'})}</li>
            ` : nothing}
            ${showEmpty ? html`
              <li>${msg('No child agents were started.', {id: 'runDialogs.noChildAgents'})}</li>
            ` : entries.map((child) => html`
              <li role="listitem">
                ${child.status}: ${child.objective || child.child_session_id || ''}
              </li>
            `)}
          </ul>
          ${this.#nextCursor !== null && !showEmpty ? html`
            <div class="roster-page-control">
              <button type="button" data-load-older-children
                      aria-busy=${this.#loadMoreState === 'loading' ? 'true' : 'false'}
                      ?disabled=${this.#loadMoreState === 'loading'}
                      @click=${this.#loadOlder}>
                ${this.#loadMoreState === 'error'
                  ? msg('Retry loading older children', {id: 'runDialogs.retryLoadOlderChildren'})
                  : msg('Load older children', {id: 'runDialogs.loadOlderChildren'})}
              </button>
            </div>
          ` : nothing}
          <span class="sr-only" data-roster-status role="status" aria-live="polite">
            ${this.#announcement}
          </span>
          <div class="ui-dialog-actions">
            <button type="button" class="ui-btn" @click=${() => void this.refresh()}>${msg('Refresh', {id: 'runDialogs.refresh'})}</button>
            <button type="submit" value="close">${msg('Close', {id: 'runDialogs.close'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'dl-continuation-dialog': DlContinuationDialog;
    'dl-children-roster': DlChildrenRoster;
  }
}

// Explicit registration: decorator emission is not stable across shared
// chunk builds, and a registered-by-import element is a hard requirement.
customElements.define('dl-continuation-dialog', DlContinuationDialog);
customElements.define('dl-children-roster', DlChildrenRoster);
