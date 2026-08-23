// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Run continuation and child-roster dialogs as first-class Lit components. */

import {html} from 'lit';
import {LightElement} from '../lib/lit_host.ts';

export type ContinuationKind = 'follow-up' | 'fork';

export interface ContinuationResult {
  kind: ContinuationKind;
  query: string | null;
}

export class DlContinuationDialog extends LightElement {
  kind: ContinuationKind = 'follow-up';

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
      dialog.showModal();
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
    const title = forking ? 'Fork this answer' : 'Follow up';
    const note = forking
      ? 'Start a new conversation from the same context. The previous answer is not carried over.'
      : 'Ask a follow-up question; the previous answer is included as context.';
    return html`
      <dialog class="confirm-dialog" aria-labelledby="dl-continuation-title"
              @close=${() => this.#emitClose()}>
        <form method="dialog">
          <h2 id="dl-continuation-title">${title}</h2>
          <p>${note}</p>
          <textarea class="ui-dialog-input" rows="3"
                    placeholder="Ask a question…"></textarea>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Cancel</button>
            <button type="submit" value="continue" class="ui-btn">Continue</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #emitClose(): void {
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

export class DlChildrenRoster extends LightElement {
  fetcher: (() => Promise<ChildRosterEntry[]>) | null = null;

  open(fetcher: () => Promise<ChildRosterEntry[]>): void {
    this.fetcher = fetcher;
    void this.updateComplete.then(() => {
      const dialog = this.querySelector<HTMLDialogElement>('dialog');
      dialog?.showModal();
      void this.refresh();
    });
  }

  async refresh(): Promise<void> {
    const list = this.querySelector<HTMLUListElement>('ul');
    if (!list) return;
    list.replaceChildren();
    let children: ChildRosterEntry[] = [];
    if (this.fetcher) {
      try {
        children = await this.fetcher();
      } catch {
        children = [];
      }
    }
    if (children.length === 0) {
      const empty = document.createElement('li');
      empty.textContent = 'No child agents were started.';
      list.appendChild(empty);
      return;
    }
    for (const child of children) {
      const item = document.createElement('li');
      item.textContent = `${child.status}: ${child.objective || child.child_session_id || ''}`;
      list.appendChild(item);
    }
  }

  override render() {
    return html`
      <dialog class="confirm-dialog" aria-labelledby="dl-roster-title">
        <form method="dialog">
          <h2 id="dl-roster-title">Child agents</h2>
          <ul class="roster-list"></ul>
          <div class="ui-dialog-actions">
            <button type="button" class="ui-btn" @click=${() => void this.refresh()}>Refresh</button>
            <button type="submit" value="close">Close</button>
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
