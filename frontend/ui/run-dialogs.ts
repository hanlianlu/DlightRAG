// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Run continuation dialog as a first-class Lit component. */

import {msg} from '@lit/localize';
import {html} from 'lit';
import {raise} from '../lib/dom.ts';
import {LightElement} from '../lib/lit-host.ts';
import {publishModalState, showOwnedModal} from './modal.ts';

export interface ContinuationResult {
  query: string | null;
}

export class DlContinuationDialog extends LightElement {
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
    raise(this, 'dl-continuation-result', {
      query: value === 'continue' ? (this.#input()?.value.trim() ?? null) : null,
    });
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'dl-continuation-dialog': DlContinuationDialog;
  }

  interface HTMLElementEventMap {
    'dl-continuation-result': CustomEvent<ContinuationResult>;
  }
}

customElements.define('dl-continuation-dialog', DlContinuationDialog);
