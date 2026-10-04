// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Conversation Sessions: how many conversations there are, and the one way to delete them.
 *
 * The conversation sidebar owns deleting them and the dialog that asks first; this page only asks
 * it to, and reports the count the store already holds.
 */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, type PropertyValues, type TemplateResult} from 'lit';
import {PHONE_DIALOG_MEDIA} from '../lib/breakpoints.ts';
import {LightElement, MediaController, StoreController} from '../lib/lit-host.ts';
import {productionHandles, type AppHandles} from '../stores/app-handles.ts';
import shared from '../styles/settings-page.module.css';
import {reportSettingsSummary} from './settings-summary.ts';

export class DlSettingsConversations extends LightElement {
  static properties = {
    handles: {attribute: false},
    deleteAll: {attribute: false},
  };

  declare handles: AppHandles;
  /** Ask the owner of the conversation list to delete every conversation; true once it did. */
  declare deleteAll: (returnFocus?: HTMLElement | null) => Promise<boolean>;

  readonly #phone = new MediaController(this, PHONE_DIALOG_MEDIA);
  #reported = -1;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.handles = productionHandles();
    this.deleteAll = async () => false;
    /** Store reads: conversations.length. */
    new StoreController(this, this.handles.conversations);
  }

  protected override updated(_changed: PropertyValues<this>): void {
    const count = this.handles.conversations.conversations.length;
    if (count === this.#reported) return;
    this.#reported = count;
    reportSettingsSummary(this, {section: 'conversations', count});
  }

  #delete = (event: Event): void => {
    const trigger = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    void this.deleteAll(trigger);
  };

  protected override render(): TemplateResult {
    const total = this.handles.conversations.conversations.length;
    return html`
      <div class=${shared.stack}>
        <div class="${shared.card} ${shared.row} ${shared.dangerRow}">
          <span class=${shared.rowText}>
            <span class=${shared.rowLabel}>${msg('Delete all conversations', {id: 'settings.deleteAllConversations'})}</span>
            <span id="conversation-count" class=${shared.rowCaption} aria-live="polite">${total === 1
              ? msg('1 conversation', {id: 'settings.oneConversation'})
              : msg(str`${total} conversations`, {id: 'settings.nConversations'})}</span>
          </span>
          <button type="button" id="delete-all-btn" class="dl-btn dl-btn-danger-text"
                  aria-label=${msg('Delete all conversations', {id: 'settings.deleteAllConversations'})}
                  @click=${this.#delete}>${this.#phone.matches
            ? msg('Delete all conversations', {id: 'settings.deleteAllConversations'})
            : msg('Delete…', {id: 'settings.deleteButton'})}</button>
        </div>
      </div>`;
  }
}

customElements.define('dl-settings-conversations', DlSettingsConversations);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-conversations': DlSettingsConversations;
  }
}
