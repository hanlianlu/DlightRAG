// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings Dialog Feature: memory state, conversation commands, and dialog lifecycle. */

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {
  currentLanguagePreference,
  setLanguagePreference,
} from '../i18n/locale.ts';
import {parseLanguagePreference, type LanguagePreference} from '../lib/language.ts';
import {
  clearMemory,
  getMemorySettings,
  putMemorySettings,
  undoMemoryChange,
  type MemorySettings,
} from '../api/memory.ts';
import {icon} from '../design-system/index.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {conversationStore} from '../stores/conversation-store.ts';
import type {ChatMemoryOperationDetail} from './chat-feature.ts';
import {modalResult, publishModalState, showOwnedModal} from './modal.ts';
import type {ToastRequestDetail} from './toast.ts';

const MAX_SEEN_MEMORY_OPERATIONS = 500;
type MemoryReadResult = 'loaded' | 'stale' | 'failed';

function memorySummary(event: ChatMemoryOperationDetail): string {
  const body = String(event.body || '').replace(/\s+/g, ' ').trim();
  const concise = body.length > 120 ? `${body.slice(0, 117)}…` : body;
  if (event.outcome === 'unchanged') {
    return msg('Already remembered.', {id: 'settings.memory.alreadyRemembered'});
  }
  if (event.outcome === 'conflict') {
    return msg('Profile Memory changed; recall it before retrying.', {
      id: 'settings.memory.conflict',
    });
  }
  if (event.outcome === 'rejected') {
    return msg('Profile Memory operation was rejected.', {id: 'settings.memory.rejected'});
  }
  if (event.operation === 'forget') {
    return concise
      ? msg(str`Forgot: ${concise}`, {id: 'settings.memory.forgot'})
      : msg('Profile Memory forgotten.', {id: 'settings.memory.forgotten'});
  }
  if (event.operation === 'undo') {
    return concise
      ? msg(str`Restored: ${concise}`, {id: 'settings.memory.restored'})
      : msg('Profile Memory restored.', {id: 'settings.memory.restoredEmpty'});
  }
  return concise
    ? msg(str`Remembered: ${concise}`, {id: 'settings.memory.remembered'})
    : msg('Saved to Profile Memory.', {id: 'settings.memory.saved'});
}

/** Owns Settings state, asynchronous mutations, focus, and native Dialog semantics. */
export class DlSettingsDialog extends LightElement {
  static properties = {
    deleteAllConversations: {attribute: false},
    memory: {state: true},
    memoryLoading: {state: true},
    memoryPending: {state: true},
    language: {state: true},
  };

  declare deleteAllConversations: (returnFocus?: HTMLElement | null) => Promise<boolean>;
  declare memory: MemorySettings | null;
  declare memoryLoading: boolean;
  declare memoryPending: boolean;
  declare language: LanguagePreference;

  #events: AbortController | null = null;
  #returnFocus: HTMLElement | null = null;
  #seenMemoryOperations = new Set<string>();
  #memoryReadGeneration = 0;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.deleteAllConversations = async () => false;
    this.memory = null;
    this.memoryLoading = false;
    this.memoryPending = false;
    this.language = currentLanguagePreference();
    /** Store reads: conversations.length. */
    new StoreController(this, conversationStore);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#events = new AbortController();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    this.#memoryReadGeneration += 1;
    document.body.classList.remove('settings-open');
    super.disconnectedCallback();
  }

  /** Open Settings and refresh its authoritative memory projection. */
  async open(returnFocus?: HTMLElement | null): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return;
    this.#returnFocus = returnFocus ?? (
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    );
    if (!this.memoryPending) {
      this.memoryLoading = true;
      const read = await this.#readMemory();
      if (read === 'failed') {
        this.memory = null;
        this.#requestToast({
          message: msg('Could not load memory settings.', {id: 'settings.memoryLoadFailed'}),
          duration: 3000,
        });
      }
      if (!signal.aborted) this.memoryLoading = false;
    }
    if (signal.aborted) return;
    await this.updateComplete;
    const dialog = this.#dialog();
    if (!dialog || dialog.open) return;
    dialog.returnValue = '';
    showOwnedModal(this, dialog);
    document.body.classList.add('settings-open');
  }

  /** Consume one live Profile Memory domain fact from Chat composition. */
  handleMemoryOperation(event: ChatMemoryOperationDetail): void {
    if (!event.live) return;
    const identity = event.change_id || `${event.intent_id || ''}:${event.operation}:${event.outcome}`;
    if (!identity || this.#seenMemoryOperations.has(identity)) return;
    if (this.#seenMemoryOperations.size >= MAX_SEEN_MEMORY_OPERATIONS) {
      const oldest = this.#seenMemoryOperations.values().next().value;
      if (oldest) this.#seenMemoryOperations.delete(oldest);
    }
    this.#seenMemoryOperations.add(identity);
    const message = memorySummary(event);
    if (event.outcome !== 'changed' || !event.change_id) {
      this.#requestToast({message, duration: 3000});
      return;
    }
    const changeId = event.change_id;
    const signal = this.#events?.signal;
    this.#requestToast({
      message,
      action: {
        actionLabel: msg('Undo', {id: 'settings.memory.undo'}),
        duration: 3000,
        onAction: async () => {
          this.#invalidateMemoryReads();
          const receipt = await undoMemoryChange(changeId, signal);
          if (receipt.outcome !== 'changed') throw new Error('Memory undo conflicted');
          await this.#refreshMemory();
          return msg('Profile Memory change undone.', {id: 'settings.memory.changeUndone'});
        },
      },
    });
    void this.#refreshMemory();
  }

  protected override render(): TemplateResult {
    const total = conversationStore.conversations.length;
    const active = this.memory?.active_count;
    return html`
      <dialog id="settings-dialog" class="settings-dialog" aria-labelledby="settings-title"
              @click=${this.#scrimClick} @close=${this.#closed}>
        <form method="dialog">
          <div class="settings-drawer-body">
            <div class="settings-header">
              <h2 id="settings-title">${msg('Settings', {id: 'settings.title'})}</h2>
              <button class="panel-close settings-close" type="submit" value="close-settings"
                      aria-label=${msg('Close settings', {id: 'settings.close'})}>${icon('close', {size: 'sm'})}</button>
            </div>
            <section class="settings-section">
              <h3 id="settings-memory">${msg('Profile Memory', {id: 'settings.profileMemory'})}</h3>
              <label class="dl-dialog-checkbox">
                <input type="checkbox" id="memory-enabled-toggle"
                       .checked=${this.memory?.enabled ?? false}
                       ?disabled=${this.memoryLoading || this.memoryPending || !this.memory}
                       @change=${this.#toggleMemory}>
                ${msg('Activate profile memories', {id: 'settings.activateMemories'})}
              </label>
              <p id="memory-active-count" class="settings-count" aria-live="polite"
                 ?hidden=${active === null || active === undefined}>
                ${active === 1
                  ? msg('1 stored item', {id: 'settings.oneStoredItem'})
                  : msg(str`${active ?? 0} stored items`, {id: 'settings.nStoredItems'})}
              </p>
              <div class="settings-actions">
                <button type="button" id="memory-clear-btn" class="dl-btn dl-btn-danger-text"
                        ?hidden=${!this.memory?.enabled} ?disabled=${this.memoryPending}
                        @click=${this.#clearMemory}>${msg('Clear memory', {id: 'settings.clearMemory'})}</button>
              </div>
            </section>
            <section class="settings-section">
              <h3 id="settings-data">${msg('Conversation Sessions', {id: 'settings.conversationSessions'})}</h3>
              <p class="settings-note">${msg('Conversations retain 365 days', {id: 'settings.retentionNote'})}</p>
              <p id="conversation-count" class="settings-count" aria-live="polite">
                ${total === 1
                  ? msg('1 conversation', {id: 'settings.oneConversation'})
                  : msg(str`${total} conversations`, {id: 'settings.nConversations'})}
              </p>
              <div class="settings-actions">
                <button type="button" id="delete-all-btn" class="dl-btn dl-btn-danger-text"
                        @click=${this.#deleteAll}>${msg('Delete all conversations', {id: 'settings.deleteAllConversations'})}</button>
              </div>
            </section>
            <section class="settings-section">
              <h3 id="settings-language">${msg('Language', {id: 'settings.language'})}</h3>
              <div id="language-options" role="radiogroup" aria-labelledby="settings-language">
                <label class="dl-dialog-checkbox">
                  <input type="radio" name="language" value="auto"
                         .checked=${this.language === 'auto'}
                         @change=${this.#setLanguage}>
                  ${msg('Automatic', {id: 'settings.language.automatic'})}
                </label>
                <label class="dl-dialog-checkbox">
                  <input type="radio" name="language" value="en"
                         .checked=${this.language === 'en'}
                         @change=${this.#setLanguage}>
                  ${msg('English', {id: 'settings.language.english'})}
                </label>
                <label class="dl-dialog-checkbox">
                  <input type="radio" name="language" value="zh"
                         .checked=${this.language === 'zh'}
                         @change=${this.#setLanguage}>
                  中文
                </label>
              </div>
            </section>
          </div>
        </form>
      </dialog>
      <dialog id="clear-memory-dialog" class="confirm-dialog" aria-labelledby="clear-memory-title">
        <form method="dialog">
          <h2 id="clear-memory-title">${msg('Clear Profile memory?', {id: 'settings.clearMemoryTitle'})}</h2>
          <p>${msg('Remembered preferences and facts will be forgotten. Conversations are not affected.', {
            id: 'settings.clearMemoryBody',
          })}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'settings.cancel'})}</button>
            <button type="submit" value="clear" class="dl-dialog-danger">${msg('Clear memory', {id: 'settings.clearMemoryConfirm'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #dialog(): HTMLDialogElement | null {
    return this.querySelector<HTMLDialogElement>('#settings-dialog');
  }

  #scrimClick = (event: MouseEvent): void => {
    const dialog = this.#dialog();
    if (dialog && event.target === dialog) dialog.close();
  };

  #closed = (): void => {
    publishModalState(this);
    document.body.classList.remove('settings-open');
    const returnFocus = this.#returnFocus;
    this.#returnFocus = null;
    if (returnFocus?.isConnected && !returnFocus.inert) returnFocus.focus();
  };

  #toggleMemory = async (event: Event): Promise<void> => {
    const input = event.currentTarget as HTMLInputElement;
    const requested = input.checked;
    const signal = this.#events?.signal;
    if (!signal || signal.aborted || this.memoryPending) return;
    this.memoryPending = true;
    this.#invalidateMemoryReads();
    try {
      const memory = await putMemorySettings(requested, signal);
      if (!signal.aborted) this.memory = memory;
    } catch {
      if (!signal.aborted) {
        input.checked = !requested;
        this.#requestToast({
          message: msg('Could not save memory settings.', {id: 'settings.memorySaveFailed'}),
          duration: 3000,
        });
      }
    } finally {
      if (!signal.aborted) this.memoryPending = false;
    }
  };

  #clearMemory = async (event: Event): Promise<void> => {
    const signal = this.#events?.signal;
    const button = event.currentTarget as HTMLButtonElement;
    const confirm = this.querySelector<HTMLDialogElement>('#clear-memory-dialog');
    if (!signal || signal.aborted || !confirm || this.memoryPending) return;
    if (await modalResult(this, confirm, () => button.focus(), signal) !== 'clear') return;
    this.memoryPending = true;
    this.#invalidateMemoryReads();
    try {
      await clearMemory(signal);
      if (await this.#readMemory() === 'failed') throw new Error('Memory refresh failed');
      if (!signal.aborted) {
        this.#requestToast({
          message: msg('Memory cleared.', {id: 'settings.memoryCleared'}),
          duration: 3000,
        });
      }
    } catch {
      if (!signal.aborted) {
        this.#requestToast({
          message: msg('Could not clear memory.', {id: 'settings.memoryClearFailedToast'}),
          duration: 3000,
        });
      }
    } finally {
      if (!signal.aborted) this.memoryPending = false;
    }
  };

  #deleteAll = async (event: Event): Promise<void> => {
    const returnFocus = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (await this.deleteAllConversations(returnFocus)) this.#dialog()?.close();
  };

  #setLanguage = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    const preference = parseLanguagePreference(input.value);
    this.language = preference;
    void setLanguagePreference(preference);
  };

  #requestToast(detail: ToastRequestDetail): void {
    this.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail,
      bubbles: true,
      composed: true,
    }));
  }

  #invalidateMemoryReads(): void {
    this.#memoryReadGeneration += 1;
    this.memoryLoading = false;
  }

  async #readMemory(): Promise<MemoryReadResult> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return 'stale';
    const generation = ++this.#memoryReadGeneration;
    try {
      const memory = await getMemorySettings(signal);
      if (signal.aborted || generation !== this.#memoryReadGeneration) return 'stale';
      this.memory = memory;
      return 'loaded';
    } catch {
      if (signal.aborted || generation !== this.#memoryReadGeneration) return 'stale';
      return 'failed';
    }
  }

  async #refreshMemory(): Promise<void> {
    if (this.memoryPending) return;
    await this.#readMemory();
  }
}

customElements.define('dl-settings-dialog', DlSettingsDialog);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-dialog': DlSettingsDialog;
  }
}
