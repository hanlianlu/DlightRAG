// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings Dialog Feature: memory state, conversation commands, and dialog lifecycle. */

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {KeysetPager} from '../lib/paged.ts';
import './settings-connections.ts';
import './toast.ts';
import type {ToastRequestDetail} from './toast.ts';
import {
  currentLanguagePreference,
  setLanguagePreference,
} from '../i18n/locale.ts';
import {parseLanguagePreference, type LanguagePreference} from '../lib/language.ts';
import {
  clearMemory,
  forgetMemory,
  listMemories,
  type MemoryRecord,
  getMemorySettings,
  putMemorySettings,
  undoMemoryChange,
  type MemorySettings,
} from '../api/memory.ts';
import {icon} from '../design-system/index.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {productionHandles, type AppHandles} from '../stores/app-handles.ts';
import {requestToast} from './toast-request.ts';
import type {ChatMemoryOperationDetail} from './chat-feature.ts';
import {modalResult, publishModalState, showOwnedModal} from './modal.ts';

const MAX_SEEN_MEMORY_OPERATIONS = 500;
type MemoryReadResult = 'loaded' | 'stale' | 'failed';

function memorySummary(event: ChatMemoryOperationDetail): string {
  const body = String(event.body || '').replace(/\s+/g, ' ').trim();
  const concise = body.length > 120 ? `${body.slice(0, 117)}…` : body;
  if (event.outcome === 'unchanged') {
    return event.operation === 'forget'
      ? msg('Already forgotten.', {id: 'settings.memory.alreadyForgotten'})
      : msg('Already remembered.', {id: 'settings.memory.alreadyRemembered'});
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
    personalMcpConnections: {attribute: false},
    showConnections: {state: true},
    handles: {attribute: false},
    deleteAllConversations: {attribute: false},
    memory: {state: true},
    memoryLoading: {state: true},
    memoryPending: {state: true},
    memoryRecords: {state: true},
    memoryListOpen: {state: true},
    language: {state: true},
  };

  declare personalMcpConnections: boolean;
  declare showConnections: boolean;
  declare handles: AppHandles;
  declare deleteAllConversations: (returnFocus?: HTMLElement | null) => Promise<boolean>;
  declare memory: MemorySettings | null;
  declare memoryLoading: boolean;
  declare memoryPending: boolean;
  declare memoryRecords: MemoryRecord[] | null;
  declare memoryListOpen: boolean;
  declare language: LanguagePreference;

  #events: AbortController | null = null;
  #returnFocus: HTMLElement | null = null;
  #seenMemoryOperations = new Set<string>();
  #memoryReadGeneration = 0;
  readonly #memoryPager = new KeysetPager<MemoryRecord>(
    (cursor, signal) => listMemories(cursor || null, signal),
    () => this.requestUpdate(),
  );

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.personalMcpConnections = false;
    this.showConnections = false;
    this.handles = productionHandles();
    this.deleteAllConversations = async () => false;
    this.memory = null;
    this.memoryLoading = false;
    this.memoryPending = false;
    this.memoryRecords = null;
    this.memoryListOpen = false;
    this.language = currentLanguagePreference();
    /** Store reads: conversations.length. */
    new StoreController(this, this.handles.conversations);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#events = new AbortController();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    this.#invalidateMemoryReads();
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
    this.showConnections = true;
    if (!this.memoryPending) this.memoryLoading = true;
    await this.updateComplete;
    if (signal.aborted) return;
    const dialog = this.#dialog();
    if (!dialog) return;
    if (!dialog.open) {
      dialog.returnValue = '';
      showOwnedModal(this, dialog);
      document.body.classList.add('settings-open');
    }
    // The visible dialog and Connections never wait for the Memory service.
    void this.#refreshMemory();
  }

  /** Consume one live Profile Memory domain fact from Chat composition. */
  handleMemoryOperation(event: ChatMemoryOperationDetail): void {
    if (!event.live) return;
    const identity = event.changeId || `${event.intent_id || ''}:${event.operation}:${event.outcome}`;
    if (!identity || this.#seenMemoryOperations.has(identity)) return;
    if (this.#seenMemoryOperations.size >= MAX_SEEN_MEMORY_OPERATIONS) {
      const oldest = this.#seenMemoryOperations.values().next().value;
      if (oldest) this.#seenMemoryOperations.delete(oldest);
    }
    this.#seenMemoryOperations.add(identity);
    const message = memorySummary(event);
    if (event.outcome !== 'changed' || !event.changeId) {
      this.#notifyMemory({message, duration: 3000});
      return;
    }
    const changeId = event.changeId;
    const signal = this.#events?.signal;
    this.#notifyMemory({
      message,
      action: {
        actionLabel: msg('Undo', {id: 'settings.memory.undo'}),
        duration: 3000,
        onAction: async () => {
          if (this.memoryPending) throw new Error('Memory operation in progress');
          this.memoryPending = true;
          this.#invalidateMemoryReads();
          try {
            const receipt = await undoMemoryChange(changeId, signal);
            if (receipt.outcome !== 'changed') throw new Error('Memory undo conflicted');
          } catch (error) {
            if (!signal?.aborted) this.#notifyMemory({
              message: msg('Could not undo the change.', {id: 'toast.undoFailed'}),
            });
            throw error;
          } finally {
            this.memoryPending = false;
            void this.#refreshMemory();
          }
          if (!signal?.aborted) this.#notifyMemory({
            message: msg('Profile Memory change undone.', {id: 'settings.memory.changeUndone'}),
          });
          return msg('Profile Memory change undone.', {id: 'settings.memory.changeUndone'});
        },
      },
    });
    void this.#refreshMemory();
  }

  protected override render(): TemplateResult {
    const total = this.handles.conversations.conversations.length;
    const active = this.memory?.activeCount;
    return html`
      <dialog id="settings-dialog" class="settings-dialog" aria-labelledby="settings-title"
              @click=${this.#scrimClick} @close=${this.#closed}>
        <form method="dialog" novalidate>
          <div class="settings-drawer-body">
            <div class="settings-header">
              <h2 id="settings-title">${msg('Settings', {id: 'settings.title'})}</h2>
              <button class="panel-close settings-close" type="submit" value="close-settings"
                      aria-label=${msg('Close settings', {id: 'settings.close'})}>${icon('close', {size: 'sm'})}</button>
            </div>
            ${this.personalMcpConnections ? html`<section class="settings-section">
              <h3 id="settings-connections">${msg('Connections', {id: 'settings.connections'})}</h3>
              ${this.showConnections ? html`<dl-settings-connections></dl-settings-connections>` : nothing}
            </section>` : nothing}
            <section class="settings-section">
              <h3 id="settings-memory">${msg('Profile Memory', {id: 'settings.profileMemory'})}</h3>
              ${this.memory ? html`<label class="dl-dialog-checkbox">
                <input type="checkbox" id="memory-enabled-toggle"
                       .checked=${this.memory.enabled}
                       ?disabled=${this.memoryLoading || this.memoryPending}
                       @change=${this.#toggleMemory}>
                ${msg('Activate profile memories', {id: 'settings.activateMemories'})}
              </label>` : html`<p class="settings-note" role="status">${this.memoryLoading
                ? msg('Loading memory settings…', {id: 'settings.memoryLoading'})
                : msg('Could not load memory settings.', {id: 'settings.memoryLoadFailed'})}</p>`}
              <p id="memory-active-count" class="settings-count" aria-live="polite"
                 ?hidden=${active === null || active === undefined}>
                ${active === 1
                  ? msg('1 stored item', {id: 'settings.oneStoredItem'})
                  : msg(str`${active ?? 0} stored items`, {id: 'settings.nStoredItems'})}
              </p>
              ${this.memory?.enabled ? this.#memoryList() : nothing}
              ${this.showConnections ? html`
                <dl-toast-region class="settings-memory-feedback" role="status" aria-live="polite"></dl-toast-region>
              ` : nothing}
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
        <form method="dialog" novalidate>
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

  /** The OAuth return path lands on the Connections surface with its group already open. */
  expandConnections(): void {
    this.querySelector('dl-settings-connections')?.expand();
  }

  #closed = (): void => {
    const toast = this.querySelector('dl-toast-region');
    const notice = toast?.request;
    if (notice?.action && !toast?.pending) requestToast(this, {message: notice.message, action: notice.action});
    this.showConnections = false;
    this.memoryListOpen = false;
    this.#invalidateMemoryReads();
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
      if (!signal.aborted) {
        this.memory = memory;
        if (!memory.enabled) this.memoryListOpen = false;
      }
    } catch {
      if (!signal.aborted) {
        input.checked = !requested;
        this.#notifyMemory({
          message: msg('Could not save memory settings.', {id: 'settings.memorySaveFailed'}),
          duration: 3000,
        });
      }
    } finally {
      if (!signal.aborted) {
        this.memoryPending = false;
        this.#reloadMemoryList();
      }
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
      if (!signal.aborted) {
        this.#notifyMemory({
          message: msg('Memory cleared.', {id: 'settings.memoryCleared'}),
          duration: 3000,
        });
      }
    } catch {
      if (!signal.aborted) {
        this.#notifyMemory({
          message: msg('Could not clear memory.', {id: 'settings.memoryClearFailedToast'}),
          duration: 3000,
        });
      }
    } finally {
      if (!signal.aborted) {
        this.memoryPending = false;
        await this.#refreshMemory();
      }
    }
  };

  #notifyMemory(detail: ToastRequestDetail): void {
    const toast = this.#dialog()?.open ? this.querySelector('dl-toast-region') : null;
    if (!toast) requestToast(this, detail);
    else if (detail.action) toast.showAction(detail.message, detail.action);
    else toast.show(detail.message, detail.duration);
  }

  #memoryList(): TemplateResult {
    const pager = this.#memoryPager;
    return html`
      <details class="memory-list" ?open=${this.memoryListOpen} @toggle=${this.#toggleMemoryList}>
        <summary>${msg('View memories', {id: 'settings.memory.view'})}</summary>
        <ul aria-label=${msg('Stored memories', {id: 'settings.memory.stored'})}>
          ${repeat(this.memoryRecords ?? [], (record) => record.memoryId, (record) => html`
            <li>
              <p>${record.body}</p>
              <button type="button" class="dl-btn dl-btn-danger-text"
                      ?disabled=${this.memoryPending || this.memoryLoading}
                      @click=${() => { void this.#forgetMemory(record); }}>
                ${msg('Forget', {id: 'settings.memory.forget'})}
              </button>
            </li>
          `)}
        </ul>
        <p class="settings-note" role="status">${pager.state === 'loading'
          ? msg('Loading memories…', {id: 'settings.memory.listLoading'})
          : pager.state === 'error'
            ? msg('Could not load memories.', {id: 'settings.memory.listFailed'})
            : this.memoryRecords?.length === 0
              ? msg('No stored memories.', {id: 'settings.memory.empty'}) : nothing}</p>
        ${pager.hasOlder ? html`
          <button type="button" class="dl-btn" data-memory-load-more
                  ?disabled=${this.memoryPending || pager.state === 'loading'}
                  @click=${() => { void this.#loadMemoryPage(); }}>
            ${pager.state === 'error'
              ? msg('Retry', {id: 'settings.memory.retry'})
              : msg('Load more', {id: 'settings.memory.loadMore'})}
          </button>
        ` : nothing}
      </details>`;
  }

  #toggleMemoryList = (event: Event): void => {
    this.memoryListOpen = (event.currentTarget as HTMLDetailsElement).open;
    if (this.memoryListOpen && this.memoryRecords === null && !this.#memoryPager.hasOlder) {
      this.#reloadMemoryList();
    }
  };

  #reloadMemoryList(): void {
    if (!this.memoryListOpen || !this.memory?.enabled || !this.#dialog()?.open) return;
    this.memoryRecords = null;
    // Empty cursor denotes the first page; subsequent cursors come from the server.
    this.#memoryPager.reset('');
    void this.#loadMemoryPage();
  }

  async #loadMemoryPage(): Promise<void> {
    const trigger = this.querySelector<HTMLButtonElement>('[data-memory-load-more]');
    const restoreFocus = trigger !== null && document.activeElement === trigger;
    await this.#memoryPager.loadNext((page) => {
      const records = new Map((this.memoryRecords ?? []).map((record) => [record.memoryId, record]));
      for (const record of page.items) records.set(record.memoryId, record);
      this.memoryRecords = [...records.values()];
    });
    await this.updateComplete;
    if (restoreFocus && !this.#memoryPager.hasOlder && this.memoryListOpen) {
      this.querySelector<HTMLElement>('.memory-list summary')?.focus();
    }
  }

  async #forgetMemory(record: MemoryRecord): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted || this.memoryPending || this.memoryLoading) return;
    this.memoryPending = true;
    this.#invalidateMemoryReads();
    try {
      const receipt = await forgetMemory(record.memoryId, signal);
      if (signal.aborted) return;
      this.handleMemoryOperation({
        live: true, operation: receipt.action, outcome: receipt.outcome,
        changeId: receipt.changeId, body: receipt.body || record.body,
      });
    } catch {
      if (!signal.aborted) this.#notifyMemory({
        message: msg('Could not forget this memory.', {id: 'settings.memory.forgetFailed'}), duration: 3000,
      });
    } finally {
      if (!signal.aborted) {
        this.memoryPending = false;
        void this.#refreshMemory();
        await this.updateComplete;
        const toast = this.querySelector('dl-toast-region');
        await toast?.updateComplete;
        if (this.#dialog()?.open) {
          (toast?.querySelector<HTMLButtonElement>('button')
            ?? this.querySelector<HTMLElement>('.memory-list summary'))?.focus();
        }
      }
    }
  }

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


  #invalidateMemoryReads(): void {
    this.#memoryReadGeneration += 1;
    this.memoryLoading = false;
    this.#memoryPager.reset(null);
    this.memoryRecords = null;
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
      this.memory = null;
      return 'failed';
    } finally {
      if (!signal.aborted && generation === this.#memoryReadGeneration) this.memoryLoading = false;
    }
  }

  async #refreshMemory(): Promise<void> {
    if (this.memoryPending) return;
    this.#memoryPager.reset(null);
    this.memoryRecords = null;
    if (await this.#readMemory() === 'loaded') this.#reloadMemoryList();
  }
}

customElements.define('dl-settings-dialog', DlSettingsDialog);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-dialog': DlSettingsDialog;
  }
}
