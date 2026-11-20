// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Profile Memory: the capability switch, the stored memories, and their receipts.
 *
 * Unlike the other pages this element stays in the dialog while Settings is closed, because a live
 * Memory change (the agent remembered something in a Run) arrives with its Undo whenever Chat
 * says so. While Settings is closed it only turns that fact into a toast. `active` says Settings
 * is open: only then does the page read, list, or report, and closing drops everything it read.
 *
 * The page is a window of its own: the switch holds the top, Clear holds the foot, and the stored
 * memories scroll in the card between them (the dialog's pane never scrolls this page).
 */

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  clearMemory,
  forgetMemory,
  getMemorySettings,
  listMemories,
  putMemorySettings,
  undoMemoryChange,
  type MemoryOperationEvent,
  type MemoryPage,
  type MemoryRecord,
  type MemorySettings,
} from '../api/memory.ts';
import {icon} from '../design-system/index.ts';
import {PHONE_DIALOG_MEDIA} from '../lib/breakpoints.ts';
import {raise} from '../lib/dom.ts';
import {LightElement, MediaController} from '../lib/lit-host.ts';
import {KeysetPager} from '../lib/paged.ts';
import shared from '../styles/settings-page.module.css';
import styles from '../styles/settings-memory.module.css';
import {loadOlderControl} from './load-older.ts';
import {memoryKindLabel, memorySummary} from './memory-words.ts';
import {modalResult} from './modal.ts';
import {commitSwitch, dangerCard, loadState, switchCard} from './settings-parts.ts';
import {reportSettingsSummary} from './settings-summary.ts';
import {requestToast} from './toast-request.ts';

const MAX_SEEN_MEMORY_OPERATIONS = 500;
type MemoryReadResult = 'loaded' | 'stale' | 'failed';

/** Owns Profile Memory state: its asynchronous reads and mutations, receipts, and focus. */
class DlSettingsMemory extends LightElement {
  static properties = {
    active: {attribute: false},
    current: {attribute: false},
    memory: {state: true},
    loading: {state: true},
    pending: {state: true},
    records: {state: true},
  };

  /** Settings is open: this page may read, and its navigation row wants its status. */
  declare active: boolean;
  /** This page is the one showing, so its list is worth loading. */
  declare current: boolean;
  declare memory: MemorySettings | null;
  declare loading: boolean;
  declare pending: boolean;
  declare records: MemoryRecord[] | null;

  readonly #seenOperations = new Set<string>();
  #readGeneration = 0;
  readonly #pager = new KeysetPager<MemoryPage>(
    (cursor, signal) => listMemories(cursor, signal),
    () => this.requestUpdate(),
  );
  readonly #phone = new MediaController(this, PHONE_DIALOG_MEDIA);

  constructor() {
    super();
    this.active = false;
    this.current = false;
    this.memory = null;
    this.loading = false;
    this.pending = false;
    this.records = null;
  }

  override disconnectedCallback(): void {
    this.#invalidateReads();
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('active')) {
      if (this.active) {
        // The first frame of an open page says "loading", never "off" for a switch nobody has read.
        if (!this.pending) this.loading = true;
        void this.#refresh();
      } else if (changed.get('active')) {
        this.#invalidateReads();
        this.memory = null;
      }
    } else if (changed.has('current') && this.current && this.records === null
      && this.#pager.state !== 'loading') {
      this.#reloadList();
    }
  }

  protected override updated(changed: PropertyValues<this>): void {
    // This page's top is its window's, because the dialog's pane never scrolls it: a page that
    // opens again opens on the newest memories.
    if (changed.has('current') && this.current) this.#listWindow()?.scrollTo({top: 0});
    if (this.active && (changed.has('memory') || changed.has('active'))) {
      reportSettingsSummary(this, {
        section: 'memory',
        enabled: this.memory?.enabled ?? null,
        count: this.memory?.activeCount ?? null,
      });
    }
  }

  /** Consume one live Profile Memory domain fact from Chat composition. */
  handleOperation(event: MemoryOperationEvent): void {
    this.#receive(event, false);
  }

  /** Turn one live fact into its receipt; `takeFocus` is for a change whose own control is gone. */
  #receive(event: MemoryOperationEvent, takeFocus: boolean): void {
    if (!event.live) return;
    const identity = event.changeId || `${event.intentId || ''}:${event.operation}:${event.outcome}`;
    if (!identity || this.#seenOperations.has(identity)) return;
    if (this.#seenOperations.size >= MAX_SEEN_MEMORY_OPERATIONS) {
      const oldest = this.#seenOperations.values().next().value;
      if (oldest) this.#seenOperations.delete(oldest);
    }
    this.#seenOperations.add(identity);
    const message = memorySummary(event);
    if (event.outcome !== 'changed' || !event.changeId) {
      const refused = event.outcome === 'conflict' || event.outcome === 'rejected';
      requestToast(this, {message, tone: refused ? 'error' : 'receipt'});
      return;
    }
    const changeId = event.changeId;
    const signal = this.lifetime;
    requestToast(this, {
      message,
      action: {
        actionLabel: msg('Undo', {id: 'settings.memory.undo'}),
        focus: takeFocus,
        onAction: async () => {
          if (this.pending) throw new Error('Memory operation in progress');
          this.pending = true;
          this.#invalidateReads();
          try {
            const receipt = await undoMemoryChange(changeId, signal);
            if (receipt.outcome !== 'changed') throw new Error('Memory undo conflicted');
            this.#profileChanged();
          } finally {
            this.pending = false;
            void this.#refresh();
          }
          if (!signal.aborted) requestToast(this, {
            message: msg('Profile Memory change undone.', {id: 'settings.memory.changeUndone'}),
          });
          return msg('Profile Memory change undone.', {id: 'settings.memory.changeUndone'});
        },
      },
    });
    void this.#refresh();
  }

  protected override render(): TemplateResult {
    // A closed dialog shows nothing, and nothing read is kept in it.
    if (!this.active) return html``;
    return html`
      ${this.memory ? this.#page(this.memory) : loadState({
    host: this,
    failed: !this.loading && !this.pending,
    loading: msg('Loading memory settings…', {id: 'settings.memoryLoading'}),
    failure: msg('Could not load memory settings.', {id: 'settings.memoryLoadFailed'}),
    retry: msg('Retry', {id: 'settings.memory.retry'}),
    onRetry: () => this.#refresh(),
  })}
      <dialog id="clear-memory-dialog" class="confirm-dialog" aria-labelledby="clear-memory-title">
        <form method="dialog" novalidate>
          <h2 id="clear-memory-title">${msg('Clear Profile memory?', {id: 'settings.clearMemoryTitle'})}</h2>
          <p>${this.#clearBody()}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'settings.cancel'})}</button>
            <button type="submit" value="clear" class="dl-dialog-danger">${msg('Clear memory', {id: 'settings.clearMemoryConfirm'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #clearBody(): string {
    return msg('Remembered preferences and facts will be forgotten. Conversations are not affected.', {
      id: 'settings.clearMemoryBody',
    });
  }

  #page(memory: MemorySettings): TemplateResult {
    return html`
      <div class="${shared.stack} ${memory.enabled ? styles.window : ''}">
        ${switchCard({
          id: 'memory-enabled-toggle',
          label: msg('Activate profile memories', {id: 'settings.activateMemories'}),
          caption: memory.enabled
            ? msg('The agent remembers your preferences and facts across conversations', {
              id: 'settings.memoryOn',
            })
            : msg('Off: stored memories are kept, but the agent neither reads nor writes them', {
              id: 'settings.memoryOff',
            }),
          checked: memory.enabled,
          disabled: this.loading || this.pending,
          onToggle: this.#toggle,
        })}
        ${memory.enabled ? html`${this.#list(memory)}${this.#clearCard()}` : html`
          <div class=${styles.info}>
            <span class=${styles.infoIcon}>${icon('info', {size: 'sm'})}</span>
            <span>${msg('Turn it on to view, forget or clear the stored memories.', {
              id: 'settings.memoryOffNote',
            })}</span>
          </div>`}
      </div>`;
  }

  #list(memory: MemorySettings): TemplateResult {
    const pager = this.#pager;
    const firstPage = this.records === null;
    const count = memory.activeCount ?? this.records?.length ?? null;
    const rows = this.records ?? [];
    return html`
      <section class="${shared.card} ${shared.divided} ${styles.listCard}"
               aria-labelledby="memory-list-title">
        <div class=${styles.listHeader}>
          <h4 id="memory-list-title" class=${styles.listTitle} tabindex="-1">${
            msg('Stored memories', {id: 'settings.memory.stored'})}</h4>
          ${count === null ? nothing : html`<span class=${styles.badge}>${count}</span>`}
        </div>
        ${rows.length ? html`
          <div class=${styles.listWindow} data-memory-window role="region" tabindex="0"
               aria-labelledby="memory-list-title">
            <ul class="${shared.list} ${shared.divided}">
              ${repeat(rows, (record) => record.memoryId, (record) => html`
                <li class=${styles.memory}>
                  <p class=${styles.text}>${record.body}</p>
                  <span class=${styles.kind}>${memoryKindLabel(record.kind)}</span>
                  <dl-icon-button class="${styles.forget} ${shared.iconAction}" name="close" size="sm"
                    aria-label=${msg('Forget this memory', {id: 'settings.memory.forget'})}
                    ?disabled=${this.pending || this.loading}
                    @click=${() => { void this.#forget(record); }}></dl-icon-button>
                </li>
              `)}
            </ul>
          </div>` : nothing}
        <p class=${styles.status} role="status">${pager.starting
          ? msg('Loading memories…', {id: 'settings.memory.listLoading'})
          : firstPage && pager.state === 'error'
            ? msg('Could not load memories.', {id: 'settings.memory.listFailed'})
            : this.records?.length === 0
              ? msg('No stored memories.', {id: 'settings.memory.empty'}) : nothing}</p>
        ${firstPage && pager.state === 'error' ? html`
          <div class=${styles.footer}>
            <button type="button" class="dl-btn" @click=${() => { this.#reloadList(); }}>
              ${msg('Retry', {id: 'settings.memory.retry'})}
            </button>
          </div>` : nothing}
        ${loadOlderControl({
          list: 'memories',
          pages: pager,
          label: msg('Load more', {id: 'settings.memory.loadMore'}),
          retryLabel: msg('Retry', {id: 'settings.memory.retry'}),
          loading: msg('Loading more memories…', {id: 'settings.memory.loadingMore'}),
          loaded: msg('Loaded more memories.', {id: 'settings.memory.loadedMore'}),
          failed: msg('More memories could not be loaded.', {id: 'settings.memory.moreFailed'}),
          onLoad: this.#loadMore,
          rowClass: styles.footer,
          buttonClass: 'dl-btn',
        })}
      </section>`;
  }

  #clearCard(): TemplateResult {
    return dangerCard({
      action: msg('Clear all memories', {id: 'settings.clearAll'}),
      short: msg('Clear…', {id: 'settings.clearButton'}),
      caption: this.#clearBody(),
      phone: this.#phone.matches,
      disabled: this.pending,
      onClick: this.#clear,
    });
  }

  #toggle = async (event: Event): Promise<void> => {
    const memory = this.memory;
    if (!memory) return;
    await commitSwitch(this, event, {
      signal: this.lifetime,
      send: (signal) => putMemorySettings(!memory.enabled, signal),
      accept: (reply) => {
        this.memory = reply;
        this.#profileChanged();
      },
      failure: msg('Could not save memory settings.', {id: 'settings.memorySaveFailed'}),
      begin: () => { this.#invalidateReads(); },
      settled: () => { this.#reloadList(); },
    });
  };

  #clear = async (event: Event): Promise<void> => {
    const signal = this.lifetime;
    const button = event.currentTarget as HTMLButtonElement;
    const confirm = this.querySelector<HTMLDialogElement>('#clear-memory-dialog');
    if (signal.aborted || !confirm || this.pending) return;
    if (await modalResult(this, confirm, () => button.focus(), signal) !== 'clear') return;
    this.pending = true;
    this.#invalidateReads();
    try {
      await clearMemory(signal);
      if (!signal.aborted) {
        this.#profileChanged();
        requestToast(this, {
          message: msg('Memory cleared.', {id: 'settings.memoryCleared'}),
        });
      }
    } catch {
      if (!signal.aborted) {
        requestToast(this, {
          tone: 'error',
          message: msg('Could not clear memory.', {id: 'settings.memoryClearFailedToast'}),
        });
      }
    } finally {
      if (!signal.aborted) {
        this.pending = false;
        await this.#refresh();
      }
    }
  };

  /** The owner changed their profile here: the notices under their answers read it again. */
  #profileChanged(): void {
    raise(this, 'dl-profile-memory-change');
  }

  /** The list is dropped, not refreshed in place, so its next page goes with it. */
  #reloadList(): void {
    if (!this.active || !this.current || !this.memory?.enabled) return;
    this.#pager.reset(null);
    this.records = null;
    void this.#pager.start((page) => { this.#add(page.items); });
  }

  #loadMore = async (event: Event): Promise<void> => {
    const restoreFocus = document.activeElement === event.currentTarget;
    await this.#pager.loadNext((page) => { this.#add(page.items); });
    await this.updateComplete;
    if (restoreFocus && !this.#pager.hasOlder) this.#listTitle()?.focus();
  };

  #add(items: readonly MemoryRecord[]): void {
    const records = new Map((this.records ?? []).map((record) => [record.memoryId, record]));
    for (const record of items) records.set(record.memoryId, record);
    this.records = [...records.values()];
  }

  #listTitle(): HTMLElement | null {
    return this.querySelector<HTMLElement>('#memory-list-title');
  }

  /** The box the stored memories scroll in, once there are any. */
  #listWindow(): HTMLElement | null {
    return this.querySelector<HTMLElement>('[data-memory-window]');
  }

  async #forget(record: MemoryRecord): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted || this.pending || this.loading) return;
    this.pending = true;
    this.#invalidateReads();
    // The row the reader is on is about to go: the Undo that replaces it takes focus, or the list's
    // title does when nothing can be undone.
    let undoable = false;
    try {
      const receipt = await forgetMemory(record.memoryId, signal);
      if (signal.aborted) return;
      this.#profileChanged();
      undoable = receipt.outcome === 'changed' && receipt.changeId !== '';
      this.#receive({
        live: true, operation: 'forget', outcome: receipt.outcome,
        changeId: receipt.changeId, intentId: null, body: receipt.body || record.body,
      }, undoable);
    } catch {
      if (!signal.aborted) requestToast(this, {
        tone: 'error',
        message: msg('Could not forget this memory.', {id: 'settings.memory.forgetFailed'}),
      });
    } finally {
      if (!signal.aborted) {
        this.pending = false;
        void this.#refresh();
        await this.updateComplete;
        if (!undoable) this.#listTitle()?.focus();
      }
    }
  }

  #invalidateReads(): void {
    this.#readGeneration += 1;
    this.loading = false;
    this.#pager.reset(null);
    this.records = null;
  }

  async #read(): Promise<MemoryReadResult> {
    const signal = this.lifetime;
    if (signal.aborted) return 'stale';
    const generation = ++this.#readGeneration;
    try {
      const memory = await getMemorySettings(signal);
      if (signal.aborted || generation !== this.#readGeneration) return 'stale';
      this.memory = memory;
      return 'loaded';
    } catch {
      if (signal.aborted || generation !== this.#readGeneration) return 'stale';
      this.memory = null;
      return 'failed';
    } finally {
      if (!signal.aborted && generation === this.#readGeneration) this.loading = false;
    }
  }

  async #refresh(): Promise<void> {
    if (this.pending || !this.active) return;
    this.#pager.reset(null);
    this.records = null;
    if (await this.#read() === 'loaded') this.#reloadList();
  }
}

customElements.define('dl-settings-memory', DlSettingsMemory);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-memory': DlSettingsMemory;
  }

  interface HTMLElementEventMap {
    /** The owner changed their Profile Memory from Settings: they turned it on or off, cleared it, or forgot or
     *  restored a record. What the page read of it is stale. */
    'dl-profile-memory-change': CustomEvent<void>;
  }
}
