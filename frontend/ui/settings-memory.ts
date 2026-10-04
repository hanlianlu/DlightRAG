// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Profile Memory: the capability switch, the stored memories, and their receipts.
 *
 * Unlike the other pages this element stays in the dialog while Settings is closed, because a live
 * Memory change (the agent remembered something in a Run) arrives with its Undo whenever Chat
 * says so. While Settings is closed it only turns that fact into a toast. `active` says Settings
 * is open: only then does the page read, list, or report, and closing drops everything it read.
 */

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
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
import {LightElement, MediaController} from '../lib/lit-host.ts';
import {KeysetPager} from '../lib/paged.ts';
import shared from '../styles/settings-page.module.css';
import styles from '../styles/settings-memory.module.css';
import {loadOlderControl} from './load-older.ts';
import {modalResult} from './modal.ts';
import {switchCard} from './settings-parts.ts';
import {reportSettingsSummary} from './settings-summary.ts';
import {requestToast} from './toast-request.ts';

const MAX_SEEN_MEMORY_OPERATIONS = 500;
type MemoryReadResult = 'loaded' | 'stale' | 'failed';

function memorySummary(event: MemoryOperationEvent): string {
  const body = event.body.replace(/\s+/g, ' ').trim();
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

/** Owns Profile Memory state: its asynchronous reads and mutations, receipts, and focus. */
export class DlSettingsMemory extends LightElement {
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

  #events: AbortController | null = null;
  readonly #seenOperations = new Set<string>();
  #readGeneration = 0;
  readonly #pager = new KeysetPager<MemoryPage>(
    (cursor, signal) => listMemories(cursor, signal),
    () => this.requestUpdate(),
  );
  readonly #phone = new MediaController(this, PHONE_DIALOG_MEDIA);

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.active = false;
    this.current = false;
    this.memory = null;
    this.loading = false;
    this.pending = false;
    this.records = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#events = new AbortController();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
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
      requestToast(this, {message, duration: 3000});
      return;
    }
    const changeId = event.changeId;
    const signal = this.#events?.signal;
    requestToast(this, {
      message,
      action: {
        actionLabel: msg('Undo', {id: 'settings.memory.undo'}),
        duration: 3000,
        focus: takeFocus,
        onAction: async () => {
          if (this.pending) throw new Error('Memory operation in progress');
          this.pending = true;
          this.#invalidateReads();
          try {
            const receipt = await undoMemoryChange(changeId, signal);
            if (receipt.outcome !== 'changed') throw new Error('Memory undo conflicted');
          } catch (error) {
            if (!signal?.aborted) requestToast(this, {
              message: msg('Could not undo the change.', {id: 'toast.undoFailed'}),
            });
            throw error;
          } finally {
            this.pending = false;
            void this.#refresh();
          }
          if (!signal?.aborted) requestToast(this, {
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
      ${this.memory ? this.#page(this.memory) : html`<p class=${shared.hint} role="status">${
        this.loading || this.pending
          ? msg('Loading memory settings…', {id: 'settings.memoryLoading'})
          : msg('Could not load memory settings.', {id: 'settings.memoryLoadFailed'})}</p>`}
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
      <div class=${shared.stack}>
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
    return html`
      <section class="${shared.card} ${shared.divided}" aria-labelledby="memory-list-title">
        <div class=${styles.listHeader}>
          <h4 id="memory-list-title" class=${styles.listTitle} tabindex="-1">${
            msg('Stored memories', {id: 'settings.memory.stored'})}</h4>
          ${count === null ? nothing : html`<span class=${styles.badge}>${count}</span>`}
        </div>
        ${this.records?.length ? html`<ul class=${styles.list}>
          ${repeat(this.records, (record) => record.memoryId, (record) => html`
            <li class=${styles.memory}>
              <span class=${styles.kind}>${record.kind === 'fact'
                ? msg('Fact', {id: 'settings.memory.kindFact'})
                : msg('Preference', {id: 'settings.memory.kindPreference'})}</span>
              <p class=${styles.body}>${record.body}</p>
              <dl-icon-button class=${styles.forget} name="close" size="sm"
                aria-label=${msg('Forget this memory', {id: 'settings.memory.forget'})}
                ?disabled=${this.pending || this.loading}
                @click=${() => { void this.#forget(record); }}></dl-icon-button>
            </li>
          `)}
        </ul>` : nothing}
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
    return html`
      <div class="${shared.card} ${shared.row} ${shared.dangerRow}">
        <span class=${shared.rowText}>
          <span class=${shared.rowLabel}>${msg('Clear all memories', {id: 'settings.clearAll'})}</span>
          <span class=${shared.rowCaption}>${this.#clearBody()}</span>
        </span>
        <button type="button" id="memory-clear-btn" class="dl-btn dl-btn-danger-text"
                aria-label=${msg('Clear all memories', {id: 'settings.clearAll'})}
                ?disabled=${this.pending} @click=${this.#clear}>${this.#phone.matches
          ? msg('Clear all memories', {id: 'settings.clearAll'})
          : msg('Clear…', {id: 'settings.clearButton'})}</button>
      </div>`;
  }

  #toggle = async (event: Event): Promise<void> => {
    const signal = this.#events?.signal;
    const toggle = event.currentTarget as HTMLElement;
    if (!signal || signal.aborted || this.pending || !this.memory) return;
    const requested = !this.memory.enabled;
    const focused = document.activeElement === toggle;
    this.pending = true;
    this.#invalidateReads();
    try {
      const memory = await putMemorySettings(requested, signal);
      if (!signal.aborted) this.memory = memory;
    } catch {
      if (!signal.aborted) {
        requestToast(this, {
          message: msg('Could not save memory settings.', {id: 'settings.memorySaveFailed'}),
          duration: 3000,
        });
      }
    } finally {
      if (!signal.aborted) {
        this.pending = false;
        this.#reloadList();
        await this.updateComplete;
        // A switch that was disabled for the request drops focus in some engines; give it back.
        if (focused) toggle.focus();
      }
    }
  };

  #clear = async (event: Event): Promise<void> => {
    const signal = this.#events?.signal;
    const button = event.currentTarget as HTMLButtonElement;
    const confirm = this.querySelector<HTMLDialogElement>('#clear-memory-dialog');
    if (!signal || signal.aborted || !confirm || this.pending) return;
    if (await modalResult(this, confirm, () => button.focus(), signal) !== 'clear') return;
    this.pending = true;
    this.#invalidateReads();
    try {
      await clearMemory(signal);
      if (!signal.aborted) {
        requestToast(this, {
          message: msg('Memory cleared.', {id: 'settings.memoryCleared'}),
          duration: 3000,
        });
      }
    } catch {
      if (!signal.aborted) {
        requestToast(this, {
          message: msg('Could not clear memory.', {id: 'settings.memoryClearFailedToast'}),
          duration: 3000,
        });
      }
    } finally {
      if (!signal.aborted) {
        this.pending = false;
        await this.#refresh();
      }
    }
  };

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

  async #forget(record: MemoryRecord): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted || this.pending || this.loading) return;
    this.pending = true;
    this.#invalidateReads();
    // The row the reader is on is about to go: the Undo that replaces it takes focus, or the list's
    // title does when nothing can be undone.
    let undoable = false;
    try {
      const receipt = await forgetMemory(record.memoryId, signal);
      if (signal.aborted) return;
      undoable = receipt.outcome === 'changed' && receipt.changeId !== '';
      this.#receive({
        live: true, operation: receipt.action, outcome: receipt.outcome,
        changeId: receipt.changeId, intentId: null, body: receipt.body || record.body,
      }, undoable);
    } catch {
      if (!signal.aborted) requestToast(this, {
        message: msg('Could not forget this memory.', {id: 'settings.memory.forgetFailed'}), duration: 3000,
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
    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return 'stale';
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
}
