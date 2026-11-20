// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The messages a reader queued behind the conversation's live Run.
 *
 * A send that arrives while a Run is working is held for the next turn; this strip is where the
 * reader sees, edits, reorders and drops those held words before they are delivered. It draws
 * nothing while the queue is empty, so a conversation without held work costs no space.
 *
 * One caption line when collapsed — the count and the promise that the words follow this run —
 * and a bounded list when the reader opens it. The words themselves are the reader's: the strip
 * only shortens them for the row and never rewrites what the server will deliver.
 *
 * The strip owns its read; the chat calls `refresh()` after a delivery or a settlement, the two
 * moments the queue can change under it.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  editQueuedMessage,
  type QueuedMessage,
  readConversationQueue,
  removeQueuedMessage,
  reorderConversationQueue,
} from '../api/conversations.ts';
import {icon } from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import styles from '../styles/queue-strip.module.css';
import {requestToast} from './toast-request.ts';

export class DlQueueStrip extends LightElement {
  static override properties = {
    conversationId: {attribute: false},
  };

  declare conversationId: string | null;
  #items: readonly QueuedMessage[] = [];
  #expanded = false;
  #editingId: string | null = null;
  #draft = '';
  #busy = false;

  override connectedCallback(): void {
    super.connectedCallback();
    void this.#load();
  }

  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    if (changed.has('conversationId')) {
      this.#expanded = false;
      this.#editingId = null;
      this.#draft = '';
      void this.#load();
    }
  }

  /** Re-read the queue; the chat calls this after a delivery or a settlement. */
  refresh(): void {
    void this.#load();
  }

  async #load(): Promise<void> {
    const conversationId = this.conversationId;
    if (!conversationId) {
      this.#items = [];
      return;
    }
    try {
      const items = await readConversationQueue(conversationId, this.lifetime);
      if (this.conversationId !== conversationId || !this.isConnected) return;
      this.#items = items;
    } catch {
      if (this.conversationId !== conversationId || !this.isConnected) return;
      // A strip that cannot read shows nothing; the conversation and the held words are unaffected.
      this.#items = [];
    }
    if (this.#items.length === 0) {
      this.#expanded = false;
      this.#editingId = null;
    }
    this.requestUpdate();
  }

  #toggle = (): void => {
    this.#expanded = !this.#expanded;
    this.requestUpdate();
  };

  #beginEdit(item: QueuedMessage): void {
    this.#editingId = item.messageId;
    this.#draft = item.content;
    this.requestUpdate();
    void this.updateComplete.then(() => {
      this.querySelector<HTMLInputElement>(`.${styles.input}`)?.focus();
    });
  }

  #cancelEdit(): void {
    this.#editingId = null;
    this.#draft = '';
    this.requestUpdate();
  }

  async #saveEdit(messageId: string): Promise<void> {
    const conversationId = this.conversationId;
    const content = this.#draft.trim();
    if (!conversationId || !content || this.#busy) return;
    this.#busy = true;
    try {
      const rewritten = await editQueuedMessage(conversationId, messageId, content, this.lifetime);
      if (this.conversationId !== conversationId) return;
      this.#items = this.#items.map((item) => (
        item.messageId === rewritten.messageId ? rewritten : item
      ));
      this.#editingId = null;
      this.#draft = '';
    } catch {
      if (this.isConnected) {
        requestToast(this, {
          tone: 'error',
          message: msg('Could not edit this queued message.', {id: 'queueStrip.editFailed'}),
        });
      }
    } finally {
      this.#busy = false;
      if (this.isConnected) this.requestUpdate();
    }
  }

  async #remove(messageId: string): Promise<void> {
    const conversationId = this.conversationId;
    if (!conversationId || this.#busy) return;
    this.#busy = true;
    try {
      await removeQueuedMessage(conversationId, messageId, this.lifetime);
      if (this.conversationId !== conversationId) return;
      this.#items = this.#items.filter((item) => item.messageId !== messageId);
    } catch {
      if (this.isConnected) {
        requestToast(this, {
          tone: 'error',
          message: msg('Could not remove this queued message.', {
            id: 'queueStrip.removeFailed',
          }),
        });
      }
    } finally {
      this.#busy = false;
      if (this.isConnected) this.requestUpdate();
    }
  }

  async #move(index: number, delta: number): Promise<void> {
    const conversationId = this.conversationId;
    const target = index + delta;
    if (!conversationId || this.#busy || target < 0 || target >= this.#items.length) return;
    this.#busy = true;
    const order = this.#items.map((item) => item.messageId);
    [order[index], order[target]] = [order[target], order[index]];
    try {
      const reordered = await reorderConversationQueue(conversationId, order, this.lifetime);
      if (this.conversationId === conversationId) this.#items = reordered;
    } catch {
      if (this.isConnected) {
        requestToast(this, {
          tone: 'error',
          message: msg('Could not reorder the queue.', {id: 'queueStrip.reorderFailed'}),
        });
      }
    } finally {
      this.#busy = false;
      if (this.isConnected) this.requestUpdate();
    }
  }

  #iconButton(
    name: 'check' | 'close' | 'edit' | 'next' | 'previous',
    label: string,
    onClick: () => void,
    options: {danger?: boolean} = {},
  ): TemplateResult {
    return html`
      <button
        class="dl-icon-button ${options.danger ? styles.danger : ''}"
        type="button"
        aria-label=${label}
        title=${label}
        ?disabled=${this.#busy}
        @click=${onClick}
      >${icon(name, {size: 'xs'})}</button>
    `;
  }

  #row(item: QueuedMessage, index: number): TemplateResult {
    if (this.#editingId === item.messageId) {
      return html`
        <li class=${styles.row}>
          <input
            class=${styles.input}
            .value=${this.#draft}
            aria-label=${msg('Queued message', {id: 'queueStrip.messageAria'})}
            @input=${(event: Event) => {
              this.#draft = (event.target as HTMLInputElement).value;
            }}
            @keydown=${(event: KeyboardEvent) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void this.#saveEdit(item.messageId);
              } else if (event.key === 'Escape') {
                event.preventDefault();
                this.#cancelEdit();
              }
            }}
          >
          ${this.#iconButton('check', msg('Save', {id: 'queueStrip.save'}), () => {
            void this.#saveEdit(item.messageId);
          })}
          ${this.#iconButton('close', msg('Cancel', {id: 'queueStrip.cancel'}), () => {
            this.#cancelEdit();
          })}
        </li>
      `;
    }
    return html`
      <li class=${styles.row}>
        <span class=${styles.content} title=${item.content}>${item.content}</span>
        ${this.#iconButton('edit', msg('Edit', {id: 'queueStrip.edit'}), () => {
          this.#beginEdit(item);
        })}
        ${index > 0
          ? this.#iconButton('previous', msg('Move earlier', {id: 'queueStrip.moveEarlier'}), () => {
            void this.#move(index, -1);
          })
          : nothing}
        ${index < this.#items.length - 1
          ? this.#iconButton('next', msg('Move later', {id: 'queueStrip.moveLater'}), () => {
            void this.#move(index, 1);
          })
          : nothing}
        ${this.#iconButton('close', msg('Remove', {id: 'queueStrip.remove'}), () => {
          void this.#remove(item.messageId);
        }, {danger: true})}
      </li>
    `;
  }

  protected override render(): TemplateResult | typeof nothing {
    if (this.#items.length === 0) return nothing;
    const count = this.#items.length;
    return html`
      <div class=${styles.region}>
        <section class=${styles.strip} aria-label=${msg('Queued messages', {id: 'queueStrip.aria'})}>
          <button
            type="button"
            class=${styles.summary}
            aria-expanded=${String(this.#expanded)}
            @click=${this.#toggle}
          >
            ${icon('disclosure', {size: 'xs'})}
            <span class=${styles.count} aria-live="polite">
              ${msg(str`${count} queued after this run`, {id: 'queueStrip.summary'})}
            </span>
          </button>
          ${this.#expanded
            ? html`
              <ol class=${styles.list}>
                ${repeat(this.#items, (item) => item.messageId, (item, index) => this.#row(item, index))}
              </ol>`
            : nothing}
        </section>
      </div>
    `;
  }
}

customElements.define('dl-queue-strip', DlQueueStrip);

declare global {
  interface HTMLElementTagNameMap {
    'dl-queue-strip': DlQueueStrip;
  }
}
