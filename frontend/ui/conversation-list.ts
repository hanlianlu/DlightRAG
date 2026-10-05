// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {ConversationSummary} from '../api/conversations.ts';
import {
  type DlMenu,
  menuButtonFocus,
  type MenuDismissDetail,
  type MenuFocus,
} from '../design-system/index.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import {loadOlderControl} from './load-older.ts';

export interface ConversationIntentDetail {
  conversationId: string;
}

export interface ConversationRenameDetail extends ConversationIntentDetail {
  title: string;
}

export interface ConversationRetryDetail {
  kind: 'reload' | 'new';
}

const SKELETON_COUNT = 3;

function actionsMenuId(conversationId: string): string {
  return `conversation-actions-${conversationId}`;
}

/** Conversation rows, row accessibility, and item intent. */
export class DlConversationList extends LightElement {
  static properties = {
    handles: {attribute: false},
    busy: {attribute: false},
    openMenuId: {state: true},
    renameId: {state: true},
  };

  declare handles: AppHandles;
  declare busy: boolean;
  declare openMenuId: string | null;
  declare renameId: string | null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.busy = false;
    this.openMenuId = null;
    this.renameId = null;
    /** Store reads: conversations, listState, olderConversations, activeConversationId. */
    new StoreController(this, this.handles.conversations);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('click', (event) => {
      if (!this.openMenuId || !(event.target instanceof Node)) return;
      if (this.#row(this.openMenuId)?.contains(event.target)) return;
      this.openMenuId = null;
    }, {signal: this.lifetime});
  }

  get menuOpen(): boolean {
    return this.openMenuId !== null;
  }

  closeMenu(restoreFocus = false): void {
    const conversationId = this.openMenuId;
    if (conversationId === null) return;
    this.openMenuId = null;
    if (restoreFocus) void this.focusActions(conversationId);
  }

  async focusConversation(conversationId: string): Promise<boolean> {
    return this.#focusAfterRender('.conversation-select', conversationId);
  }

  async focusActions(conversationId: string): Promise<boolean> {
    return this.#focusAfterRender('.conversation-actions-button', conversationId);
  }

  #row(conversationId: string): HTMLElement | null {
    return this.querySelector<HTMLElement>(
      `[data-conversation-id="${CSS.escape(conversationId)}"]`,
    );
  }

  async #focusAfterRender(selector: string, conversationId: string): Promise<boolean> {
    await this.updateComplete;
    const target = this.#row(conversationId)?.querySelector<HTMLElement>(selector);
    if (!target) return false;
    target.focus();
    return true;
  }

  #emit<D>(type: string, detail: D): void {
    this.dispatchEvent(new CustomEvent<D>(type, {detail, bubbles: true, composed: true}));
  }

  #openMenu(conversationId: string, focus: MenuFocus = 'first'): void {
    this.openMenuId = conversationId;
    this.renameId = null;
    void this.updateComplete.then(() => {
      this.#row(conversationId)?.querySelector<DlMenu>('dl-menu')?.focusItem(focus);
    });
  }

  #rowIdFromEvent(event: Event): string | null {
    const target = event.target;
    if (!(target instanceof Element)) return null;
    if (target.closest('dl-menu, .conversation-actions-button, input')) return null;
    return target.closest('[data-conversation-id]')?.getAttribute('data-conversation-id') ?? null;
  }

  #selectFromPointer = (event: MouseEvent): void => {
    if (this.busy) return;
    if (event.detail >= 2) return;
    const conversationId = this.#rowIdFromEvent(event);
    if (!conversationId) return;
    this.#emit<ConversationIntentDetail>('dl-conversation-select', {conversationId});
  };

  #renameFromPointer = (event: MouseEvent): void => {
    const conversationId = this.#rowIdFromEvent(event);
    if (conversationId) this.#startRename(conversationId);
  };

  #startRename(conversationId: string): void {
    this.openMenuId = null;
    this.renameId = conversationId;
    void this.updateComplete.then(() => {
      const input = this.#row(conversationId)?.querySelector('input');
      input?.focus();
      input?.select();
    });
  }

  #commitRename(conversation: ConversationSummary, input: HTMLInputElement): void {
    if (this.renameId !== conversation.conversationId) return;
    this.renameId = null;
    const title = input.value.trim();
    if (!title || title === (conversation.title ?? '').trim()) return;
    this.#emit<ConversationRenameDetail>('dl-conversation-rename', {
      conversationId: conversation.conversationId,
      title,
    });
  }

  #finishKeyboardRename(
    conversation: ConversationSummary,
    input: HTMLInputElement,
    commit: boolean,
  ): void {
    if (commit) this.#commitRename(conversation, input);
    else this.renameId = null;
    void this.focusActions(conversation.conversationId);
  }

  #renderStatus(
    message: string,
    retryLabel: string,
    kind: ConversationRetryDetail['kind'],
  ): TemplateResult {
    return html`
      <div class="conversation-list-status" role="status">
        <span>${message}</span>
        <button type="button" @click=${() => {
          this.#emit<ConversationRetryDetail>('dl-conversation-retry', {kind});
        }}>${retryLabel}</button>
      </div>
    `;
  }

  #renderRenameInput(conversation: ConversationSummary): TemplateResult {
    return html`
      <input
        type="text"
        aria-label=${msg('Conversation title', {id: 'conversationList.conversationTitle'})}
        maxlength="120"
        .value=${conversation.title ?? ''}
        @keydown=${(event: KeyboardEvent) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            this.#finishKeyboardRename(
              conversation,
              event.currentTarget as HTMLInputElement,
              true,
            );
          } else if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            this.#finishKeyboardRename(
              conversation,
              event.currentTarget as HTMLInputElement,
              false,
            );
          }
        }}
        @blur=${(event: FocusEvent) => {
          this.#commitRename(conversation, event.currentTarget as HTMLInputElement);
        }}
      >
    `;
  }

  #renderMenu(conversation: ConversationSummary): TemplateResult {
    const conversationId = conversation.conversationId;
    return html`
      <dl-menu
        id=${actionsMenuId(conversationId)}
        class="conversation-actions-menu dl-anchored dl-anchored--end"
        aria-label=${msg('Conversation actions', {id: 'conversationList.conversationActions'})}
        @dl-menu-dismiss=${(event: CustomEvent<MenuDismissDetail>) => {
          this.closeMenu(event.detail.restoreFocus);
        }}
      >
        <button
          type="button"
          role="menuitem"
          tabindex="-1"
          @click=${() => { this.#startRename(conversationId); }}
        >${msg('Rename', {id: 'conversationList.rename'})}</button>
        <button
          type="button"
          role="menuitem"
          tabindex="-1"
          class="conversation-delete-action"
          aria-disabled=${this.busy ? 'true' : nothing}
          @click=${() => {
            this.openMenuId = null;
            this.#emit<ConversationIntentDetail>('dl-conversation-delete', {conversationId});
          }}
        >${msg('Delete', {id: 'conversationList.delete'})}</button>
      </dl-menu>
    `;
  }

  #renderRow(conversation: ConversationSummary): TemplateResult {
    const conversationId = conversation.conversationId;
    const active = conversationId === this.handles.conversations.activeConversationId;
    const renaming = this.renameId === conversationId;
    const expanded = this.openMenuId === conversationId;
    return html`
      <div
        class="conversation-row"
        role="listitem"
        data-conversation-id=${conversationId}
        aria-current=${active ? 'page' : nothing}
      >
        <div class="conversation-row-main">
          ${renaming ? this.#renderRenameInput(conversation) : html`
            <button
              type="button"
              class="conversation-select"
              ?disabled=${this.busy}
              aria-label=${conversation.title
                ? nothing
                : msg('Open untitled conversation', {id: 'conversationList.openUntitled'})}
            >${conversation.title || msg('New chat', {id: 'conversationList.newChat'})}</button>
          `}
          ${conversation.forkedFromTitle ? html`
            <span
              class="conversation-lineage"
              title=${msg('Forked from another conversation', {id: 'conversationList.forkedFromTitle'})}
            >
              ${msg(str`Forked from ${conversation.forkedFromTitle}`, {id: 'conversationList.forkedFrom'})}
            </span>
          ` : nothing}
        </div>
        <button
          type="button"
          class="conversation-actions-button"
          aria-label=${msg('Conversation actions', {id: 'conversationList.conversationActionsButton'})}
          aria-haspopup="menu"
          aria-controls=${expanded ? actionsMenuId(conversationId) : nothing}
          aria-expanded=${expanded ? 'true' : 'false'}
          @click=${(event: MouseEvent) => {
            event.stopPropagation();
            if (expanded) this.closeMenu();
            else this.#openMenu(conversationId);
          }}
          @keydown=${(event: KeyboardEvent) => {
            const focus = menuButtonFocus(event);
            if (!focus) return;
            event.preventDefault();
            this.#openMenu(conversationId, focus);
          }}
        >•••</button>
        ${expanded ? this.#renderMenu(conversation) : nothing}
      </div>
    `;
  }

  protected override render(): TemplateResult | TemplateResult[] {
    const conversations = this.handles.conversations.conversations;
    const listState = this.handles.conversations.listState;
    if (listState === 'loading' && conversations.length === 0) {
      return html`
        ${Array.from({length: SKELETON_COUNT}, () => html`
          <div class="conversation-skeleton" aria-hidden="true"></div>
        `)}
        <span class="dl-sr-only">${msg('Loading conversations', {id: 'conversationList.loadingConversations'})}</span>
      `;
    }
    return html`
      ${listState === 'error'
        ? this.#renderStatus(
            msg('Could not load conversations.', {id: 'conversationList.couldNotLoad'}),
            msg('Retry', {id: 'conversationList.retry'}),
            'reload',
          )
        : nothing}
      ${listState === 'empty-error'
        ? this.#renderStatus(
            msg('No conversation is open.', {id: 'conversationList.noConversationOpen'}),
            msg('Retry New chat', {id: 'conversationList.retryNewChat'}),
            'new',
          )
        : nothing}
      <div class="conversation-items" role="list"
           @click=${this.#selectFromPointer}
           @dblclick=${this.#renameFromPointer}>
        ${repeat(
          conversations,
          (conversation) => conversation.conversationId,
          (conversation) => this.#renderRow(conversation),
        )}
      </div>
      ${loadOlderControl({
        list: 'conversations',
        pages: this.handles.conversations.olderConversations,
        label: msg('Load older conversations', {id: 'conversationList.loadOlder'}),
        retryLabel: msg('Retry loading older conversations', {id: 'conversationList.retryLoadOlder'}),
        loading: msg('Loading older conversations…', {id: 'conversationList.olderLoading'}),
        loaded: msg('Loaded older conversations.', {id: 'conversationList.olderLoaded'}),
        failed: msg('Could not load older conversations.', {id: 'conversationList.couldNotLoadOlder'}),
        onLoad: () => { void this.handles.conversations.loadOlder(); },
        rowClass: 'conversation-load-older-row',
        buttonClass: 'conversation-load-older',
      })}
    `;
  }
}

customElements.define('dl-conversation-list', DlConversationList);

declare global {
  interface HTMLElementTagNameMap {
    'dl-conversation-list': DlConversationList;
  }

  interface HTMLElementEventMap {
    'dl-conversation-select': CustomEvent<ConversationIntentDetail>;
    'dl-conversation-delete': CustomEvent<ConversationIntentDetail>;
    'dl-conversation-rename': CustomEvent<ConversationRenameDetail>;
    'dl-conversation-retry': CustomEvent<ConversationRetryDetail>;
  }
}
