// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {clearMemory} from '../api/memory.ts';
import {icon, isImeKey} from '../design-system/index.ts';
import {DESKTOP_SHELL_MEDIA} from '../lib/breakpoints.ts';
import {raise, tabbables, wrapTabFocus} from '../lib/dom.ts';
import {LightElement, MediaController, StoreController} from '../lib/lit-host.ts';
import {conversationRoute, newChatRoute, type WebRoute} from '../lib/router.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import type {
  ConversationMutationResult,
} from '../stores/conversation-store.ts';
import conversationStyles from '../styles/conversations.module.css';
import type {ConversationRouteController} from './conversation-route-controller.ts';
import type {
  ConversationIntentDetail,
  ConversationRenameDetail,
  ConversationRetryDetail,
  DlConversationList,
} from './conversation-list.ts';
import type {DlScheduleDialogs} from './schedule-dialogs.ts';
import type {ScheduleIntentDetail} from './schedule-section.ts';
import {requestToast} from './toast-request.ts';
import './conversation-list.ts';
import './schedule-dialogs.ts';
import './schedule-section.ts';
import {type FocusRestorer, modalResult } from './modal.ts';
import {webRouter} from './router.ts';
import {readStored, writeStored} from '../lib/storage.ts';

const COLLAPSED_KEY = 'dlightrag.conversation_sidebar_collapsed';

export interface ConversationSidebarStateDetail {
  expanded: boolean;
  compact: boolean;
}

/** The conversation list and its drawer, the commands that change the list, and the sidebar's accessibility.
 *  Where the page is, and what the Chat is told of it, is the route controller's; the sidebar asks it. */
export class DlConversationSidebar extends LightElement {
  static properties = {
    handles: {attribute: false},
    enabled: {attribute: false},
    routes: {attribute: false},
    drawerOpen: {state: true},
    desktopCollapsed: {state: true},
    pendingLifecycleAction: {state: true},
    shellInert: {state: true},
  };

  declare handles: AppHandles;
  declare enabled: boolean;
  /** Where the page is and what leaving it costs; null where nothing joins the sidebar to a Chat. */
  declare routes: ConversationRouteController | null;
  declare drawerOpen: boolean;
  declare desktopCollapsed: boolean;
  declare pendingLifecycleAction: boolean;
  declare shellInert: boolean;

  #drawerReturnFocus: HTMLElement | null = null;
  readonly #desktop = new MediaController(this, DESKTOP_SHELL_MEDIA, () => { this.#breakpointCrossed(); });
  #stateSignature = '';

  constructor() {
    super();
    this.handles = productionHandles();
    this.enabled = false;
    this.routes = null;
    this.drawerOpen = false;
    this.desktopCollapsed = this.#collapsedPreference();
    this.pendingLifecycleAction = false;
    this.shellInert = false;
    /** Store reads: activeConversationId, fallbackConversationId, mutationPending. */
    new StoreController(this, () => [this.handles.conversations]);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('keydown', this.#documentKeydown, {signal: this.lifetime});
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#stateSignature = '';
  }

  protected override updated(): void {
    this.classList.toggle('open', this.#expanded && this.#desktop.matches);
    this.#publishSidebarState();
  }

  /** Opens the sidebar and moves focus to its primary action. */
  async open(trigger: HTMLElement | null = null): Promise<boolean> {
    if (!this.enabled) return false;
    if (this.#desktop.matches) {
      this.desktopCollapsed = false;
      this.#setCollapsedPreference(false);
      await this.updateComplete;
      this.#newButton()?.focus();
      return true;
    }

    const opening = new CustomEvent('dl-conversation-sidebar-opening', {
      bubbles: true,
      composed: true,
      cancelable: true,
    });
    if (!this.dispatchEvent(opening)) return false;
    this.#drawerReturnFocus = trigger;
    this.drawerOpen = true;
    await this.updateComplete;
    this.#newButton()?.focus();
    return true;
  }

  /** Closes only the compact drawer; desktop collapse is an explicit user action. */
  async close(restoreFocus = false): Promise<void> {
    if (this.#desktop.matches || !this.drawerOpen) return;
    this.drawerOpen = false;
    await this.updateComplete;
    if (restoreFocus && this.#drawerReturnFocus?.isConnected) {
      this.#drawerReturnFocus.focus();
    }
    this.#drawerReturnFocus = null;
  }

  /** Settings delegates its destructive command to the navigation owner. */
  async deleteAll(returnFocus?: HTMLElement | null): Promise<boolean> {
    if (this.#busy || this.#lifecycleBlocked()) return false;
    const discardsDraft = this.#hasUnsavedDraft();
    const dialog = this.#dialog('delete-all-conversations-dialog');
    const warning = this.querySelector<HTMLElement>(
      '#delete-all-conversations-draft-warning',
    );
    if (warning) warning.hidden = !discardsDraft;
    if (dialog) {
      if (discardsDraft) {
        dialog.setAttribute('aria-describedby', 'delete-all-conversations-draft-warning');
      } else {
        dialog.removeAttribute('aria-describedby');
      }
    }
    if (!dialog || await modalResult(
      this,
      dialog,
      () => {
        if (returnFocus?.isConnected && !returnFocus.inert) returnFocus.focus();
        else void this.#focusSurvivingConversation();
      },
      this.lifetime,
    ) !== 'delete-all') {
      return false;
    }
    if (this.#lifecycleBlocked()) return false;

    const signal = this.lifetime;
    if (signal.aborted) return false;
    const alsoClearMemory = this.querySelector<HTMLInputElement>(
      '#delete-all-also-clear-memory',
    )?.checked;
    let result: ConversationMutationResult = 'error';

    this.#holdNavigation(true);
    try {
      this.routes?.beforeDeleting();
      result = await this.handles.conversations.deleteAll(signal);
      if (signal.aborted) return false;
      if (result === 'error') {
        requestToast(this, {
          tone: 'error',
          message: msg('Could not delete conversations.', {id: 'conversationSidebar.deleteAllFailed'}),
        });
      } else {
        this.routes?.afterDeleting();
        if (alsoClearMemory) {
          try {
            await clearMemory(signal);
          } catch {
            if (!signal.aborted) {
              requestToast(this, {
                tone: 'error',
                message: msg('Conversations deleted; could not clear Profile memory.', {
                  id: 'conversationSidebar.memoryClearFailed',
                }),
              });
            }
          }
        }
        if (signal.aborted) return false;
        // The store has already closed the conversation that was open, so the move to a new chat below
        // may find nothing to leave: what belonged to it is told to go here, whichever way the route moves.
        this.routes?.conversationsGone();
        await webRouter.navigate(newChatRoute(), {replace: true, bypassGuard: true});
      }
    } finally {
      this.#holdNavigation(false);
    }
    if (signal.aborted) return false;
    await this.updateComplete;
    if (result === 'error') await this.#focusSurvivingConversation();
    else await this.#focusNewConversation();
    return result !== 'error';
  }

  /** The page does not move while a deletion is under way, and the controls that start another wait. */
  #holdNavigation(held: boolean): void {
    this.pendingLifecycleAction = held;
    if (this.routes) this.routes.lifecycleInFlight = held;
  }

  /** Ask the reader whether the unsent message may go before the page moves to `next`. The dialog is the
   *  sidebar's; the route controller asks for it when its guard meets a draft. */
  async confirmDiscardDraft(next: WebRoute): Promise<boolean> {
    const dialog = this.#dialog('discard-draft-dialog');
    if (!dialog) return false;
    return await modalResult(this, dialog, this.#routeFocusTarget(next), this.lifetime) === 'discard';
  }

  get #busy(): boolean {
    return this.pendingLifecycleAction || this.handles.conversations.mutationPending;
  }

  #collapsedPreference(): boolean {
    return readStored(COLLAPSED_KEY) === 'true';
  }

  #setCollapsedPreference(value: boolean): void {
    writeStored(COLLAPSED_KEY, value ? 'true' : 'false');
  }

  #list(): DlConversationList | null {
    return this.querySelector<DlConversationList>('dl-conversation-list');
  }

  #newButton(): HTMLButtonElement | null {
    return this.querySelector<HTMLButtonElement>('#new-conversation-btn');
  }

  #openButton(): HTMLButtonElement | null {
    return document.querySelector<HTMLButtonElement>('#conversation-sidebar-open');
  }

  #dialog(id: string): HTMLDialogElement | null {
    return this.querySelector<HTMLDialogElement>(`#${id}`);
  }

  async #focusNewConversation(): Promise<void> {
    await this.updateComplete;
    this.#newButton()?.focus();
  }

  async #focusSurvivingConversation(): Promise<void> {
    const active = this.handles.conversations.activeConversationId;
    if (active && await this.#list()?.focusConversation(active)) return;
    await this.#focusNewConversation();
  }

  async #focusConversationActions(conversationId: string): Promise<void> {
    if (await this.#list()?.focusActions(conversationId)) return;
    await this.#focusSurvivingConversation();
  }

  #routeFocusTarget(route: WebRoute): FocusRestorer {
    if (route.kind === 'conversation') {
      return async () => {
        if (!await this.#list()?.focusConversation(route.conversationId)) {
          await this.#focusNewConversation();
        }
      };
    }
    return () => this.#focusNewConversation();
  }

  #hasUnsavedDraft(): boolean {
    return this.routes?.hasDraft ?? false;
  }

  #lifecycleBlocked(): boolean {
    return this.routes?.blocksLeaving() ?? false;
  }

  /** Having chosen where to be, close the drawer and let the reader type. */
  async #backToComposer(): Promise<void> {
    await this.close(true);
    window.requestAnimationFrame(() => { raise(this, 'dl-conversation-chosen'); });
  }

  async #requestSelectConversation(conversationId: string): Promise<void> {
    if (conversationId === this.handles.conversations.activeConversationId
      || await webRouter.navigate(conversationRoute(conversationId))) {
      await this.#backToComposer();
    }
  }

  async #requestNewConversation(): Promise<void> {
    if (webRouter.current.kind === 'new' || await webRouter.navigate(newChatRoute())) {
      await this.#backToComposer();
    }
  }

  async #commitRename(conversationId: string, title: string): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted) return;
    const result = await this.handles.conversations.rename(conversationId, title, signal);
    if (signal.aborted || result === 'ok') return;
    if (result === 'missing') {
      requestToast(this, {
        tone: 'error',
        message: msg('Conversation unavailable.', {id: 'conversationSidebar.renameMissing'}),
      });
      return;
    }
    requestToast(this, {
      tone: 'error',
      message: title.trim().length > 120
        ? msg('Conversation titles must be 1 to 120 characters.', {
            id: 'conversationSidebar.renameTooLong',
          })
        : msg('Could not rename the conversation.', {id: 'conversationSidebar.renameFailed'}),
    });
  }

  async #requestDelete(conversationId: string): Promise<void> {
    if (this.#busy || this.#lifecycleBlocked()) return;
    const wasActive = this.handles.conversations.activeConversationId === conversationId;
    const discardsDraft = wasActive && this.#hasUnsavedDraft();
    const dialog = this.#dialog('delete-conversation-dialog');
    const warning = this.querySelector<HTMLElement>('#delete-conversation-draft-warning');
    if (warning) warning.hidden = !discardsDraft;
    if (dialog) {
      dialog.setAttribute(
        'aria-describedby',
        discardsDraft
          ? 'delete-conversation-message delete-conversation-draft-warning'
          : 'delete-conversation-message',
      );
    }
    if (!dialog || await modalResult(
      this,
      dialog,
      () => this.#focusConversationActions(conversationId),
      this.lifetime,
    ) !== 'delete') return;
    if (this.#lifecycleBlocked()) return;

    const signal = this.lifetime;
    if (signal.aborted) return;
    let result: ConversationMutationResult = 'error';
    this.#holdNavigation(true);
    try {
      if (wasActive) this.routes?.beforeDeleting();
      result = await this.handles.conversations.delete(conversationId, signal);
      if (signal.aborted) return;
      if (result === 'error') {
        requestToast(this, {
          tone: 'error',
          message: msg('Could not delete the conversation.', {
            id: 'conversationSidebar.deleteFailed',
          }),
        });
      } else if (wasActive) {
        this.routes?.afterDeleting();
        const fallback = this.handles.conversations.fallbackConversationId;
        await webRouter.navigate(
          fallback ? conversationRoute(fallback) : newChatRoute(),
          {replace: true, bypassGuard: true},
        );
      }
    } finally {
      this.#holdNavigation(false);
    }
    if (signal.aborted) return;
    await this.updateComplete;
    if (result === 'error') await this.#focusConversationActions(conversationId);
    else await this.#focusSurvivingConversation();
  }


  /**
   * Puts the sidebar away: where it is drawn beside the chat it collapses to its rail, and where it
   * covers the chat it closes. What its own control does, and what a drag that takes its width away
   * asks for — the panel decides what its state means, the split only reports the width.
   */
  collapse(restoreFocus = true): void {
    if (this.#desktop.matches) {
      this.desktopCollapsed = true;
      this.#setCollapsedPreference(true);
      void this.updateComplete.then(() => { if (restoreFocus) this.#openButton()?.focus(); });
      return;
    }
    void this.close(restoreFocus);
  }

  #toggleSidebar = (): void => {
    this.collapse(true);
  };

  #focusTrap = (event: KeyboardEvent): void => {
    if (this.#desktop.matches || !this.drawerOpen || event.key !== 'Tab') return;
    const nav = this.querySelector('nav');
    if (nav) wrapTabFocus(tabbables(nav), event);
  };

  #documentKeydown = (event: KeyboardEvent): void => {
    // An Escape that cancels an IME composition belongs to the composition, not to the drawer.
    if (isImeKey(event) || event.key !== 'Escape' || this.#desktop.matches || !this.drawerOpen) return;
    if (document.querySelector('dialog[open]') || this.#list()?.menuOpen) return;
    event.preventDefault();
    void this.close(true);
  };

  /** Crossing the breakpoint: the drawer is only for the compact layout, and focus must not be stranded in it. */
  #breakpointCrossed(): void {
    const navigation = this.querySelector<HTMLElement>('#chat-sidebar');
    const focusWasInNavigation = Boolean(
      document.activeElement instanceof Node && navigation?.contains(document.activeElement),
    );
    this.drawerOpen = false;
    this.#drawerReturnFocus = null;
    if (focusWasInNavigation) {
      void this.updateComplete.then(() => {
        if (navigation?.inert) this.#openButton()?.focus();
      });
    }
  }

  /** Whether the conversation list shows: the column on desktop unless collapsed, the drawer when compact. */
  get #expanded(): boolean {
    return this.enabled && (this.#desktop.matches ? !this.desktopCollapsed : this.drawerOpen);
  }

  #publishSidebarState(): void {
    const expanded = this.#expanded;
    const compact = !this.#desktop.matches;
    const signature = `${expanded}:${compact}`;
    if (signature === this.#stateSignature) return;
    this.#stateSignature = signature;
    raise(this, 'dl-conversation-sidebar-state-change', {expanded, compact});
  }

  #selectConversation = (event: CustomEvent<ConversationIntentDetail>): void => {
    event.stopPropagation();
    void this.#requestSelectConversation(event.detail.conversationId);
  };

  #deleteConversation = (event: CustomEvent<ConversationIntentDetail>): void => {
    event.stopPropagation();
    void this.#requestDelete(event.detail.conversationId);
  };

  #renameConversation = (event: CustomEvent<ConversationRenameDetail>): void => {
    event.stopPropagation();
    void this.#commitRename(event.detail.conversationId, event.detail.title);
  };

  #retryConversation = (event: CustomEvent<ConversationRetryDetail>): void => {
    event.stopPropagation();
    void (event.detail.kind === 'reload' ? this.routes?.reload() : this.#requestNewConversation());
  };

  /** What the reader asked of the Schedules (the Scheduled row asks for their list) is carried out by the dialogs, which stand outside the navigation. */
  #scheduleIntent = (event: CustomEvent<ScheduleIntentDetail>): void => {
    event.stopPropagation();
    void this.querySelector<DlScheduleDialogs>('dl-schedule-dialogs')?.handle(event.detail);
  };

  #requestSettings = (): void => {
    raise(this, 'dl-settings-request');
  };

  protected override render(): TemplateResult {
    const expanded = this.#expanded;
    const modal = this.enabled && !this.#desktop.matches && this.drawerOpen;
    return html`
      <nav
        id="chat-sidebar"
        class=${conversationStyles.root}
        aria-label=${msg('Conversations', {id: 'conversationSidebar.conversations'})}
        aria-hidden=${expanded ? nothing : 'true'}
        role=${modal ? 'dialog' : nothing}
        aria-modal=${modal ? 'true' : nothing}
        .inert=${this.shellInert || !expanded}
        @keydown=${this.#focusTrap}
      >
        <div class="conversation-top-row">
          <button
            id="new-conversation-btn"
            class="dl-nav-item"
            type="button"
            ?disabled=${this.#busy}
            @click=${() => { void this.#requestNewConversation(); }}
          >
            <span class="dl-nav-item-icon">${icon('edit', {size: 'sm'})}</span>
            <span class="dl-nav-item-text">
              <span class="dl-nav-item-label">${msg('New chat', {id: 'conversationSidebar.newChat'})}</span>
            </span>
          </button>
          <button
            id="conversation-sidebar-toggle"
            class="dl-icon-button"
            type="button"
            aria-label=${this.#desktop.matches
              ? msg('Collapse conversations', {id: 'conversationSidebar.collapseConversations'})
              : msg('Close conversations', {id: 'conversationSidebar.closeConversations'})}
            aria-controls="chat-sidebar"
            @click=${this.#toggleSidebar}
          >
            ${icon('panel-collapse', {size: 'sm'})}
          </button>
        </div>
        <dl-schedule-section
          .handles=${this.handles}
          .enabled=${this.enabled}
          @dl-schedule-intent=${this.#scheduleIntent}
        ></dl-schedule-section>
        <p class="dl-nav-group-label">${msg('Recents', {id: 'conversationSidebar.recents'})}</p>
        <dl-conversation-list
          .handles=${this.handles}
          .busy=${this.#busy}
          @dl-conversation-select=${this.#selectConversation}
          @dl-conversation-delete=${this.#deleteConversation}
          @dl-conversation-rename=${this.#renameConversation}
          @dl-conversation-retry=${this.#retryConversation}
        ></dl-conversation-list>
        <button id="settings-btn" class="dl-nav-item" type="button"
                aria-label=${msg('Settings', {id: 'conversationSidebar.settings'})}
                aria-controls="settings-dialog"
                @click=${this.#requestSettings}>
          <span class="dl-nav-item-icon">${icon('settings', {size: 'sm'})}</span>
          <span class="dl-nav-item-text">
            <span class="dl-nav-item-label">${msg('Settings', {id: 'conversationSidebar.settingsLabel'})}</span>
          </span>
        </button>
      </nav>
      <div
        id="conversation-sidebar-backdrop"
        ?hidden=${!modal}
        @click=${() => { void this.close(true); }}
      ></div>
      ${this.#conversationDialogs()}
    `;
  }

  #conversationDialogs(): TemplateResult {
    return html`
      <dl-schedule-dialogs
        .handles=${this.handles}
        @dl-conversation-select=${this.#selectConversation}
      ></dl-schedule-dialogs>
      <dialog id="delete-conversation-dialog" class="confirm-dialog"
              aria-labelledby="delete-conversation-title"
              aria-describedby="delete-conversation-message">
        <form method="dialog">
          <h2 id="delete-conversation-title">${msg('Delete conversation', {id: 'conversationSidebar.deleteConversationTitle'})}</h2>
          <p id="delete-conversation-message">
            ${msg('This conversation and its history will be permanently deleted.', {
              id: 'conversationSidebar.deleteConversationBody',
            })}
          </p>
          <p id="delete-conversation-draft-warning" hidden>
            ${msg('Your unsent draft and attachments will also be discarded.', {
              id: 'conversationSidebar.deleteConversationDraftWarning',
            })}
          </p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'conversationSidebar.cancel'})}</button>
            <button type="submit" value="delete" class="dl-dialog-danger">${msg('Delete', {id: 'conversationSidebar.delete'})}</button>
          </div>
        </form>
      </dialog>
      <dialog id="delete-all-conversations-dialog" class="confirm-dialog"
              aria-labelledby="delete-all-conversations-title">
        <form method="dialog">
          <h2 id="delete-all-conversations-title">${msg('Delete all conversations?', {id: 'conversationSidebar.deleteAllTitle'})}</h2>
          <p id="delete-all-conversations-draft-warning" hidden>
            ${msg('Draft and attachments will also be deleted.', {
              id: 'conversationSidebar.deleteAllDraftWarning',
            })}
          </p>
          <label class="dl-dialog-checkbox">
            <input type="checkbox" id="delete-all-also-clear-memory">
            ${msg('Also clear profile memories', {id: 'conversationSidebar.alsoClearMemories'})}
          </label>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'conversationSidebar.cancelDeleteAll'})}</button>
            <button type="submit" value="delete-all" class="dl-dialog-danger">${msg('Delete all', {id: 'conversationSidebar.deleteAll'})}</button>
          </div>
        </form>
      </dialog>
      <dialog id="discard-draft-dialog" class="confirm-dialog"
              aria-labelledby="discard-draft-title">
        <form method="dialog">
          <h2 id="discard-draft-title">${msg('Discard draft?', {id: 'conversationSidebar.discardDraftTitle'})}</h2>
          <p>${msg('Your unsent message and attachments will not move to another conversation.', {
            id: 'conversationSidebar.discardDraftBody',
          })}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Keep editing', {id: 'conversationSidebar.keepEditing'})}</button>
            <button type="submit" value="discard">${msg('Discard and continue', {id: 'conversationSidebar.discardAndContinue'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }
}

customElements.define('dl-conversation-sidebar', DlConversationSidebar);

declare global {
  interface HTMLElementTagNameMap {
    'dl-conversation-sidebar': DlConversationSidebar;
  }

  interface HTMLElementEventMap {
    'dl-conversation-sidebar-opening': CustomEvent<void>;
    'dl-conversation-sidebar-state-change': CustomEvent<ConversationSidebarStateDetail>;
    /** The reader chose where to be and the drawer is put away: the Chat's box is theirs to type in. */
    'dl-conversation-chosen': CustomEvent<void>;
  }
}
