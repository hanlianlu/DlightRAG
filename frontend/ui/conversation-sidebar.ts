// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {clearMemory} from '../api/memory.ts';
import {icon} from '../design-system/index.ts';
import {DESKTOP_SHELL_MEDIA} from '../lib/breakpoints.ts';
import {wrapTabFocus} from '../lib/dom.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {conversationRoute, newChatRoute, type WebRoute} from '../lib/router.ts';
import {
  conversationStore,
  type ConversationMutationResult,
} from '../stores/conversation-store.ts';
import conversationStyles from '../styles/conversations.module.css';
import type {ChatView, ChatViewActionDetail} from './chat-feature.ts';
import {
  type ConversationIntentDetail,
  type ConversationRenameDetail,
  type ConversationRetryDetail,
  type DlConversationList,
} from './conversation-list.ts';
import './conversation-list.ts';
import {modalResult, type FocusRestorer} from './modal.ts';
import {webRouter} from './router.ts';
import type {ToastRequestDetail} from './toast.ts';

const COLLAPSED_KEY = 'dlightrag.conversation_sidebar_collapsed';

export interface ConversationSidebarStateDetail {
  expanded: boolean;
  compact: boolean;
}

export interface ConversationRouteChangeDetail {
  previousConversationId: string | null;
  nextConversationId: string | null;
}

/** The Chat seam required by conversation route ownership. */
export interface ConversationChat {
  view: ChatView;
  readonly hasDraft: boolean;
  readonly hasUnresolvedSubmission?: boolean;
  readonly submissionPending: boolean;
  clearDraft(): void;
  detachRun(): void;
  focusComposer(): void;
}

/** Conversation route lifecycle, navigation commands, and sidebar accessibility. */
export class DlConversationSidebar extends LightElement {
  static properties = {
    enabled: {attribute: false},
    chatFeature: {attribute: false},
    drawerOpen: {state: true},
    desktopCollapsed: {state: true},
    desktop: {state: true},
    pendingLifecycleAction: {state: true},
    shellInert: {state: true},
  };

  declare enabled: boolean;
  declare chatFeature: ConversationChat | null;
  declare drawerOpen: boolean;
  declare desktopCollapsed: boolean;
  declare desktop: boolean;
  declare pendingLifecycleAction: boolean;
  declare shellInert: boolean;

  #drawerReturnFocus: HTMLElement | null = null;
  #events: AbortController | null = null;
  #releaseRouter: (() => void) | null = null;
  #renderedViewRevision = -1;
  #stateSignature = '';

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.enabled = false;
    this.chatFeature = null;
    this.drawerOpen = false;
    this.desktopCollapsed = this.#collapsedPreference();
    this.desktop = window.matchMedia(DESKTOP_SHELL_MEDIA).matches;
    this.pendingLifecycleAction = false;
    this.shellInert = false;
    /** Store reads: activeConversationId, fallbackConversationId, mutationPending. */
    new StoreController(this, conversationStore);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.desktop = window.matchMedia(DESKTOP_SHELL_MEDIA).matches;
    this.requestUpdate();
    const events = new AbortController();
    this.#events = events;
    document.addEventListener('keydown', this.#documentKeydown, {signal: events.signal});
    window.addEventListener('resize', this.#resize, {signal: events.signal});
    window.addEventListener('beforeunload', this.#beforeUnload, {signal: events.signal});
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#events?.abort();
    this.#events = null;
    this.#releaseRouter?.();
    this.#releaseRouter = null;
    webRouter.setGuard(null);
    conversationStore.dispose();
    this.#stateSignature = '';
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('chatFeature')) this.#renderedViewRevision = -1;
    this.#startIfReady();
    this.#renderCurrentConversationView();
    this.#publishSidebarState();
  }

  /** Opens the sidebar and moves focus to its primary action. */
  async open(trigger: HTMLElement | null = null): Promise<boolean> {
    if (!this.enabled) return false;
    if (this.desktop) {
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
    if (this.desktop || !this.drawerOpen) return;
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
      this.#events?.signal,
    ) !== 'delete-all') {
      return false;
    }
    if (this.#lifecycleBlocked()) return false;

    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return false;
    const alsoClearMemory = this.querySelector<HTMLInputElement>(
      '#delete-all-also-clear-memory',
    )?.checked;
    const previousConversationId = conversationStore.activeConversationId;
    let result: ConversationMutationResult = 'error';

    this.pendingLifecycleAction = true;
    try {
      this.chatFeature?.detachRun();
      result = await conversationStore.deleteAll(signal);
      if (signal.aborted) return false;
      if (result === 'error') {
        this.#requestToast({
          message: msg('Could not delete conversations.', {id: 'conversationSidebar.deleteAllFailed'}),
          duration: 3000,
        });
      } else {
        this.chatFeature?.clearDraft();
        if (alsoClearMemory) {
          try {
            await clearMemory(signal);
          } catch {
            if (!signal.aborted) {
              this.#requestToast({
                message: msg('Conversations deleted; could not clear Profile memory.', {
                  id: 'conversationSidebar.memoryClearFailed',
                }),
                duration: 3000,
              });
            }
          }
        }
        if (signal.aborted) return false;
        if (previousConversationId === null) this.#announceRouteChange(null, null);
        await webRouter.navigate(newChatRoute(), {replace: true, bypassGuard: true});
      }
    } finally {
      this.pendingLifecycleAction = false;
    }
    if (signal.aborted) return false;
    await this.updateComplete;
    if (result === 'error') await this.#focusSurvivingConversation();
    else await this.#focusNewConversation();
    return result !== 'error';
  }

  handleChatViewAction(action: ChatViewActionDetail['action']): void {
    if (action === 'new') {
      void webRouter.navigate(newChatRoute());
      return;
    }
    if (action === 'recent') {
      const fallback = conversationStore.fallbackConversationId;
      if (fallback) void webRouter.navigate(conversationRoute(fallback));
      return;
    }
    const conversationId = conversationStore.activeConversationId;
    if (conversationId) void conversationStore.open(conversationId);
    else void conversationStore.loadList();
  }

  get #busy(): boolean {
    return this.pendingLifecycleAction || conversationStore.mutationPending;
  }

  #collapsedPreference(): boolean {
    try {
      return window.localStorage.getItem(COLLAPSED_KEY) === 'true';
    } catch {
      return false;
    }
  }

  #setCollapsedPreference(value: boolean): void {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, value ? 'true' : 'false');
    } catch {
      // Storage may be unavailable under hardened browser settings.
    }
  }

  #list(): DlConversationList | null {
    return this.querySelector<DlConversationList>('dl-conversation-list');
  }

  #newButton(): HTMLButtonElement | null {
    return this.querySelector<HTMLButtonElement>('#new-conversation-btn');
  }

  #openButton(): HTMLButtonElement | null {
    return this.querySelector<HTMLButtonElement>('#conversation-sidebar-open');
  }

  #dialog(id: string): HTMLDialogElement | null {
    return this.querySelector<HTMLDialogElement>(`#${id}`);
  }

  async #focusNewConversation(): Promise<void> {
    await this.updateComplete;
    this.#newButton()?.focus();
  }

  async #focusSurvivingConversation(): Promise<void> {
    const active = conversationStore.activeConversationId;
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
    return this.chatFeature?.hasDraft ?? false;
  }

  #lifecycleBlocked(): boolean {
    if (!this.chatFeature?.submissionPending) return false;
    this.#requestToast({
      message: msg('Wait for the current question to be accepted.', {
        id: 'conversationSidebar.waitForAcceptedQuestion',
      }),
      duration: 3000,
    });
    return true;
  }

  async #confirmDiscardDraft(restoreFocus: FocusRestorer): Promise<boolean> {
    if (!this.#hasUnsavedDraft()) return true;
    const dialog = this.#dialog('discard-draft-dialog');
    if (!dialog) return false;
    return await modalResult(this, dialog, restoreFocus, this.#events?.signal) === 'discard';
  }

  async #guardNavigation(next: WebRoute): Promise<boolean> {
    if (this.pendingLifecycleAction || this.#lifecycleBlocked()) return false;
    if (!this.#hasUnsavedDraft()) return true;
    const accepted = await this.#confirmDiscardDraft(this.#routeFocusTarget(next));
    if (accepted) this.chatFeature?.clearDraft();
    return accepted;
  }

  #startIfReady(): void {
    if (!this.enabled || !this.chatFeature || this.#releaseRouter) return;
    webRouter.setGuard((next) => this.#guardNavigation(next));
    this.#releaseRouter = webRouter.start((route) => this.#applyRoute(route));
    queueMicrotask(() => {
      if (this.#releaseRouter) void this.#initialize();
    });
  }

  async #initialize(): Promise<void> {
    await conversationStore.loadList();
    if (this.#releaseRouter) await this.#applyRoute(webRouter.current, false);
  }

  #renderCurrentConversationView(): void {
    if (!this.chatFeature || this.#renderedViewRevision === conversationStore.viewRevision) return;
    this.#renderedViewRevision = conversationStore.viewRevision;

    if (conversationStore.viewState === 'new') {
      this.chatFeature.view = {kind: 'new'};
      return;
    }
    if (conversationStore.viewState === 'loading') {
      this.chatFeature.view = {kind: 'loading'};
      return;
    }
    if (conversationStore.viewState === 'ready') {
      const history = conversationStore.history;
      if (!history) return;
      const lineage = conversationStore.conversations.find(
        (summary) => summary.conversation_id === history.conversation.conversation_id,
      )?.forked_from_title ?? null;
      this.chatFeature.view = {
        kind: 'ready',
        conversationId: history.conversation.conversation_id,
        history: history.turns,
        lineage,
        hasOlderMessages: conversationStore.hasOlderMessages,
        olderMessagesState: conversationStore.historyLoadMoreState,
      };
      return;
    }
    if (conversationStore.viewState === 'unavailable') {
      this.chatFeature.view = {
        kind: 'unavailable',
        hasRecent: Boolean(conversationStore.fallbackConversationId),
      };
      return;
    }
    this.chatFeature.view = {kind: 'error'};
  }

  async #applyRoute(route: WebRoute, closeDrawer = true): Promise<void> {
    const previous = conversationStore.activeConversationId;
    const next = route.kind === 'conversation' ? route.conversationId : null;
    if (previous !== next) {
      this.chatFeature?.detachRun();
      this.#announceRouteChange(previous, next);
    }

    if (route.kind === 'conversation') {
      await conversationStore.open(route.conversationId);
    } else if (route.kind === 'new') {
      conversationStore.openNew();
    } else {
      conversationStore.openNew();
      await this.updateComplete;
      if (this.chatFeature) this.chatFeature.view = {kind: 'unavailable', hasRecent: false};
    }
    if (closeDrawer) await this.close(false);
  }

  #announceRouteChange(previous: string | null, next: string | null): void {
    this.dispatchEvent(new CustomEvent<ConversationRouteChangeDetail>(
      'dl-conversation-route-change',
      {
        bubbles: true,
        composed: true,
        detail: {previousConversationId: previous, nextConversationId: next},
      },
    ));
  }

  async #requestSelectConversation(conversationId: string): Promise<void> {
    if (conversationId === conversationStore.activeConversationId) {
      await this.close(true);
      window.requestAnimationFrame(() => this.chatFeature?.focusComposer());
      return;
    }
    if (await webRouter.navigate(conversationRoute(conversationId))) {
      await this.close(true);
      window.requestAnimationFrame(() => this.chatFeature?.focusComposer());
    }
  }

  async #requestNewConversation(): Promise<void> {
    if (webRouter.current.kind === 'new') {
      await this.close(true);
      window.requestAnimationFrame(() => this.chatFeature?.focusComposer());
      return;
    }
    if (await webRouter.navigate(newChatRoute())) {
      await this.close(true);
      window.requestAnimationFrame(() => this.chatFeature?.focusComposer());
    }
  }

  async #commitRename(conversationId: string, title: string): Promise<void> {
    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return;
    const result = await conversationStore.rename(conversationId, title, signal);
    if (signal.aborted || result === 'ok') return;
    if (result === 'missing') {
      this.#requestToast({
        message: msg('Conversation unavailable.', {id: 'conversationSidebar.renameMissing'}),
        duration: 3000,
      });
      return;
    }
    this.#requestToast({
      message: title.trim().length > 120
        ? msg('Conversation titles must be 1 to 120 characters.', {
            id: 'conversationSidebar.renameTooLong',
          })
        : msg('Could not rename the conversation.', {id: 'conversationSidebar.renameFailed'}),
      duration: 3000,
    });
  }

  async #requestDelete(conversationId: string): Promise<void> {
    if (this.#busy || this.#lifecycleBlocked()) return;
    const wasActive = conversationStore.activeConversationId === conversationId;
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
      this.#events?.signal,
    ) !== 'delete') return;
    if (this.#lifecycleBlocked()) return;

    const signal = this.#events?.signal;
    if (!signal || signal.aborted) return;
    let result: ConversationMutationResult = 'error';
    this.pendingLifecycleAction = true;
    try {
      if (wasActive) this.chatFeature?.detachRun();
      result = await conversationStore.delete(conversationId, signal);
      if (signal.aborted) return;
      if (result === 'error') {
        this.#requestToast({
          message: msg('Could not delete the conversation.', {
            id: 'conversationSidebar.deleteFailed',
          }),
          duration: 3000,
        });
      } else if (wasActive) {
        this.chatFeature?.clearDraft();
        const fallback = conversationStore.fallbackConversationId;
        await webRouter.navigate(
          fallback ? conversationRoute(fallback) : newChatRoute(),
          {replace: true, bypassGuard: true},
        );
      }
    } finally {
      this.pendingLifecycleAction = false;
    }
    if (signal.aborted) return;
    await this.updateComplete;
    if (result === 'error') await this.#focusConversationActions(conversationId);
    else await this.#focusSurvivingConversation();
  }

  #requestToast(detail: ToastRequestDetail): void {
    this.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail,
      bubbles: true,
      composed: true,
    }));
  }

  #toggleSidebar = (): void => {
    if (this.desktop) {
      this.desktopCollapsed = true;
      this.#setCollapsedPreference(true);
      void this.updateComplete.then(() => { this.#openButton()?.focus(); });
      return;
    }
    void this.close(true);
  };

  #focusTrap = (event: KeyboardEvent): void => {
    if (this.desktop || !this.drawerOpen || event.key !== 'Tab') return;
    const focusable = Array.from(this.querySelectorAll<HTMLElement>(
      'nav button:not([disabled]), nav input:not([disabled]), nav [tabindex]:not([tabindex="-1"])',
    )).filter((element) => !element.hidden);
    wrapTabFocus(focusable, event);
  };

  #documentKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape' || this.desktop || !this.drawerOpen) return;
    if (document.querySelector('dialog[open]') || this.#list()?.menuOpen) return;
    event.preventDefault();
    void this.close(true);
  };

  #resize = (): void => {
    const desktop = window.matchMedia(DESKTOP_SHELL_MEDIA).matches;
    if (desktop === this.desktop) return;
    const navigation = this.querySelector<HTMLElement>('#chat-sidebar');
    const focusWasInNavigation = Boolean(
      document.activeElement instanceof Node && navigation?.contains(document.activeElement),
    );
    this.drawerOpen = false;
    this.#drawerReturnFocus = null;
    this.desktop = desktop;
    if (focusWasInNavigation) {
      void this.updateComplete.then(() => {
        if (navigation?.inert) this.#openButton()?.focus();
      });
    }
  };

  #beforeUnload = (event: BeforeUnloadEvent): void => {
    if (!this.#hasUnsavedDraft() && !this.chatFeature?.hasUnresolvedSubmission) return;
    event.preventDefault();
    event.returnValue = '';
  };

  #publishSidebarState(): void {
    const expanded = this.enabled && (this.desktop ? !this.desktopCollapsed : this.drawerOpen);
    const compact = !this.desktop;
    const signature = `${expanded}:${compact}`;
    if (signature === this.#stateSignature) return;
    this.#stateSignature = signature;
    this.dispatchEvent(new CustomEvent<ConversationSidebarStateDetail>(
      'dl-conversation-sidebar-state-change',
      {bubbles: true, composed: true, detail: {expanded, compact}},
    ));
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
    void (event.detail.kind === 'reload' ? this.#initialize() : this.#requestNewConversation());
  };

  #requestSettings = (): void => {
    this.dispatchEvent(new CustomEvent('dl-settings-request', {
      bubbles: true,
      composed: true,
    }));
  };

  protected override render(): TemplateResult {
    const expanded = this.enabled && (this.desktop ? !this.desktopCollapsed : this.drawerOpen);
    const modal = this.enabled && !this.desktop && this.drawerOpen;
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
            type="button"
            ?disabled=${this.#busy}
            @click=${() => { void this.#requestNewConversation(); }}
          >
            ${icon('edit', {size: 'sm', className: 'new-chat-icon'})}
            ${msg('New chat', {id: 'conversationSidebar.newChat'})}
          </button>
          <button
            id="conversation-sidebar-toggle"
            type="button"
            aria-label=${this.desktop
              ? msg('Collapse conversations', {id: 'conversationSidebar.collapseConversations'})
              : msg('Close conversations', {id: 'conversationSidebar.closeConversations'})}
            aria-controls="chat-sidebar"
            @click=${this.#toggleSidebar}
          >
            ${icon('panel-collapse', {size: 'sm'})}
          </button>
        </div>
        <dl-conversation-list
          .busy=${this.#busy}
          @dl-conversation-select=${this.#selectConversation}
          @dl-conversation-delete=${this.#deleteConversation}
          @dl-conversation-rename=${this.#renameConversation}
          @dl-conversation-retry=${this.#retryConversation}
        ></dl-conversation-list>
        <button id="settings-btn" type="button"
                aria-label=${msg('Settings', {id: 'conversationSidebar.settings'})}
                aria-controls="settings-dialog"
                @click=${this.#requestSettings}>
          ${icon('settings', {size: 'sm'})}
          <span class="sidebar-action-label">${msg('Settings', {id: 'conversationSidebar.settingsLabel'})}</span>
        </button>
      </nav>
      <div
        id="conversation-sidebar-backdrop"
        ?hidden=${!modal}
        @click=${() => { void this.close(true); }}
      ></div>
      <button
        id="conversation-sidebar-open"
        type="button"
        aria-label=${msg('Open conversations', {id: 'conversationSidebar.openConversations'})}
        aria-controls="chat-sidebar"
        aria-expanded=${expanded ? 'true' : 'false'}
        aria-hidden=${modal ? 'true' : nothing}
        .inert=${modal}
        @click=${(event: MouseEvent) => {
          void this.open(event.currentTarget as HTMLElement);
        }}
      >
        ${icon('panel-expand', {size: 'md'})}
      </button>
      ${this.#conversationDialogs()}
    `;
  }

  #conversationDialogs(): TemplateResult {
    return html`
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
    'dl-conversation-route-change': CustomEvent<ConversationRouteChangeDetail>;
    'dl-settings-request': CustomEvent<void>;
  }
}
