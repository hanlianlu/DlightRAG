// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {clearMemory} from '../api/memory.ts';
import {DESKTOP_SHELL_MEDIA} from '../lib/breakpoints.ts';
import {wrapTabFocus} from '../lib/dom.ts';
import {LightElement, StoreController} from '../lib/lit_host.ts';
import {conversationRoute, newChatRoute, type WebRoute} from '../lib/router.ts';
import {
  conversationStore,
  type ConversationMutationResult,
} from '../stores/conversationStore.ts';
import conversationStyles from '../styles/conversations.module.css';
import type {ChatView, ChatViewActionDetail} from './chat_feature.ts';
import {
  type ConversationIntentDetail,
  type ConversationRenameDetail,
  type ConversationRetryDetail,
  type DlConversationList,
} from './conversation_list.ts';
import './conversation_list.ts';
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
    this.enabled = false;
    this.chatFeature = null;
    this.drawerOpen = false;
    this.desktopCollapsed = this.#collapsedPreference();
    this.desktop = window.matchMedia(DESKTOP_SHELL_MEDIA).matches;
    this.pendingLifecycleAction = false;
    this.shellInert = false;
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
        this.#requestToast({message: 'Could not delete conversations.', duration: 3000});
      } else {
        this.chatFeature?.clearDraft();
        if (alsoClearMemory) {
          try {
            await clearMemory(signal);
          } catch {
            if (!signal.aborted) {
              this.#requestToast({
                message: 'Conversations deleted; could not clear Profile memory.',
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
    this.#requestToast({message: 'Wait for the current question to be accepted.', duration: 3000});
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
    if (this.#releaseRouter) await this.#applyRoute(webRouter.current);
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

  async #applyRoute(route: WebRoute): Promise<void> {
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
    await this.close(false);
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
      this.#requestToast({message: 'Conversation unavailable.', duration: 3000});
      return;
    }
    this.#requestToast({
      message: title.trim().length > 120
        ? 'Conversation titles must be 1 to 120 characters.'
        : 'Could not rename the conversation.',
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
        this.#requestToast({message: 'Could not delete the conversation.', duration: 3000});
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
    const navigation = this.querySelector<HTMLElement>('nav[aria-label="Conversations"]');
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
    if (!this.#hasUnsavedDraft()) return;
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
        aria-label="Conversations"
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
            <svg class="new-chat-icon" width="14" height="14" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="1.8"
                 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
            </svg>
            New chat
          </button>
          <button
            id="conversation-sidebar-toggle"
            type="button"
            aria-label=${this.desktop ? 'Collapse conversations' : 'Close conversations'}
            aria-controls="chat-sidebar"
            @click=${this.#toggleSidebar}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
                 stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2"></rect>
              <path d="M9 3v18"></path><path d="M14 10l-2 2 2 2"></path>
            </svg>
          </button>
        </div>
        <dl-conversation-list
          role="list"
          aria-live="polite"
          .busy=${this.#busy}
          @dl-conversation-select=${this.#selectConversation}
          @dl-conversation-delete=${this.#deleteConversation}
          @dl-conversation-rename=${this.#renameConversation}
          @dl-conversation-retry=${this.#retryConversation}
        ></dl-conversation-list>
        <button id="settings-btn" type="button" aria-label="Settings" aria-controls="settings-dialog"
                @click=${this.#requestSettings}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.08a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.08a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06-.06a1.65 1.65 0 0 0-.33 1.82v.08a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
          <span class="sidebar-action-label">Settings</span>
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
        aria-label="Open conversations"
        aria-controls="chat-sidebar"
        aria-expanded=${expanded ? 'true' : 'false'}
        aria-hidden=${modal ? 'true' : nothing}
        .inert=${modal}
        @click=${(event: MouseEvent) => {
          void this.open(event.currentTarget as HTMLElement);
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
             stroke-linejoin="round" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2"></rect>
          <path d="M9 3v18"></path><path d="M10 10l2 2-2 2"></path>
        </svg>
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
          <h2 id="delete-conversation-title">Delete conversation</h2>
          <p id="delete-conversation-message">
            This conversation and its history will be permanently deleted.
          </p>
          <p id="delete-conversation-draft-warning" hidden>
            Your unsent draft and attachments will also be discarded.
          </p>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Cancel</button>
            <button type="submit" value="delete" class="ui-dialog-danger">Delete</button>
          </div>
        </form>
      </dialog>
      <dialog id="delete-all-conversations-dialog" class="confirm-dialog"
              aria-labelledby="delete-all-conversations-title">
        <form method="dialog">
          <h2 id="delete-all-conversations-title">Delete all conversations?</h2>
          <p id="delete-all-conversations-draft-warning" hidden>
            Draft and attachments will also be deleted.
          </p>
          <label class="ui-dialog-checkbox">
            <input type="checkbox" id="delete-all-also-clear-memory">
            Also clear profile memories
          </label>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Cancel</button>
            <button type="submit" value="delete-all" class="ui-dialog-danger">Delete all</button>
          </div>
        </form>
      </dialog>
      <dialog id="discard-draft-dialog" class="confirm-dialog"
              aria-labelledby="discard-draft-title">
        <form method="dialog">
          <h2 id="discard-draft-title">Discard draft?</h2>
          <p>Your unsent message and attachments will not move to another conversation.</p>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Keep editing</button>
            <button type="submit" value="discard">Discard and continue</button>
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
