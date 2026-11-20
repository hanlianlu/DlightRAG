// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Which conversation the page shows, and what leaving it costs.
 *
 * The address names a conversation, the conversation store loads it, and the Chat draws what the store
 * holds. This is the one place that joins the three: it starts the router, answers each route by opening the
 * conversation or a new chat, projects what the store holds into the `ChatView` the Shell hands the Chat,
 * and stands guard where a move would lose the reader's work. The conversation list and its drawer are the
 * sidebar's, and the Chat's own inputs are the Chat's; neither is written from here except through what the
 * Chat offers, which is to stop following a Run and to forget an unsent message.
 */

import type {ReactiveController, ReactiveControllerHost} from 'lit';
import {conversationRoute, newChatRoute, type WebRoute} from '../lib/router.ts';
import type {AppHandles} from '../stores/app-handles.ts';
import type {ConversationStore} from '../stores/conversation-store.ts';
import type {ChatView, ChatViewActionDetail} from './chat-message-list.ts';
import {webRouter} from './router.ts';

/** The Chat as the route meets it: what moving away would lose, and what must stop first. */
export interface ConversationRouteChat {
  readonly hasDraft: boolean;
  /** Staged files, which a reload loses; the unsent message is stored and comes back. */
  readonly hasStagedAttachments: boolean;
  readonly hasUnresolvedSubmission: boolean;
  readonly submissionPending: boolean;
  clearDraft(): void;
  detachRun(): void;
}

/** What the route asks the rest of the Shell to show or do. */
export interface ConversationRouteUi {
  /** Ask the reader whether the unsent message may go before the page moves to `next`. */
  confirmDiscardDraft(next: WebRoute): Promise<boolean>;
  /** Tell the reader the page cannot move while a question is still being accepted. */
  waitForAcceptedQuestion(): void;
  /** What belonged to the conversation the page showed goes with it: its dock, its Canvas. */
  conversationLeft(): void;
  /** Put the compact drawer away once the page has moved. */
  closeDrawer(): Promise<void>;
}

interface ConversationRouteOptions {
  /** Asked at each use: the bag a Shell holds is the one it was handed, not the one it was built with. */
  handles(): AppHandles;
  /** The Chat on the page, if it is there now. */
  chat(): ConversationRouteChat | null;
  ui: ConversationRouteUi;
}

/** What the store holds, as the Chat is told it. Every projection is a view of its own, even one that reads
 *  as the last did: the Chat draws again for each thing that happened, and a new chat opened over a new
 *  chat whose answer is on show must clear it. A conversation that is ready but whose history is not
 *  loaded is one the Chat is already showing, so it keeps the view it has. */
function project(store: ConversationStore, current: ChatView): ChatView {
  switch (store.viewState) {
    case 'new':
      return {kind: 'new'};
    case 'loading':
      return {kind: 'loading'};
    case 'ready': {
      const history = store.history;
      if (!history) return current;
      const lineage = store.conversations.find(
        (summary) => summary.conversationId === history.conversation.conversationId,
      )?.forkedFromTitle ?? null;
      return {
        kind: 'ready',
        conversationId: history.conversation.conversationId,
        history: history.turns,
        lineage,
        olderMessages: store.olderMessages,
      };
    }
    case 'unavailable':
      return {kind: 'unavailable', hasRecent: Boolean(store.fallbackConversationId)};
    default:
      return {kind: 'error'};
  }
}

interface Projection {
  store: ConversationStore;
  revision: number;
  notFound: boolean;
  view: ChatView;
}

export class ConversationRouteController implements ReactiveController {
  readonly #host: ReactiveControllerHost;
  readonly #options: ConversationRouteOptions;
  #release: (() => void) | null = null;
  #projection: Projection | null = null;
  /** The view revision at which the address named no page: until the store publishes another, the Chat
   *  is told so, whatever the store holds for a new chat. */
  #notFoundAt = -1;
  #unload: AbortController | null = null;

  /** Set by whoever is deleting conversations while that runs: the page does not move under it. */
  lifecycleInFlight = false;

  constructor(host: ReactiveControllerHost, options: ConversationRouteOptions) {
    this.#host = host;
    this.#options = options;
    host.addController(this);
  }

  hostConnected(): void {
    this.#unload = new AbortController();
    window.addEventListener('beforeunload', this.#beforeUnload, {signal: this.#unload.signal});
  }

  hostDisconnected(): void {
    this.#unload?.abort();
    this.#unload = null;
    this.stop();
  }

  /** The Chat's view of the conversation now open. It changes identity only when the store publishes a
   *  new view, so binding it to the Chat sets the Chat's input once for each thing that happened. */
  get view(): ChatView {
    const store = this.#options.handles().conversations;
    const revision = store.viewRevision;
    const notFound = this.#notFoundAt === revision;
    const held = this.#projection;
    if (held && held.store === store && held.revision === revision && held.notFound === notFound) {
      return held.view;
    }
    const view: ChatView = notFound
      ? {kind: 'unavailable', hasRecent: false}
      : project(store, held?.view ?? {kind: 'new'});
    this.#projection = {store, revision, notFound, view};
    return view;
  }

  /** Whether the reader has an unsent message or staged files that moving away would lose. */
  get hasDraft(): boolean {
    return this.#options.chat()?.hasDraft ?? false;
  }

  /** Follow the address and answer it. The Shell calls this once it can show a conversation. */
  start(): void {
    if (this.#release) return;
    webRouter.setGuard((next) => this.#guard(next));
    this.#release = webRouter.start((route) => this.#apply(route));
    queueMicrotask(() => {
      if (this.#release) void this.reload();
    });
  }

  /** Let go of the address, as the Shell does when the page signs out or leaves. */
  stop(): void {
    this.#release?.();
    this.#release = null;
    webRouter.setGuard(null);
    this.#options.handles().conversations.dispose();
  }

  /** Read the list again and open what the address names. */
  async reload(): Promise<void> {
    await this.#options.handles().conversations.loadList();
    if (this.#release) await this.#apply(webRouter.current, false);
  }

  /** What the Chat's own buttons ask for when the conversation cannot be shown. */
  chatViewAction(action: ChatViewActionDetail['action']): void {
    const {conversations} = this.#options.handles();
    if (action === 'new') {
      void webRouter.navigate(newChatRoute());
      return;
    }
    if (action === 'recent') {
      const fallback = conversations.fallbackConversationId;
      if (fallback) void webRouter.navigate(conversationRoute(fallback));
      return;
    }
    const conversationId = conversations.activeConversationId;
    if (conversationId) void conversations.open(conversationId);
    else void conversations.loadList();
  }

  /** Whether a question is still being accepted, which nothing may leave behind. Says so when it is. */
  blocksLeaving(): boolean {
    if (!this.#options.chat()?.submissionPending) return false;
    this.#options.ui.waitForAcceptedQuestion();
    return true;
  }

  /** The conversation on show, or every conversation, is about to be deleted: stop following its Run
   *  first, so the end of its stream does not read back what is gone. */
  beforeDeleting(): void {
    this.#options.chat()?.detachRun();
  }

  /** A deletion took the conversation the unsent message was written for. */
  afterDeleting(): void {
    this.#options.chat()?.clearDraft();
  }

  /** The conversations the page showed are all gone, whether or not the address moved with them. */
  conversationsGone(): void {
    this.#options.ui.conversationLeft();
  }

  async #apply(route: WebRoute, closeDrawer = true): Promise<void> {
    const {conversations} = this.#options.handles();
    const previous = conversations.activeConversationId;
    const next = route.kind === 'conversation' ? route.conversationId : null;
    if (previous !== next) {
      this.#options.chat()?.detachRun();
      this.#options.ui.conversationLeft();
    }

    if (route.kind === 'conversation') {
      await conversations.open(route.conversationId);
    } else {
      conversations.openNew();
      if (route.kind === 'not-found') {
        this.#notFoundAt = conversations.viewRevision;
        this.#host.requestUpdate();
      }
    }
    if (closeDrawer) await this.#options.ui.closeDrawer();
  }

  async #guard(next: WebRoute): Promise<boolean> {
    if (this.lifecycleInFlight || this.blocksLeaving()) return false;
    if (!this.hasDraft) return true;
    const accepted = await this.#options.ui.confirmDiscardDraft(next);
    if (accepted) this.#options.chat()?.clearDraft();
    return accepted;
  }

  readonly #beforeUnload = (event: BeforeUnloadEvent): void => {
    // Only what a reload would really take: staged files, and a question still being accepted.
    // A typed draft is stored, so the browser is not asked about leaving it behind.
    const chat = this.#options.chat();
    if (!chat?.hasStagedAttachments && !chat?.hasUnresolvedSubmission) return;
    event.preventDefault();
    event.returnValue = '';
  };
}
