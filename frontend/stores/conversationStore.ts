// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  ConversationApiError,
  deleteAllConversations,
  deleteConversation,
  getConversationHistory,
  listConversations,
  renameConversation,
  type ConversationHistory,
  type ConversationPage,
  type ConversationSummary,
} from '../api/conversations.ts';
import {isAbortError} from '../lib/errors.ts';
import {Store} from './base.ts';

export type ConversationListState = 'loading' | 'ready' | 'error' | 'empty-error';
export type ConversationLoadMoreState = 'idle' | 'loading' | 'error';
export type ConversationViewState = 'new' | 'loading' | 'ready' | 'unavailable' | 'error';
export type ConversationOpenResult = 'ready' | 'unavailable' | 'error' | 'stale';
export type ConversationMutationResult = 'ok' | 'missing' | 'error';

export interface ConversationApi {
  list(cursor: string | null, signal?: AbortSignal): Promise<ConversationPage>;
  history(conversationId: string, signal?: AbortSignal): Promise<ConversationHistory>;
  rename(
    conversationId: string,
    title: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary>;
  delete(conversationId: string, signal?: AbortSignal): Promise<void>;
  deleteAll(signal?: AbortSignal): Promise<void>;
}

const browserConversationApi: ConversationApi = {
  list: listConversations,
  history: getConversationHistory,
  rename: renameConversation,
  delete: deleteConversation,
  deleteAll: deleteAllConversations,
};

/** Route-selected conversation data and its async lifecycle. */
export class ConversationStore extends Store {
  readonly #api: ConversationApi;
  #conversations: ConversationSummary[] = [];
  #activeConversationId: string | null = null;
  #history: ConversationHistory | null = null;
  #listState: ConversationListState = 'loading';
  #loadMoreState: ConversationLoadMoreState = 'idle';
  #nextCursor: string | null = null;
  #loadMoreFlight: Promise<void> | null = null;
  #viewState: ConversationViewState = 'new';
  #viewRevision = 0;
  #mutationPending = false;
  #listGeneration = 0;
  #viewGeneration = 0;
  #listController: AbortController | null = null;
  #viewController: AbortController | null = null;

  constructor(api: ConversationApi = browserConversationApi) {
    super();
    this.#api = api;
  }

  get conversations(): readonly ConversationSummary[] {
    return this.#conversations;
  }

  get activeConversationId(): string | null {
    return this.#activeConversationId;
  }

  get history(): ConversationHistory | null {
    return this.#history;
  }

  get listState(): ConversationListState {
    return this.#listState;
  }

  get loadMoreState(): ConversationLoadMoreState {
    return this.#loadMoreState;
  }

  get hasOlderConversations(): boolean {
    return this.#nextCursor !== null;
  }

  get viewState(): ConversationViewState {
    return this.#viewState;
  }

  get viewRevision(): number {
    return this.#viewRevision;
  }

  get mutationPending(): boolean {
    return this.#mutationPending;
  }

  get canAnswer(): boolean {
    return this.#viewState === 'new' || this.#viewState === 'ready';
  }

  get answerConversationId(): string | null {
    return this.#viewState === 'ready' ? this.#activeConversationId : null;
  }

  get fallbackConversationId(): string | null {
    return this.#conversations[0]?.conversation_id ?? null;
  }

  async loadList(): Promise<void> {
    this.#listController?.abort();
    const controller = new AbortController();
    const generation = ++this.#listGeneration;
    this.#listController = controller;
    this.#loadMoreFlight = null;
    this.#loadMoreState = 'idle';
    this.#listState = 'loading';
    this.changed();
    try {
      const page = await this.#api.list(null, controller.signal);
      if (generation !== this.#listGeneration) return;
      this.#conversations = this.#merge([], page.items);
      this.#nextCursor = page.next_cursor;
      this.#listState = 'ready';
      this.changed();
    } catch (error) {
      if (isAbortError(error) || generation !== this.#listGeneration) return;
      this.#listState = this.#conversations.length > 0 ? 'error' : 'empty-error';
      this.changed();
    } finally {
      if (this.#listController === controller) this.#listController = null;
    }
  }

  loadOlder(): Promise<void> {
    if (this.#loadMoreFlight !== null) return this.#loadMoreFlight;
    if (this.#nextCursor === null || this.#listState === 'loading') return Promise.resolve();
    const flight = this.#loadOlderPage(this.#nextCursor, this.#listGeneration);
    this.#loadMoreFlight = flight;
    void flight.finally(() => {
      if (this.#loadMoreFlight === flight) this.#loadMoreFlight = null;
    });
    return flight;
  }

  async #loadOlderPage(cursor: string, generation: number): Promise<void> {
    this.#listController?.abort();
    const controller = new AbortController();
    this.#listController = controller;
    this.#loadMoreState = 'loading';
    this.changed();
    try {
      const page = await this.#api.list(cursor, controller.signal);
      if (generation !== this.#listGeneration) return;
      this.#conversations = this.#merge(this.#conversations, page.items);
      this.#nextCursor = page.next_cursor;
      this.#loadMoreState = 'idle';
      this.changed();
    } catch (error) {
      if (isAbortError(error) || generation !== this.#listGeneration) return;
      this.#loadMoreState = 'error';
      this.changed();
    } finally {
      if (this.#listController === controller) this.#listController = null;
    }
  }

  openNew(): void {
    this.#abortView();
    this.#activeConversationId = null;
    this.#history = null;
    this.#viewState = 'new';
    this.#publishView();
  }

  async open(
    conversationId: string,
    options: {showLoading?: boolean; preserveOnError?: boolean} = {},
  ): Promise<ConversationOpenResult> {
    this.#viewController?.abort();
    const controller = new AbortController();
    const generation = ++this.#viewGeneration;
    const sameConversation = this.#activeConversationId === conversationId;
    this.#viewController = controller;
    this.#activeConversationId = conversationId;
    if (!sameConversation) this.#history = null;
    if (options.showLoading !== false) {
      this.#viewState = 'loading';
      this.#publishView();
    }

    try {
      const history = await this.#api.history(conversationId, controller.signal);
      if (generation !== this.#viewGeneration) return 'stale';
      this.#history = history;
      this.#activeConversationId = conversationId;
      this.#upsert(history.conversation);
      this.#viewState = 'ready';
      this.#publishView();
      return 'ready';
    } catch (error) {
      if (isAbortError(error) || generation !== this.#viewGeneration) return 'stale';
      if (this.#isRouteUnavailable(error)) {
        this.#removeSummary(conversationId);
        this.#history = null;
        this.#activeConversationId = conversationId;
        this.#viewState = 'unavailable';
        this.#publishView();
        return 'unavailable';
      }
      if (options.preserveOnError && sameConversation && this.#history !== null) {
        return 'error';
      }
      this.#history = null;
      this.#viewState = 'error';
      this.#publishView();
      return 'error';
    } finally {
      if (this.#viewController === controller) this.#viewController = null;
    }
  }

  async refreshActive(): Promise<ConversationOpenResult> {
    const conversationId = this.#activeConversationId;
    if (!conversationId) return 'stale';
    return this.open(conversationId, {showLoading: false, preserveOnError: true});
  }

  adoptCreatedConversation(summary: ConversationSummary): void {
    this.#abortView();
    this.#upsert(summary);
    this.#activeConversationId = summary.conversation_id;
    this.#history = null;
    this.#viewState = 'ready';
    // The live answer already owns the viewport; only list consumers update.
    this.changed();
  }

  upsertSummary(summary: ConversationSummary): void {
    this.#upsert(summary);
    this.changed();
  }

  async rename(
    conversationId: string,
    title: string,
    signal?: AbortSignal,
  ): Promise<ConversationMutationResult> {
    return this.#mutate(async () => {
      try {
        this.#upsert(await this.#api.rename(conversationId, title, signal));
        return 'ok';
      } catch (error) {
        if (!this.#isMissing(error)) return 'error';
        this.#removeSummary(conversationId);
        if (this.#activeConversationId === conversationId) {
          this.#history = null;
          this.#viewState = 'unavailable';
          this.#publishView();
        }
        return 'missing';
      }
    });
  }

  async delete(
    conversationId: string,
    signal?: AbortSignal,
  ): Promise<ConversationMutationResult> {
    return this.#mutate(async () => {
      let result: ConversationMutationResult = 'ok';
      try {
        await this.#api.delete(conversationId, signal);
      } catch (error) {
        if (!this.#isMissing(error)) return 'error';
        result = 'missing';
      }
      this.#removeSummary(conversationId);
      return result;
    });
  }

  async deleteAll(signal?: AbortSignal): Promise<ConversationMutationResult> {
    return this.#mutate(async () => {
      try {
        await this.#api.deleteAll(signal);
      } catch {
        return 'error';
      }
      this.#conversations = [];
      return 'ok';
    });
  }

  dispose(): void {
    this.#listController?.abort();
    this.#listController = null;
    this.#listGeneration += 1;
    this.#loadMoreFlight = null;
    this.#loadMoreState = 'idle';
    this.#abortView();
  }

  async #mutate(
    operation: () => Promise<ConversationMutationResult>,
  ): Promise<ConversationMutationResult> {
    if (this.#mutationPending) return 'error';
    this.#mutationPending = true;
    this.changed();
    try {
      return await operation();
    } finally {
      this.#mutationPending = false;
      this.changed();
    }
  }

  #publishView(): void {
    this.#viewRevision += 1;
    this.changed();
  }

  #abortView(): void {
    this.#viewController?.abort();
    this.#viewController = null;
    this.#viewGeneration += 1;
  }

  #upsert(summary: ConversationSummary): void {
    this.#conversations = this.#merge(this.#conversations, [summary]);
  }

  #removeSummary(conversationId: string): void {
    this.#conversations = this.#conversations.filter(
      (conversation) => conversation.conversation_id !== conversationId,
    );
  }

  #merge(
    existing: readonly ConversationSummary[],
    incoming: readonly ConversationSummary[],
  ): ConversationSummary[] {
    const byId = new Map(existing.map((conversation) => [
      conversation.conversation_id,
      conversation,
    ]));
    for (const conversation of incoming) byId.set(conversation.conversation_id, conversation);
    return this.#sort([...byId.values()]);
  }

  #sort(conversations: readonly ConversationSummary[]): ConversationSummary[] {
    return [...conversations].sort((left, right) => {
      const leftMillis = Date.parse(left.updated_at);
      const rightMillis = Date.parse(right.updated_at);
      if (leftMillis !== rightMillis) return rightMillis - leftMillis;
      const microsecondOrder = this.#microsecondRemainder(right.updated_at)
        - this.#microsecondRemainder(left.updated_at);
      if (microsecondOrder !== 0) return microsecondOrder;
      return right.conversation_id.localeCompare(left.conversation_id);
    });
  }

  #microsecondRemainder(timestamp: string): number {
    const fraction = /\.(\d+)(?:Z|[+-]\d\d:\d\d)$/.exec(timestamp)?.[1] ?? '';
    return Number(fraction.padEnd(6, '0').slice(3, 6) || 0);
  }

  #isMissing(error: unknown): boolean {
    return error instanceof ConversationApiError && error.status === 404;
  }

  #isRouteUnavailable(error: unknown): boolean {
    return this.#isMissing(error)
      || (error instanceof ConversationApiError && error.status === 422);
  }
}

export const conversationStore = new ConversationStore();
