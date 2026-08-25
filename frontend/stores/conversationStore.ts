// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  ConversationApiError,
  deleteAllConversations,
  deleteConversation,
  getConversationHistory,
  listConversations,
  renameConversation,
  type ConversationHistory,
  type ConversationSummary,
} from '../api/conversations.ts';
import {isAbortError} from '../lib/errors.ts';
import {Store} from './base.ts';

export type ConversationListState = 'loading' | 'ready' | 'error' | 'empty-error';
export type ConversationViewState = 'new' | 'loading' | 'ready' | 'unavailable' | 'error';
export type ConversationOpenResult = 'ready' | 'unavailable' | 'error' | 'stale';
export type ConversationMutationResult = 'ok' | 'missing' | 'error';

export interface ConversationApi {
  list(signal?: AbortSignal): Promise<ConversationSummary[]>;
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
    this.#listState = 'loading';
    this.changed();
    try {
      const conversations = await this.#api.list(controller.signal);
      if (generation !== this.#listGeneration) return;
      this.#conversations = this.#sort(conversations);
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
    const next = this.#conversations.filter(
      (conversation) => conversation.conversation_id !== summary.conversation_id,
    );
    next.push(summary);
    this.#conversations = this.#sort(next);
  }

  #removeSummary(conversationId: string): void {
    this.#conversations = this.#conversations.filter(
      (conversation) => conversation.conversation_id !== conversationId,
    );
  }

  #sort(conversations: readonly ConversationSummary[]): ConversationSummary[] {
    return [...conversations].sort((left, right) => right.updated_at.localeCompare(left.updated_at));
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
