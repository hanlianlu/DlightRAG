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
  history(
    conversationId: string,
    cursor?: string | null,
    limit?: number,
    signal?: AbortSignal,
  ): Promise<ConversationHistory>;
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
  #historyPageController: AbortController | null = null;
  #historyNextCursor: string | null = null;
  #historyLoadMoreState: ConversationLoadMoreState = 'idle';
  #historyLoadMoreFlight: Promise<void> | null = null;

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

  get historyLoadMoreState(): ConversationLoadMoreState {
    return this.#historyLoadMoreState;
  }

  get hasOlderMessages(): boolean {
    return this.#historyNextCursor !== null;
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
    return this.#conversations[0]?.conversationId ?? null;
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
      this.#nextCursor = page.nextCursor;
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
      this.#nextCursor = page.nextCursor;
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
    this.#historyNextCursor = null;
    this.#historyLoadMoreState = 'idle';
    this.#viewState = 'new';
    this.#publishView();
  }

  async open(
    conversationId: string,
    options: {showLoading?: boolean; preserveOnError?: boolean} = {},
  ): Promise<ConversationOpenResult> {
    const sameConversation = this.#activeConversationId === conversationId;
    const hadHistory = sameConversation && this.#history !== null;
    this.#abortView();
    const controller = new AbortController();
    const generation = this.#viewGeneration;
    this.#viewController = controller;
    this.#activeConversationId = conversationId;
    this.#historyLoadMoreState = 'idle';
    if (!sameConversation) {
      this.#history = null;
      this.#historyNextCursor = null;
    }
    if (options.showLoading !== false) {
      this.#viewState = 'loading';
      this.#publishView();
    }

    try {
      const recent = await this.#api.history(conversationId, null, undefined, controller.signal);
      if (generation !== this.#viewGeneration) return 'stale';
      if (recent.conversation.conversationId !== conversationId) {
        throw new Error('conversation history response changed conversation identity');
      }
      const replaceHistory = sameConversation && this.#history !== null
        && this.#recentPageHasGap(this.#history.turns, recent.turns);
      const turns = sameConversation && this.#history !== null && !replaceHistory
        ? this.#mergeTurns(this.#history.turns, recent.turns, true)
        : this.#mergeTurns([], recent.turns, true);
      if (!hadHistory || replaceHistory) this.#historyNextCursor = recent.nextCursor ?? null;
      this.#history = {
        ...recent,
        turns,
        nextCursor: this.#historyNextCursor,
      };
      this.#activeConversationId = conversationId;
      this.#upsert(recent.conversation);
      this.#viewState = 'ready';
      this.#publishView();
      return 'ready';
    } catch (error) {
      if (isAbortError(error) || generation !== this.#viewGeneration) return 'stale';
      if (this.#isRouteUnavailable(error)) {
        this.#removeSummary(conversationId);
        this.#history = null;
        this.#historyNextCursor = null;
        this.#activeConversationId = conversationId;
        this.#viewState = 'unavailable';
        this.#publishView();
        return 'unavailable';
      }
      if (options.preserveOnError && sameConversation && this.#history !== null) {
        return 'error';
      }
      this.#history = null;
      this.#historyNextCursor = null;
      this.#viewState = 'error';
      this.#publishView();
      return 'error';
    } finally {
      if (this.#viewController === controller) this.#viewController = null;
    }
  }

  loadOlderMessages(): Promise<void> {
    if (this.#historyLoadMoreFlight !== null) return this.#historyLoadMoreFlight;
    const conversationId = this.#activeConversationId;
    const cursor = this.#historyNextCursor;
    if (!conversationId || cursor === null || this.#viewState !== 'ready') {
      return Promise.resolve();
    }
    const flight = this.#loadOlderMessagesPage(
      conversationId,
      cursor,
      this.#viewGeneration,
    );
    this.#historyLoadMoreFlight = flight;
    void flight.finally(() => {
      if (this.#historyLoadMoreFlight === flight) this.#historyLoadMoreFlight = null;
    });
    return flight;
  }

  async #loadOlderMessagesPage(
    conversationId: string,
    cursor: string,
    generation: number,
  ): Promise<void> {
    this.#historyPageController?.abort();
    const controller = new AbortController();
    this.#historyPageController = controller;
    this.#historyLoadMoreState = 'loading';
    this.#publishView();
    try {
      const older = await this.#api.history(
        conversationId,
        cursor,
        undefined,
        controller.signal,
      );
      if (
        generation !== this.#viewGeneration
        || this.#activeConversationId !== conversationId
        || this.#historyNextCursor !== cursor
      ) return;
      if (older.conversation.conversationId !== conversationId || this.#history === null) {
        throw new Error('older history response changed conversation identity');
      }
      this.#history = {
        ...this.#history,
        conversation: older.conversation,
        turns: this.#mergeTurns(this.#history.turns, older.turns, false),
        nextCursor: older.nextCursor ?? null,
      };
      this.#historyNextCursor = older.nextCursor ?? null;
      this.#historyLoadMoreState = 'idle';
      this.#upsert(older.conversation);
      this.#publishView();
    } catch (error) {
      if (isAbortError(error) || generation !== this.#viewGeneration) return;
      this.#historyLoadMoreState = 'error';
      this.#publishView();
    } finally {
      if (this.#historyPageController === controller) this.#historyPageController = null;
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
    this.#activeConversationId = summary.conversationId;
    this.#history = null;
    this.#historyNextCursor = null;
    this.#historyLoadMoreState = 'idle';
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
          this.#abortView();
          this.#history = null;
          this.#historyNextCursor = null;
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
    if (this.#activeConversationId === conversationId) this.#abortView();
    return this.#mutate(async () => {
      let result: ConversationMutationResult = 'ok';
      try {
        await this.#api.delete(conversationId, signal);
      } catch (error) {
        if (!this.#isMissing(error)) return 'error';
        result = 'missing';
      }
      this.#removeSummary(conversationId);
      if (this.#activeConversationId === conversationId) {
        this.#abortView();
        this.#activeConversationId = conversationId;
        this.#history = null;
        this.#historyNextCursor = null;
        this.#viewState = 'unavailable';
        this.#publishView();
      }
      return result;
    });
  }

  async deleteAll(signal?: AbortSignal): Promise<ConversationMutationResult> {
    this.#abortView();
    return this.#mutate(async () => {
      try {
        await this.#api.deleteAll(signal);
      } catch {
        return 'error';
      }
      this.#conversations = [];
      this.openNew();
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
    this.#abortHistoryPage();
    this.#viewGeneration += 1;
  }

  #abortHistoryPage(): void {
    this.#historyPageController?.abort();
    this.#historyPageController = null;
    this.#historyLoadMoreFlight = null;
    this.#historyLoadMoreState = 'idle';
  }

  #upsert(summary: ConversationSummary): void {
    this.#conversations = this.#merge(this.#conversations, [summary]);
  }

  #removeSummary(conversationId: string): void {
    this.#conversations = this.#conversations.filter(
      (conversation) => conversation.conversationId !== conversationId,
    );
  }

  #merge(
    existing: readonly ConversationSummary[],
    incoming: readonly ConversationSummary[],
  ): ConversationSummary[] {
    const byId = new Map(existing.map((conversation) => [
      conversation.conversationId,
      conversation,
    ]));
    for (const conversation of incoming) byId.set(conversation.conversationId, conversation);
    return this.#sort([...byId.values()]);
  }

  #recentPageHasGap(
    existing: readonly ConversationHistory['turns'][number][],
    recent: readonly ConversationHistory['turns'][number][],
  ): boolean {
    if (existing.length === 0 || recent.length === 0) return false;
    const newestExisting = Math.max(...existing.map((turn) => turn.turnNumber));
    const oldestRecent = Math.min(...recent.map((turn) => turn.turnNumber));
    return oldestRecent > newestExisting + 1;
  }

  #mergeTurns(
    existing: readonly ConversationHistory['turns'][number][],
    incoming: readonly ConversationHistory['turns'][number][],
    incomingWins: boolean,
  ): ConversationHistory['turns'] {
    const byId = new Map(existing.map((turn) => [turn.turnId, turn]));
    const byNumber = new Map(existing.map((turn) => [turn.turnNumber, turn]));
    for (const turn of incoming) {
      const sameId = byId.get(turn.turnId);
      const sameNumber = byNumber.get(turn.turnNumber);
      if (
        (sameId && (
          sameId.turnNumber !== turn.turnNumber
          || sameId.answerRunId !== turn.answerRunId
        ))
        || (sameNumber && sameNumber.turnId !== turn.turnId)
      ) {
        throw new Error('conversation turn identity changed across history pages');
      }
      if (!sameId || incomingWins) byId.set(turn.turnId, turn);
      byNumber.set(turn.turnNumber, byId.get(turn.turnId)!);
    }
    return [...byId.values()].sort((left, right) => left.turnNumber - right.turnNumber);
  }

  #sort(conversations: readonly ConversationSummary[]): ConversationSummary[] {
    return [...conversations].sort((left, right) => {
      const leftMillis = Date.parse(left.updatedAt);
      const rightMillis = Date.parse(right.updatedAt);
      if (leftMillis !== rightMillis) return rightMillis - leftMillis;
      const microsecondOrder = this.#microsecondRemainder(right.updatedAt)
        - this.#microsecondRemainder(left.updatedAt);
      if (microsecondOrder !== 0) return microsecondOrder;
      return right.conversationId.localeCompare(left.conversationId);
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
