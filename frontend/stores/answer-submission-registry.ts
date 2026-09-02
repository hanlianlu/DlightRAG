// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {AnswerSubmissionAdapter, AnswerSubmissionIntent} from '../api/answer-submission.ts';
import type {AttachmentLease} from './attachment-store.ts';
import {
  answerSubmissionSnapshot,
  createAnswerSubmissionActor,
  type AnswerSubmissionActor,
  type AnswerSubmissionSnapshot,
} from './answer-submission-machine.ts';

const NEW_CONVERSATION_KEY = '__new_chat__';

export class AnswerSubmissionRegistry {
  readonly #actors = new Map<string, AnswerSubmissionActor>();
  readonly #subscribers = new Set<() => void>();

  start(
    intent: AnswerSubmissionIntent,
    lease: AttachmentLease,
    adapter: AnswerSubmissionAdapter,
  ): AnswerSubmissionActor | null {
    const key = intent.conversationId ?? NEW_CONVERSATION_KEY;
    if (this.#actors.has(key)) return null;
    const actor = createAnswerSubmissionActor({intent, lease, adapter});
    this.#actors.set(key, actor);
    actor.subscribe({
      next: () => this.#notify(),
      complete: () => {
        if (this.#actors.get(key) === actor) this.#actors.delete(key);
        this.#notify();
      },
    });
    actor.start();
    this.#notify();
    return actor;
  }

  actor(conversationId: string | null): AnswerSubmissionActor | null {
    return this.#actors.get(conversationId ?? NEW_CONVERSATION_KEY) ?? null;
  }

  list(): readonly AnswerSubmissionSnapshot[] {
    return [...this.#actors.values()].map(answerSubmissionSnapshot);
  }

  subscribe(handler: () => void): () => void {
    this.#subscribers.add(handler);
    return () => this.#subscribers.delete(handler);
  }

  dispose(): void {
    for (const actor of this.#actors.values()) {
      actor.getSnapshot().context.lease.discard();
      actor.stop();
    }
    this.#actors.clear();
    this.#notify();
  }

  #notify(): void {
    for (const subscriber of this.#subscribers) subscriber();
  }
}

export const answerSubmissionRegistry = new AnswerSubmissionRegistry();
