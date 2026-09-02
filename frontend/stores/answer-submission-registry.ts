// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {AnswerSubmissionAdapter, AnswerSubmissionIntent} from '../api/answer-submission.ts';
import type {AttachmentLease} from './attachment-store.ts';
import {
  answerSubmissionSnapshot,
  createAnswerSubmissionActor,
  type AnswerSubmissionActor,
  type AnswerSubmissionSnapshot,
} from './answer-submission-machine.ts';
import {Store} from './base.ts';

const NEW_CONVERSATION_KEY = '__new_chat__';

export class AnswerSubmissionRegistry extends Store {
  readonly #actors = new Map<string, AnswerSubmissionActor>();

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
      next: () => this.changed(),
      complete: () => {
        if (this.#actors.get(key) === actor) this.#actors.delete(key);
        this.changed();
      },
    });
    actor.start();
    this.changed();
    return actor;
  }

  actor(conversationId: string | null): AnswerSubmissionActor | null {
    return this.#actors.get(conversationId ?? NEW_CONVERSATION_KEY) ?? null;
  }

  list(): readonly AnswerSubmissionSnapshot[] {
    return [...this.#actors.values()].map(answerSubmissionSnapshot);
  }

  dispose(): void {
    for (const actor of this.#actors.values()) {
      actor.getSnapshot().context.lease.discard();
      actor.stop();
    }
    this.#actors.clear();
    this.changed();
  }

}

export const answerSubmissionRegistry = new AnswerSubmissionRegistry();
