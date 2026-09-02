// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {ReactiveController, ReactiveControllerHost} from 'lit';
import type {AnswerSubmissionAdapter, AnswerSubmissionIntent} from '../api/answer-submission.ts';
import type {AttachmentLease} from './attachment-store.ts';
import {
  answerSubmissionRegistry,
  type AnswerSubmissionRegistry,
} from './answer-submission-registry.ts';
import type {AnswerSubmissionActor, AnswerSubmissionSnapshot} from './answer-submission-machine.ts';

/** Thin Lit bridge; the registry and actors own all submission behavior. */
export class AnswerSubmissionController implements ReactiveController {
  #unsubscribe: (() => void) | null = null;
  readonly #host: ReactiveControllerHost;
  readonly #registry: AnswerSubmissionRegistry;

  constructor(
    host: ReactiveControllerHost,
    registry: AnswerSubmissionRegistry = answerSubmissionRegistry,
  ) {
    this.#host = host;
    this.#registry = registry;
    host.addController(this);
  }

  get snapshots(): readonly AnswerSubmissionSnapshot[] {
    return this.#registry.list();
  }

  actor(conversationId: string | null): AnswerSubmissionActor | null {
    return this.#registry.actor(conversationId);
  }

  start(
    intent: AnswerSubmissionIntent,
    lease: AttachmentLease,
    adapter: AnswerSubmissionAdapter,
  ): AnswerSubmissionActor | null {
    return this.#registry.start(intent, lease, adapter);
  }

  hostConnected(): void {
    this.#unsubscribe = this.#registry.subscribe(() => this.#host.requestUpdate());
  }

  hostDisconnected(): void {
    this.#unsubscribe?.();
    this.#unsubscribe = null;
  }
}
