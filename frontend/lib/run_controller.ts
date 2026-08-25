// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  cancelAnswerRun,
  getAnswerRun,
  type ConversationTurn,
} from '../api/conversations.ts';
import {createSSEParser} from './sse.ts';
import {answerRunStore} from '../stores/answerRunStore.ts';

const DEFAULT_MAX_RECONNECT_ATTEMPTS = 5;
const DEFAULT_RECONNECT_DELAY_MS = 500;

export type FollowResult =
  | {kind: 'aborted'}
  | {kind: 'terminal'; stored?: ConversationTurn}
  | {kind: 'retryable'; stored: ConversationTurn}
  | {kind: 'error'; message: string};

export interface RunControllerOptions {
  fetch?: typeof fetch;
  getRun?: typeof getAnswerRun;
  cancelRun?: typeof cancelAnswerRun;
  maxReconnectAttempts?: number;
  reconnectDelayMs?: number;
  onStateChange?: () => void;
}

/** Owns the transport and lifecycle resources for this tab's one followed run. */
export class RunController {
  readonly #fetch: typeof fetch;
  readonly #getRun: typeof getAnswerRun;
  readonly #cancelRun: typeof cancelAnswerRun;
  readonly #maxReconnectAttempts: number;
  readonly #reconnectDelayMs: number;
  readonly #onStateChange: () => void;

  #lifecycleController: AbortController | null = null;
  readonly #cancelControllers = new Map<string, AbortController>();
  #timer: ReturnType<typeof setTimeout> | null = null;
  #submissionPending = false;
  #active = false;
  #stopping = false;
  #runId: string | null = null;

  constructor(options: RunControllerOptions = {}) {
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.#getRun = options.getRun ?? getAnswerRun;
    this.#cancelRun = options.cancelRun ?? cancelAnswerRun;
    this.#maxReconnectAttempts = options.maxReconnectAttempts ?? DEFAULT_MAX_RECONNECT_ATTEMPTS;
    this.#reconnectDelayMs = options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS;
    this.#onStateChange = options.onStateChange ?? (() => {});
  }

  get active(): boolean {
    return this.#active;
  }

  get stopping(): boolean {
    return this.#stopping;
  }

  get submissionPending(): boolean {
    return this.#submissionPending;
  }

  get runId(): string | null {
    return this.#runId;
  }

  /** Return the current run lifecycle signal only while that run remains attached. */
  signalFor(runId: string): AbortSignal | null {
    const signal = this.#lifecycleController?.signal;
    return this.#runId === runId && signal && !signal.aborted ? signal : null;
  }

  /** Start the short, navigation-blocking window before a submission is durable. */
  beginSubmission(): AbortSignal | null {
    if (this.#active) return null;
    this.#startLifecycle();
    this.#submissionPending = true;
    this.#active = true;
    this.#notify();
    return this.#lifecycleController?.signal ?? null;
  }

  /** Bind the accepted durable run without replacing the submission's abort signal. */
  acceptSubmission(runId: string, cancelRequested: boolean): void {
    if (!this.#lifecycleController) return;
    this.#submissionPending = false;
    this.#runId = runId;
    this.#stopping = cancelRequested || this.#cancelControllers.has(runId);
    this.#notify();
  }

  /** Begin following a durable run discovered from conversation history. */
  beginFollow(runId: string, cancelRequested: boolean): AbortSignal | null {
    if (this.#active) return null;
    this.#startLifecycle();
    this.#active = true;
    this.#runId = runId;
    this.#stopping = cancelRequested || this.#cancelControllers.has(runId);
    this.#notify();
    return this.#lifecycleController?.signal ?? null;
  }

  /** Stop this tab's reader and timers without asking the server to cancel. */
  detach(): void {
    this.#clearTimer();
    this.#lifecycleController?.abort();
    this.#lifecycleController = null;
    this.#resetState();
  }

  /** Release a completed or failed local lifecycle if it is still current. */
  finish(runId?: string): void {
    if (runId !== undefined && this.#runId !== runId) return;
    this.#clearTimer();
    this.#lifecycleController?.abort();
    this.#lifecycleController = null;
    this.#resetState();
  }

  /** Ask the server to stop the durable run; disconnecting alone never does this. */
  async cancel(): Promise<void> {
    const runId = this.#runId;
    if (!runId || this.#stopping || this.#cancelControllers.has(runId)) return;
    this.#stopping = true;
    this.#notify();
    const controller = new AbortController();
    this.#cancelControllers.set(runId, controller);
    try {
      await this.#cancelRun(runId, controller.signal);
    } catch {
      if (!controller.signal.aborted && this.#runId === runId) {
        this.#stopping = false;
        this.#notify();
      }
    } finally {
      if (this.#cancelControllers.get(runId) === controller) {
        this.#cancelControllers.delete(runId);
      }
    }
  }

  /** Abort every owned request when the Feature itself leaves the document. */
  disconnect(): void {
    this.detach();
    for (const controller of this.#cancelControllers.values()) controller.abort();
    this.#cancelControllers.clear();
  }

  /** Follow and replay one durable event stream from its last consumed sequence. */
  async follow(
    conversationId: string,
    runId: string,
    onEvent: (eventType: string, data: string) => void,
  ): Promise<FollowResult> {
    const signal = this.#lifecycleController?.signal;
    if (!signal || signal.aborted || this.#runId !== runId) return {kind: 'aborted'};
    let barrenAttempts = 0;

    const mayRetry = (before: number): boolean => {
      barrenAttempts = answerRunStore.lastSequence(conversationId, runId) > before
        ? 0
        : barrenAttempts + 1;
      return barrenAttempts <= this.#maxReconnectAttempts;
    };

    while (!signal.aborted) {
      const after = answerRunStore.lastSequence(conversationId, runId);
      let response: Response;
      try {
        response = await this.#fetch(`/web/api/answer/${encodeURIComponent(runId)}/events`, {
          signal,
          headers: after > 0 ? {'Last-Event-ID': String(after)} : undefined,
        });
      } catch {
        if (signal.aborted) return {kind: 'aborted'};
        if (!mayRetry(after)) break;
        if (!await this.#delay(signal)) return {kind: 'aborted'};
        continue;
      }
      if (response.status === 404 || response.status === 410) break;
      if (!response.ok) return {kind: 'error', message: 'Service error. Please try again.'};

      let terminal = false;
      try {
        terminal = await this.#readEvents(
          response,
          conversationId,
          runId,
          signal,
          onEvent,
        );
      } catch {
        if (signal.aborted) return {kind: 'aborted'};
        if (!mayRetry(after)) break;
        if (!await this.#delay(signal)) return {kind: 'aborted'};
        continue;
      }
      if (terminal) return {kind: 'terminal'};
      if (!mayRetry(after)) break;
      if (!await this.#delay(signal)) return {kind: 'aborted'};
    }

    if (signal.aborted) return {kind: 'aborted'};
    try {
      const stored = await this.#getRun(runId, signal);
      if (stored.status === 'queued' || stored.status === 'running') {
        return {kind: 'retryable', stored};
      }
      return {kind: 'terminal', stored};
    } catch {
      if (signal.aborted) return {kind: 'aborted'};
      return {kind: 'error', message: 'Service error. Please try again.'};
    }
  }

  #startLifecycle(): void {
    this.#clearTimer();
    this.#lifecycleController?.abort();
    this.#lifecycleController = new AbortController();
    this.#submissionPending = false;
    this.#runId = null;
    this.#stopping = false;
  }

  #resetState(): void {
    const changed = this.#active || this.#submissionPending || this.#stopping || this.#runId !== null;
    this.#submissionPending = false;
    this.#active = false;
    this.#stopping = false;
    this.#runId = null;
    if (changed) this.#notify();
  }

  #notify(): void {
    this.#onStateChange();
  }

  #clearTimer(): void {
    if (this.#timer === null) return;
    clearTimeout(this.#timer);
    this.#timer = null;
  }

  #delay(signal: AbortSignal): Promise<boolean> {
    if (signal.aborted) return Promise.resolve(false);
    return new Promise((resolve) => {
      const onAbort = (): void => {
        this.#clearTimer();
        resolve(false);
      };
      signal.addEventListener('abort', onAbort, {once: true});
      this.#timer = setTimeout(() => {
        this.#timer = null;
        signal.removeEventListener('abort', onAbort);
        resolve(!signal.aborted);
      }, this.#reconnectDelayMs);
    });
  }

  async #readEvents(
    response: Response,
    conversationId: string,
    runId: string,
    signal: AbortSignal,
    onEvent: (eventType: string, data: string) => void,
  ): Promise<boolean> {
    if (!response.body) throw new Error('Response body is not streamable');
    let terminal = false;
    const parser = createSSEParser((eventType, data, id) => {
      const sequence = Number(id);
      if (Number.isFinite(sequence) && sequence > 0) {
        if (sequence <= answerRunStore.lastSequence(conversationId, runId)) return;
        answerRunStore.recordSequence(conversationId, runId, sequence);
      }
      onEvent(eventType, data);
      if (eventType === 'done' || eventType === 'error') terminal = true;
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    try {
      while (!signal.aborted) {
        const result = await reader.read();
        if (result.done) break;
        parser.push(decoder.decode(result.value, {stream: true}));
      }
      if (!signal.aborted) {
        parser.push(decoder.decode());
        parser.flush();
      }
    } finally {
      void reader.cancel().catch(() => {});
    }
    return terminal;
  }
}
