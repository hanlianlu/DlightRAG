// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {bus, type DlightragEvents} from '../events/bus.ts';

export interface SubscribableStore {
  subscribe(handler: () => void): () => void;
}

/** Common local change subscription; semantic cross-domain events are transitional. */
export abstract class Store implements SubscribableStore {
  readonly #subscribers = new Set<() => void>();

  subscribe(handler: () => void): () => void {
    this.#subscribers.add(handler);
    return () => { this.#subscribers.delete(handler); };
  }

  protected changed(): void {
    for (const handler of this.#subscribers) handler();
  }

  protected emit<E extends keyof DlightragEvents>(
    event: E,
    ...payload: Parameters<DlightragEvents[E]>
  ): void {
    this.changed();
    bus.emit(event, ...payload);
  }
}
