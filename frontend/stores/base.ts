// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Common local change subscription. */
export abstract class Store {
  readonly #subscribers = new Set<() => void>();

  subscribe(handler: () => void): () => void {
    this.#subscribers.add(handler);
    return () => { this.#subscribers.delete(handler); };
  }

  protected changed(): void {
    for (const handler of this.#subscribers) handler();
  }

}
