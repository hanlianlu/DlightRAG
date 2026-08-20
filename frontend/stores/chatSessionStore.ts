// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {Store} from './base.ts';

/** Browser-local state of the one answer submission/event reader this tab owns. */
class ChatSessionStore extends Store {
  #active = false;

  get active(): boolean {
    return this.#active;
  }

  setActive(active: boolean): void {
    if (this.#active === active) return;
    this.#active = active;
    this.changed();
  }
}

export const chatSessionStore = new ChatSessionStore();
