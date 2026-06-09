// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const SESSION_KEY = 'dlightrag.session_id';

class SessionStore {
  #sessionId: string | null = null;

  get sessionId(): string {
    if (this.#sessionId) return this.#sessionId;
    const stored = window.localStorage.getItem(SESSION_KEY);
    if (stored) {
      this.#sessionId = stored;
      return stored;
    }
    const id = window.crypto?.randomUUID
      ? window.crypto.randomUUID()
      : 'session_' + Date.now().toString(36) + '_' + performance.now().toString(36).replace('.', '');
    window.localStorage.setItem(SESSION_KEY, id);
    this.#sessionId = id;
    return id;
  }

  reset(): void {
    this.#sessionId = null;
  }
}

export const sessionStore = new SessionStore();
