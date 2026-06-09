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
    let id: string;
    if (window.crypto?.randomUUID) {
      id = window.crypto.randomUUID();
    } else {
      // crypto.randomUUID is supported in all modern browsers; fall back
      // to getRandomValues for truly old engines instead of predictable
      // Date.now() + performance.now().
      const bytes = new Uint8Array(16);
      window.crypto.getRandomValues(bytes);
      id = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
    }
    window.localStorage.setItem(SESSION_KEY, id);
    this.#sessionId = id;
    return id;
  }

  reset(): void {
    this.#sessionId = null;
  }
}

export const sessionStore = new SessionStore();
