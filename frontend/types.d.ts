// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

interface Window {
  __DLIGHTRAG_STATIC_VERSION__: string;
  htmx: HTMXGlobal;
}

interface HTMXGlobal {
  process(element: Element): void;
  ajax(method: string, url: string, options?: HTMXAjaxOptions): Promise<void>;
  trigger(element: Element | null, event: string): void;
}

interface HTMXAjaxOptions {
  target?: string;
  swap?: string;
  values?: Record<string, string>;
}

// nanoevents types (pre-declaration before Phase 2 npm install)
declare module 'nanoevents' {
  export interface Unsubscribe {
    (): void;
  }
  export interface Emitter<Events extends Record<string, unknown>> {
    on<E extends keyof Events>(event: E, fn: (payload: Events[E]) => void): Unsubscribe;
    emit<E extends keyof Events>(event: E, payload: Events[E]): void;
  }
  export function createNanoEvents<Events extends Record<string, unknown>>(): Emitter<Events>;
}
