// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

declare module '*.module.css' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

declare module '*.css' {}

// ── Window extensions ────────────────────────────────────────────────

interface MathJaxConfig {
  tex?: {
    inlineMath?: [string, string][];
    displayMath?: [string, string][];
    processEscapes?: boolean;
    tags?: string;
  };
  options?: {
    skipHtmlTags?: string[];
  };
  typesetPromise?: (elements?: Element[]) => Promise<void>;
}

interface Window {
  __DLIGHTRAG_STATIC_VERSION__: string;
  htmx: HTMXGlobal;
  MathJax: MathJaxConfig;
}

// ── HTMX ──────────────────────────────────────────────────────────────

interface HTMXGlobal {
  process(element: Element): void;
  ajax(method: string, url: string, options?: HTMXAjaxOptions): Promise<void>;
  trigger(element: Element | null, event: string, detail?: unknown): void;
}

interface HTMXAjaxOptions {
  target?: string;
  swap?: string;
  values?: Record<string, string>;
}

// Custom events emitted by HTMX after swaps
interface HTMXAfterSwapEvent extends Event {
  detail: {
    target: Element;
    xhr: XMLHttpRequest;
    pathInfo: { requestPath: string };
  };
}

// ── nanoevents ────────────────────────────────────────────────────────

declare module 'nanoevents' {
  export interface Unsubscribe {
    (): void;
  }
  export interface Emitter<Events extends { [key: string]: any }> {
    on<E extends keyof Events>(event: E, fn: (payload: Events[E]) => void): Unsubscribe;
    emit<E extends keyof Events>(event: E, payload: Events[E]): void;
  }
  export function createNanoEvents<Events extends { [key: string]: any }>(): Emitter<Events>;
}
