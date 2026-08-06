// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

declare module '*.module.css' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

declare module '*.css' {}

// ── Window extensions ────────────────────────────────────────────────

interface MathJaxConfig {
  loader?: {
    paths?: {[key: string]: string};
  };
  svg?: {
    dynamicPrefix?: string;
  };
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
  htmx: HTMXGlobal;
  MathJax?: MathJaxConfig;
}

declare const htmx: HTMXGlobal;

// ── HTMX ──────────────────────────────────────────────────────────────

interface HTMXGlobal {
  process(element: Element): void;
  ajax(method: string, url: string, options?: HTMXAjaxOptions): Promise<void>;
}

interface HTMXAjaxOptions {
  swap?: string;
  values?: Record<string, string>;
}

// Custom events emitted by HTMX after swaps
interface HTMXEvent extends Event {
  detail: {
    isError?: boolean;
    serverResponse?: string;
    shouldSwap?: boolean;
    successful?: boolean;
    target?: Element;
    xhr: XMLHttpRequest;
  };
}
