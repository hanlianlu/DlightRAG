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
  MathJax?: MathJaxConfig;
}


