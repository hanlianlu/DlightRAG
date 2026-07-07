// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import DOMPurify, {type Config} from 'dompurify';

// htmx action attributes: only trusted server-rendered fragments (file panel,
// htmx swaps) ever need these. They must NOT survive on untrusted LLM answer
// content, where an injected hx-delete/hx-post would be a CSRF-like action sink
// if that content were ever processed by htmx.
const HX_ATTRS = [
  'hx-confirm',
  'hx-delete',
  'hx-encoding',
  'hx-get',
  'hx-indicator',
  'hx-post',
  'hx-swap',
  'hx-target',
  'hx-trigger',
  'hx-vals',
];

// Attributes shared by both profiles. data-action / data-* are kept for LLM
// content because answer image strips and reference lists rely on them
// (open-lightbox / open-ref-source), and their click handlers re-validate input.
const BASE_ADD_ATTR = [
  'aria-label',
  'data-action',
  'data-chunk',
  'data-full-src',
  'data-ref',
  'data-src',
  'download',
  'hidden',
  'role',
  'stroke',
  'stroke-linecap',
  'stroke-linejoin',
  'stroke-width',
  'tabindex',
  'viewBox',
  'x1',
  'x2',
  'y1',
  'y2',
];

// Permissive profile — trusted server fragments (includes htmx attributes).
const SANITIZE_CONFIG: Config = {
  USE_PROFILES: {html: true, svg: true},
  ALLOW_DATA_ATTR: true,
  ADD_TAGS: ['line', 'polyline'],
  ADD_ATTR: [...BASE_ADD_ATTR, ...HX_ATTRS],
};

// Strict profile — untrusted LLM answer/preview/highlight content (no htmx).
const LLM_SANITIZE_CONFIG: Config = {
  USE_PROFILES: {html: true, svg: true},
  ALLOW_DATA_ATTR: true,
  ADD_TAGS: ['line', 'polyline'],
  ADD_ATTR: [...BASE_ADD_ATTR],
};

export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, SANITIZE_CONFIG) as string;
}

export function setSanitizedHtml(element: Element, html: string): void {
  element.innerHTML = sanitizeHtml(html);
}

export function fragmentFromSanitizedHtml(html: string): DocumentFragment {
  const template = document.createElement('template');
  setSanitizedHtml(template, html);
  return template.content;
}

export function setSanitizedLlmHtml(element: Element, html: string): void {
  element.innerHTML = DOMPurify.sanitize(html, LLM_SANITIZE_CONFIG) as string;
}

export function llmFragmentFromSanitizedHtml(html: string): DocumentFragment {
  const template = document.createElement('template');
  setSanitizedLlmHtml(template, html);
  return template.content;
}
