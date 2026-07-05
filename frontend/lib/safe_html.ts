// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import DOMPurify, {type Config} from 'dompurify';

const SANITIZE_CONFIG: Config = {
  USE_PROFILES: {html: true, svg: true},
  ALLOW_DATA_ATTR: true,
  ADD_TAGS: ['line', 'polyline'],
  ADD_ATTR: [
    'aria-label',
    'data-action',
    'data-chunk',
    'data-full-src',
    'data-ref',
    'data-src',
    'download',
    'hidden',
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
  ],
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
