// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import DOMPurify, {type Config} from 'dompurify';

// Attributes shared by both profiles. Citation identities use data attributes;
// server-created resource slots use ordinary span classes. Image and Artifact
// intent is rendered by Lit, never admitted as active model-authored HTML.
const BASE_ADD_ATTR = [
  'aria-label',
  'data-chunk',
  'data-ref',
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

// Strict profile — untrusted LLM answer/preview/highlight content. It is the second pass behind
// the server's nh3 allowlist (src/dlightrag/adapters/http/browser/safe_html.py), which emits no
// form controls, no <style> and no style or id attributes; this pass refuses the same, so the page
// stays safe if the server pass ever lets one through:
//  - <style> (HTML or inside an inline <svg>) is document-global CSS: it can hide or overlay the
//    application, or fetch URLs, from answer text;
//  - a form with its controls posts what the reader types to an address the model chose;
//  - an id can shadow one the application looks up by id.
const LLM_SANITIZE_CONFIG: Config = {
  USE_PROFILES: {html: true, svg: true},
  ALLOW_DATA_ATTR: false,
  ADD_TAGS: ['line', 'polyline'],
  ADD_ATTR: [...BASE_ADD_ATTR],
  FORBID_TAGS: ['style', 'form', 'input', 'textarea', 'select', 'option', 'optgroup'],
  FORBID_ATTR: ['style', 'id'],
};

export function setSanitizedLlmHtml(element: Element, html: string): void {
  element.innerHTML = DOMPurify.sanitize(html, LLM_SANITIZE_CONFIG) as string;
}

// Mermaid renders untrusted model output into SVG. We display it as an isolated
// <img> (secure static mode already blocks scripts and external references) and
// sanitize the markup first as defense in depth: strip scripting and HTML-in-SVG
// (foreignObject) while keeping the scoped <style> Mermaid needs for its baked-in
// theme colors.
const SVG_SANITIZE_CONFIG: Config = {
  USE_PROFILES: {svg: true, svgFilters: true},
  FORBID_TAGS: ['foreignObject', 'script'],
  ALLOW_UNKNOWN_PROTOCOLS: false,
};

export function sanitizeSvg(svg: string): string {
  return DOMPurify.sanitize(svg, SVG_SANITIZE_CONFIG) as string;
}
