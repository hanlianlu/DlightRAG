// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import DOMPurify, {type Config} from 'dompurify';

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

// Strict profile — untrusted LLM answer/preview/highlight content.
const LLM_SANITIZE_CONFIG: Config = {
  USE_PROFILES: {html: true, svg: true},
  ALLOW_DATA_ATTR: true,
  ADD_TAGS: ['line', 'polyline'],
  ADD_ATTR: [...BASE_ADD_ATTR],
};

export function setSanitizedLlmHtml(element: Element, html: string): void {
  element.innerHTML = DOMPurify.sanitize(html, LLM_SANITIZE_CONFIG) as string;
}

export function llmFragmentFromSanitizedHtml(html: string): DocumentFragment {
  const template = document.createElement('template');
  setSanitizedLlmHtml(template, html);
  return template.content;
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
