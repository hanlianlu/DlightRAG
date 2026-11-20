// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What a reader searched for, marked where it appears, so a narrowed pane says why it is narrowed. */

import {html, type TemplateResult} from 'lit';

/** The text with every term marked; plain text when there is nothing to mark. */
export function markedText(text: string, terms: readonly string[]): TemplateResult | string {
  const wanted = terms.filter((term) => term !== '');
  if (wanted.length === 0) return text;
  const pattern = new RegExp(`(${wanted.map(escapeTerm).join('|')})`, 'gi');
  return html`${text
    .split(pattern)
    .map((part, index) => (index % 2 === 1 ? html`<mark>${part}</mark>` : part))}`;
}

function escapeTerm(term: string): string {
  return term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
