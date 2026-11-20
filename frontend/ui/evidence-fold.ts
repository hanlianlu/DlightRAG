// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Evidence an agent holds, as a fold of its source lines: the main agent's cited sources, or what a
 * child merged into the answer. A line leads with its citation number, and when the answer's Sources hold
 * that number the line opens the source. */

import {msg, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import type {AnswerPresentation} from '../api/conversations.ts';
import {raise} from '../lib/dom.ts';
import {foldView} from './fold.ts';
import styles from '../styles/agent-session.module.css';

export function evidenceFold(
  host: HTMLElement,
  handles: readonly string[],
  presentation: AnswerPresentation | null,
): TemplateResult {
  const known = new Set(presentation?.sources.map((source) => source.id));
  const open = (event: Event): void => {
    const trigger = event.currentTarget as HTMLElement;
    const referenceId = trigger.dataset.evidenceRef;
    // A line is a button only when the answer's Sources hold its number, so both are there.
    if (presentation === null || referenceId === undefined) return;
    raise(host, 'dl-answer-source-open', {presentation, referenceId, returnFocus: trigger});
  };
  const title = msg(str`Evidence · ${handles.length}`, {id: 'evidence.title'});
  return foldView({
    label: title,
    wordless: true,
    as: 'div',
    body: html`<ul class=${styles.mono} tabindex="0" aria-label=${title}>
      ${handles.map((handle) => {
        const reference = /^\[([^\]]+)\]/.exec(handle)?.[1];
        return reference !== undefined && known.has(reference)
          ? html`<li><button type="button" class=${styles.evidence} data-evidence-ref=${reference}
                             @click=${open}>${handle}</button></li>`
          : html`<li>${handle}</li>`;
      })}
    </ul>`,
  });
}
