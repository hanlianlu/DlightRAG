// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Evidence an agent holds, as a fold of its source lines: the main agent's cited sources, or what a
 * child merged into the answer. A line leads with its citation number, and when the answer's Sources hold
 * that number the line opens the source. */

import {msg, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import type {AnswerPresentation} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {raise} from '../lib/dom.ts';
import styles from '../styles/agent-session.module.css';

export function evidenceFold(
  host: HTMLElement,
  handles: readonly string[],
  presentation: AnswerPresentation | null,
): TemplateResult {
  const known = new Set(presentation?.sources.map((source) => source.id));
  const open = (event: Event): void => {
    const trigger = event.currentTarget as HTMLElement;
    raise(host, 'dl-answer-source-open', {
      presentation: presentation!,
      referenceId: trigger.dataset.evidenceRef!,
      returnFocus: trigger,
    });
  };
  return html`
    <details class=${styles.fold}>
      <summary class=${styles.summary}>${icon('disclosure', {size: 'xs', className: styles.chevron})}
        ${msg(str`Evidence · ${handles.length}`, {id: 'evidence.title'})}</summary>
      <ul class=${styles.mono}>
        ${handles.map((handle) => {
          const reference = /^\[([^\]]+)\]/.exec(handle)?.[1];
          return reference !== undefined && known.has(reference)
            ? html`<li><button type="button" class=${styles.evidence} data-evidence-ref=${reference}
                               @click=${open}>${handle}</button></li>`
            : html`<li>${handle}</li>`;
        })}
      </ul>
    </details>
  `;
}
