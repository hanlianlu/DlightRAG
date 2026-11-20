// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * The one line that opens, and the window it opens: a summary a reader presses, and a body that
 * scrolls in its own box instead of running the length of the page.
 *
 * Every fold that leads with a line — a tool call, an agent's conclusion, the evidence it holds —
 * is this shape, so they look and behave alike and there is only one place to change them. It is
 * deliberately not a custom element: the callers draw list items of their own, and a `<details>`
 * inside a slot cannot pass a flex chain through, which the pane's layout depends on.
 */
import {html, nothing, type TemplateResult} from 'lit';
import {icon} from '../design-system/index.ts';
import styles from '../styles/activity.module.css';

interface FoldView {
  /** The words on the line a reader presses. */
  label: TemplateResult | string;
  /** What opens: words for a window, or markup when the fold holds something richer. */
  body: TemplateResult | string;
  /** The element the body is drawn in; a `pre` keeps a result's own line breaks, a `div` does not. */
  wordless?: boolean;
  /** A state the line carries, such as a running or failed call. */
  tone?: string;
  /** What this fold adds to the step it is drawn as, if it is one. */
  className?: string;
  /** A list item where the fold is a step of a list, and a plain box where it is not. */
  as?: 'li' | 'div';
  /** A window of an agent's own words keeps the page's type instead of a result's monospace. */
  prose?: boolean;
  /** The step this fold stands for, so a search can walk back to it; the same value the steps
   *  drawn by hand carry in `data-sequence`, key included. */
  sequence?: number | symbol;
  /** The Tool call this fold stands for, so the Run timeline can name its row from the outside. */
  callId?: string;
  /** What kind of step this is, so the Run timeline can find the row it came from. */
  stepKind?: string;
}

export function foldView({label, body, wordless = false, prose = false, tone = '', className = '', as = 'li', sequence, callId, stepKind}: FoldView): TemplateResult {
  const inside = html`
      <details>
        <summary class=${styles.stepSummary}>
          <span class=${styles.stepLine}>${label}</span>
          ${icon('disclosure', {size: 'xs', className: styles.chevron})}
        </summary>
        ${wordless || prose
          ? html`<div class="${styles.full} ${prose ? styles.prose : ''}" tabindex="0">${body}</div>`
          : html`<pre class=${styles.full} tabindex="0">${body}</pre>`}
      </details>
  `;
  // A list item outside a list is a violation, so the one the callers that lead a list ask for is
  // the one drawn as `li`; anything else gets a box that looks the same.
  return as === 'li'
    ? html`<li class="${styles.step} ${tone} ${className}" data-sequence=${sequence ?? nothing}
            data-call-id=${callId ?? nothing} data-step-kind=${stepKind ?? nothing}>${inside}</li>`
    : inside;
}
