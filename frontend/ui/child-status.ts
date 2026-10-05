// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Where a child stands, as the Child agents dock says it: its state in words, how long it has run,
 * and the mark that shows it at a glance. The roster's rows and the page of one child share these. */

import {msg} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import type {AgentChildStatus} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {elapsed} from '../lib/date-format.ts';
import styles from '../styles/child-session.module.css';

/** The word for a child's state; a cancellation says who cancelled. */
export function childStateText(child: AgentChildStatus): string {
  switch (child.status) {
    case 'running': return msg('Running', {id: 'childSession.state.running'});
    case 'succeeded': return msg('Done', {id: 'childSession.state.done'});
    case 'failed': return msg('Failed', {id: 'childSession.state.failed'});
    case 'cancelled':
      switch (child.cancellationOrigin) {
        case 'user': return msg('Cancelled by you', {id: 'childSession.state.cancelledByYou'});
        case 'parent': return msg('Cancelled by the agent', {id: 'childSession.state.cancelledByAgent'});
        case 'run': return msg('Stopped with the run', {id: 'childSession.state.stoppedWithRun'});
        default: return msg('Cancelled', {id: 'childSession.state.cancelled'});
      }
    default: return child.status;
  }
}

/** How long the child's current Operation has run, or took; nothing when the server gave no start. */
export function childElapsed(child: AgentChildStatus, now: number): string {
  const start = child.startedAt === null ? Number.NaN : Date.parse(child.startedAt);
  const end = child.status === 'running'
    ? now
    : child.finishedAt === null ? Number.NaN : Date.parse(child.finishedAt);
  return Number.isNaN(start) || Number.isNaN(end) ? '' : elapsed(end - start, getLocale());
}

/** The mark that says a child's state at a glance: a pulsing dot while it runs. */
export function childGlyph(child: AgentChildStatus): TemplateResult {
  switch (child.status) {
    case 'running':
      return html`<span class=${styles.glyph} data-state="running">${icon('status-dot', {size: 'lg', className: styles.pulse})}</span>`;
    case 'succeeded':
      return html`<span class=${styles.glyph} data-state="done">${icon('check', {size: 'sm'})}</span>`;
    case 'failed':
      return html`<span class=${styles.glyph} data-state="failed">${icon('close', {size: 'sm'})}</span>`;
    default:
      return html`<span class=${styles.glyph} data-state="stopped">${icon('stop', {size: 'xs'})}</span>`;
  }
}
