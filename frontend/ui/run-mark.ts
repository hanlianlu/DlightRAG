// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The decision words a marked step wears, in every surface that shows one.
 *
 *  A Run states one mechanical decision about a step as a mark, and the words for it live here so a
 *  search row and the Run overview's attention list cannot drift apart: the badge names the mark for
 *  the eye, and its aria label says the whole sentence for a screen reader. A colour is never the
 *  only carrier — the badge's words are the mark.
 */

import {msg} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import type {DecisionMark} from '../api/conversations.ts';
import styles from '../styles/run-overview.module.css';

interface MarkCopy {
  readonly label: {readonly source: string; readonly id: string};
  readonly aria: {readonly source: string; readonly id: string};
}

/** The words one decision badge draws, and what a screen reader hears for it. The ids are the ones
 *  the Inspector's step rows already use, so a mark reads the same wherever a Run is read. */
const MARK_TEXT: Record<DecisionMark, MarkCopy> = {
  failed: {
    label: {source: 'Failed', id: 'tracesPanel.markFailed'},
    aria: {source: 'This step failed.', id: 'tracesPanel.markFailedAria'},
  },
  degraded: {
    label: {source: 'Degraded', id: 'tracesPanel.markDegraded'},
    aria: {source: 'The search degraded.', id: 'tracesPanel.markDegradedAria'},
  },
  repeated: {
    label: {source: 'Repeated', id: 'tracesPanel.markRepeated'},
    aria: {source: 'This repeats an earlier call on the same subject.', id: 'tracesPanel.markRepeatedAria'},
  },
  redirect: {
    label: {source: 'Redirected', id: 'tracesPanel.markRedirect'},
    aria: {source: 'The main agent changed what it looks at after a child answered.', id: 'tracesPanel.markRedirectAria'},
  },
};

/** The word a mark is called by, for a row that names it in passing rather than wearing its badge. */
export function markLabel(mark: DecisionMark): string {
  return msg(MARK_TEXT[mark].label.source, {id: MARK_TEXT[mark].label.id});
}

/** The full sentence a screen reader hears for a mark. */
function markAriaLabel(mark: DecisionMark): string {
  return msg(MARK_TEXT[mark].aria.source, {id: MARK_TEXT[mark].aria.id});
}

/** One decision a row states, named for the eye and for a screen reader: a badge, never a colour. */
export function markBadge(mark: DecisionMark): TemplateResult {
  return html`
    <span class=${styles.mark} data-run-mark=${mark} role="note"
          aria-label=${markAriaLabel(mark)}>${markLabel(mark)}</span>
  `;
}
