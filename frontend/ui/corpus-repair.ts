// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The one copy Files and document recovery show for a Run parked for operator repair. */

import {msg} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import type {CorpusRunResumeOutcome, TrackedCorpusRun} from '../lib/corpus-run-tracker.ts';

/** Why the Run is parked and what to do, in the server's words when it gave them. */
export function corpusRepairNotice(run: TrackedCorpusRun): TemplateResult {
  return html`<strong>${run.repairReason ?? msg('The Corpus outcome needs operator repair.', {
    id: 'corpusRepair.reason',
  })}</strong> <span>${run.repairRemedy ?? msg('Repair the Corpus, then resume this same Run.', {
    id: 'corpusRepair.remedy',
  })}</span>`;
}

export function resumeRepairLabel(): string {
  return msg('Resume after repair', {id: 'corpusRepair.resume'});
}

/** What a settled resume request tells the reader. */
export function resumeRepairResult(outcome: Exclude<CorpusRunResumeOutcome, 'stale'>): string {
  return outcome === 'accepted'
    ? msg('Corpus repair resume accepted.', {id: 'corpusRepair.resumeAccepted'})
    : msg('Corpus repair resume failed.', {id: 'corpusRepair.resumeFailed'});
}
