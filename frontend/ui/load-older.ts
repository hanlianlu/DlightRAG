// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The one control every keyset-paged list offers for its next page. */

import {html, nothing, type TemplateResult} from 'lit';
import type {KeysetPagerStatus} from '../lib/paged.ts';

export interface LoadOlderControl {
  /** Names the list: the button carries it as `data-load-older`, its status as `data-load-older-status`. */
  list: string;
  pages: KeysetPagerStatus;
  label: string;
  retryLabel: string;
  /** What the polite status says while a next page loads, once it arrived, and when it failed. */
  loading: string;
  loaded: string;
  failed: string;
  onLoad: (event: Event) => void;
  /** The owning Feature's layout classes for the button row and the button. */
  rowClass?: string;
  buttonClass?: string;
}

function status(control: LoadOlderControl): string {
  const {pages} = control;
  // A first page replaces the list rather than extending it: the list's own
  // status speaks for that load, so only a next page is announced here.
  if (pages.state === 'loading' && !pages.starting) return control.loading;
  if (pages.outcome === 'loaded') return control.loaded;
  if (pages.outcome === 'failed') return control.failed;
  return '';
}

/** A busy-aware next-page button plus the polite status that outlives it.
 *  While any page loads the button is aria-disabled and ignores activation;
 *  it is never natively disabled, which would drop the reader's focus. */
export function loadOlderControl(control: LoadOlderControl): TemplateResult {
  const {pages} = control;
  const busy = pages.state === 'loading';
  return html`
    ${pages.hasOlder ? html`
      <div class=${control.rowClass ?? nothing}>
        <button type="button" class=${control.buttonClass ?? nothing}
                data-load-older=${control.list}
                aria-busy=${busy ? 'true' : 'false'}
                aria-disabled=${busy ? 'true' : nothing}
                @click=${(event: Event) => { if (!busy) control.onLoad(event); }}>
          ${pages.outcome === 'failed' ? control.retryLabel : control.label}
        </button>
      </div>
    ` : nothing}
    <span class="dl-sr-only" data-load-older-status=${control.list} role="status"
          aria-live="polite">${status(control)}</span>
  `;
}
