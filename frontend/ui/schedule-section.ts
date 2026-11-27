// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The sidebar's Scheduled row: one nav item, with how many Schedules there are, that opens the Schedules dialog.
 *
 * The list is not in the navigation. The Agent makes and changes Schedules too, so the list can grow and
 * change under its reader, and it belongs in a window (`dl-schedule-list`). The row only says that
 * Schedules exist and asks to see them: the ask is an intent that the sidebar hands to the dialogs.
 */

import {msg} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {icon} from '../design-system/index.ts';
import {raise} from '../lib/dom.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';

/** `list` opens the Schedules dialog; the rest are asked of one Schedule (`new` of none). */
export type ScheduleIntent = 'list' | 'open' | 'new' | 'edit' | 'run' | 'pause' | 'resume' | 'delete';

/** What a reader asked, and the control they asked with, which focus returns to. */
export interface ScheduleIntentDetail {
  intent: ScheduleIntent;
  /** Null for `list` and `new`. */
  scheduleId: string | null;
  trigger: HTMLElement | null;
}

export class DlScheduleSection extends LightElement {
  static properties = {
    handles: {attribute: false},
    enabled: {attribute: false},
  };

  declare handles: AppHandles;
  /** The page is ready for the owner's data: the list is read once it is. */
  declare enabled: boolean;

  constructor() {
    super();
    this.handles = productionHandles();
    this.enabled = false;
    /** Store reads: how many Schedules there are, once the list has been read. */
    new StoreController(this, () => [this.handles.schedules]);
  }

  protected override willUpdate(): void {
    if (this.enabled && this.handles.schedules.listState === 'idle') void this.handles.schedules.load();
  }

  protected override render(): TemplateResult {
    const store = this.handles.schedules;
    const count = store.listState === 'ready' ? store.schedules.length : 0;
    return html`
      <button type="button" id="scheduled-btn" class="dl-nav-item" aria-haspopup="dialog"
        @click=${(event: Event) => {
          const trigger = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
          raise(this, 'dl-schedule-intent', {intent: 'list', scheduleId: null, trigger});
        }}>
        <span class="dl-nav-item-icon">${icon('clock', {size: 'sm'})}</span>
        <span class="dl-nav-item-text">
          <span class="dl-nav-item-label">${msg('Scheduled', {id: 'scheduleSection.title'})}</span>
        </span>
        ${count > 0 ? html`<span class="dl-nav-item-status" data-count>${count}</span>` : nothing}
      </button>`;
  }
}

customElements.define('dl-schedule-section', DlScheduleSection);

declare global {
  interface HTMLElementTagNameMap {
    'dl-schedule-section': DlScheduleSection;
  }

  interface HTMLElementEventMap {
    'dl-schedule-intent': CustomEvent<ScheduleIntentDetail>;
  }
}
