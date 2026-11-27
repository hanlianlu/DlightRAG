// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The dialog the sidebar's Scheduled row opens: every Schedule the owner has, one row each.
 *
 * It follows the store, so a Schedule the Agent makes while it is open appears at once and one that is
 * deleted goes, and a reader who cannot see it is told the count again. A row says what the Schedule is and when it runs next or that it is not running, and opens
 * that Schedule's detail, which carries every action; this dialog only draws and asks. The detail and the
 * form open on top of it as native stacked modals, so what they close back to is the row that opened them.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {Schedule} from '../api/schedules.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {raise} from '../lib/dom.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {describeCadence, nextRunText} from '../lib/schedule-cadence.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import styles from '../styles/schedule-dialogs.module.css';
import {modalResult} from './modal.ts';
import type {ScheduleIntent} from './schedule-section.ts';

/** The row's second line: the cadence in words, then when it runs next, or that it is paused or done. */
function meta(schedule: Schedule): string {
  const locale = getLocale();
  const cadence = describeCadence(schedule, locale);
  const status = ((): string => {
    switch (schedule.state) {
      case 'paused': return msg('Paused', {id: 'scheduleSection.paused'});
      case 'completed': return msg('Done', {id: 'scheduleSection.done'});
      case 'active':
        return schedule.nextFireAt === null
          ? ''
          : msg(str`Next: ${nextRunText(schedule.nextFireAt, new Date(), schedule.zone, locale)}`, {id: 'schedules.next'});
    }
  })();
  return status === '' ? cadence : `${cadence} · ${status}`;
}

/** How many there are, for the line that tells a reader who is not looking when the list changes. */
function countText(count: number): string {
  if (count === 0) return msg('No schedules', {id: 'scheduleList.count.none'});
  return count === 1
    ? msg('1 schedule', {id: 'scheduleList.count.one'})
    : msg(str`${count} schedules`, {id: 'scheduleList.count.many'});
}

export class DlScheduleList extends LightElement {
  static properties = {
    handles: {attribute: false},
    shown: {state: true},
  };

  declare handles: AppHandles;
  /** The dialog is open: only then are its rows drawn. */
  declare shown: boolean;

  /** The control that had focus at the last update was a row's, so a row that goes does not take focus with it. */
  #rowFocused = false;

  constructor() {
    super();
    this.handles = productionHandles();
    this.shown = false;
    /** Store reads: the Schedules, and whether the list has been read. */
    new StoreController(this, () => [this.handles.schedules]);
  }

  /** Open the dialog; resolves once it has closed and focus is back on `trigger`. */
  async open(trigger: HTMLElement | null): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted || this.shown) return;
    this.shown = true;
    // A page left open holds a list gone stale (its "Next: …", its count, what the Agent has made since), so
    // opening reads it again; one that was never read, or could not be, is read for the first time.
    const store = this.handles.schedules;
    void (store.listState === 'ready' ? store.refresh() : store.load());
    await this.updateComplete;
    const dialog = this.querySelector<HTMLDialogElement>('dialog');
    if (!dialog) return;
    await modalResult(this, dialog, () => { if (trigger?.isConnected) trigger.focus(); }, signal);
    this.shown = false;
  }

  protected override willUpdate(): void {
    const active = document.activeElement;
    this.#rowFocused = active instanceof HTMLElement && this.contains(active) && active.closest('[data-schedule-id]') !== null;
  }

  protected override updated(): void {
    // The row that had focus is gone (its Schedule was deleted): focus goes to the list's entry, not to the page.
    if (this.#rowFocused && !this.contains(document.activeElement)) void this.focusEntry();
  }

  close(): void {
    this.querySelector<HTMLDialogElement>('dialog')?.close();
  }

  /** Put focus on the first row, or on New schedule when there is none: where it goes when the row it was on is gone. */
  async focusEntry(): Promise<void> {
    await this.updateComplete;
    (this.querySelector<HTMLElement>('[data-schedule-id] button') ?? this.querySelector<HTMLElement>('[data-new]'))?.focus();
  }

  #ask(intent: ScheduleIntent, scheduleId: string | null, trigger: EventTarget | null): void {
    raise(this, 'dl-schedule-intent', {intent, scheduleId, trigger: trigger instanceof HTMLElement ? trigger : null});
  }

  #row(schedule: Schedule): TemplateResult {
    const id = schedule.scheduleId;
    // The row is named by what it shows, both lines: a name that left the second out would not hold the
    // words a reader sees on it (WCAG 2.5.3), as the Settings rows' can only because their detail is hidden.
    return html`
      <li data-schedule-id=${id} data-state=${schedule.state}>
        <button type="button" class="dl-nav-item dl-nav-item--list"
          @click=${(event: Event) => { this.#ask('open', id, event.currentTarget); }}>
          <span class="dl-nav-item-text">
            <span class="dl-nav-item-label" data-name>${schedule.name}</span>
            <span class="dl-nav-item-detail" data-meta>${meta(schedule)}</span>
          </span>
          <span class="dl-nav-item-disclosure" aria-hidden="true">${icon('disclosure', {size: 'sm'})}</span>
        </button>
      </li>`;
  }

  #body(): TemplateResult {
    const store = this.handles.schedules;
    if (store.listState === 'error') {
      return html`
        <p class="dl-note" role="status" data-failed>${msg('Could not load schedules.', {id: 'scheduleSection.loadFailed'})}
          <button type="button" class="dl-text-action" @click=${() => { void store.load(); }}>${msg('Retry', {id: 'scheduleSection.retry'})}</button></p>`;
    }
    if (store.listState !== 'ready') {
      return html`<p class="dl-note" role="status">${msg('Loading schedules…', {id: 'scheduleList.loading'})}</p>`;
    }
    if (store.schedules.length === 0) {
      return html`<p class="dl-note" data-empty>${msg('Nothing is scheduled yet. Ask the Agent in a chat, or add one here.', {id: 'scheduleList.empty'})}</p>`;
    }
    return html`
      <ul class="dl-list-card" role="list" data-schedules>
        ${repeat(store.schedules, (schedule) => schedule.scheduleId, (schedule) => this.#row(schedule))}
      </ul>`;
  }

  protected override render(): TemplateResult {
    const store = this.handles.schedules;
    const ready = store.listState === 'ready';
    return html`
      <dialog class="confirm-dialog ${styles.dialog}" aria-labelledby="schedule-list-title">
        ${this.shown ? html`
          <div class=${styles.frame}>
            <h2 id="schedule-list-title" class=${styles.title}>${msg('Scheduled', {id: 'scheduleSection.title'})}</h2>
            <div class="dl-sr-only" role="status" data-count-status>${ready ? countText(store.schedules.length) : ''}</div>
            <div class=${styles.body} data-body>${this.#body()}</div>
            <div class="dl-dialog-actions ${styles.actions}">
              <button type="button" data-close @click=${() => { this.close(); }}>${msg('Close', {id: 'scheduleDetail.close'})}</button>
              <button type="button" class="dl-dialog-primary" data-new ?disabled=${!ready}
                @click=${(event: Event) => { this.#ask('new', null, event.currentTarget); }}>${msg('New schedule', {id: 'scheduleSection.new'})}</button>
            </div>
          </div>` : nothing}
      </dialog>`;
  }
}

customElements.define('dl-schedule-list', DlScheduleList);

declare global {
  interface HTMLElementTagNameMap {
    'dl-schedule-list': DlScheduleList;
  }
}
