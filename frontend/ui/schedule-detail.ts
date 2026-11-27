// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The dialog a Schedule opens into: what it is, when it runs next or why it does not, what it asks, the
 * actions on it, and its recent runs.
 *
 * It follows the Schedule in the store, so a pause, a resume or an edit made from anywhere shows here at
 * once, and it closes when the Schedule is gone. It carries out nothing itself: an action is an intent that
 * the dialogs act on, the same one a row's quick actions raise. A run's Fire history is read here, newest
 * first and a page at a time, because nothing else shows it.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {listScheduleFires, type Schedule, type ScheduleFire} from '../api/schedules.ts';
import {ApiError} from '../api/wire.ts';
import {getLocale} from '../i18n/locale.ts';
import {dateTime} from '../lib/date-format.ts';
import {raise} from '../lib/dom.ts';
import {LatestRequest} from '../lib/latest-request.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {describeCadence, nextRunText, slotText} from '../lib/schedule-cadence.ts';
import {browserZone} from '../lib/zoned-time.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import styles from '../styles/schedule-dialogs.module.css';
import {modalResult} from './modal.ts';
import type {ScheduleIntent} from './schedule-section.ts';

/** A prompt longer than this is folded to a few lines until the reader asks for the rest. */
const FOLDED_PROMPT_CHARS = 280;
const FOLDED_PROMPT_LINES = 4;

type History =
  | {kind: 'loading'}
  | {kind: 'loaded'; items: readonly ScheduleFire[]; nextCursor: string | null}
  | {kind: 'failed'};

type Tone = 'live' | 'good' | 'bad' | 'quiet';

function fireState(fire: ScheduleFire): {label: string; tone: Tone} {
  switch (fire.state) {
    case 'pending': return {label: msg('Starting', {id: 'schedules.fire.pending'}), tone: 'live'};
    case 'started': return {label: msg('Running', {id: 'schedules.fire.started'}), tone: 'live'};
    case 'succeeded': return {label: msg('Succeeded', {id: 'schedules.fire.succeeded'}), tone: 'good'};
    case 'failed': return {label: msg('Failed', {id: 'schedules.fire.failed'}), tone: 'bad'};
    case 'refused': return {label: msg('Refused', {id: 'schedules.fire.refused'}), tone: 'bad'};
    case 'cancelled': return {label: msg('Cancelled', {id: 'schedules.fire.cancelled'}), tone: 'quiet'};
    case 'skipped_busy': return {label: msg('Skipped, still busy', {id: 'schedules.fire.skippedBusy'}), tone: 'quiet'};
    case 'skipped_limit': return {label: msg('Skipped, at the run limit', {id: 'schedules.fire.skippedLimit'}), tone: 'quiet'};
    case 'missed':
      return {
        label: fire.missedCount !== null && fire.missedCount > 1
          ? msg(str`Missed ${fire.missedCount} times`, {id: 'schedules.fire.missedMany'})
          : msg('Missed', {id: 'schedules.fire.missed'}),
        tone: 'quiet',
      };
  }
}

export class DlScheduleDetail extends LightElement {
  static properties = {
    handles: {attribute: false},
    scheduleId: {state: true},
    expanded: {state: true},
    history: {state: true},
  };

  declare handles: AppHandles;
  /** The Schedule shown, or null while the dialog is closed. */
  declare scheduleId: string | null;
  /** The whole prompt shows, not its first lines. */
  declare expanded: boolean;
  declare history: History;

  readonly #reads = new LatestRequest();

  constructor() {
    super();
    this.handles = productionHandles();
    this.scheduleId = null;
    this.expanded = false;
    this.history = {kind: 'loading'};
    /** Store reads: the Schedule shown, and whether a command on it is in flight. */
    new StoreController(this, () => [this.handles.schedules]);
  }

  override disconnectedCallback(): void {
    this.#reads.cancel();
    super.disconnectedCallback();
  }

  protected override updated(): void {
    // A Schedule deleted from anywhere takes its dialog with it.
    if (this.scheduleId !== null && this.handles.schedules.get(this.scheduleId) === null) this.close();
  }

  get shows(): string | null {
    return this.scheduleId;
  }

  /** Open the dialog on a Schedule; resolves once it has closed and focus is back. */
  async open(scheduleId: string, trigger: HTMLElement | null): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted || this.scheduleId !== null) return;
    this.scheduleId = scheduleId;
    this.expanded = false;
    void this.#read(null);
    await this.updateComplete;
    const dialog = this.querySelector<HTMLDialogElement>('dialog');
    if (!dialog) return;
    await modalResult(this, dialog, () => { if (trigger?.isConnected) trigger.focus(); }, signal);
    this.#reads.cancel();
    this.scheduleId = null;
  }

  close(): void {
    this.querySelector<HTMLDialogElement>('dialog')?.close();
  }

  /** Read the history again, as after a run that has just added its newest line. */
  reload(): void {
    if (this.scheduleId !== null) void this.#read(null);
  }

  /** Read one page of the Schedule's Fires: the first page replaces what was read, a later one follows it. */
  async #read(cursor: string | null): Promise<void> {
    const scheduleId = this.scheduleId;
    if (scheduleId === null) return;
    const before = this.history;
    if (cursor === null && before.kind !== 'loaded') this.history = {kind: 'loading'};
    const signal = this.#reads.start();
    try {
      const page = await listScheduleFires(scheduleId, cursor, signal);
      if (!this.#reads.isCurrent(signal)) return;
      const earlier = cursor !== null && before.kind === 'loaded' ? before.items : [];
      this.history = {kind: 'loaded', items: [...earlier, ...page.items], nextCursor: page.nextCursor};
    } catch (error) {
      if (!this.#reads.isCurrent(signal)) return;
      if (error instanceof ApiError && error.status === 404) this.handles.schedules.drop(scheduleId);
      else this.history = {kind: 'failed'};
    } finally {
      this.#reads.finish(signal);
    }
  }

  #ask(intent: ScheduleIntent): void {
    if (this.scheduleId !== null) raise(this, 'dl-schedule-intent', {intent, scheduleId: this.scheduleId, trigger: null});
  }

  #open(conversationId: string): void {
    this.close();
    raise(this, 'dl-conversation-select', {conversationId});
  }

  #when(schedule: Schedule): TemplateResult | typeof nothing {
    const locale = getLocale();
    switch (schedule.state) {
      case 'active': {
        if (schedule.nextFireAt === null) return nothing;
        const now = new Date();
        const mine = browserZone();
        return html`
          <p class=${styles.status} data-when>
            ${msg(str`Next run: ${nextRunText(schedule.nextFireAt, now, schedule.zone, locale)}`, {id: 'scheduleDetail.next'})}
            ${schedule.zone === mine ? nothing : html`<span class="dl-note">${
              msg(str`(your time ${slotText(schedule.nextFireAt, now, mine, locale)})`, {id: 'schedules.yourTime'})}</span>`}
          </p>`;
      }
      case 'paused':
        return html`
          <p class=${styles.status} data-when>${msg('Paused', {id: 'scheduleSection.paused'})}${
            schedule.pauseReason === null ? nothing : html`<span class="dl-note" data-pause-reason> · ${schedule.pauseReason}</span>`}</p>`;
      case 'completed':
        return html`<p class=${styles.status} data-when>${msg('Done', {id: 'scheduleSection.done'})}</p>`;
    }
  }

  #prompt(schedule: Schedule): TemplateResult {
    const long = schedule.prompt.length > FOLDED_PROMPT_CHARS || schedule.prompt.split('\n').length > FOLDED_PROMPT_LINES;
    return html`
      <section class=${styles.section}>
        <h3 class=${styles.caption}>${schedule.goalMaxRounds !== null
          ? msg('Goal objective', {id: 'schedules.objective'})
          : msg('Prompt', {id: 'schedules.prompt'})}</h3>
        <p class="${styles.promptText} ${long && !this.expanded ? styles.folded : ''}" data-prompt>${schedule.prompt}</p>
        ${long ? html`
          <button type="button" class="dl-text-action dl-text-action--disclosure" data-expand
            aria-expanded=${String(this.expanded)}
            @click=${() => { this.expanded = !this.expanded; }}>${
              this.expanded ? msg('Show less', {id: 'scheduleDetail.less'}) : msg('Show more', {id: 'scheduleDetail.more'})}</button>` : nothing}
      </section>`;
  }

  #actions(schedule: Schedule): TemplateResult {
    const pending = this.handles.schedules.isPending(schedule.scheduleId);
    const done = schedule.state === 'completed';
    const paused = schedule.state === 'paused';
    return html`
      <div class=${styles.buttons}>
        ${done ? nothing : html`
          <button type="button" class="dl-btn" data-action="run" ?disabled=${pending} @click=${() => { this.#ask('run'); }}>${
            msg('Run now', {id: 'scheduleSection.run'})}</button>
          <button type="button" class="dl-btn" data-action=${paused ? 'resume' : 'pause'} ?disabled=${pending}
            @click=${() => { this.#ask(paused ? 'resume' : 'pause'); }}>${
              paused ? msg('Resume', {id: 'scheduleSection.resume'}) : msg('Pause', {id: 'scheduleSection.pause'})}</button>
          <button type="button" class="dl-btn" data-action="edit" ?disabled=${pending} @click=${() => { this.#ask('edit'); }}>${
            msg('Edit', {id: 'scheduleSection.edit'})}</button>`}
        <button type="button" class="dl-btn dl-btn-danger-text" data-action="delete" ?disabled=${pending}
          @click=${() => { this.#ask('delete'); }}>${msg('Delete', {id: 'scheduleSection.delete'})}</button>
      </div>`;
  }

  #openLink(conversationId: string): TemplateResult {
    return html`
      <button type="button" class="dl-text-action ${styles.open}" data-open=${conversationId}
        @click=${() => { this.#open(conversationId); }}>${msg('Open', {id: 'scheduleDetail.open'})}</button>`;
  }

  #fires(): TemplateResult {
    const history = this.history;
    if (history.kind === 'failed') {
      return html`
        <p class="dl-note" role="alert">${msg('Could not load the history.', {id: 'schedules.historyFailed'})}
          <button type="button" class="dl-text-action" @click=${() => { void this.#read(null); }}>${msg('Retry', {id: 'scheduleSection.retry'})}</button></p>`;
    }
    if (history.kind === 'loading') {
      return html`<p class="dl-note" role="status">${msg('Loading history…', {id: 'schedules.historyLoading'})}</p>`;
    }
    if (history.items.length === 0) return html`<p class="dl-note">${msg('It has not run yet.', {id: 'scheduleDetail.noRuns'})}</p>`;
    const locale = getLocale();
    return html`
      <ul class=${styles.fires} data-fires>
        ${history.items.map((fire) => {
          const state = fireState(fire);
          return html`
            <li class=${styles.fire} data-fire=${fire.state}>
              <span class=${styles.slot}>${dateTime(fire.slot.toISOString(), locale)}</span>
              <span class=${styles.state} data-tone=${state.tone}>${state.label}</span>
              ${fire.conversationId === null ? nothing : this.#openLink(fire.conversationId)}
              ${fire.reason === null ? nothing : html`<span class=${styles.reason}>${fire.reason}</span>`}
            </li>`;
        })}
      </ul>
      ${history.nextCursor === null ? nothing : html`
        <button type="button" class="dl-btn ${styles.older}" data-older
          @click=${() => { void this.#read(history.nextCursor); }}>${msg('Show older', {id: 'schedules.older'})}</button>`}`;
  }

  protected override render(): TemplateResult {
    const schedule = this.scheduleId === null ? null : this.handles.schedules.get(this.scheduleId);
    return html`
      <dialog class="confirm-dialog ${styles.dialog}" aria-labelledby="schedule-detail-title">
        ${schedule === null ? nothing : html`
          <div class=${styles.frame} data-detail=${schedule.scheduleId}>
            <h2 id="schedule-detail-title" class=${styles.title}>${schedule.name}</h2>
            <div class=${styles.body}>
              <div class=${styles.lead}>
                <p class=${styles.cadence} data-cadence>${describeCadence(schedule, getLocale())} · ${schedule.zone}</p>
                ${this.#when(schedule)}
                ${schedule.changedBy === 'agent' ? html`<p class="dl-note" data-changed-by>${msg('Last changed by the Agent.', {id: 'scheduleDetail.changedByAgent'})}</p>` : nothing}
              </div>
              ${this.#prompt(schedule)}
              ${this.#actions(schedule)}
              <section class=${styles.section}>
                <h3 class=${styles.caption}>${msg('Recent runs', {id: 'scheduleDetail.recent'})}</h3>
                ${this.#fires()}
              </section>
            </div>
            <div class="dl-dialog-actions ${styles.actions}">
              <button type="button" @click=${() => { this.close(); }}>${msg('Close', {id: 'scheduleDetail.close'})}</button>
            </div>
          </div>`}
      </dialog>`;
  }
}

customElements.define('dl-schedule-detail', DlScheduleDetail);

declare global {
  interface HTMLElementTagNameMap {
    'dl-schedule-detail': DlScheduleDetail;
  }
}
