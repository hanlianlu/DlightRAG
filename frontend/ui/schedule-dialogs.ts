// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What a reader's ask of the Schedules opens or does: their list, a detail, a form, a confirmation, a command.
 *
 * The Scheduled row, the list and the detail dialog only say what was asked (`dl-schedule-intent`); this
 * carries it out, so every ask has one act and one set of words. The store answers how a command ended without
 * wording it, and the reader is told here: a started run, the reason a Schedule would not start, a Schedule
 * that is gone. The dialogs stand outside the sidebar's navigation, as the conversation dialogs do, so the
 * drawer's focus trap never sees them. The detail and the form open on top of the list, which stays open
 * under them.
 */

import {msg, str} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import type {Schedule, ScheduleFire} from '../api/schedules.ts';
import {LightElement} from '../lib/lit-host.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import type {ScheduleOutcome} from '../stores/schedule-store.ts';
import {modalResult} from './modal.ts';
import './schedule-detail.ts';
import './schedule-form.ts';
import './schedule-list.ts';
import type {DlScheduleDetail} from './schedule-detail.ts';
import type {DlScheduleForm} from './schedule-form.ts';
import type {DlScheduleList} from './schedule-list.ts';
import type {ScheduleIntentDetail} from './schedule-section.ts';
import {requestToast} from './toast-request.ts';

export class DlScheduleDialogs extends LightElement {
  static properties = {
    handles: {attribute: false},
    deleting: {state: true},
  };

  declare handles: AppHandles;
  /** The name of the Schedule the delete dialog is asking about, so its copy can name it. */
  declare deleting: string | null;

  /** The control the detail dialog was opened from (a row of the list), which what it opens in turn returns focus to. */
  #origin: HTMLElement | null = null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.deleting = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    // The detail dialog asks for its own actions the way a row does.
    this.addEventListener('dl-schedule-intent', (event) => {
      event.stopPropagation();
      void this.handle(event.detail);
    }, {signal: this.lifetime});
    // A run's conversation opened from the detail takes the reader away from the Schedules: the list does not stay
    // modal over it. The event goes on to the sidebar, which carries out the navigation.
    this.addEventListener('dl-conversation-select', () => { this.#list()?.close(); }, {signal: this.lifetime});
  }

  #list(): DlScheduleList | null {
    return this.querySelector<DlScheduleList>('dl-schedule-list');
  }

  #detail(): DlScheduleDetail | null {
    return this.querySelector<DlScheduleDetail>('dl-schedule-detail');
  }

  #form(): DlScheduleForm | null {
    return this.querySelector<DlScheduleForm>('dl-schedule-form');
  }

  /** Carry out what was asked of a Schedule. */
  async handle(ask: ScheduleIntentDetail): Promise<void> {
    const store = this.handles.schedules;
    if (ask.intent === 'list') {
      await this.#list()?.open(ask.trigger);
      return;
    }
    if (ask.intent === 'new') {
      await this.#form()?.open(null, ask.trigger);
      return;
    }
    const schedule = ask.scheduleId === null ? null : store.get(ask.scheduleId);
    if (schedule === null) return;
    const trigger = ask.trigger ?? this.#origin;
    switch (ask.intent) {
      case 'open':
        this.#origin = ask.trigger;
        await this.#detail()?.open(schedule.scheduleId, ask.trigger);
        this.#origin = null;
        await this.#backToList(ask.trigger);
        return;
      case 'edit':
        // The form takes the detail's place, and focus returns to what opened the detail.
        this.#detail()?.close();
        await this.#form()?.open(schedule, trigger);
        await this.#backToList(trigger);
        return;
      case 'run': return this.#run(schedule);
      case 'pause': return this.#setPaused(schedule, true);
      case 'resume': return this.#setPaused(schedule, false);
      case 'delete': return this.#delete(schedule, trigger);
    }
  }

  /** A dialog opened from a row has closed: when the row is gone (its Schedule was deleted), focus goes to the list's entry. */
  async #backToList(trigger: HTMLElement | null): Promise<void> {
    if (trigger?.isConnected === false) await this.#list()?.focusEntry();
  }

  /** Tell the reader how a command ended when it did not end well; whether it did. */
  #told<T>(outcome: ScheduleOutcome<T>, schedule: Schedule, failure: string): outcome is {kind: 'ok'; value: T} {
    if (outcome.kind === 'gone') {
      requestToast(this, {
        tone: 'error',
        message: msg(str`${schedule.name} no longer exists, so it was removed from the list.`, {id: 'schedules.gone'}),
      });
    } else if (outcome.kind === 'failed') {
      requestToast(this, {tone: 'error', message: outcome.reason ?? failure});
    }
    return outcome.kind === 'ok';
  }

  async #setPaused(schedule: Schedule, pause: boolean): Promise<void> {
    const store = this.handles.schedules;
    this.#told(
      await (pause ? store.pause(schedule.scheduleId) : store.resume(schedule.scheduleId)),
      schedule,
      pause
        ? msg(str`Could not pause ${schedule.name}.`, {id: 'schedules.pauseFailed'})
        : msg(str`Could not resume ${schedule.name}.`, {id: 'schedules.resumeFailed'}),
    );
  }

  async #run(schedule: Schedule): Promise<void> {
    const outcome = await this.handles.schedules.run(schedule.scheduleId);
    if (!this.#told(outcome, schedule, msg(str`Could not run ${schedule.name}.`, {id: 'schedules.runFailed'}))) return;
    const fire: ScheduleFire = outcome.value;
    if (fire.state === 'pending' || fire.state === 'started') {
      requestToast(this, {message: msg(str`Started ${schedule.name}.`, {id: 'schedules.runStarted'})});
    } else {
      // A Fire that started nothing says why: the owner's own limit, a busy Run, an access they lost.
      requestToast(this, {
        tone: 'error',
        message: fire.reason ?? msg(str`${schedule.name} did not start.`, {id: 'schedules.runNotStarted'}),
      });
    }
    // The Fire is the newest line of the history the detail may be showing.
    if (this.#detail()?.shows === schedule.scheduleId) this.#detail()?.reload();
  }

  async #delete(schedule: Schedule, trigger: HTMLElement | null): Promise<void> {
    const signal = this.lifetime;
    const dialog = this.querySelector<HTMLDialogElement>('#schedule-delete-dialog');
    if (signal.aborted || !dialog || this.handles.schedules.isPending(schedule.scheduleId)) return;
    this.deleting = schedule.name;
    await this.updateComplete;
    const answer = await modalResult(this, dialog, () => { if (trigger?.isConnected) trigger.focus(); }, signal);
    this.deleting = null;
    if (answer !== 'delete' || signal.aborted) return;
    const outcome = await this.handles.schedules.remove(schedule.scheduleId);
    if (signal.aborted) return;
    if (this.#told(outcome, schedule, msg(str`Could not delete ${schedule.name}.`, {id: 'schedules.deleteFailed'}))) {
      requestToast(this, {message: msg(str`Deleted ${schedule.name}.`, {id: 'schedules.deleted'})});
    }
    // The Schedule is gone, and its row and focus with it: the detail's close, which resolves its `open`, brings focus back.
    if (outcome.kind !== 'failed' && this.#detail()?.shows === schedule.scheduleId) this.#detail()?.close();
  }

  protected override render(): TemplateResult {
    return html`
      <dl-schedule-list .handles=${this.handles}></dl-schedule-list>
      <dl-schedule-detail .handles=${this.handles}></dl-schedule-detail>
      <dl-schedule-form .handles=${this.handles}></dl-schedule-form>
      <dialog id="schedule-delete-dialog" class="confirm-dialog" aria-labelledby="schedule-delete-title">
        <form method="dialog" novalidate>
          <h2 id="schedule-delete-title">${this.deleting
            ? msg(str`Delete ${this.deleting}?`, {id: 'schedules.deleteTitle'})
            : msg('Delete this schedule?', {id: 'schedules.deleteTitleEmpty'})}</h2>
          <p>${msg(
            'This stops its future runs and deletes its history. The conversations its runs wrote stay.',
            {id: 'schedules.deleteBody'},
          )}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'schedules.cancel'})}</button>
            <button type="submit" value="delete" class="dl-dialog-danger">${msg('Delete schedule', {id: 'schedules.deleteConfirm'})}</button>
          </div>
        </form>
      </dialog>`;
  }
}

customElements.define('dl-schedule-dialogs', DlScheduleDialogs);

declare global {
  interface HTMLElementTagNameMap {
    'dl-schedule-dialogs': DlScheduleDialogs;
  }
}
