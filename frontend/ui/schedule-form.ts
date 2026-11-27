// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The dialog that makes or edits a Schedule: a name, what to ask, and when, with everything else defaulted.
 *
 * "When" is one choice of frequency and the few controls it needs, which compile to cron here. A line under it
 * reads the choice back in words, and asks the server what it would be and whether it would be refused (the
 * server holds the cadence to the rules saving it would, so a refusal shows before the reader saves). Editing
 * sends only what changed. What a Schedule is made as stays: where its fires go and whether it starts a goal
 * are chosen when it is made.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {live} from 'lit/directives/live.js';
import type {Schedule, ScheduleChanges, ScheduleMode, ScheduleTarget} from '../api/schedules.ts';
import {previewCadence} from '../api/schedules.ts';
import {ApiError} from '../api/wire.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {LatestRequest} from '../lib/latest-request.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {
  type CadenceDraft,
  cadenceDraftOf,
  cadenceOfDraft,
  describeOnce,
  describeRecurrence,
  type Frequency,
  LAST_PRESET_DAY,
  newCadenceDraft,
  nextRunText,
  parseRecurrence,
  WEEK,
  weekdayShort,
} from '../lib/schedule-cadence.ts';
import {browserZone, instantInZone} from '../lib/zoned-time.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import type {ScheduleOutcome} from '../stores/schedule-store.ts';
import styles from '../styles/schedule-dialogs.module.css';
import {modalResult} from './modal.ts';
import {requestToast} from './toast-request.ts';

/** The rounds a new goal Schedule may run, until its owner says otherwise: the Goal's own default. */
const DEFAULT_GOAL_ROUNDS = '16';

/** The daily limit is read and written in millions of tokens: "2", not "2000000". */
function millionsText(tokens: number): string {
  return String(Math.round(tokens / 100_000) / 10);
}

function tokensOfMillions(text: string): number {
  const millions = Number(text);
  return text.trim() === '' || !Number.isFinite(millions) ? Number.NaN : Math.round(millions * 1_000_000);
}

/** How long the cadence may rest before the server is asked what it reads as. */
const PREVIEW_DELAY_MS = 300;

const FREQUENCIES: readonly Frequency[] = ['once', 'hourly', 'daily', 'weekdays', 'weekly', 'monthly', 'custom'];

function frequencyLabel(frequency: Frequency): string {
  switch (frequency) {
    case 'once': return msg('Once', {id: 'scheduleForm.once'});
    case 'hourly': return msg('Hourly', {id: 'scheduleForm.hourly'});
    case 'daily': return msg('Daily', {id: 'scheduleForm.daily'});
    case 'weekdays': return msg('Weekdays', {id: 'scheduleForm.weekdays'});
    case 'weekly': return msg('Weekly', {id: 'scheduleForm.weekly'});
    case 'monthly': return msg('Monthly', {id: 'scheduleForm.monthly'});
    case 'custom': return msg('Custom', {id: 'scheduleForm.custom'});
  }
}

/** A Schedule as the form holds it: text the reader is typing, and the choices they have made. */
interface Draft {
  /** The Schedule being edited, or null for a new one. */
  id: string | null;
  name: string;
  prompt: string;
  cadence: CadenceDraft;
  zone: string;
  zoneShown: boolean;
  advanced: boolean;
  mode: ScheduleMode;
  workspaces: string[];
  target: ScheduleTarget;
  /** Whether the Schedule starts a goal, which is fixed when it is made. */
  goal: boolean;
  goalRounds: string;
  ceiling: string;
  saving: boolean;
  /** Why the last save did not happen, in words; null while nothing has failed. */
  problem: string | null;
}

type Preview =
  | {kind: 'idle'}
  | {kind: 'loading'}
  | {kind: 'next'; slot: Date; zone: string}
  | {kind: 'refused'; reason: string};

/** What the cadence asks the server and what saving sends: a recurrence, or one instant. */
type Cadence = {zone: string; cron: string} | {zone: string; at: Date};

export class DlScheduleForm extends LightElement {
  static properties = {
    handles: {attribute: false},
    draft: {state: true},
    preview: {state: true},
  };

  declare handles: AppHandles;
  declare draft: Draft | null;
  /** What the form's cadence reads as to the server. */
  declare preview: Preview;

  readonly #previews = new LatestRequest();
  #previewTimer: ReturnType<typeof setTimeout> | undefined;

  constructor() {
    super();
    this.handles = productionHandles();
    this.draft = null;
    this.preview = {kind: 'idle'};
    /** Store reads: the Schedule being edited, and the workspaces a Schedule may name. */
    new StoreController(this, () => [this.handles.schedules, this.handles.workspaces]);
  }

  override disconnectedCallback(): void {
    clearTimeout(this.#previewTimer);
    this.#previews.cancel();
    super.disconnectedCallback();
  }

  /** Open the form for a new Schedule (null) or an edit of one; resolves once it has closed and focus is back. */
  async open(schedule: Schedule | null, trigger: HTMLElement | null): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted || this.draft !== null) return;
    this.draft = schedule === null ? this.#newDraft() : this.#editDraft(schedule);
    this.#readCadence();
    await this.updateComplete;
    const dialog = this.querySelector<HTMLDialogElement>('dialog');
    if (!dialog) return;
    const closed = modalResult(this, dialog, () => { if (trigger?.isConnected) trigger.focus(); }, signal);
    this.querySelector<HTMLElement>('[data-field="name"]')?.focus();
    await closed;
    clearTimeout(this.#previewTimer);
    this.#previews.cancel();
    this.preview = {kind: 'idle'};
    this.draft = null;
  }

  #newDraft(): Draft {
    const {workspaces, schedules} = this.handles;
    return {
      id: null,
      name: '',
      prompt: '',
      cadence: newCadenceDraft(),
      zone: browserZone(),
      zoneShown: false,
      advanced: false,
      mode: 'research',
      workspaces: workspaces.active.length > 0 ? [...workspaces.active] : workspaces.records.slice(0, 1).map((record) => record.workspace),
      target: 'new_conversation',
      goal: false,
      goalRounds: DEFAULT_GOAL_ROUNDS,
      ceiling: millionsText(schedules.defaultCeiling),
      saving: false,
      problem: null,
    };
  }

  #editDraft(schedule: Schedule): Draft {
    return {
      id: schedule.scheduleId,
      name: schedule.name,
      prompt: schedule.prompt,
      cadence: cadenceDraftOf(schedule),
      zone: schedule.zone,
      zoneShown: false,
      advanced: false,
      mode: schedule.mode,
      workspaces: [...schedule.workspaces],
      target: schedule.target,
      goal: schedule.goalMaxRounds !== null,
      goalRounds: String(schedule.goalMaxRounds ?? DEFAULT_GOAL_ROUNDS),
      ceiling: millionsText(schedule.maxTokensPerDay),
      saving: false,
      problem: null,
    };
  }

  #close(returnValue: string): void {
    this.querySelector<HTMLDialogElement>('dialog')?.close(returnValue);
  }

  #set(changes: Partial<Draft>, cadence = false): void {
    if (!this.draft) return;
    this.draft = {...this.draft, ...changes};
    if (cadence) this.#readCadence();
  }

  #setCadence(changes: Partial<CadenceDraft>): void {
    if (this.draft) this.#set({cadence: {...this.draft.cadence, ...changes}}, true);
  }

  /** What the form's cadence asks the server, or null while it holds nothing to ask about. */
  #cadenceOf(draft: Draft): Cadence | null {
    const zone = draft.zone.trim();
    const cadence = cadenceOfDraft(draft.cadence);
    if (zone === '' || cadence === null) return null;
    if ('cron' in cadence) return {zone, cron: cadence.cron};
    // A zone this browser does not know has no instant of its own: the server names the zone it refused.
    const at = instantInZone(cadence.local, zone) ?? new Date(`${cadence.local}:00Z`);
    return Number.isNaN(at.getTime()) ? null : {zone, at};
  }

  /** Ask what the cadence reads as once the reader has paused typing; only the latest question is answered. */
  #readCadence(): void {
    clearTimeout(this.#previewTimer);
    this.#previews.cancel();
    const cadence = this.draft ? this.#cadenceOf(this.draft) : null;
    if (cadence === null) {
      this.preview = {kind: 'idle'};
      return;
    }
    this.preview = {kind: 'loading'};
    this.#previewTimer = setTimeout(() => { void this.#ask(cadence); }, PREVIEW_DELAY_MS);
  }

  async #ask(cadence: Cadence): Promise<void> {
    const signal = this.#previews.start();
    try {
      const [slot] = await previewCadence('cron' in cadence ? cadence : {zone: cadence.zone, at: cadence.at.toISOString()}, signal);
      if (this.#previews.isCurrent(signal)) this.preview = slot ? {kind: 'next', slot, zone: cadence.zone} : {kind: 'idle'};
    } catch (error) {
      if (!this.#previews.isCurrent(signal)) return;
      this.preview = {
        kind: 'refused',
        reason: reasonOf(error, msg('The schedule could not be checked.', {id: 'schedules.previewFailed'})),
      };
    } finally {
      this.#previews.finish(signal);
    }
  }

  /** What saving would change in an existing Schedule, as the names the server takes; null when nothing. */
  #changes(draft: Draft, schedule: Schedule, cadence: Cadence, ceiling: number): ScheduleChanges | null {
    const same = (left: readonly string[], right: readonly string[]): boolean => (
      left.length === right.length && left.every((item) => right.includes(item))
    );
    const changes: {-readonly [Key in keyof ScheduleChanges]: ScheduleChanges[Key]} = {};
    if (draft.name.trim() !== schedule.name) changes.name = draft.name.trim();
    if (draft.prompt.trim() !== schedule.prompt) changes.prompt = draft.prompt.trim();
    if (draft.mode !== schedule.mode) changes.mode = draft.mode;
    if (!same(draft.workspaces, schedule.workspaces)) changes.workspaces = draft.workspaces;
    if (ceiling !== schedule.maxTokensPerDay) changes.maxTokensPerDay = ceiling;
    if (cadence.zone !== schedule.zone) changes.zone = cadence.zone;
    if ('cron' in cadence) {
      if (cadence.cron !== schedule.cron) changes.cron = cadence.cron;
    } else if (cadence.at.getTime() !== schedule.at?.getTime()) {
      changes.at = cadence.at.toISOString();
    }
    return Object.keys(changes).length === 0 ? null : changes;
  }

  /** Why the form cannot be saved yet, in words, and the part of it the reader should look at. */
  #missing(draft: Draft, cadence: Cadence | null, ceiling: number, rounds: number): {problem: string; advanced: boolean} | null {
    if (draft.name.trim() === '') return {problem: msg('Give the schedule a name.', {id: 'schedules.needName'}), advanced: false};
    if (draft.prompt.trim() === '') return {problem: msg('Write what the schedule should ask.', {id: 'schedules.needPrompt'}), advanced: false};
    if (cadence === null) return {problem: msg('Say when it should run.', {id: 'schedules.needCadence'}), advanced: false};
    if (draft.workspaces.length === 0) return {problem: msg('Choose at least one workspace.', {id: 'schedules.needWorkspace'}), advanced: true};
    if (!Number.isInteger(ceiling) || ceiling < 1) {
      return {problem: msg('The daily limit is a number of million tokens above 0.', {id: 'schedules.badCeiling'}), advanced: true};
    }
    if (draft.id === null && draft.goal && (!Number.isInteger(rounds) || rounds < 1)) {
      return {problem: msg('The most rounds is a whole number of 1 or more.', {id: 'schedules.badRounds'}), advanced: true};
    }
    return null;
  }

  #told(outcome: ScheduleOutcome<Schedule>, name: string): void {
    if (outcome.kind === 'ok') {
      requestToast(this, {message: msg(str`Saved ${outcome.value.name}.`, {id: 'schedules.saved'})});
      this.#close('saved');
    } else if (outcome.kind === 'gone') {
      requestToast(this, {tone: 'error', message: msg(str`${name} no longer exists, so it was removed from the list.`, {id: 'schedules.gone'})});
      this.#close('gone');
    } else if (outcome.kind === 'failed') {
      this.#set({saving: false, problem: outcome.reason ?? msg('Could not save the schedule.', {id: 'schedules.saveFailed'})});
    } else {
      this.#set({saving: false});
    }
  }

  #submit = async (event: Event): Promise<void> => {
    event.preventDefault();
    const draft = this.draft;
    const signal = this.lifetime;
    if (!draft || draft.saving || signal.aborted) return;
    const ceiling = tokensOfMillions(draft.ceiling);
    const rounds = Number(draft.goalRounds);
    const cadence = this.#cadenceOf(draft);
    const missing = this.#missing(draft, cadence, ceiling, rounds);
    if (missing !== null || cadence === null) {
      this.#set({problem: missing?.problem ?? null, ...(missing?.advanced ? {advanced: true} : {})});
      return;
    }
    const store = this.handles.schedules;
    const editing = draft.id === null ? null : store.get(draft.id);
    if (draft.id !== null && editing === null) {
      this.#close('gone');
      return;
    }
    const changes = editing === null ? null : this.#changes(draft, editing, cadence, ceiling);
    if (editing !== null && changes === null) {
      this.#close('unchanged');
      return;
    }
    this.#set({saving: true, problem: null});
    const outcome = editing === null || changes === null
      ? await store.create({
        name: draft.name.trim(),
        prompt: draft.prompt.trim(),
        mode: draft.mode,
        workspaces: draft.workspaces,
        target: draft.target,
        maxTokensPerDay: ceiling,
        goalMaxRounds: draft.goal ? rounds : undefined,
        zone: cadence.zone,
        ...('cron' in cadence ? {cron: cadence.cron} : {at: cadence.at.toISOString()}),
      })
      : await store.update(editing.scheduleId, changes);
    if (signal.aborted) return;
    this.#told(outcome, editing?.name ?? draft.name.trim());
  };

  // ── Rendering ──

  /** A control and the words that name it; the label wraps the control, so the words are its name. */
  #field(caption: string, control: TemplateResult, hint?: string): TemplateResult {
    return html`
      <label class=${styles.field}>
        <span class=${styles.caption}>${caption}</span>
        ${control}
        ${hint ? html`<span class="dl-note">${hint}</span>` : nothing}
      </label>`;
  }

  /** The controls the chosen frequency needs, for the line under the choice; nothing when it needs none. */
  #frequencyControls(draft: Draft): TemplateResult | typeof nothing {
    const cadence = draft.cadence;
    const word = (text: string): TemplateResult => html`<span class=${styles.word}>${text}</span>`;
    const time = html`
      <input class="dl-input ${styles.time}" data-field="time" type="time" required .value=${live(cadence.time)}
        aria-label=${msg('Time of day', {id: 'scheduleForm.timeLabel'})}
        @input=${(event: Event) => { this.#setCadence({time: (event.currentTarget as HTMLInputElement).value}); }}>`;
    switch (cadence.frequency) {
      case 'hourly':
        return html`${word(msg('at minute', {id: 'scheduleForm.atMinute'}))}
          <input class="dl-input ${styles.number}" data-field="minute" type="number" min="0" max="59" step="1"
            .value=${live(cadence.minute)} aria-label=${msg('Minute of the hour', {id: 'scheduleForm.minuteLabel'})}
            @input=${(event: Event) => { this.#setCadence({minute: (event.currentTarget as HTMLInputElement).value}); }}>`;
      case 'daily':
      case 'weekdays':
      case 'weekly':
        return html`${word(msg('at', {id: 'scheduleForm.at'}))}${time}`;
      case 'monthly':
        return html`${word(msg('on day', {id: 'scheduleForm.onDay'}))}
          <input class="dl-input ${styles.number}" data-field="day" type="number" min="1" max=${LAST_PRESET_DAY} step="1"
            .value=${live(cadence.dayOfMonth)} aria-label=${msg('Day of the month', {id: 'scheduleForm.dayLabel'})}
            @input=${(event: Event) => { this.#setCadence({dayOfMonth: (event.currentTarget as HTMLInputElement).value}); }}>
          ${word(msg('at', {id: 'scheduleForm.at'}))}${time}`;
      case 'once':
        return html`
          <input class="dl-input ${styles.date}" data-field="date" type="date" required .value=${live(cadence.date)}
            aria-label=${msg('Date', {id: 'scheduleForm.dateLabel'})}
            @click=${openPicker}
            @input=${(event: Event) => { this.#setCadence({date: (event.currentTarget as HTMLInputElement).value}); }}>
          ${word(msg('at', {id: 'scheduleForm.at'}))}${time}`;
      case 'custom':
        return nothing;
    }
  }

  #toggleDay(day: number, box: HTMLInputElement): void {
    const chosen = this.draft?.cadence.days ?? [];
    const days = chosen.includes(day) ? chosen.filter((other) => other !== day) : [...chosen, day];
    // A Schedule that runs on no day is no Schedule: the last day stays chosen.
    if (days.length > 0) this.#setCadence({days});
    else box.checked = true;
  }

  #frequencies(draft: Draft): TemplateResult {
    return html`
      <div class="dl-chips" role="radiogroup" aria-label=${msg('How often', {id: 'scheduleForm.frequencyLabel'})} data-field="frequency">
        ${FREQUENCIES.map((frequency) => html`
          <label class="dl-chip"><input type="radio" name="schedule-frequency" value=${frequency}
            .checked=${live(draft.cadence.frequency === frequency)}
            @change=${() => { this.#setCadence({frequency}); }}>${frequencyLabel(frequency)}</label>`)}
      </div>`;
  }

  #days(draft: Draft): TemplateResult {
    const locale = getLocale();
    return html`
      <div class="dl-chips" role="group" aria-label=${msg('Days of the week', {id: 'scheduleForm.daysLabel'})}>
        ${WEEK.map((day) => html`
          <label class="dl-chip"><input type="checkbox" data-day=${day} .checked=${draft.cadence.days.includes(day)}
            @change=${(event: Event) => { this.#toggleDay(day, event.currentTarget as HTMLInputElement); }}>${weekdayShort(day, locale)}</label>`)}
      </div>`;
  }

  /** The cadence in words, on its zone's clock, and what the server says of it. */
  #summary(draft: Draft): TemplateResult {
    const locale = getLocale();
    const cadence = this.#cadenceOf(draft);
    const words = cadence === null
      ? msg('Choose when it should run.', {id: 'scheduleForm.choose'})
      : 'cron' in cadence
        ? describeRecurrence(parseRecurrence(cadence.cron), locale)
        : describeOnce(cadence.at, cadence.zone, locale);
    const preview = this.preview;
    return html`
      <p class=${styles.summary} data-summary aria-live="polite">
        <span>${words}</span>
        <span aria-hidden="true">·</span>
        <button type="button" class="dl-text-action" data-zone-toggle
          aria-expanded=${String(draft.zoneShown)} aria-controls="schedule-zone"
          aria-label=${msg(str`Time zone ${draft.zone}`, {id: 'scheduleForm.zoneLabel'})}
          @click=${() => { this.#set({zoneShown: !this.draft?.zoneShown}); }}>${draft.zone}</button>
        ${preview.kind === 'loading'
          ? html`<span aria-hidden="true">·</span><span>${msg('Checking…', {id: 'schedules.checking'})}</span>`
          : preview.kind === 'next'
            ? html`<span aria-hidden="true">·</span><span data-next>${
              msg(str`Next: ${nextRunText(preview.slot, new Date(), preview.zone, locale)}`, {id: 'schedules.next'})}</span>`
            : preview.kind === 'refused'
              ? html`<span aria-hidden="true">·</span><span class=${styles.refused} data-refused role="alert">${preview.reason}</span>`
              : nothing}
      </p>
      <div id="schedule-zone">
        ${draft.zoneShown ? html`
          <label class=${styles.field}>
            <span class=${styles.caption}>${msg('Time zone', {id: 'schedules.zone'})}</span>
            <input class="dl-input ${styles.wide}" data-field="zone" list="schedule-zones" autocomplete="off" spellcheck="false"
              .value=${live(draft.zone)}
              @input=${(event: Event) => { this.#set({zone: (event.currentTarget as HTMLInputElement).value}, true); }}>
            <span class="dl-note">${msg('An IANA zone such as Europe/Stockholm. A recurrence follows its clock through daylight saving.', {id: 'schedules.zoneHint'})}</span>
          </label>
          <datalist id="schedule-zones">${Intl.supportedValuesOf('timeZone').map((zone) => html`<option value=${zone}></option>`)}</datalist>` : nothing}
      </div>`;
  }

  #workspaces(draft: Draft): TemplateResult | typeof nothing {
    const known = this.handles.workspaces.records.map((record) => ({id: record.workspace, name: record.displayName}));
    // A workspace the Schedule names that this page no longer lists is still its own, and stays choosable.
    const kept = draft.workspaces.filter((id) => !known.some((record) => record.id === id)).map((id) => ({id, name: id}));
    const all = [...known, ...kept];
    // With one workspace there is nothing to choose, and the Schedule runs in it.
    if (all.length < 2) return nothing;
    return html`
      <fieldset class=${styles.group}>
        <legend class=${styles.caption}>${msg('Workspaces', {id: 'schedules.workspaces'})}</legend>
        ${all.map((workspace) => html`
          <label class="dl-dialog-checkbox ${styles.choice}">
            <input type="checkbox" data-workspace=${workspace.id} .checked=${draft.workspaces.includes(workspace.id)}
              @change=${(event: Event) => {
                const on = (event.currentTarget as HTMLInputElement).checked;
                const chosen = (this.draft?.workspaces ?? []).filter((id) => id !== workspace.id);
                this.#set({workspaces: on ? [...chosen, workspace.id] : chosen});
              }}>
            <span>${workspace.name}</span>
          </label>`)}
      </fieldset>`;
  }

  /** One choice among a few as radio rows, in the form's own rhythm; `data-field` names the group. */
  #choice<Value extends string>(
    name: string,
    caption: string,
    options: ReadonlyArray<readonly [Value, string]>,
    current: Value,
    chosen: (value: Value) => void,
    {disabled = false, hint}: {disabled?: boolean; hint?: string} = {},
  ): TemplateResult {
    return html`
      <fieldset class=${styles.group} data-field=${name} aria-describedby=${hint ? `schedule-${name}-hint` : nothing}>
        <legend class=${styles.caption}>${caption}</legend>
        ${options.map(([value, label]) => html`
          <label class="dl-dialog-checkbox ${styles.choice}">
            <input type="radio" name=${`schedule-${name}`} value=${value} .checked=${live(current === value)} ?disabled=${disabled}
              @change=${() => { chosen(value); }}>
            <span>${label}</span>
          </label>`)}
        ${hint ? html`<span id=${`schedule-${name}-hint`} class="dl-note">${hint}</span>` : nothing}
      </fieldset>`;
  }

  #advanced(draft: Draft): TemplateResult {
    const creating = draft.id === null;
    return html`
      <div class=${styles.advanced}>
        <button type="button" class="dl-text-action dl-text-action--disclosure" data-advanced
          aria-expanded=${String(draft.advanced)} aria-controls="schedule-advanced"
          @click=${() => { this.#set({advanced: !this.draft?.advanced}); }}>${msg('Advanced', {id: 'scheduleForm.advanced'})}${icon('chevron-down', {size: 'xs'})}</button>
        <div id="schedule-advanced" class=${styles.advancedBody} ?hidden=${!draft.advanced}>
          ${draft.advanced ? html`
            <div class=${styles.pair}>
              ${this.#choice('mode', msg('Mode', {id: 'schedules.mode'}), [
                ['research', msg('Research', {id: 'schedules.modeResearch'})],
                ['fast', msg('Fast', {id: 'schedules.modeFast'})],
              ], draft.mode, (mode: ScheduleMode) => { this.#set({mode}); }, {
                disabled: draft.goal,
                hint: draft.goal ? msg('A goal runs in Research.', {id: 'schedules.goalResearch'}) : undefined,
              })}
              ${creating ? this.#choice('target', msg('Each run goes to', {id: 'schedules.target'}), [
                ['new_conversation', msg('A new conversation each time', {id: 'schedules.targetNew'})],
                ['same_conversation', msg('One ongoing conversation', {id: 'schedules.targetSame'})],
              ], draft.target, (target: ScheduleTarget) => { this.#set({target}); }) : nothing}
            </div>
            ${this.#field(msg('Daily limit', {id: 'schedules.ceiling'}), html`
              <span class=${styles.line}>
                <input class="dl-input ${styles.number}" data-field="ceiling" type="number" min="0.1" step="0.1" inputmode="decimal"
                  .value=${live(draft.ceiling)}
                  @input=${(event: Event) => { this.#set({ceiling: (event.currentTarget as HTMLInputElement).value}); }}>
                <span class=${styles.word}>${msg('million tokens', {id: 'schedules.ceilingUnit'})}</span>
              </span>`, msg('The schedule pauses once its runs have spent this much in 24 hours.', {id: 'schedules.ceilingHint'}))}
            ${this.#workspaces(draft)}
            ${creating ? html`
              <label class="dl-dialog-checkbox ${styles.choice}">
                <input type="checkbox" data-field="goal" .checked=${draft.goal}
                  @change=${(event: Event) => {
                    const goal = (event.currentTarget as HTMLInputElement).checked;
                    this.#set(goal ? {goal, mode: 'research'} : {goal});
                  }}>
                <span>${msg('Run as a goal', {id: 'schedules.goal'})}</span>
              </label>
              ${draft.goal ? this.#field(msg('Most rounds', {id: 'schedules.rounds'}), html`
                <input class="dl-input ${styles.number}" data-field="rounds" type="number" min="1" step="1" .value=${live(draft.goalRounds)}
                  @input=${(event: Event) => { this.#set({goalRounds: (event.currentTarget as HTMLInputElement).value}); }}>`) : nothing}` : nothing}` : nothing}
        </div>
      </div>`;
  }

  #form(draft: Draft): TemplateResult {
    const creating = draft.id === null;
    const controls = this.#frequencyControls(draft);
    return html`
      <form class=${styles.frame} novalidate data-form @submit=${this.#submit}>
        <h2 id="schedule-form-title" class=${styles.title}>${
          creating ? msg('New schedule', {id: 'scheduleForm.titleNew'}) : msg('Edit schedule', {id: 'scheduleForm.titleEdit'})}</h2>
        <div class=${styles.body}>
          ${this.#field(msg('Name', {id: 'schedules.name'}), html`
            <input class="dl-input ${styles.wide}" data-field="name" maxlength="120" required autocomplete="off" .value=${live(draft.name)}
              @input=${(event: Event) => { this.#set({name: (event.currentTarget as HTMLInputElement).value}); }}>`)}
          ${this.#field(
            draft.goal ? msg('Goal objective', {id: 'schedules.objective'}) : msg('Prompt', {id: 'schedules.prompt'}),
            html`<textarea class="dl-dialog-input ${styles.prompt}" data-field="prompt" rows="5" required
              placeholder=${msg('What should the agent do each time?', {id: 'scheduleForm.promptPlaceholder'})}
              .value=${live(draft.prompt)}
              @input=${(event: Event) => { this.#set({prompt: (event.currentTarget as HTMLTextAreaElement).value}); }}></textarea>`,
            draft.goal ? msg('Each run starts a goal with this objective, and its rounds continue it.', {id: 'schedules.objectiveHint'}) : undefined,
          )}
          <fieldset class=${styles.when}>
            <legend class=${styles.caption}>${msg('When', {id: 'scheduleForm.when'})}</legend>
            ${this.#frequencies(draft)}
            ${draft.cadence.frequency === 'weekly' ? this.#days(draft) : nothing}
            ${controls === nothing ? nothing : html`<div class=${styles.line}>${controls}</div>`}
            ${draft.cadence.frequency === 'custom' ? html`
              <label class=${styles.field}>
                <input class="dl-input ${styles.wide} ${styles.mono}" data-field="cron" autocomplete="off" spellcheck="false"
                  placeholder="0 8 * * 1" .value=${live(draft.cadence.cron)}
                  aria-label=${msg('Cron expression', {id: 'scheduleForm.cronLabel'})}
                  @input=${(event: Event) => { this.#setCadence({cron: (event.currentTarget as HTMLInputElement).value}); }}>
                <span class="dl-note">${msg('Five fields: minute, hour, day of month, month, day of week.', {id: 'schedules.cronHint'})}</span>
              </label>` : nothing}
            ${this.#summary(draft)}
          </fieldset>
          ${this.#advanced(draft)}
          ${draft.problem === null ? nothing : html`<p class=${styles.problem} role="alert" data-problem>${draft.problem}</p>`}
        </div>
        <div class="dl-dialog-actions ${styles.actions}">
          <button type="button" ?disabled=${draft.saving} @click=${() => { this.#close('cancel'); }}>${msg('Cancel', {id: 'schedules.cancel'})}</button>
          <button type="submit" class="dl-dialog-primary" ?disabled=${draft.saving}>${
            creating ? msg('Create', {id: 'scheduleForm.create'}) : msg('Save', {id: 'scheduleForm.save'})}</button>
        </div>
      </form>`;
  }

  protected override render(): TemplateResult {
    return html`
      <dialog class="confirm-dialog ${styles.dialog}" aria-labelledby="schedule-form-title">
        ${this.draft ? this.#form(this.draft) : nothing}
      </dialog>`;
  }
}

/** The date field draws no picker mark of its own, so pressing it opens the browser's calendar. */
function openPicker(event: Event): void {
  try {
    (event.currentTarget as HTMLInputElement).showPicker();
  } catch {
    // A browser without showPicker, or one that refuses it here, still takes the date typed.
  }
}

/** The server's reason for a refusal when it gave one, else the page's own words. */
function reasonOf(error: unknown, fallback: string): string {
  return error instanceof ApiError && error.detail !== null ? error.detail : fallback;
}

customElements.define('dl-schedule-form', DlScheduleForm);

declare global {
  interface HTMLElementTagNameMap {
    'dl-schedule-form': DlScheduleForm;
  }
}
