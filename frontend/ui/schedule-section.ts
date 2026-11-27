// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The sidebar's Scheduled section: the owner's Schedules, one compact row each, above the conversations.
 *
 * A row says what the Schedule is, when it runs and whether it is paused, in the density of a conversation
 * row; its quick actions (run now, pause or resume, and a menu for edit and delete) show on hover and
 * focus, as a conversation's do. The section only draws and asks: every command is an intent that the
 * sidebar hands to the dialogs, which own what asking needs (a form, a detail, a confirmation) and the
 * words a refusal is told in. Whether the section is open is this browser's own preference; with none
 * stored it is open once there is a Schedule to show.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {Schedule} from '../api/schedules.ts';
import {
  type DlMenu,
  icon,
  menuButtonFocus,
  type MenuDismissDetail,
  type MenuFocus,
} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {raise} from '../lib/dom.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {describeCadence} from '../lib/schedule-cadence.ts';
import {readStored, writeStored} from '../lib/storage.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import styles from '../styles/schedule-section.module.css';

const OPEN_KEY = 'dlightrag.schedule_section_open';

export type ScheduleIntent = 'open' | 'new' | 'edit' | 'run' | 'pause' | 'resume' | 'delete';

/** What a reader asked of a Schedule, and the control they asked with, which focus returns to. */
export interface ScheduleIntentDetail {
  intent: ScheduleIntent;
  /** Null for `new`. */
  scheduleId: string | null;
  trigger: HTMLElement | null;
}

function menuId(scheduleId: string): string {
  return `schedule-actions-${scheduleId}`;
}

export class DlScheduleSection extends LightElement {
  static properties = {
    handles: {attribute: false},
    enabled: {attribute: false},
    stored: {state: true},
    openMenuId: {state: true},
  };

  declare handles: AppHandles;
  /** The page is ready for the owner's data: the list is read once it is. */
  declare enabled: boolean;
  /** The open state this browser kept, or null while it has kept none. */
  declare stored: boolean | null;
  declare openMenuId: string | null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.enabled = false;
    const kept = readStored(OPEN_KEY);
    this.stored = kept === null ? null : kept === 'true';
    this.openMenuId = null;
    /** Store reads: schedules, listState, and which Schedule has a command in flight. */
    new StoreController(this, () => [this.handles.schedules]);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('click', (event) => {
      if (!this.openMenuId || !(event.target instanceof Node)) return;
      if (this.#row(this.openMenuId)?.contains(event.target)) return;
      this.openMenuId = null;
    }, {signal: this.lifetime});
  }

  protected override willUpdate(): void {
    if (this.enabled && this.handles.schedules.listState === 'idle') void this.handles.schedules.load();
  }

  get menuOpen(): boolean {
    return this.openMenuId !== null;
  }

  /** Close the open menu; Escape also returns focus to the button that opened it. */
  closeMenu(restoreFocus = false): void {
    const scheduleId = this.openMenuId;
    if (scheduleId === null) return;
    this.openMenuId = null;
    if (restoreFocus) void this.updateComplete.then(() => { this.#row(scheduleId)?.querySelector<HTMLElement>('[data-menu-button]')?.focus(); });
  }

  /** Put focus on the section's own header, where it goes when the control it came from is gone. */
  focusHeader(): void {
    this.querySelector<HTMLElement>('[data-toggle]')?.focus();
  }

  get #open(): boolean {
    return this.stored ?? this.handles.schedules.schedules.length > 0;
  }

  #setOpen(open: boolean): void {
    this.stored = open;
    writeStored(OPEN_KEY, open ? 'true' : 'false');
  }

  #row(scheduleId: string): HTMLElement | null {
    return this.querySelector<HTMLElement>(`[data-schedule-id="${CSS.escape(scheduleId)}"]`);
  }

  #ask(intent: ScheduleIntent, scheduleId: string | null, trigger: EventTarget | null): void {
    this.openMenuId = null;
    raise(this, 'dl-schedule-intent', {intent, scheduleId, trigger: trigger instanceof HTMLElement ? trigger : null});
  }

  #openMenu(scheduleId: string, focus: MenuFocus = 'first'): void {
    this.openMenuId = scheduleId;
    void this.updateComplete.then(() => {
      this.#row(scheduleId)?.querySelector<DlMenu>('dl-menu')?.focusItem(focus);
    });
  }

  #status(schedule: Schedule): string {
    switch (schedule.state) {
      case 'paused': return msg('Paused', {id: 'scheduleSection.paused'});
      case 'completed': return msg('Done', {id: 'scheduleSection.done'});
      // The row names when it runs; the detail says when it runs next.
      case 'active': return '';
    }
  }

  #menu(schedule: Schedule): TemplateResult {
    const id = schedule.scheduleId;
    return html`
      <dl-menu
        id=${menuId(id)}
        class="conversation-actions-menu dl-anchored dl-anchored--end"
        aria-label=${msg(str`Actions for ${schedule.name}`, {id: 'scheduleSection.actionsFor'})}
        @dl-menu-dismiss=${(event: CustomEvent<MenuDismissDetail>) => { this.closeMenu(event.detail.restoreFocus); }}
      >
        ${schedule.state === 'completed' ? nothing : html`
          <button type="button" role="menuitem" tabindex="-1" data-menu="edit"
            @click=${(event: Event) => { this.#ask('edit', id, this.#row(id)?.querySelector('[data-menu-button]') ?? event.currentTarget); }}>${
              msg('Edit', {id: 'scheduleSection.edit'})}</button>`}
        <button type="button" role="menuitem" tabindex="-1" data-menu="delete" class="conversation-delete-action"
          @click=${(event: Event) => { this.#ask('delete', id, this.#row(id)?.querySelector('[data-menu-button]') ?? event.currentTarget); }}>${
            msg('Delete', {id: 'scheduleSection.delete'})}</button>
      </dl-menu>`;
  }

  #quick(schedule: Schedule, pending: boolean): TemplateResult | typeof nothing {
    if (schedule.state === 'completed') return nothing;
    const paused = schedule.state === 'paused';
    const id = schedule.scheduleId;
    const toggle = paused ? 'resume' : 'pause';
    const toggleLabel = paused
      ? msg(str`Resume ${schedule.name}`, {id: 'scheduleSection.resumeLabel'})
      : msg(str`Pause ${schedule.name}`, {id: 'scheduleSection.pauseLabel'});
    return html`
      <button type="button" class="dl-icon-button" data-action="run" ?disabled=${pending}
        aria-label=${msg(str`Run ${schedule.name} now`, {id: 'scheduleSection.runLabel'})}
        title=${msg('Run now', {id: 'scheduleSection.run'})}
        @click=${(event: Event) => { this.#ask('run', id, event.currentTarget); }}>${icon('zap', {size: 'sm'})}</button>
      <button type="button" class="dl-icon-button" data-action=${toggle} ?disabled=${pending}
        aria-label=${toggleLabel}
        title=${paused ? msg('Resume', {id: 'scheduleSection.resume'}) : msg('Pause', {id: 'scheduleSection.pause'})}
        @click=${(event: Event) => { this.#ask(toggle, id, event.currentTarget); }}>${icon(paused ? 'play' : 'pause', {size: 'sm'})}</button>`;
  }

  #renderRow(schedule: Schedule): TemplateResult {
    const id = schedule.scheduleId;
    const pending = this.handles.schedules.isPending(id);
    const expanded = this.openMenuId === id;
    const status = this.#status(schedule);
    const cadence = describeCadence(schedule, getLocale());
    const meta = status === '' ? cadence : `${cadence} · ${status}`;
    return html`
      <div class="conversation-row ${styles.row}" role="listitem" data-schedule-id=${id} data-state=${schedule.state}>
        <button type="button" class="conversation-select ${styles.select}" title=${`${schedule.name}\n${meta}`}
          @click=${(event: Event) => { this.#ask('open', id, event.currentTarget); }}>
          <span class=${styles.name}>${schedule.name}</span>
          <span class=${styles.meta}>${schedule.state === 'paused' ? icon('pause', {size: 'xs'}) : nothing}<span>${meta}</span></span>
        </button>
        <span class=${styles.actions}>
          ${this.#quick(schedule, pending)}
          <button type="button" class="dl-icon-button conversation-actions-button" data-menu-button ?disabled=${pending}
            aria-label=${msg(str`Actions for ${schedule.name}`, {id: 'scheduleSection.actionsFor'})}
            aria-haspopup="menu"
            aria-controls=${expanded ? menuId(id) : nothing}
            aria-expanded=${expanded ? 'true' : 'false'}
            @click=${(event: MouseEvent) => {
              event.stopPropagation();
              if (expanded) this.closeMenu();
              else this.#openMenu(id);
            }}
            @keydown=${(event: KeyboardEvent) => {
              const focus = menuButtonFocus(event);
              if (!focus) return;
              event.preventDefault();
              this.#openMenu(id, focus);
            }}>${icon('more', {size: 'sm'})}</button>
        </span>
        ${expanded ? this.#menu(schedule) : nothing}
      </div>`;
  }

  #body(): TemplateResult | typeof nothing {
    const store = this.handles.schedules;
    if (store.listState === 'error') {
      return html`
        <div class="conversation-list-status" role="status">
          <span>${msg('Could not load schedules.', {id: 'scheduleSection.loadFailed'})}</span>
          <button type="button" @click=${() => { void store.load(); }}>${msg('Retry', {id: 'scheduleSection.retry'})}</button>
        </div>`;
    }
    if (store.schedules.length === 0) return nothing;
    return html`
      <div class=${styles.list} role="list" aria-label=${msg('Scheduled', {id: 'scheduleSection.title'})}>
        ${repeat(store.schedules, (schedule) => schedule.scheduleId, (schedule) => this.#renderRow(schedule))}
      </div>`;
  }

  protected override render(): TemplateResult {
    const store = this.handles.schedules;
    const ready = store.listState === 'ready';
    const open = this.#open;
    return html`
      <div class=${styles.head}>
        <button type="button" class="dl-nav-item ${styles.toggle}" data-toggle aria-expanded=${String(open)} aria-controls="scheduled-rows"
          @click=${(event: Event) => {
            // With nothing to show, the row's one use is to make the first Schedule.
            if (ready && store.schedules.length === 0) this.#ask('new', null, event.currentTarget);
            else this.#setOpen(!open);
          }}>
          <span class="dl-nav-item-icon">${icon('clock', {size: 'sm'})}</span>
          <span class="dl-nav-item-text">
            <span class="dl-nav-item-label">${msg('Scheduled', {id: 'scheduleSection.title'})}</span>
          </span>
          ${ready && store.schedules.length > 0 ? html`<span class="dl-nav-item-status" data-count>${store.schedules.length}</span>` : nothing}
        </button>
        <button type="button" class="dl-icon-button ${styles.add}" data-new ?disabled=${!ready}
          aria-label=${msg('New schedule', {id: 'scheduleSection.new'})}
          title=${msg('New schedule', {id: 'scheduleSection.new'})}
          @click=${(event: Event) => {
            this.#setOpen(true);
            this.#ask('new', null, event.currentTarget);
          }}>${icon('add', {size: 'sm'})}</button>
      </div>
      <div id="scheduled-rows" class=${styles.rows} ?hidden=${!open}>${open ? this.#body() : nothing}</div>`;
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
