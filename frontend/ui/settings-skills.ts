// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Settings → Skills: the Agent Skills of the owner's own, to switch off, read, or delete.
 *
 * Only the owner's own tier is here: a built-in or operator-global Skill belongs to the deployment,
 * so it never lists and cannot be turned off. A Skill that is off stays in the list, marked, so the
 * owner can switch it back on. Each command answers the one Skill it changed, so a reply replaces
 * its row and nothing else, and a row that is gone stays gone however late a reply for it arrives.
 * A Skill's SKILL.md is what the agent installed, so the owner can read it here, as plain text and
 * never as markup.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  deleteOwnerSkill,
  getOwnerSkillDocument,
  listOwnerSkills,
  type OwnerSkill,
  setOwnerSkillEnabled,
} from '../api/skills.ts';
import {ApiError} from '../api/wire.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import shared from '../styles/settings-page.module.css';
import styles from '../styles/settings-skills.module.css';
import {modalResult} from './modal.ts';
import {reportSettingsSummary} from './settings-summary.ts';
import {requestToast} from './toast-request.ts';

/** A Skill's SKILL.md as far as the page has read it. */
type SkillDocument = {state: 'loading'} | {state: 'loaded'; text: string} | {state: 'failed'};

/** The server no longer has the Skill: it was deleted from another tab or by the agent. */
function isGone(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

export class DlSettingsSkills extends LightElement {
  static properties = {
    skills: {state: true},
    limit: {state: true},
    error: {state: true},
    deleting: {state: true},
  };

  /** The owner's Skills by name, null until the first read answers. */
  declare skills: readonly OwnerSkill[] | null;
  /** How many Skills the owner may keep. */
  declare limit: number;
  /** The first read failed. Once one has succeeded the page never reads again: every command answers its own row. */
  declare error: boolean;
  /** The Skill the delete dialog is asking about, so its copy can name it. */
  declare deleting: string | null;

  /** The Skills with a command in flight; each is answered by its own request, so rows never wait for each other. */
  readonly #pending = new Set<string>();
  /** The Skills whose SKILL.md is showing. */
  readonly #open = new Set<string>();
  /** What was read of each SKILL.md; a text that loaded stays for the page's life. */
  readonly #documents = new Map<string, SkillDocument>();

  constructor() {
    super();
    this.skills = null;
    this.limit = 0;
    this.error = false;
    this.deleting = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    void this.#load();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('skills') && this.skills) {
      reportSettingsSummary(this, {
        section: 'skills',
        enabled: this.skills.filter((skill) => skill.enabled).length,
        total: this.skills.length,
      });
    }
  }

  async #load(): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted) return;
    this.error = false;
    try {
      const {skills, limit} = await listOwnerSkills(signal);
      if (signal.aborted) return;
      this.skills = skills;
      this.limit = limit;
    } catch {
      if (!signal.aborted) this.error = true;
    }
  }

  #indexOf(name: string): number {
    return this.skills?.findIndex((skill) => skill.name === name) ?? -1;
  }

  /** A command's reply is the Skill as it now stands; a Skill that is no longer listed stays unlisted. */
  #replace(row: OwnerSkill): void {
    this.skills = this.skills?.map((skill) => (skill.name === row.name ? row : skill)) ?? null;
  }

  #drop(name: string): void {
    this.skills = this.skills?.filter((skill) => skill.name !== name) ?? null;
    this.#open.delete(name);
    this.#documents.delete(name);
  }

  #gone(name: string): void {
    this.#drop(name);
    requestToast(this, {
      message: msg(str`${name} no longer exists, so it was removed from the list.`, {id: 'settings.skills.gone'}),
    });
  }

  /** Give focus back after a command: to the control the reader was on, or, when its Skill is gone,
   * to the View of the one that took its place, else to the note that nothing is left. */
  async #restoreFocus(trigger: HTMLElement, index: number): Promise<void> {
    await this.updateComplete;
    if (trigger.isConnected) {
      trigger.focus();
      return;
    }
    const views = this.querySelectorAll<HTMLElement>('[data-view]');
    (views[Math.min(index, views.length - 1)] ?? this.querySelector<HTMLElement>('#skills-empty-title'))?.focus();
  }

  async #setEnabled(skill: OwnerSkill, toggle: HTMLElement): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted || this.#pending.has(skill.name)) return;
    const focused = document.activeElement === toggle;
    const index = this.#indexOf(skill.name);
    this.#pending.add(skill.name);
    this.requestUpdate();
    try {
      const row = await setOwnerSkillEnabled(skill.name, !skill.enabled, signal);
      if (!signal.aborted) this.#replace(row);
    } catch (error) {
      if (signal.aborted) return;
      if (isGone(error)) this.#gone(skill.name);
      else requestToast(this, {message: msg(str`Could not update ${skill.name}.`, {id: 'settings.skills.updateFailed'})});
    } finally {
      if (!signal.aborted) {
        this.#pending.delete(skill.name);
        this.requestUpdate();
        // A switch that was disabled for the request drops focus in some engines; give it back.
        if (focused) await this.#restoreFocus(toggle, index);
      }
    }
  }

  async #delete(skill: OwnerSkill, trigger: HTMLElement): Promise<void> {
    const signal = this.lifetime;
    const dialog = this.querySelector<HTMLDialogElement>('#skills-delete');
    if (signal.aborted || !dialog || this.#pending.has(skill.name)) return;
    this.deleting = skill.name;
    await this.updateComplete;
    const outcome = await modalResult(this, dialog, () => trigger.focus(), signal);
    this.deleting = null;
    if (outcome !== 'delete' || signal.aborted) return;
    const index = this.#indexOf(skill.name);
    this.#pending.add(skill.name);
    this.requestUpdate();
    try {
      await deleteOwnerSkill(skill.name, signal);
      if (signal.aborted) return;
      this.#drop(skill.name);
      requestToast(this, {message: msg(str`Deleted ${skill.name}.`, {id: 'settings.skills.deleted'})});
    } catch (error) {
      if (signal.aborted) return;
      if (isGone(error)) this.#gone(skill.name);
      else requestToast(this, {message: msg(str`Could not delete ${skill.name}.`, {id: 'settings.skills.deleteFailed'})});
    } finally {
      if (!signal.aborted) {
        this.#pending.delete(skill.name);
        this.requestUpdate();
        await this.#restoreFocus(trigger, index);
      }
    }
  }

  /** Show or hide a Skill's SKILL.md. It is read the first time it shows, and a text that loaded is not read again. */
  #toggleView(name: string): void {
    if (this.#open.delete(name)) {
      this.requestUpdate();
      return;
    }
    this.#open.add(name);
    const state = this.#documents.get(name)?.state;
    if (state === 'loaded' || state === 'loading') this.requestUpdate();
    else void this.#fetchDocument(name, null);
  }

  /** Read one SKILL.md. `retry` is the button that asked again, which the reply replaces, so focus follows to what does. */
  async #fetchDocument(name: string, retry: HTMLElement | null): Promise<void> {
    const signal = this.lifetime;
    if (signal.aborted) return;
    const focused = retry !== null && document.activeElement === retry;
    this.#documents.set(name, {state: 'loading'});
    this.requestUpdate();
    try {
      const text = await getOwnerSkillDocument(name, signal);
      if (signal.aborted) return;
      this.#documents.set(name, {state: 'loaded', text});
    } catch (error) {
      if (signal.aborted) return;
      if (isGone(error)) {
        const row = this.querySelector<HTMLElement>(`[data-skill="${CSS.escape(name)}"]`);
        const index = this.#indexOf(name);
        const inRow = row?.contains(document.activeElement) ?? false;
        this.#gone(name);
        if (row && inRow) await this.#restoreFocus(row, index);
        return;
      }
      this.#documents.set(name, {state: 'failed'});
    }
    this.requestUpdate();
    if (focused) {
      await this.updateComplete;
      this.querySelector<HTMLElement>(`[data-document="${CSS.escape(name)}"]`)?.focus();
    }
  }

  #text(name: string): TemplateResult {
    const read = this.#documents.get(name);
    if (read?.state === 'loaded') {
      // Text, never markup: the agent wrote this file, and the owner is here to judge it.
      return html`<pre class=${styles.document} role="region" tabindex="0" data-document=${name}
        aria-label=${msg(str`SKILL.md of ${name}`, {id: 'settings.skills.documentLabel'})}>${read.text}</pre>`;
    }
    if (read?.state === 'failed') {
      return html`
        <div class=${shared.failed}>
          <p class=${shared.note} role="alert">${msg('Could not load SKILL.md.', {id: 'settings.skills.documentFailed'})}</p>
          <button type="button" class="dl-btn" data-document=${name}
            @click=${(event: Event) => { void this.#fetchDocument(name, event.currentTarget as HTMLElement); }}>
            ${msg('Retry', {id: 'settings.skills.retry'})}
          </button>
        </div>`;
    }
    return html`<p class=${shared.hint} role="status">${msg('Loading SKILL.md…', {id: 'settings.skills.documentLoading'})}</p>`;
  }

  #row(skill: OwnerSkill): TemplateResult {
    const open = this.#open.has(skill.name);
    const pending = this.#pending.has(skill.name);
    const nameId = `skill-name-${skill.name}`;
    return html`
      <li class=${styles.skill} data-skill=${skill.name}>
        <div class=${styles.main}>
          <div class=${styles.text}>
            <span class=${styles.heading}>
              <span id=${nameId} class="${shared.rowLabel} ${styles.name} ${skill.enabled ? '' : shared.rowLabelMuted}">${skill.name}</span>
              ${skill.enabled ? nothing : html`<span class=${styles.state}>${msg('Disabled', {id: 'settings.skills.disabled'})}</span>`}
            </span>
            <p class="${shared.rowCaption} ${styles.description} ${open ? styles.whole : ''}">${skill.description}</p>
          </div>
          <button class="dl-switch dl-switch--dense" type="button" role="switch" data-switch=${skill.name}
            aria-checked=${String(skill.enabled)} aria-labelledby=${nameId} ?disabled=${pending}
            @click=${(event: Event) => { void this.#setEnabled(skill, event.currentTarget as HTMLElement); }}></button>
        </div>
        <div class=${styles.actions}>
          <button type="button" class="dl-btn ${styles.view}" data-view=${skill.name} aria-expanded=${String(open)}
            aria-label=${msg(str`View ${skill.name}`, {id: 'settings.skills.viewLabel'})}
            @click=${() => { this.#toggleView(skill.name); }}>
            ${msg('View', {id: 'settings.skills.view'})}
            <span class=${styles.chevron}>${icon('disclosure', {size: 'sm'})}</span>
          </button>
          <button type="button" class="dl-btn dl-btn-danger-text" data-delete=${skill.name} ?disabled=${pending}
            aria-label=${msg(str`Delete ${skill.name}`, {id: 'settings.skills.deleteLabel'})}
            @click=${(event: Event) => { void this.#delete(skill, event.currentTarget as HTMLElement); }}>
            ${msg('Delete…', {id: 'settings.skills.delete'})}
          </button>
        </div>
        ${open ? this.#text(skill.name) : nothing}
      </li>`;
  }

  #empty(): TemplateResult {
    return html`
      <div class="${shared.card} ${shared.empty}">
        <span class=${shared.emptyIcon}>${icon('skills', {size: 'md'})}</span>
        <h4 id="skills-empty-title" class=${shared.emptyTitle} tabindex="-1">${
          msg('No skills of your own yet', {id: 'settings.skills.emptyTitle'})}</h4>
        <span class=${shared.emptyBody}>${msg(
          'Ask the agent to create one and it will appear here.',
          {id: 'settings.skills.emptyBody'},
        )}</span>
      </div>`;
  }

  #page(skills: readonly OwnerSkill[]): TemplateResult {
    return html`
      <div class=${shared.stack}>
        <p class=${shared.hint}>${msg(str`${skills.length} of ${this.limit} skills`, {id: 'settings.skills.quota'})}</p>
        ${skills.length === 0 ? this.#empty() : html`
          <ul class="${shared.card} ${shared.list} ${shared.divided}">
            ${repeat(skills, (skill) => skill.name, (skill) => this.#row(skill))}
          </ul>`}
      </div>`;
  }

  protected override render(): TemplateResult {
    return html`
      ${this.skills ? this.#page(this.skills) : this.error ? html`
        <div class=${shared.failed}>
          <p class=${shared.note} role="alert">${msg('Could not load skills.', {id: 'settings.skills.loadFailed'})}</p>
          <button type="button" class="dl-btn" @click=${() => { void this.#load(); }}>${
            msg('Retry', {id: 'settings.skills.retry'})}</button>
        </div>` : html`<p class=${shared.hint} role="status">${
          msg('Loading skills…', {id: 'settings.skills.loading'})}</p>`}
      <dialog id="skills-delete" class="confirm-dialog" aria-labelledby="skills-delete-title">
        <form method="dialog" novalidate>
          <h2 id="skills-delete-title">${this.deleting
            ? msg(str`Delete ${this.deleting}?`, {id: 'settings.skills.deleteTitle'})
            : msg('Delete this skill?', {id: 'settings.skills.deleteTitleEmpty'})}</h2>
          <p>${msg(
            'This removes the skill for good, and it cannot be undone. To keep it without the agent using it, turn it off instead.',
            {id: 'settings.skills.deleteBody'},
          )}</p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'settings.skills.cancel'})}</button>
            <button type="submit" value="delete" class="dl-dialog-danger">${
              msg('Delete skill', {id: 'settings.skills.deleteConfirm'})}</button>
          </div>
        </form>
      </dialog>`;
  }
}

customElements.define('dl-settings-skills', DlSettingsSkills);

declare global {
  interface HTMLElementTagNameMap {
    'dl-settings-skills': DlSettingsSkills;
  }
}
