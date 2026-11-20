// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Inspector's one-line agent switcher: which agent the read-only page shows, and the way to any
 *  other agent of the Run.
 *
 *  Closed, it is one line — previous, the current agent's name, state, elapsed time, its objective,
 *  next, and the shortcut hint. Opening it lists the roster the dock already reads: the Run's main
 *  agent first, then its children indented, each with its state and the question it is waiting on.
 *  Picking one is a switch, never a command: the Inspector only reads, and the conversation is where
 *  a reader acts. */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import type {AgentChildStatus} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {type AgentId, agentKey, agentOfRow, RUN_AGENT, sameAgent} from '../lib/agent-id.ts';
import {agentName} from '../lib/agent-name.ts';
import {ClockController} from '../lib/clock.ts';
import {LightElement} from '../lib/lit-host.ts';
import styles from '../styles/agent-switcher.module.css';
import {agentElapsed, agentGlyph, agentIsCounting, agentStateText} from './agent-status.ts';

export interface AgentSwitchDetail {
  agent: AgentId;
}

/** The roster paging the switcher shows inside its list: the dock owns the read, the switcher asks. */
export interface AgentSwitcherPaging {
  readonly hasOlder: boolean;
  readonly loading: boolean;
  readonly failed: boolean;
}

export class DlAgentSwitcher extends LightElement {
  static properties = {
    main: {attribute: false},
    entries: {attribute: false},
    selected: {attribute: false},
    hits: {attribute: false},
    paging: {attribute: false},
    warm: {attribute: false},
    active: {attribute: false},
  };

  /** The Run's own agent as a roster row: its question, state, and clock. */
  declare main: (AgentChildStatus & {childSessionId: string}) | null;
  /** The Run's children as the roster pages state them, newest first. */
  declare entries: readonly (AgentChildStatus & {childSessionId: string})[] | null;
  /** The agent the page below is showing. */
  declare selected: AgentId;
  /** Match counts by agent key while a search narrows the Run; null when nothing is searched. */
  declare hits: ReadonlyMap<string, number> | null;
  /** Whether the roster has another page of children, and the state of reading it. */
  declare paging: AgentSwitcherPaging | null;
  /** Read an agent ahead of the click, so the page does not wait: the dock keeps its own page cache. */
  declare warm: ((agent: AgentId) => void) | null;
  /** Whether the dock showing this switcher is the panel on show. Its shortcuts only answer then, so a
   *  closed Inspector's switcher cannot take keys from the conversation. */
  declare active: boolean;

  #open = false;
  /** The words the reader typed into the list's filter; empty shows the whole roster. */
  #filter = '';
  /** The row a keyboard reader would pick while the list is open. */
  #active = 0;
  /** The clock every elapsed time is drawn against: the switcher's own, so the dock's redraws do not
   *  decide whether a running time moves. */
  #now = Date.now();

  constructor() {
    super();
    // The host holds the controller for as long as the element lives, so it needs no field here.
    new ClockController(this, () => this.#moving());
    this.main = null;
    this.entries = null;
    this.selected = RUN_AGENT;
    this.hits = null;
    this.paging = null;
    this.warm = null;
    this.active = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('keydown', this.#keydown, {signal: this.lifetime});
  }

  override disconnectedCallback(): void {
    this.#open = false;
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    // One answer per frame for every time this render draws, the clock's tick and the dock's redraws alike.
    this.#now = Date.now();
    if (changed.has('entries') || changed.has('hits')) {
      if (this.#active >= this.#offered().length) this.#active = 0;
    }
  }

  /** Whether any row this switcher can draw still counts time; only then is a clock needed. */
  #moving(): boolean {
    return (this.main !== null && agentIsCounting(this.main))
      || (this.entries ?? []).some(agentIsCounting);
  }

  /** The agents the list offers: the Run's own first, then its children, or only the ones a search
   *  found something in while words narrow the Run. */
  #offered(): AgentId[] {
    const children = (this.entries ?? []).map(agentOfRow);
    const all = [RUN_AGENT, ...children];
    const matching = this.hits === null
      ? all
      : all.filter((agent) => (this.hits?.get(agentKey(agent)) ?? 0) > 0);
    const words = this.#filter.trim().toLocaleLowerCase();
    if (words === '') return matching;
    return matching.filter((agent) => this.#searchText(agent).includes(words));
  }

  /** The words a filter matches an agent on: its name, its objective, and what it handed back. */
  #searchText(agent: AgentId): string {
    const row = this.#rowOf(agent);
    return [
      sameAgent(agent, RUN_AGENT) ? agentName(agent) : agentKey(agent),
      row?.objective ?? '',
      row?.summary ?? '',
    ].join(' ').toLocaleLowerCase();
  }

  /** The roster row one agent's id names, the main agent's row included. */
  #rowOf(agent: AgentId): (AgentChildStatus & {childSessionId: string}) | null {
    if (sameAgent(agent, RUN_AGENT)) return this.main;
    return (this.entries ?? []).find((child) => sameAgent(agentOfRow(child), agent)) ?? null;
  }

  #switchTo(agent: AgentId): void {
    this.#open = false;
    this.requestUpdate();
    if (!sameAgent(agent, this.selected)) {
      this.dispatchEvent(new CustomEvent<AgentSwitchDetail>('dl-agent-switch', {
        detail: {agent}, bubbles: true, composed: true,
      }));
    }
  }

  /** Move to the next or previous agent in the roster's own order, wrapping at the ends. */
  #step(delta: number): void {
    const agents = this.#offered();
    if (agents.length === 0) return;
    const at = agents.findIndex((agent) => sameAgent(agent, this.selected));
    const next = at < 0
      ? (delta > 0 ? 0 : agents.length - 1)
      : (at + delta + agents.length) % agents.length;
    const agent = agents[next];
    if (agent !== undefined) this.#switchTo(agent);
  }

  #filtered = (event: Event): void => {
    this.#filter = (event.currentTarget as HTMLInputElement).value;
    this.#active = 0;
    this.requestUpdate();
  };

  #toggle = (): void => {
    this.#open = !this.#open;
    if (this.#open) {
      this.#filter = '';
      const at = this.#offered().findIndex((agent) => sameAgent(agent, this.selected));
      this.#active = Math.max(at, 0);
    }
    this.requestUpdate();
  };

  #close = (): void => {
    if (!this.#open) return;
    this.#open = false;
    this.requestUpdate();
  };

  /** Keys the Inspector owns while a switcher is on it: the agent shortcuts and the list's own keys.
   *  A reader typing in any box keeps their keystrokes, and keys chosen outside the Inspector are not
   *  this component's to answer. */
  #keydown = (event: KeyboardEvent): void => {
    if (!this.active) return;
    if (event.defaultPrevented || event.isComposing) return;
    const target = event.composedPath()[0];
    if (target instanceof HTMLElement && typingIn(target)) return;

    if (event.altKey && (event.code === 'BracketLeft' || event.key === '[')) {
      event.preventDefault();
      this.#step(-1);
      return;
    }
    if (event.altKey && (event.code === 'BracketRight' || event.key === ']')) {
      event.preventDefault();
      this.#step(1);
      return;
    }
    if (event.metaKey && (event.code === 'Digit1' || event.key === '1')) {
      event.preventDefault();
      this.#switchTo(RUN_AGENT);
      return;
    }
    if (!this.#open) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      this.#close();
      return;
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const agents = this.#offered();
      if (agents.length === 0) return;
      this.#active = (this.#active + (event.key === 'ArrowDown' ? 1 : -1) + agents.length) % agents.length;
      this.requestUpdate();
      return;
    }
    if (event.key === 'Enter') {
      const agent = this.#offered()[this.#active];
      if (agent === undefined) return;
      // A focused control answers its own Enter through native activation — a roster row, Load older,
      // or the prev/next buttons. Only the closed line (which opens the list) and a keystroke on the
      // page itself pick the highlighted row here.
      const control = target instanceof HTMLElement
        ? target.closest('button, a[href], input, select, textarea, [contenteditable="true"]')
        : null;
      if (control !== null && !control.hasAttribute('data-switcher-current')) return;
      event.preventDefault();
      this.#switchTo(agent);
    }
  };

  #loadOlder = (): void => {
    this.dispatchEvent(new CustomEvent('dl-agent-load-older', {bubbles: true, composed: true}));
  };

  /** One agent's row in the list: its glyph, what it is, where it stands, and what it waits on. */
  #row(agent: AgentId): TemplateResult {
    const row = this.#rowOf(agent);
    const main = sameAgent(agent, RUN_AGENT);
    const id = agentKey(agent);
    const found = this.hits?.get(id) ?? null;
    const meta = [
      row === null ? '' : agentStateText(row),
      row === null ? '' : agentElapsed(row, this.#now),
      found === null
        ? ''
        : found === 1
          ? msg('1 match', {id: 'tracesPanel.rowOneMatch'})
          : msg(str`${found} matches`, {id: 'tracesPanel.rowMatches'}),
    ].filter(Boolean).join(' · ');
    return html`
      <li role="none">
        <button type="button" class=${styles.switcherRow} role="option"
                data-agent-session=${id} data-main=${main ? 'true' : nothing}
                aria-selected=${sameAgent(agent, this.selected) ? 'true' : 'false'}
                aria-current=${sameAgent(agent, this.selected) ? 'true' : nothing}
                @mouseenter=${() => this.warm?.(agent)}
                @focus=${() => this.warm?.(agent)}
                @click=${() => this.#switchTo(agent)}>
          ${row === null ? nothing : agentGlyph(row)}
          <span class=${styles.switcherText}>
            <span class=${styles.switcherName}>${main
              ? agentName(agent)
              : row?.objective || agentName(agent)}</span>
            <span class=${styles.switcherMeta}>
              <span>${meta}</span>
              ${(row?.pendingQuestions ?? 0) > 0
                ? html`<span class=${styles.switcherPill}>
                    ${(row?.pendingQuestions ?? 0) === 1
                      ? msg('Asks parent', {id: 'tracesPanel.asksParent'})
                      : msg(str`Asks parent · ${row?.pendingQuestions ?? 0}`, {id: 'tracesPanel.asksParentMany'})}
                  </span>`
                : nothing}
            </span>
            ${main && row?.objective
              ? html`<span class=${styles.switcherSummary}>${row.objective}</span>`
              : nothing}
            ${!main && row?.summary && row.status !== 'running' && row.status !== 'cancelled'
              ? html`<span class=${styles.switcherSummary}>${row.summary}</span>`
              : nothing}
          </span>
        </button>
      </li>
    `;
  }

  protected override render(): TemplateResult {
    const rows = this.#offered();
    return html`
      <div class=${styles.root} data-agent-switcher>
        <button type="button" class=${styles.current} data-switcher-current
                aria-haspopup="listbox" aria-expanded=${this.#open ? 'true' : 'false'}
                @click=${this.#toggle}>
          ${this.#currentLine()}
        </button>
        <div class="dl-popover dl-anchored ${styles.list}" data-switcher-popover role="listbox"
             aria-label=${msg('Agents of this Run', {id: 'tracesPanel.switcherListAria'})}
             ?hidden=${!this.#open}>
          <label class=${styles.filter}>
            <span class="dl-sr-only">${msg('Filter agents…', {id: 'tracesPanel.filterAgents'})}</span>
            <input class="dl-input ${styles.filterInput}" type="search" data-switcher-filter
                   placeholder=${msg('Filter agents…', {id: 'tracesPanel.filterAgents'})}
                   .value=${this.#filter} autocomplete="off" spellcheck="false"
                   @input=${this.#filtered}>
          </label>
          ${rows.length === 0
            ? html`<p class=${styles.empty}>${this.#filter.trim() === ''
              ? msg('No agent holds those words.', {id: 'tracesPanel.switcherEmpty'})
              : msg('No agent matches that filter.', {id: 'tracesPanel.switcherNoMatch'})}</p>`
            : html`<ul class=${styles.rows} role="list">${rows.map((agent) => this.#row(agent))}</ul>`}
          ${this.paging === null || !this.paging.hasOlder ? nothing : html`
            <div class=${styles.older}>
              <button type="button" class="dl-btn" data-load-older="children"
                      aria-disabled=${this.paging.loading ? 'true' : nothing}
                      @click=${() => { if (!this.paging?.loading) this.#loadOlder(); }}>
                ${this.paging.failed
                  ? msg('Retry loading older children', {id: 'tracesPanel.retryLoadOlder'})
                  : this.paging.loading
                    ? msg('Loading older children…', {id: 'tracesPanel.loadingOlder'})
                    : msg('Load older children', {id: 'tracesPanel.loadOlder'})}
              </button>
            </div>
          `}
        </div>
      </div>
    `;
  }

  /** The closed line: the dot, who, where it stands, and the task it was given. */
  #currentLine(): TemplateResult {
    const row = this.#rowOf(this.selected);
    const main = sameAgent(this.selected, RUN_AGENT);
    const index = (this.entries ?? []).findIndex((child) => sameAgent(agentOfRow(child), this.selected));
    const name = main || index < 0
      ? agentName(this.selected)
      : msg(str`Child ${index + 1}`, {id: 'tracesPanel.childNumber'});
    const state = row === null ? '' : agentStateText(row);
    const elapsed = row === null ? '' : agentElapsed(row, this.#now);
    return html`
      ${row === null ? nothing : agentGlyph(row)}
      <span class=${styles.currentName} data-switcher-name>${name}</span>
      ${state === '' ? nothing : html`<span class=${styles.currentState}>${state}</span>`}
      ${elapsed === '' ? nothing : html`<span class=${styles.currentElapsed}>${elapsed}</span>`}
      ${!row?.objective ? nothing : html`<span class=${styles.currentObjective}>${row.objective}</span>`}
      ${icon('disclosure', {size: 'xs', className: styles.chevron})}
    `;
  }
}

/** Whether a keystroke belongs to a box a reader is writing in. */
function typingIn(target: HTMLElement): boolean {
  return target.isContentEditable
    || target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement;
}

customElements.define('dl-agent-switcher', DlAgentSwitcher);

declare global {
  interface HTMLElementTagNameMap {
    'dl-agent-switcher': DlAgentSwitcher;
  }

  interface HTMLElementEventMap {
    'dl-agent-switch': CustomEvent<AgentSwitchDetail>;
    'dl-agent-load-older': CustomEvent<void>;
  }
}
