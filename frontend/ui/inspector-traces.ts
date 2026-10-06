// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Inspector's Child agents dock: one Run's agents as a roster, and the one the reader is watching.
 *
 * The roster is a small tree: the Run's main agent, with its children indented beneath it. A narrow dock
 * shows the list or one agent at a time, and a wide one shows them side by side, opening on the main agent.
 * A Run with no children has no list, only the main agent. The roster follows its Run while the dock is
 * open: the Run's own events and the commands the reader sends keep it current, and one clock a second
 * redraws the elapsed times without asking the server.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {
  ActivityPage,
  AgentChildRosterPage,
  AgentChildStatus,
  AnswerPresentation,
  ChildControlReceipt,
  ChildObservation,
} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {LightElement, NarrowController} from '../lib/lit-host.ts';
import {MAIN_AGENT, mainAgentStatus} from '../lib/main-agent.ts';
import {KeysetPager} from '../lib/paged.ts';
import styles from '../styles/inspector-traces.module.css';
import {agentElapsed, agentGlyph, agentStateText} from './agent-status.ts';
import './agent-session.ts';
import {loadOlderControl} from './load-older.ts';

/** A followed roster refetches at most this often while its run streams child activity. */
const FOLLOW_REFRESH_INTERVAL_MS = 1000;
/** From this width, in rem, the dock shows the list beside the child. */
const WIDE_REM = 40;
const CLOCK_MILLISECONDS = 1000;
/** A Run in one of these statuses is over: the server refuses to steer, continue or cancel its children. */
const OVER_RUN_STATUSES: ReadonlySet<string> = new Set(['succeeded', 'failed', 'cancelled']);

/** What the dock reads and does for one Run's agents. */
export interface TracesSource {
  readonly runId: string;
  /** The Run's main agent as a status row: what the turn it answers says of it now. */
  mainAgent(): AgentChildStatus & {childSessionId: string};
  /** The Run's answer, once it has one: what any agent's Evidence line can open. */
  presentation(): AnswerPresentation | null;
  page(cursor: string | null, signal: AbortSignal): Promise<AgentChildRosterPage>;
  observe(childSessionId: string, signal: AbortSignal): Promise<ChildObservation>;
  /** One page of an agent's transcript: the Run's main agent when `agent` is null. */
  activity(agent: string | null, cursor: string | null, signal: AbortSignal): Promise<ActivityPage>;
  control(
    childSessionId: string,
    action: 'steer' | 'continue' | 'cancel',
    content: string,
    reauthorize?: boolean,
    operationId?: string | null,
    signal?: AbortSignal,
  ): Promise<ChildControlReceipt>;
  reply(requestId: string, content: string, signal?: AbortSignal): Promise<ChildControlReceipt>;
}

/** A roster row the reader can open: the wire leaves the id optional, and a row without one has nothing to open. */
type ListedChild = AgentChildStatus & {childSessionId: string};

function listed(children: readonly AgentChildStatus[]): ListedChild[] {
  return children.filter((child): child is ListedChild => Boolean(child.childSessionId));
}

export class DlInspectorTraces extends LightElement {
  static properties = {
    source: {attribute: false},
    active: {attribute: false},
    entries: {state: true},
    lead: {state: true},
    failed: {state: true},
    picked: {state: true},
    runStatus: {state: true},
  };

  declare source: TracesSource | null;
  /** Whether the Inspector is showing this content: a dock that is not does no background work. */
  declare active: boolean;
  /** The roster as loaded, newest first; null until its first page arrives. */
  declare entries: readonly ListedChild[] | null;
  /** The main agent's row, drawn again from its turn at every refresh. */
  declare lead: ListedChild;
  declare failed: boolean;
  /** The agent the reader opened: a child's id, or `MAIN_AGENT`. */
  declare picked: string | null;
  /** Where the Run stood at the latest refresh; null before the first, or from a server that does not say. */
  declare runStatus: string | null;

  readonly #narrow = new NarrowController(this, WIDE_REM);
  readonly #pager = new KeysetPager<AgentChildRosterPage>(
    (cursor, signal) => this.source!.page(cursor, signal),
    () => { this.requestUpdate(); },
  );
  #appended = 0;
  #refreshing = false;
  #refreshQueued = false;
  #lastRefresh = Number.NEGATIVE_INFINITY;
  #followTimer: ReturnType<typeof setTimeout> | null = null;
  #clock: ReturnType<typeof setInterval> | null = null;

  constructor() {
    super();
    this.source = null;
    this.active = false;
    this.entries = null;
    this.lead = mainAgentStatus(undefined);
    this.failed = false;
    this.picked = null;
    this.runStatus = null;
  }

  override disconnectedCallback(): void {
    this.#cancelFollow();
    this.#stopClock();
    super.disconnectedCallback();
  }

  /** Refresh for child activity of the followed run: at once, then at most once per interval. */
  refreshIfFollowing(runId: string): void {
    if (!this.#following(runId) || this.#followTimer !== null) return;
    const wait = this.#lastRefresh + FOLLOW_REFRESH_INTERVAL_MS - performance.now();
    if (wait <= 0) {
      this.#followRefresh();
      return;
    }
    // One trailing refresh carries every activity inside the interval.
    this.#followTimer = setTimeout(() => {
      this.#followTimer = null;
      if (this.#following(runId)) this.#followRefresh();
    }, wait);
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('source')) {
      this.#pager.reset(null);
      this.#cancelFollow();
      this.#refreshQueued = false;
      this.entries = null;
      this.failed = false;
      this.picked = null;
      this.runStatus = null;
      this.#appended = 0;
      if (this.source) {
        this.lead = this.source.mainAgent();
        void this.#refresh();
      }
    }
    if (!this.active) this.#cancelFollow();
  }

  protected override updated(): void {
    const ticking = this.active
      && Boolean(this.entries?.some((child) => child.status === 'running' || child.pendingQuestions > 0));
    if (ticking && this.#clock === null) {
      this.#clock = setInterval(() => { this.requestUpdate(); }, CLOCK_MILLISECONDS);
    } else if (!ticking) {
      this.#stopClock();
    }
  }

  #stopClock(): void {
    if (this.#clock !== null) clearInterval(this.#clock);
    this.#clock = null;
  }

  #following(runId: string): boolean {
    return this.active && this.source?.runId === runId;
  }

  #followRefresh(): void {
    this.#refreshQueued = true;
    void this.#flushRefresh();
  }

  #cancelFollow(): void {
    if (this.#followTimer !== null) clearTimeout(this.#followTimer);
    this.#followTimer = null;
  }

  async #flushRefresh(): Promise<void> {
    if (this.#refreshing) return;
    this.#refreshing = true;
    try {
      while (this.#refreshQueued) {
        this.#refreshQueued = false;
        await this.#refresh();
      }
    } finally {
      this.#refreshing = false;
      if (this.#refreshQueued) void this.#flushRefresh();
    }
  }

  async #refresh(): Promise<void> {
    const source = this.source;
    if (!source) return;
    // Opening, a retry, and followed activity share one throttle window.
    this.#lastRefresh = performance.now();
    this.failed = false;
    await this.#pager.start((page) => {
      this.entries = listed(page.children);
      this.runStatus = page.runStatus;
      this.lead = source.mainAgent();
    }, () => {
      this.failed = true;
    });
  }

  #retry = (): void => {
    void this.#refresh();
  };

  #loadOlder = (): void => {
    void this.#pager.loadNext((page) => {
      const known = new Set(this.entries?.map((child) => child.childSessionId));
      const older = listed(page.children).filter((child) => {
        if (known.has(child.childSessionId)) return false;
        known.add(child.childSessionId);
        return true;
      });
      this.#appended = older.length;
      this.entries = [...(this.entries ?? []), ...older];
    });
  };

  /** Whether the Run can still be steered: until a refresh says it is over, it can. */
  #commandable(): boolean {
    return this.runStatus === null || !OVER_RUN_STATUSES.has(this.runStatus);
  }

  /** The agent on show: the one the reader opened, else the main agent, unless a narrow dock has a list to
   * show first. Nothing shows before the roster is known. */
  #shown(): string | null {
    if (this.entries === null) return null;
    if (this.picked !== null) return this.picked;
    return this.#narrow.narrow && this.#hasChildren() ? null : MAIN_AGENT;
  }

  #hasChildren(): boolean {
    return (this.entries?.length ?? 0) > 0;
  }

  #pick = (event: Event): void => {
    this.picked = (event.currentTarget as HTMLElement).dataset.agentSession!;
    // A narrow dock has replaced the list with the agent, so the reader lands on its title.
    if (this.#narrow.narrow) {
      void this.updateComplete.then(async () => {
        const session = this.querySelector('dl-agent-session');
        await session?.updateComplete;
        session?.focusTitle();
      });
    }
  };

  #back = (): void => {
    const opener = this.picked;
    this.picked = null;
    void this.updateComplete.then(() => {
      if (opener !== null) this.querySelector<HTMLElement>(`[data-agent-session="${CSS.escape(opener)}"]`)?.focus();
    });
  };

  protected override render(): TemplateResult | typeof nothing {
    const source = this.source;
    if (!source) return nothing;
    const entries = this.entries ?? [];
    const family = entries.length > 0;
    const narrow = this.#narrow.narrow;
    const shown = this.#shown();
    const entry = shown === MAIN_AGENT ? this.lead : entries.find((child) => child.childSessionId === shown) ?? null;
    // A narrow dock that shows one agent has put the list away.
    const detail = narrow && family && shown !== null;
    // One clock for the whole render, the agent on show included.
    const now = Date.now();
    return html`
      <div class=${styles.root}>
        ${this.failed ? this.#failure() : nothing}
        ${family ? this.#toolbar(entries, detail) : nothing}
        <div class="${styles.body} ${narrow ? '' : styles.wide}">
          ${family ? html`
            <div class=${styles.listPane} ?hidden=${detail}>
              <ul class=${styles.list} role="list">
                <li>
                  ${this.#row(this.lead, shown === MAIN_AGENT, now)}
                  <ul class=${styles.tree} role="list">
                    ${repeat(entries, (child) => child.childSessionId, (child) => html`
                      <li>${this.#row(child, child.childSessionId === shown, now)}</li>
                    `)}
                  </ul>
                </li>
              </ul>
              ${loadOlderControl({
                list: 'children',
                pages: this.#pager,
                label: msg('Load older children', {id: 'tracesPanel.loadOlder'}),
                retryLabel: msg('Retry loading older children', {id: 'tracesPanel.retryLoadOlder'}),
                loading: msg('Loading older children…', {id: 'tracesPanel.loadingOlder'}),
                loaded: this.#appended === 1
                  ? msg('Loaded 1 older child.', {id: 'tracesPanel.loadedOneOlder'})
                  : msg(str`Loaded ${this.#appended} older children.`, {id: 'tracesPanel.loadedOlder'}),
                failed: msg('Older children could not be loaded.', {id: 'tracesPanel.olderFailed'}),
                onLoad: this.#loadOlder,
                rowClass: styles.older,
                buttonClass: 'dl-btn',
              })}
            </div>
          ` : nothing}
          ${this.entries === null ? html`
            <p class=${styles.quiet} role="status">${msg('Loading agents…', {id: 'tracesPanel.loading'})}</p>
          ` : html`
            <dl-agent-session class=${styles.detail} ?hidden=${shown === null}
              .source=${source} .childSessionId=${shown ?? ''} .entry=${entry} .now=${now}
              .commandable=${this.#commandable()}
              @dl-child-command-settled=${this.#followRefresh}></dl-agent-session>
          `}
        </div>
      </div>
    `;
  }

  #failure(): TemplateResult {
    return html`
      <div class=${styles.empty}>
        <p class=${styles.emptyTitle} role="alert">${msg('Agents could not be loaded.', {id: 'tracesPanel.loadFailed'})}</p>
        <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'tracesPanel.retry'})}</button>
      </div>
    `;
  }

  #toolbar(entries: readonly ListedChild[], detail: boolean): TemplateResult {
    if (detail) {
      return html`
        <div class=${styles.toolbar}>
          <button type="button" class=${styles.back} @click=${this.#back}>
            ${icon('previous', {size: 'sm'})}${msg('All agents', {id: 'tracesPanel.back'})}
          </button>
        </div>
      `;
    }
    const total = `${entries.length}${this.#pager.hasOlder ? '+' : ''}`;
    const running = entries.filter((child) => child.status === 'running').length;
    return html`
      <div class=${styles.toolbar}>
        <span>${entries.length === 1 && !this.#pager.hasOlder
          ? msg('1 child', {id: 'tracesPanel.oneChild'})
          : msg(str`${total} children`, {id: 'tracesPanel.children'})}</span>
        ${running > 0 ? html`
          <span aria-hidden="true">·</span>
          <span class=${styles.live}>${msg(str`${running} running`, {id: 'tracesPanel.running'})}</span>
        ` : nothing}
      </div>
    `;
  }

  /** One agent's row, the main agent's too: it is named by what it is, a child by its task. */
  #row(child: ListedChild, selected: boolean, now: number): TemplateResult {
    const main = child.childSessionId === MAIN_AGENT;
    const meta = [agentStateText(child), agentElapsed(child, now)].filter(Boolean).join(' · ');
    return html`
      <button type="button" class="${styles.row} ${main ? styles.leadRow : ''}" data-agent-session=${child.childSessionId}
              aria-current=${selected ? 'true' : nothing} @click=${this.#pick}>
        ${agentGlyph(child)}
        <span class=${styles.text}>
          <span class=${styles.objective}>${main ? msg('Main agent', {id: 'tracesPanel.lead'}) : child.objective || child.childSessionId}</span>
          <span class=${styles.meta}>
            <span>${meta}</span>
            ${child.pendingQuestions > 0
              ? html`<span class=${styles.pill}>${msg('Question', {id: 'tracesPanel.question'})}</span>`
              : nothing}
          </span>
          ${child.status !== 'running' && child.summary
            ? html`<span class=${styles.summary}>${child.summary}</span>`
            : nothing}
        </span>
      </button>
    `;
  }
}

customElements.define('dl-inspector-traces', DlInspectorTraces);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-traces': DlInspectorTraces;
  }
}
