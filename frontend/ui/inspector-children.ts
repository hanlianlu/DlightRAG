// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Inspector's Child agents dock: one Run's children as a roster, and the child the reader is watching.
 *
 * A narrow dock shows the list or one child at a time, and a wide one shows them side by side. The
 * roster follows its Run while the dock is open: the Run's own events and the commands the reader
 * sends keep it current, and one clock a second redraws the elapsed times without asking the server.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {
  AgentChildRosterPage,
  AgentChildStatus,
  ChildControlReceipt,
  ChildObservation,
} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {LightElement, NarrowController} from '../lib/lit-host.ts';
import {KeysetPager} from '../lib/paged.ts';
import styles from '../styles/inspector-children.module.css';
import './child-session.ts';
import {childElapsed, childGlyph, childStateText} from './child-status.ts';
import {loadOlderControl} from './load-older.ts';

/** A followed roster refetches at most this often while its run streams child activity. */
const FOLLOW_REFRESH_INTERVAL_MS = 1000;
/** From this width, in rem, the dock shows the list beside the child. */
const WIDE_REM = 40;
const CLOCK_MILLISECONDS = 1000;
/** A Run in one of these statuses is over: the server refuses to steer, continue or cancel its children. */
const OVER_RUN_STATUSES: ReadonlySet<string> = new Set(['succeeded', 'failed', 'cancelled']);

/** What the dock reads and does for one Run's children. */
export interface ChildrenSource {
  readonly runId: string;
  page(cursor: string | null, signal: AbortSignal): Promise<AgentChildRosterPage>;
  observe(childSessionId: string, signal: AbortSignal): Promise<ChildObservation>;
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

export class DlInspectorChildren extends LightElement {
  static properties = {
    source: {attribute: false},
    active: {attribute: false},
    entries: {state: true},
    failed: {state: true},
    picked: {state: true},
    runStatus: {state: true},
  };

  declare source: ChildrenSource | null;
  /** Whether the Inspector is showing this content: a dock that is not does no background work. */
  declare active: boolean;
  /** The roster as loaded, newest first; null until its first page arrives. */
  declare entries: readonly ListedChild[] | null;
  declare failed: boolean;
  /** The child the reader opened. */
  declare picked: string | null;
  /** Where the Run stood at the latest refresh; null before the first, or from a server that does not say. */
  declare runStatus: string | null;

  readonly #narrow = new NarrowController(this, WIDE_REM);
  readonly #pager = new KeysetPager<AgentChildRosterPage>(
    (cursor, signal) => this.source!.page(cursor, signal),
    () => { this.requestUpdate(); },
  );
  /** The newest child: what a wide dock shows until the reader opens another. */
  #newest: string | null = null;
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
      this.#newest = null;
      this.#appended = 0;
      if (this.source) void this.#refresh();
    }
    if (!this.active) this.#cancelFollow();
    const entries = this.entries;
    if (entries && !this.#narrow.narrow && !entries.some((child) => child.childSessionId === this.#newest)) {
      this.#newest = entries[0]?.childSessionId ?? null;
    }
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
    if (!this.source) return;
    // Opening, a retry, and followed activity share one throttle window.
    this.#lastRefresh = performance.now();
    this.failed = false;
    await this.#pager.start((page) => {
      this.entries = listed(page.children);
      this.runStatus = page.runStatus;
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

  /** The child on show: the one the reader opened, else, in a wide dock, the newest. */
  #shown(): string | null {
    return this.picked ?? (this.#narrow.narrow ? null : this.#newest);
  }

  #pick = (event: Event): void => {
    this.picked = (event.currentTarget as HTMLElement).dataset.childSession!;
    // A narrow dock has replaced the list with the child, so the reader lands on its title.
    if (this.#narrow.narrow) {
      void this.updateComplete.then(async () => {
        const session = this.querySelector('dl-child-session');
        await session?.updateComplete;
        session?.focusTitle();
      });
    }
  };

  #back = (): void => {
    const opener = this.picked;
    this.picked = null;
    void this.updateComplete.then(() => {
      if (opener) this.querySelector<HTMLElement>(`[data-child-session="${CSS.escape(opener)}"]`)?.focus();
    });
  };

  protected override render(): TemplateResult | typeof nothing {
    const source = this.source;
    if (!source) return nothing;
    const entries = this.entries;
    if (entries === null || entries.length === 0) {
      return html`<div class=${styles.root}>${this.#unlisted(entries)}</div>`;
    }
    const narrow = this.#narrow.narrow;
    const shown = this.#shown();
    const entry = entries.find((child) => child.childSessionId === shown) ?? null;
    // One clock for the whole render, the child on show included.
    const now = Date.now();
    return html`
      <div class=${styles.root}>
        ${this.#toolbar(entries, narrow && shown !== null)}
        <div class="${styles.body} ${narrow ? '' : styles.wide}">
          <div class=${styles.listPane} ?hidden=${narrow && shown !== null}>
            ${this.failed ? this.#failure() : nothing}
            <ul class=${styles.list} role="list">
              ${repeat(entries, (child) => child.childSessionId, (child) => this.#row(child, child.childSessionId === shown, now))}
            </ul>
            ${loadOlderControl({
              list: 'children',
              pages: this.#pager,
              label: msg('Load older children', {id: 'childrenPanel.loadOlder'}),
              retryLabel: msg('Retry loading older children', {id: 'childrenPanel.retryLoadOlder'}),
              loading: msg('Loading older children…', {id: 'childrenPanel.loadingOlder'}),
              loaded: this.#appended === 1
                ? msg('Loaded 1 older child.', {id: 'childrenPanel.loadedOneOlder'})
                : msg(str`Loaded ${this.#appended} older children.`, {id: 'childrenPanel.loadedOlder'}),
              failed: msg('Older children could not be loaded.', {id: 'childrenPanel.olderFailed'}),
              onLoad: this.#loadOlder,
              rowClass: styles.older,
              buttonClass: 'dl-btn',
            })}
          </div>
          <dl-child-session class=${styles.detail} ?hidden=${narrow && shown === null}
            .source=${source} .childSessionId=${shown ?? ''} .entry=${entry} .now=${now}
            .commandable=${this.#commandable()}
            @dl-child-command-settled=${this.#followRefresh}></dl-child-session>
        </div>
      </div>
    `;
  }

  /** What shows while no child is listed: the load, what stopped it, or the empty roster. */
  #unlisted(entries: readonly ListedChild[] | null): TemplateResult {
    if (this.failed) return this.#failure();
    if (entries === null) {
      return html`<p class=${styles.quiet} role="status">${msg('Loading child agents…', {id: 'childrenPanel.loading'})}</p>`;
    }
    return html`
      <div class=${styles.empty}>
        <h3 class=${styles.emptyTitle}>${msg('No child agents were started', {id: 'childrenPanel.emptyTitle'})}</h3>
        <span class=${styles.emptyBody}>${msg('They appear here when the agent splits a task.', {id: 'childrenPanel.emptyBody'})}</span>
      </div>
    `;
  }

  #failure(): TemplateResult {
    return html`
      <div class=${styles.empty}>
        <p class=${styles.emptyTitle} role="alert">${msg('Child agents could not be loaded.', {id: 'childrenPanel.loadFailed'})}</p>
        <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'childrenPanel.retry'})}</button>
      </div>
    `;
  }

  #toolbar(entries: readonly ListedChild[], detail: boolean): TemplateResult {
    if (detail) {
      return html`
        <div class=${styles.toolbar}>
          <button type="button" class=${styles.back} @click=${this.#back}>
            ${icon('previous', {size: 'sm'})}${msg('All child agents', {id: 'childrenPanel.back'})}
          </button>
        </div>
      `;
    }
    const total = `${entries.length}${this.#pager.hasOlder ? '+' : ''}`;
    const running = entries.filter((child) => child.status === 'running').length;
    return html`
      <div class=${styles.toolbar}>
        <span>${entries.length === 1 && !this.#pager.hasOlder
          ? msg('1 child', {id: 'childrenPanel.oneChild'})
          : msg(str`${total} children`, {id: 'childrenPanel.children'})}</span>
        ${running > 0 ? html`
          <span aria-hidden="true">·</span>
          <span class=${styles.live}>${msg(str`${running} running`, {id: 'childrenPanel.running'})}</span>
        ` : nothing}
      </div>
    `;
  }

  #row(child: ListedChild, selected: boolean, now: number): TemplateResult {
    const meta = [childStateText(child), childElapsed(child, now)].filter(Boolean).join(' · ');
    return html`
      <li>
        <button type="button" class=${styles.row} data-child-session=${child.childSessionId}
                aria-current=${selected ? 'true' : nothing} @click=${this.#pick}>
          ${childGlyph(child)}
          <span class=${styles.text}>
            <span class=${styles.objective}>${child.objective || child.childSessionId}</span>
            <span class=${styles.meta}>
              <span>${meta}</span>
              ${child.pendingQuestions > 0
                ? html`<span class=${styles.pill}>${msg('Question', {id: 'childrenPanel.question'})}</span>`
                : nothing}
            </span>
            ${child.status !== 'running' && child.summary
              ? html`<span class=${styles.summary}>${child.summary}</span>`
              : nothing}
          </span>
        </button>
      </li>
    `;
  }
}

customElements.define('dl-inspector-children', DlInspectorChildren);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-children': DlInspectorChildren;
  }
}
